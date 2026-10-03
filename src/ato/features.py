"""Point-in-time behavioural features for login risk scoring.

Every feature for an event uses ONLY events strictly earlier in the stream
(windows are left-closed). This mirrors what a real-time scoring service can
know at the moment a password has been verified, and prevents label leakage.

Feature families
----------------
user baseline : is the country / city / ASN / IP / device / platform new for this user?
velocity      : distance & implied travel speed from the previous successful login
temporal      : deviation from the user's usual login hour, time since last login
pressure      : recent failed logins against the user, from the IP, and from the ASN
reputation    : hosting/VPN ASN, automation user-agent, global rarity of the country
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import reference as ref

HOUR = 3600
DAY = 24 * HOUR

FEATURES = [
    # user baseline
    "new_country", "new_city", "new_asn", "new_ip", "new_device", "new_platform",
    "user_prior_logins", "user_tenure_days",
    # velocity
    "dist_prev_km", "geo_velocity_kmh", "dist_first_km",
    # temporal
    "hours_since_prev", "hour_deviation",
    # pressure
    "user_fails_1h", "user_fails_24h", "ip_attempts_1h", "ip_fails_24h", "ip_fail_ratio_24h",
    "ip_distinct_users_24h", "ip_users_prior", "asn_fails_1h",
    # reputation / context
    "is_hosting_asn", "is_mobile_asn", "is_foreign", "ua_automation", "is_desktop",
    "country_global_share",
]


def _window_sum(group: np.ndarray, t: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """Sum of `values` over rows of the same group with time in [t - window, t), excluding self.

    Arrays must be sorted by (group, t). Vectorised via a composite sort key.
    """
    span = int(t.max()) + window + 1
    key = group.astype(np.int64) * span + t
    lo = np.searchsorted(key, key - window, side="left")
    cs = np.concatenate([[0], np.cumsum(values, dtype=np.float64)])
    idx = np.arange(len(key))
    return cs[idx] - cs[lo]


def _grouped_window(df: pd.DataFrame, by: str | list[str], col: str, window: int) -> pd.Series:
    order = df.sort_values(by=[*([by] if isinstance(by, str) else by), "t"], kind="mergesort").index
    sub = df.loc[order]
    codes = sub.groupby(by, sort=False).ngroup().to_numpy()
    out = _window_sum(codes, sub["t"].to_numpy(np.int64), sub[col].to_numpy(np.float64), window)
    return pd.Series(out, index=order).reindex(df.index)


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Return `events` with feature columns appended (same row order, time-sorted)."""
    df = events.sort_values("login_ts", kind="mergesort").reset_index(drop=True).copy()
    t0 = df["login_ts"].min()
    df["t"] = ((df["login_ts"] - t0).dt.total_seconds()).astype(np.int64)
    succ = df["login_successful"].astype(int)
    df["fail"] = 1 - succ
    df["one"] = 1
    lat = df["city"].map(lambda c: ref.CITIES[c][2])
    lon = df["city"].map(lambda c: ref.CITIES[c][3])
    df["platform"] = df["browser"] + "|" + df["os"]

    # ---- user baseline: seen in a PRIOR successful login? ------------------------------
    for col, name in [("country", "new_country"), ("city", "new_city"), ("asn", "new_asn"),
                      ("ip", "new_ip"), ("device_id", "new_device"), ("platform", "new_platform")]:
        prior = df.groupby(["user_id", col], sort=False)["login_successful"].cumsum() - succ
        df[name] = (prior == 0).astype(int)
    g_user = df.groupby("user_id", sort=False)
    df["user_prior_logins"] = g_user["login_successful"].cumsum() - succ

    # ---- previous successful login (strictly earlier) ----------------------------------
    prev = pd.DataFrame({
        "user_id": df["user_id"],
        "ts": df["t"].where(succ == 1),
        "lat": lat.where(succ == 1),
        "lon": lon.where(succ == 1),
    })
    gp = prev.groupby("user_id", sort=False)
    prev_ts = gp["ts"].shift(1)
    prev_lat = gp["lat"].shift(1)
    prev_lon = gp["lon"].shift(1)
    tmp = pd.DataFrame({"user_id": df["user_id"], "ts": prev_ts, "lat": prev_lat, "lon": prev_lon})
    tmp[["ts", "lat", "lon"]] = tmp.groupby("user_id", sort=False)[["ts", "lat", "lon"]].ffill()

    hours = (df["t"] - tmp["ts"]) / HOUR
    dist = ref.haversine_km(tmp["lat"], tmp["lon"], lat, lon)
    df["hours_since_prev"] = hours.fillna(24 * 30).clip(upper=24 * 30)
    df["dist_prev_km"] = dist.fillna(0)
    df["geo_velocity_kmh"] = (dist / hours.clip(lower=0.25)).fillna(0)

    first = prev.dropna(subset=["ts"]).groupby("user_id").first()
    f_ts = df["user_id"].map(first["ts"])
    valid = f_ts < df["t"]
    d_first = ref.haversine_km(df["user_id"].map(first["lat"]), df["user_id"].map(first["lon"]), lat, lon)
    df["dist_first_km"] = np.where(valid, d_first, 0.0)
    df["user_tenure_days"] = np.where(valid, (df["t"] - f_ts) / DAY, 0.0)

    # ---- usual login hour (circular mean of prior successful logins) --------------------
    hour = df["login_ts"].dt.hour + df["login_ts"].dt.minute / 60
    ang = 2 * np.pi * hour / 24
    s = (np.sin(ang) * succ)
    c = (np.cos(ang) * succ)
    ps = s.groupby(df["user_id"], sort=False).cumsum() - s
    pc = c.groupby(df["user_id"], sort=False).cumsum() - c
    mean_hour = (np.arctan2(ps, pc) % (2 * np.pi)) * 24 / (2 * np.pi)
    diff = np.abs(hour - mean_hour)
    df["hour_deviation"] = np.where(df["user_prior_logins"] > 0, np.minimum(diff, 24 - diff), 0.0)

    # ---- pressure: recent failures (left-closed windows) --------------------------------
    df["user_fails_1h"] = _grouped_window(df, "user_id", "fail", HOUR)
    df["user_fails_24h"] = _grouped_window(df, "user_id", "fail", DAY)
    df["ip_attempts_1h"] = _grouped_window(df, "ip", "one", HOUR)
    attempts_24 = _grouped_window(df, "ip", "one", DAY)
    df["ip_fails_24h"] = _grouped_window(df, "ip", "fail", DAY)
    df["ip_fail_ratio_24h"] = np.where(attempts_24 > 0, df["ip_fails_24h"] / attempts_24.clip(lower=1), 0.0)
    df["asn_fails_1h"] = _grouped_window(df, "asn", "fail", HOUR)

    # distinct users per IP in 24h: count (ip, user) pairs whose previous co-occurrence
    # is older than 24h (approximate sliding distinct count, exact for first sightings)
    pair_prev = df.groupby(["ip", "user_id"], sort=False)["t"].shift(1)
    df["pair_new_24h"] = ((df["t"] - pair_prev) > DAY).astype(int) | pair_prev.isna().astype(int)
    df["ip_distinct_users_24h"] = _grouped_window(df, "ip", "pair_new_24h", DAY)
    first_pair = (~df.duplicated(["ip", "user_id"])).astype(int)
    df["ip_users_prior"] = first_pair.groupby(df["ip"], sort=False).cumsum() - first_pair

    # ---- reputation / context ----------------------------------------------------------
    df["is_hosting_asn"] = (df["asn_type"] == "hosting").astype(int)
    df["is_mobile_asn"] = (df["asn_type"] == "mobile").astype(int)
    df["is_foreign"] = (df["country"] != "ID").astype(int)
    df["ua_automation"] = df["browser"].str.split(" ").str[0].isin(
        {"Python", "HeadlessChrome", "curl"}).astype(int)
    df["is_desktop"] = (df["device_type"] == "desktop").astype(int)
    c_prior = df.groupby("country", sort=False)["login_successful"].cumsum() - succ
    total_prior = succ.cumsum() - succ
    df["country_global_share"] = np.where(total_prior > 0, c_prior / total_prior.clip(lower=1), 0.0)

    return df.drop(columns=["t", "fail", "one", "pair_new_24h", "platform"])


def scoring_frame(feat: pd.DataFrame, warmup_days: int = 14) -> pd.DataFrame:
    """Rows the risk engine scores: successful logins after the baseline warm-up period."""
    start = feat["login_ts"].min().normalize() + pd.Timedelta(days=warmup_days)
    return feat[(feat["login_successful"] == 1) & (feat["login_ts"] >= start)].copy()
