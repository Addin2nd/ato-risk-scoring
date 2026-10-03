"""Synthetic login telemetry generator.

Produces an authentication log modelled on the schema of the public
Risk-Based Authentication (RBA) dataset (Wiefling et al.), plus a device id and
ASN type, so the pipeline can be moved to real data with a thin column mapping
(see README). Synthetic data keeps the repo fully reproducible and gives
*known* attack scenarios to evaluate against.

Legit behaviour deliberately includes the noise that makes ATO detection hard:
shared mobile CGNAT IPs, shared office IPs, holidays abroad, phone upgrades,
commercial VPN users, and password typos.

Attack scenarios (ground truth):
  * credential_stuffing - leaked-credential lists replayed from proxy/hosting IPs
  * targeted_ato        - phished credentials used from a new device/location
  * brute_force         - password guessing against one account from one IP
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import reference as ref

COLUMNS = [
    "login_ts", "user_id", "ip", "country", "region", "city", "asn", "asn_name", "asn_type",
    "user_agent", "browser", "os", "device_type", "device_id", "login_successful",
    # ---- ground truth / analysis-only columns (never used as model features) ----
    "is_attack_ip", "is_account_takeover", "scenario",
]


@dataclass(frozen=True)
class GenConfig:
    n_users: int = 2500
    start: str = "2026-05-01"
    days: int = 120
    seed: int = 42
    n_stuffing_campaigns: int = 8
    n_targeted_victims: int = 110
    n_bruteforce_victims: int = 20


class _IPPool:
    """Deterministic fake IPs: one /16 per (asn, city), CGNAT pools for mobile."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._cgnat: dict[tuple[int, str], list[str]] = {}

    @staticmethod
    def _prefix(asn: int, city: str) -> tuple[int, int]:
        h = hashlib.md5(f"{asn}-{city}".encode()).digest()
        return 11 + h[0] % 200, h[1]

    def random_ip(self, asn: int, city: str) -> str:
        a, b = self._prefix(asn, city)
        c, d = self.rng.integers(0, 256), self.rng.integers(1, 255)
        return f"{a}.{b}.{c}.{d}"

    def cgnat_ip(self, asn: int, city: str) -> str:
        """Mobile carriers put thousands of subscribers behind a small NAT pool."""
        key = (asn, city)
        if key not in self._cgnat:
            self._cgnat[key] = [self.random_ip(asn, city) for _ in range(24)]
        pool = self._cgnat[key]
        return pool[self.rng.integers(len(pool))]


def _device_id(rng: np.random.Generator) -> str:
    return "d_" + "".join(rng.choice(list("0123456789abcdef"), 12))


class _Writer:
    def __init__(self):
        self.rows: list[tuple] = []

    def add(self, ts, user_id, ip, city, asn, platform, device_id, success,
            attack_ip=0, ato=0, scenario="normal"):
        region, country, _, _ = ref.CITIES[city]
        asn_name, asn_type, _ = ref.ASNS[asn]
        browser, bver, os_name, os_ver, dtype = platform
        self.rows.append((
            ts, user_id, ip, country, region, city, asn, asn_name, asn_type,
            ref.user_agent_string(*platform), f"{browser} {bver}", f"{os_name} {os_ver}".strip(),
            dtype, device_id, int(success), int(attack_ip), int(ato), scenario,
        ))


def _pick(rng, items, weights=None):
    idx = rng.choice(len(items), p=weights)
    return items[idx]


# --------------------------------------------------------------------------------------
# Legit users
# --------------------------------------------------------------------------------------
def _simulate_users(cfg: GenConfig, rng: np.random.Generator, ips: _IPPool, out: _Writer):
    start = pd.Timestamp(cfg.start)
    n_companies = 160
    company_city = rng.choice(ref.ID_HOME_CITIES, n_companies, p=ref.ID_HOME_WEIGHTS)
    company_ips = {c: (int(rng.choice(ref.ID_FIXED)), []) for c in range(n_companies)}
    for c, (asn, lst) in company_ips.items():
        lst.extend(ips.random_ip(asn, company_city[c]) for _ in range(2))

    profiles = []
    for u in range(cfg.n_users):
        uid = f"u{u:05d}"
        home = rng.choice(ref.ID_HOME_CITIES, p=ref.ID_HOME_WEIGHTS)
        fixed_asn = int(rng.choice(ref.ID_FIXED))
        mobile_asn = int(rng.choice(ref.ID_MOBILE, p=ref.ID_MOBILE_WEIGHTS))
        local = np.flatnonzero(company_city == home)  # people work in the city they live in
        company = int(rng.choice(local)) if len(local) and rng.random() < 0.55 else None

        devices = [(_pick(rng, ref.MOBILE_PLATFORMS, ref.MOBILE_WEIGHTS), _device_id(rng))]
        if rng.random() < 0.6:
            devices.append((_pick(rng, ref.DESKTOP_PLATFORMS, ref.DESKTOP_WEIGHTS), _device_id(rng)))
        if rng.random() < 0.15:
            devices.append((_pick(rng, ref.MOBILE_PLATFORMS, ref.MOBILE_WEIGHTS), _device_id(rng)))

        rate = float(np.clip(rng.lognormal(np.log(0.8), 0.7), 0.08, 6.0))
        pref_hour = float(np.clip(rng.normal(19, 3.5), 6, 23.5))
        hour_sd = float(rng.uniform(1.5, 4.0))

        # Legit "anomalies"
        trip = None
        if rng.random() < 0.18:
            t0 = int(rng.integers(3, cfg.days - 8))
            abroad = rng.random() < 0.35
            dest = (rng.choice(ref.TRAVEL_ABROAD) if abroad
                    else rng.choice([c for c in ref.ID_HOME_CITIES if c != home]))
            trip = (t0, t0 + int(rng.integers(2, 9)), str(dest), abroad)
        new_phone_day = int(rng.integers(5, cfg.days)) if rng.random() < 0.15 else None
        new_phone = ((_pick(rng, ref.MOBILE_PLATFORMS, ref.MOBILE_WEIGHTS), _device_id(rng))
                     if new_phone_day is not None else None)
        vpn_user = rng.random() < 0.03
        vpn_city = str(rng.choice(["Singapore", "Amsterdam", "Ashburn", "Tokyo"]))
        vpn_asn = int(rng.choice([a for a in ref.VPN_PROVIDERS if vpn_city in ref.ASNS[a][2]]))

        home_ip = ips.random_ip(fixed_asn, home)
        next_rotation = int(rng.integers(7, 45))

        weekday = np.array([(start + pd.Timedelta(days=d)).dayofweek for d in range(cfg.days)])
        day_rate = rate * np.where(weekday >= 5, 0.8, 1.05)
        counts = rng.poisson(day_rate)

        for day in range(cfg.days):
            if day >= next_rotation:  # residential DHCP lease renewal
                home_ip = ips.random_ip(fixed_asn, home)
                next_rotation = day + int(rng.integers(7, 45))
            for _ in range(counts[day]):
                hour = (rng.normal(pref_hour, hour_sd) if rng.random() < 0.85
                        else rng.uniform(5, 24)) % 24
                ts = start + pd.Timedelta(days=day, hours=float(hour))

                dev_pool = list(devices)
                if new_phone is not None and day >= new_phone_day:
                    dev_pool[0] = new_phone
                weights = np.array([0.65] + [0.35 / max(1, len(dev_pool) - 1)] * (len(dev_pool) - 1))
                weights = weights / weights.sum() if len(dev_pool) > 1 else np.array([1.0])
                platform, dev_id = dev_pool[rng.choice(len(dev_pool), p=weights)]
                mobile = platform[4] == "mobile"
                scenario = "normal"

                on_trip = trip is not None and trip[0] <= day < trip[1]
                if on_trip:
                    _, _, dest, abroad = trip
                    scenario = "travel_abroad" if abroad else "travel_domestic"
                    if abroad and rng.random() < 0.45:   # roaming: IP still geolocates home
                        city, asn, ip = home, mobile_asn, ips.cgnat_ip(mobile_asn, home)
                    elif abroad:                          # hotel wifi / local SIM
                        city = dest
                        asn = ref.foreign_consumer_asn(dest)
                        ip = ips.random_ip(asn, dest)
                    else:
                        city, asn = dest, mobile_asn
                        ip = ips.cgnat_ip(asn, dest)
                elif vpn_user and rng.random() < 0.35:
                    scenario = "vpn"
                    city, asn, ip = vpn_city, vpn_asn, ips.random_ip(vpn_asn, vpn_city)
                elif mobile and rng.random() < 0.6:
                    city, asn, ip = home, mobile_asn, ips.cgnat_ip(mobile_asn, home)
                elif (not mobile) and company is not None and rng.random() < 0.4 and ts.dayofweek < 5:
                    asn, lst = company_ips[company]
                    city, ip = str(company_city[company]), lst[rng.integers(len(lst))]
                else:
                    city, asn, ip = home, fixed_asn, home_ip

                if new_phone is not None and dev_id == new_phone[1] and day < new_phone_day + 3:
                    scenario = "new_device"

                # Typos: 1-2 failures right before the successful login
                if rng.random() < 0.05:
                    for k in range(int(rng.integers(1, 3))):
                        out.add(ts - pd.Timedelta(seconds=float(rng.uniform(8, 50) * (k + 1))),
                                uid, ip, city, asn, platform, dev_id, False, scenario="typo")
                if rng.random() < 0.002:  # forgot password - gives up
                    for k in range(int(rng.integers(3, 7))):
                        out.add(ts + pd.Timedelta(seconds=30 * k), uid, ip, city, asn, platform,
                                dev_id, False, scenario="forgot_password")
                    continue
                out.add(ts, uid, ip, city, asn, platform, dev_id, True, scenario=scenario)

        profiles.append({"user_id": uid, "home_city": home})
    return pd.DataFrame(profiles)


# --------------------------------------------------------------------------------------
# Attacks
# --------------------------------------------------------------------------------------
def _attacker_node(rng, ips: _IPPool, residential_share: float):
    """An attacker egress: hosting/VPN IP, or a residential proxy (incl. Indonesian)."""
    if rng.random() < residential_share:
        if rng.random() < 0.6:
            asn = int(rng.choice(ref.ID_MOBILE + ref.ID_FIXED))
            city = str(rng.choice(ref.ID_HOME_CITIES, p=ref.ID_HOME_WEIGHTS))
            ip = ips.cgnat_ip(asn, city) if ref.ASNS[asn][1] == "mobile" else ips.random_ip(asn, city)
        else:
            city = str(rng.choice(["Hanoi", "Manila", "Mumbai", "Sao Paulo", "Lagos", "Moscow"]))
            asn = ref.foreign_consumer_asn(city)
            ip = ips.random_ip(asn, city)
    else:
        asn = int(rng.choice(ref.HOSTING))
        city = str(rng.choice(ref.ASNS[asn][2]))
        ip = ips.random_ip(asn, city)
    return ip, city, asn


def _credential_stuffing(cfg, rng, ips, out, users):
    start = pd.Timestamp(cfg.start)
    for c in range(cfg.n_stuffing_campaigns):
        t0 = start + pd.Timedelta(days=float(rng.uniform(10, cfg.days - 4)))
        dur_h = float(rng.uniform(4, 40))
        nodes = [_attacker_node(rng, ips, residential_share=0.35) for _ in range(int(rng.integers(25, 160)))]
        bot_ua = rng.random() < 0.4
        targets = rng.choice(users, size=min(len(users), int(rng.integers(250, 700))), replace=False)
        reuse_rate = rng.uniform(0.03, 0.06)  # share of targets whose leaked password still works
        for uid in targets:
            hit = rng.random() < reuse_rate
            ts = t0 + pd.Timedelta(hours=float(rng.uniform(0, dur_h)))
            platform = (_pick(rng, ref.AUTOMATION_PLATFORMS) if bot_ua
                        else _pick(rng, ref.DESKTOP_PLATFORMS + ref.MOBILE_PLATFORMS))
            dev = _device_id(rng)
            ip, city, asn = nodes[rng.integers(len(nodes))]
            n_try = 1 if hit else int(rng.integers(1, 4))
            for k in range(n_try):
                success = hit and k == n_try - 1
                out.add(ts + pd.Timedelta(seconds=20 * k), uid, ip, city, asn, platform, dev,
                        success, attack_ip=1, ato=int(success), scenario="credential_stuffing")
            if hit and rng.random() < 0.5:   # attacker returns to monetise the account
                ip2, city2, asn2 = _attacker_node(rng, ips, residential_share=0.7)
                plat2 = _pick(rng, ref.DESKTOP_PLATFORMS + ref.MOBILE_PLATFORMS)
                for _ in range(int(rng.integers(1, 4))):
                    ts2 = ts + pd.Timedelta(hours=float(rng.uniform(2, 96)))
                    out.add(ts2, uid, ip2, city2, asn2, plat2, dev, True, attack_ip=1, ato=1,
                            scenario="credential_stuffing")


def _targeted_ato(cfg, rng, ips, out, users):
    start = pd.Timestamp(cfg.start)
    victims = rng.choice(users, size=cfg.n_targeted_victims, replace=False)
    for uid in victims:
        t0 = start + pd.Timedelta(days=float(rng.uniform(10, cfg.days - 1)))
        style = rng.choice(["foreign", "domestic_stealthy", "vpn"], p=[0.4, 0.4, 0.2])
        if style == "foreign":
            city = str(rng.choice(["Lagos", "Moscow", "Sao Paulo", "Hanoi", "Manila", "Mumbai",
                                   "Amsterdam", "Ashburn"]))
            if rng.random() < 0.5 and any(city in ref.ASNS[a][2] for a in ref.HOSTING):
                asn = int(rng.choice([a for a in ref.HOSTING if city in ref.ASNS[a][2]]))
            else:
                asn = ref.foreign_consumer_asn(city) if city not in ("Amsterdam", "Ashburn") else \
                    {"Amsterdam": 1136, "Ashburn": 7922}[city]
            ip = ips.random_ip(asn, city)
            hour_jitter = float(rng.uniform(0, 24))
        elif style == "domestic_stealthy":
            # same country, mobile carrier (shared CGNAT), ordinary phone, normal hours
            city = str(rng.choice(ref.ID_HOME_CITIES, p=ref.ID_HOME_WEIGHTS))
            asn = int(rng.choice(ref.ID_MOBILE, p=ref.ID_MOBILE_WEIGHTS))
            ip = ips.cgnat_ip(asn, city)
            hour_jitter = float(rng.uniform(9, 23))
        else:
            asn = int(rng.choice(ref.VPN_PROVIDERS))
            city = str(rng.choice([c for c in ref.ASNS[asn][2] if c in ("Singapore", "Amsterdam", "Tokyo", "Ashburn")]
                                  or ref.ASNS[asn][2]))
            ip = ips.random_ip(asn, city)
            hour_jitter = float(rng.uniform(0, 24))
        platform = (_pick(rng, ref.MOBILE_PLATFORMS, ref.MOBILE_WEIGHTS) if style == "domestic_stealthy"
                    else _pick(rng, ref.DESKTOP_PLATFORMS + ref.MOBILE_PLATFORMS))
        dev = _device_id(rng)
        base = t0.normalize() + pd.Timedelta(hours=hour_jitter)
        if rng.random() < 0.3:
            out.add(base - pd.Timedelta(seconds=40), uid, ip, city, asn, platform, dev, False,
                    attack_ip=1, scenario="targeted_ato")
        for s in range(int(rng.integers(1, 5))):
            ts = base + pd.Timedelta(hours=float(s * rng.uniform(1, 30)))
            out.add(ts, uid, ip, city, asn, platform, dev, True, attack_ip=1, ato=1, scenario="targeted_ato")


def _brute_force(cfg, rng, ips, out, users):
    start = pd.Timestamp(cfg.start)
    for uid in rng.choice(users, size=cfg.n_bruteforce_victims, replace=False):
        t0 = start + pd.Timedelta(days=float(rng.uniform(10, cfg.days - 1)))
        ip, city, asn = _attacker_node(rng, ips, residential_share=0.2)
        platform = _pick(rng, ref.AUTOMATION_PLATFORMS)
        dev = _device_id(rng)
        n = int(rng.integers(25, 160))
        span = float(rng.uniform(600, 5400))
        cracked = rng.random() < 0.3
        for k in range(n):
            success = cracked and k == n - 1
            out.add(t0 + pd.Timedelta(seconds=span * k / n), uid, ip, city, asn, platform, dev,
                    success, attack_ip=1, ato=int(success), scenario="brute_force")


def generate(cfg: GenConfig = GenConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, user_profiles). Events are sorted by time."""
    rng = np.random.default_rng(cfg.seed)
    ips = _IPPool(rng)
    out = _Writer()
    profiles = _simulate_users(cfg, rng, ips, out)
    users = profiles["user_id"].to_numpy()
    _credential_stuffing(cfg, rng, ips, out, users)
    _targeted_ato(cfg, rng, ips, out, users)
    _brute_force(cfg, rng, ips, out, users)

    df = pd.DataFrame(out.rows, columns=COLUMNS)
    end = pd.Timestamp(cfg.start) + pd.Timedelta(days=cfg.days)
    df = df[df["login_ts"] < end]
    df["login_ts"] = df["login_ts"].dt.floor("s")
    df = df.sort_values(["login_ts", "user_id"], kind="mergesort").reset_index(drop=True)
    return df, profiles
