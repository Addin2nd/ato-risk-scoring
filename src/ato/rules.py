"""Deterministic detection rules (the "SIEM correlation rule" baseline) and reason codes.

Rules serve two purposes:
1. A transparent baseline that the ML layers must beat.
2. A guard-rail signal inside the hybrid score, plus human-readable reason codes
   that analysts and auditors can verify without understanding the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    code: str
    description: str
    points: int
    condition: Callable[[pd.DataFrame], pd.Series]


RULES: list[Rule] = [
    Rule("R01_IMPOSSIBLE_TRAVEL", "Implied travel speed > 900 km/h over > 500 km since last login", 40,
         lambda d: (d["geo_velocity_kmh"] > 900) & (d["dist_prev_km"] > 500)),
    Rule("R02_NEW_COUNTRY_NEW_DEVICE", "First login from this country AND on an unseen device", 35,
         lambda d: (d["new_country"] == 1) & (d["new_device"] == 1)),
    Rule("R03_AUTOMATION_UA", "User-agent is a script/headless browser", 50,
         lambda d: d["ua_automation"] == 1),
    Rule("R04_FAILED_BURST_USER", ">= 5 failed logins on this account in the last hour", 35,
         lambda d: d["user_fails_1h"] >= 5),
    Rule("R05_IP_SPRAY", "Source IP tried >= 5 accounts in 24h with >= 50% failures", 40,
         lambda d: (d["ip_distinct_users_24h"] >= 5) & (d["ip_fail_ratio_24h"] >= 0.5)),
    Rule("R06_HOSTING_ASN", "Source is a hosting / VPN provider", 20,
         lambda d: d["is_hosting_asn"] == 1),
    Rule("R07_NEW_DEVICE_NEW_NETWORK", "Unseen device on an unseen network (ASN)", 20,
         lambda d: (d["new_device"] == 1) & (d["new_asn"] == 1)),
    Rule("R08_RARE_COUNTRY", "Country accounts for < 0.1% of platform logins", 15,
         lambda d: d["country_global_share"] < 0.001),
    Rule("R09_ODD_HOUR", "> 6h away from the user's usual login hour (established users)", 10,
         lambda d: (d["hour_deviation"] > 6) & (d["user_prior_logins"] >= 20)),
]


def rule_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix: one column per rule code."""
    return pd.DataFrame({r.code: r.condition(df).astype(bool) for r in RULES}, index=df.index)


def rule_points(hits: pd.DataFrame) -> pd.Series:
    pts = np.array([r.points for r in RULES])
    return pd.Series(np.minimum(hits.to_numpy() @ pts, 100), index=hits.index, name="rule_points")


# Softer behavioural indicators used only for explanation (not scored directly)
_INDICATORS: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
    ("new device", lambda d: d["new_device"] == 1),
    ("new country", lambda d: d["new_country"] == 1),
    ("new city", lambda d: (d["new_city"] == 1) & (d["new_country"] == 0)),
    ("new network (ASN)", lambda d: d["new_asn"] == 1),
    ("new browser/OS", lambda d: d["new_platform"] == 1),
    ("far from first-seen location", lambda d: d["dist_first_km"] > 1000),
    ("recent failed logins", lambda d: (d["user_fails_24h"] >= 2) & (d["user_fails_1h"] < 5)),
    ("unusual login hour", lambda d: (d["hour_deviation"] > 4) & (d["hour_deviation"] <= 6)),
]


def reason_codes(df: pd.DataFrame, hits: pd.DataFrame, max_reasons: int = 4) -> pd.Series:
    """Ordered, human-readable reasons per event: fired rules first (by weight), then indicators."""
    by_weight = sorted(RULES, key=lambda r: -r.points)
    ind = pd.DataFrame({name: f(df).astype(bool) for name, f in _INDICATORS}, index=df.index)
    out = []
    for i in df.index:
        reasons = [r.code for r in by_weight if hits.at[i, r.code]]
        reasons += [n for n in ind.columns if ind.at[i, n]]
        out.append(reasons[:max_reasons] or ["behavioural model score"])
    return pd.Series(out, index=df.index, name="reasons")
