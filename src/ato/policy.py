"""Turn scores into actions: risk tiers, cost-based thresholds (risk tolerance) and evaluation.

Tiers / actions
---------------
LOW    (score <  t_mfa)            allow silently
MEDIUM (t_mfa <= score < t_block)  step-up authentication (OTP / passkey)
HIGH   (score >= t_block)          block session, force password reset, SOC alert

Thresholds are chosen on a *validation* period by minimising expected cost, then
reported on an untouched *test* period. The cost table is the explicit statement
of the organisation's risk tolerance and is meant to be agreed with Fraud/GRC.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class CostModel:
    """Expected cost per outcome, in IDR. Assumptions - tune with the business."""
    missed_ato: float = 7_500_000       # avg fraud loss + remediation per compromised account
    mfa_on_legit: float = 2_500         # friction: abandonment x avg margin
    block_on_legit: float = 500_000     # locked-out customer: reset, support call, churn risk
    soc_review: float = 40_000          # ~30 min analyst time per HIGH alert
    mfa_bypass_rate: float = 0.15       # share of ATOs that also phished/SIM-swapped the OTP

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Thresholds:
    mfa: float
    block: float


def assign_tier(score: np.ndarray, th: Thresholds) -> np.ndarray:
    return np.select([score >= th.block, score >= th.mfa], ["HIGH", "MEDIUM"], "LOW")


def expected_cost(y: np.ndarray, score: np.ndarray, th: Thresholds, cm: CostModel) -> float:
    tier = assign_tier(score, th)
    legit, ato = y == 0, y == 1
    cost = 0.0
    cost += cm.mfa_on_legit * np.sum(legit & (tier == "MEDIUM"))
    cost += (cm.block_on_legit + cm.soc_review) * np.sum(legit & (tier == "HIGH"))
    cost += cm.missed_ato * np.sum(ato & (tier == "LOW"))
    cost += cm.missed_ato * cm.mfa_bypass_rate * np.sum(ato & (tier == "MEDIUM"))
    cost += cm.soc_review * np.sum(ato & (tier == "HIGH"))
    return float(cost)


def optimise_thresholds(y: np.ndarray, score: np.ndarray, cm: CostModel,
                        max_mfa_rate: float = 0.03) -> tuple[Thresholds, pd.DataFrame]:
    """Grid search (t_mfa < t_block) minimising cost, subject to an MFA-friction budget."""
    grid = np.round(np.arange(0, 100.01, 0.5), 1)
    legit_s = np.sort(score[y == 0])
    ato_s = np.sort(score[y == 1])
    # number of legit / ATO events at or above each threshold
    L = len(legit_s) - np.searchsorted(legit_s, grid, side="left")
    A = len(ato_s) - np.searchsorted(ato_s, grid, side="left")
    La, Lb = L[:, None], L[None, :]
    Aa, Ab = A[:, None], A[None, :]
    cost = (cm.mfa_on_legit * (La - Lb)
            + (cm.block_on_legit + cm.soc_review) * Lb
            + cm.missed_ato * (len(ato_s) - Aa)
            + cm.missed_ato * cm.mfa_bypass_rate * (Aa - Ab)
            + cm.soc_review * Ab).astype(float)
    friction = np.broadcast_to(La / max(len(legit_s), 1), cost.shape)
    valid = (grid[:, None] < grid[None, :]) & (friction <= max_mfa_rate)
    cost_masked = np.where(valid, cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost_masked), cost.shape)
    ii, jj = np.nonzero(valid)
    res = pd.DataFrame({"t_mfa": grid[ii], "t_block": grid[jj], "cost": cost[ii, jj],
                        "legit_challenge_rate": friction[ii, jj]})
    return Thresholds(float(grid[i]), float(grid[j])), res


def evaluate(y: np.ndarray, score: np.ndarray, th: Thresholds, cm: CostModel, days: float) -> dict:
    tier = assign_tier(score, th)
    legit, ato = y == 0, y == 1
    n_ato = int(ato.sum())
    stopped = ato & (tier == "HIGH")
    challenged = ato & (tier == "MEDIUM")
    caught = stopped | challenged
    alerts = tier == "HIGH"
    return {
        "events": int(len(y)),
        "ato_events": n_ato,
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "ato_blocked_pct": round(100 * stopped.sum() / n_ato, 1),
        "ato_challenged_pct": round(100 * challenged.sum() / n_ato, 1),
        "ato_missed_pct": round(100 * (ato & (tier == "LOW")).sum() / n_ato, 1),
        "ato_intercepted_pct": round(100 * caught.sum() / n_ato, 1),
        "legit_mfa_rate_pct": round(100 * (legit & (tier == "MEDIUM")).sum() / legit.sum(), 3),
        "legit_blocked_rate_pct": round(100 * (legit & alerts).sum() / legit.sum(), 4),
        "high_alerts_per_day": round(alerts.sum() / days, 1),
        "high_alert_precision_pct": round(100 * (alerts & ato).sum() / max(alerts.sum(), 1), 1),
        "expected_cost_idr": expected_cost(y, score, th, cm),
        "cost_no_control_idr": float(cm.missed_ato * n_ato),
    }


def bootstrap_ci(y: np.ndarray, score: np.ndarray, th: Thresholds, n_boot: int = 300,
                 seed: int = 0) -> dict:
    """95% bootstrap intervals (resampling login events) for the headline test metrics."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pr, inter, blocked = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], score[idx]
        if yb.sum() == 0:
            continue
        tier = assign_tier(sb, th)
        pr.append(average_precision_score(yb, sb))
        inter.append(100 * np.mean(tier[yb == 1] != "LOW"))
        blocked.append(100 * np.mean(tier[yb == 1] == "HIGH"))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]  # noqa: E731
    return {"pr_auc_ci95": q(pr), "ato_intercepted_pct_ci95": q(inter), "ato_blocked_pct_ci95": q(blocked)}


# --------------------------------------------------------------------------------------
# Credential-stuffing IP detector (runs on ALL attempts, incl. failures)
# --------------------------------------------------------------------------------------
def ip_spray_condition(feat: pd.DataFrame, min_users: int = 5, min_fail_ratio: float = 0.6) -> pd.Series:
    """An attempt at which its source IP has hit many accounts with mostly failures."""
    return (feat["ip_distinct_users_24h"] >= min_users) & (feat["ip_fail_ratio_24h"] >= min_fail_ratio)


def ip_control_active(feat: pd.DataFrame, ttl_hours: int = 24) -> pd.Series:
    """True when the event's IP was flagged within the previous `ttl_hours` (control in force).

    Flags expire: a permanent block on a mobile-carrier CGNAT address would lock out
    thousands of innocent subscribers sharing it.
    """
    from .features import _grouped_window  # local import to avoid a cycle
    tmp = pd.DataFrame({"ip": feat["ip"].to_numpy(),
                        "t": ((feat["login_ts"] - feat["login_ts"].min()).dt.total_seconds()).astype("int64").to_numpy(),
                        "flag": ip_spray_condition(feat).astype(int).to_numpy()})
    return pd.Series(_grouped_window(tmp, "ip", "flag", ttl_hours * 3600) > 0, index=feat.index)


def ip_blocklist(feat: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """First flag per IP inside the window, with the recommended action."""
    flagged = feat[ip_spray_condition(feat) & (feat["login_ts"] >= start)]
    first = flagged.sort_values("login_ts").groupby("ip").first()
    first["action"] = np.where(first["asn_type"] == "hosting", "block_24h", "captcha_and_rate_limit_24h")
    return first[["login_ts", "action", "asn_name", "asn_type", "country",
                  "ip_distinct_users_24h", "ip_fail_ratio_24h"]].rename(columns={"login_ts": "flagged_at"})


def evaluate_ip_detector(feat: pd.DataFrame, start: pd.Timestamp, ttl_hours: int = 24) -> dict:
    """Replay the IP control over the window: what it stops and what it costs.

    Policy: hard-block hosting/VPN IPs; for residential & mobile IPs (often CGNAT,
    shared with real customers) only add CAPTCHA + rate limiting.
    """
    active = ip_control_active(feat, ttl_hours)
    period = feat["login_ts"] >= start
    p = feat[period].assign(active=active[period])
    hosting = p["asn_type"] == "hosting"
    attack, legit = p["is_attack_ip"] == 1, p["is_attack_ip"] == 0
    flagged_ips = set(p.loc[ip_spray_condition(p), "ip"])
    attack_ips = set(p.loc[attack, "ip"])
    legit_ok = legit & (p["login_successful"] == 1)
    return {
        "ttl_hours": ttl_hours,
        "attack_ips_seen": len(attack_ips),
        "attack_ips_flagged": len(flagged_ips & attack_ips),
        "ips_flagged_never_used_by_attackers": len(flagged_ips - attack_ips),
        "attack_attempts_total": int(attack.sum()),
        "attack_attempts_blocked": int((attack & p["active"] & hosting).sum()),
        "attack_attempts_captcha": int((attack & p["active"] & ~hosting).sum()),
        "ato_events_total": int(p["is_account_takeover"].sum()),
        "ato_prevented_by_block": int((p["is_account_takeover"].eq(1) & p["active"] & hosting).sum()),
        "ato_hit_captcha": int((p["is_account_takeover"].eq(1) & p["active"] & ~hosting).sum()),
        "legit_logins_total": int(legit_ok.sum()),
        "legit_logins_blocked": int((legit_ok & p["active"] & hosting).sum()),
        "legit_logins_captcha": int((legit_ok & p["active"] & ~hosting).sum()),
        "legit_logins_hit_if_all_ips_hard_blocked": int((legit_ok & p["active"]).sum()),
    }
