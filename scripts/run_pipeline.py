"""End-to-end pipeline: data -> features -> models -> thresholds -> evaluation -> alerts.

Usage:
    python scripts/run_pipeline.py            # reuse data/login_events.csv.gz if present
    python scripts/run_pipeline.py --regen    # regenerate synthetic data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ato.alerts import write_jsonl  # noqa: E402
from ato.features import FEATURES, build_features, scoring_frame  # noqa: E402
from ato.generate import GenConfig, generate  # noqa: E402
from ato.model import TARGET, RiskEngine, prob_to_score  # noqa: E402
from ato.policy import (CostModel, assign_tier, bootstrap_ci, evaluate,  # noqa: E402
                        evaluate_ip_detector, ip_blocklist, optimise_thresholds)
from ato.rules import RULES, reason_codes, rule_hits  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
TRAIN_END = pd.Timestamp("2026-07-15")   # day 75
VALID_END = pd.Timestamp("2026-08-08")   # day 99 -> test = last ~3 weeks


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_events(regen: bool) -> pd.DataFrame:
    path = DATA / "login_events.csv.gz"
    if path.exists() and not regen:
        log(f"loading {path.relative_to(ROOT)}")
        return pd.read_csv(path, parse_dates=["login_ts"])
    log("generating synthetic login telemetry ...")
    events, _ = generate(GenConfig())
    DATA.mkdir(exist_ok=True)
    events.to_csv(path, index=False, compression="gzip")
    log(f"wrote {len(events):,} events -> {path.relative_to(ROOT)}")
    return events


def main(regen: bool = False) -> dict:
    REPORTS.mkdir(exist_ok=True)
    events = load_events(regen)

    log("building point-in-time features ...")
    feat = build_features(events)
    scored = scoring_frame(feat)
    train = scored[scored["login_ts"] < TRAIN_END]
    valid = scored[(scored["login_ts"] >= TRAIN_END) & (scored["login_ts"] < VALID_END)]
    test = scored[scored["login_ts"] >= VALID_END]
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        log(f"  {name:5s}: {len(part):>7,} successful logins, {int(part[TARGET].sum()):>3} ATO")

    log("training hybrid risk engine ...")
    engine = RiskEngine().fit(train)
    sv, st = engine.score(valid), engine.score(test)

    # ---- compare detectors on identical policy machinery ----------------------------------
    variants = {
        "Rules only": (sv["rule_points"].to_numpy(float), st["rule_points"].to_numpy(float)),
        "Isolation Forest only": (100 * sv["anomaly_pct"].to_numpy(), 100 * st["anomaly_pct"].to_numpy()),
        "Gradient boosting only": (prob_to_score(sv["p_ml"].to_numpy()), prob_to_score(st["p_ml"].to_numpy())),
        "Hybrid (stacked)": (sv["risk_score"].to_numpy(), st["risk_score"].to_numpy()),
    }
    cm = CostModel()
    yv, yt = valid[TARGET].to_numpy(), test[TARGET].to_numpy()
    test_days = (test["login_ts"].max() - test["login_ts"].min()).total_seconds() / 86400
    comparison, chosen = {}, {}
    for name, (s_val, s_test) in variants.items():
        th, _ = optimise_thresholds(yv, s_val, cm)
        chosen[name] = th
        comparison[name] = {"thresholds": {"mfa": th.mfa, "block": th.block},
                            **evaluate(yt, s_test, th, cm, test_days),
                            **bootstrap_ci(yt, s_test, th)}
        c = comparison[name]
        log(f"  {name:24s} PR-AUC {c['pr_auc']:.3f} | intercepted {c['ato_intercepted_pct']:5.1f}% "
            f"| MFA on legit {c['legit_mfa_rate_pct']:.2f}% | cost Rp {c['expected_cost_idr']/1e6:,.1f} jt")

    th = chosen["Hybrid (stacked)"]
    _, cost_surface = optimise_thresholds(yv, sv["risk_score"].to_numpy(), cm)

    # ---- scored test set + alerts ------------------------------------------------------------
    out = test.join(st)
    out["tier"] = assign_tier(out["risk_score"].to_numpy(), th)
    flagged = out[out["tier"] != "LOW"].copy()
    flagged["reasons"] = reason_codes(flagged, rule_hits(flagged))
    n_alerts = write_jsonl(flagged.sort_values("login_ts"), REPORTS / "alerts_test.jsonl")
    log(f"wrote {n_alerts} ECS alerts -> reports/alerts_test.jsonl")

    keep = ["login_ts", "user_id", "ip", "country", "city", "asn_name", "browser", "os", "device_type",
            "scenario", TARGET, "p_ml", "anomaly_pct", "rule_points", "p_ato", "risk_score", "tier"]
    out[keep].to_csv(REPORTS / "scored_test.csv.gz", index=False, compression="gzip")

    # ---- explainability: permutation importance of the supervised layer ----------------------
    log("permutation importance (validation, PR-AUC) ...")
    pi = permutation_importance(engine.gbm, valid[FEATURES], yv, scoring="average_precision",
                                n_repeats=3, random_state=0, n_jobs=1)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean,
                                "std": pi.importances_std})
                  .sort_values("importance", ascending=False))
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)

    # ---- detection by scenario + rule coverage ----------------------------------------------
    ato_rows = out[out[TARGET] == 1]
    by_scenario = (ato_rows.groupby("scenario")["tier"].value_counts().unstack(fill_value=0)
                   .reindex(columns=["HIGH", "MEDIUM", "LOW"], fill_value=0))
    hits_test = rule_hits(test)
    rule_cov = []
    for r in RULES:
        h = hits_test[r.code]
        rule_cov.append({"rule": r.code, "points": r.points, "description": r.description,
                         "fires_per_day": round(h.sum() / test_days, 1),
                         "precision_pct": round(100 * test.loc[h, TARGET].mean(), 1) if h.any() else None,
                         "ato_coverage_pct": round(100 * h[test[TARGET] == 1].mean(), 1)})

    # ---- credential-stuffing IP detector (all attempts) -----------------------------------------
    ip_eval = evaluate_ip_detector(feat, VALID_END)
    ip_blocklist(feat, VALID_END).to_csv(REPORTS / "ip_blocklist_test.csv")
    log(f"  IP spray detector: {ip_eval['attack_ips_flagged']}/{ip_eval['attack_ips_seen']} attack IPs flagged, "
        f"{ip_eval['attack_attempts_blocked'] + ip_eval['attack_attempts_captcha']}/{ip_eval['attack_attempts_total']} "
        f"attack attempts hit; legit logins blocked {ip_eval['legit_logins_blocked']} "
        f"(vs {ip_eval['legit_logins_hit_if_all_ips_hard_blocked']} if every flagged IP were hard-blocked)")

    # ---- robustness: an attack type the labels have never seen -------------------------------
    log("robustness check: hide credential-stuffing labels from training ...")
    blind = train.copy()
    blind.loc[blind["scenario"] == "credential_stuffing", TARGET] = 0   # fraud ops never confirmed them
    eng_b = RiskEngine().fit(blind)
    sb = eng_b.score(test)
    novel = ((test["scenario"] == "credential_stuffing") & (test[TARGET] == 1)).to_numpy()
    legit_mask = (test[TARGET] == 0).to_numpy()
    robustness = {"hidden_attack": "credential_stuffing", "legit_challenge_budget_pct": 2.0,
                  "novel_ato_events": int(novel.sum())}
    for name, sc in [("Gradient boosting only", prob_to_score(sb["p_ml"].to_numpy())),
                     ("Isolation Forest only", 100 * sb["anomaly_pct"].to_numpy()),
                     ("Rules only", sb["rule_points"].to_numpy(float)),
                     ("Hybrid (stacked)", sb["risk_score"].to_numpy())]:
        cut = np.quantile(sc[legit_mask], 0.98)
        robustness[name] = round(100 * float(np.mean(sc[novel] > cut)), 1)
    log(f"  recall on never-labelled attack @2% legit challenged: "
        + ", ".join(f"{k} {v}%" for k, v in robustness.items() if k.endswith(("only", "(stacked)"))))

    # ---- daily series for dashboards ---------------------------------------------------------
    daily = feat.assign(day=feat["login_ts"].dt.floor("D")).groupby("day").agg(
        logins=("login_successful", "sum"),
        failed=("login_successful", lambda s: int((s == 0).sum())),
        attack_attempts=("is_attack_ip", "sum"),
        ato=("is_account_takeover", "sum"))
    daily.to_csv(REPORTS / "daily_volume.csv")
    out.assign(day=out["login_ts"].dt.floor("D")).groupby(["day", "tier"]).size() \
        .unstack(fill_value=0).to_csv(REPORTS / "daily_tiers_test.csv")
    cost_surface.to_csv(REPORTS / "cost_surface_valid.csv.gz", index=False, compression="gzip")

    metrics = {
        "dataset": {"events": int(len(feat)), "users": int(feat["user_id"].nunique()),
                    "successful_logins": int(feat["login_successful"].sum()),
                    "failed_logins": int((feat["login_successful"] == 0).sum()),
                    "ato_events": int(feat[TARGET].sum()),
                    "attack_attempts": int(feat["is_attack_ip"].sum()),
                    "start": str(feat["login_ts"].min().date()), "end": str(feat["login_ts"].max().date())},
        "splits": {n: {"from": str(p["login_ts"].min().date()), "to": str(p["login_ts"].max().date()),
                       "rows": int(len(p)), "ato": int(p[TARGET].sum())}
                   for n, p in [("train", train), ("valid", valid), ("test", test)]},
        "cost_model_idr": cm.as_dict(),
        "chosen_thresholds": {"mfa": th.mfa, "block": th.block},
        "stacker_weights": engine.stacker_weights(),
        "test_comparison": comparison,
        "test_by_scenario": {k: {c: int(v) for c, v in row.items()} for k, row in by_scenario.iterrows()},
        "rule_coverage_test": rule_cov,
        "ip_detector_test": ip_eval,
        "robustness_unseen_attack": robustness,
        "top_features": importance.head(10).round(4).to_dict(orient="records"),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log("wrote reports/metrics.json")

    from ato.report import build_all
    figs = build_all(ROOT)
    log(f"wrote {len(figs)} figures -> reports/figures/")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate synthetic data")
    main(ap.parse_args().regen)
