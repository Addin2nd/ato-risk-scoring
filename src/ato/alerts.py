"""Emit alerts as Elastic Common Schema (ECS)-style JSON lines.

The output can be shipped as-is to Elasticsearch/Kibana (Filebeat/Logstash),
Splunk HEC or QRadar (via a JSON DSM), so SOC analysts see risk-engine decisions
next to the rest of their authentication telemetry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ACTIONS = {"MEDIUM": "step_up_mfa", "HIGH": "block_session_and_reset_password"}
ENGINE_VERSION = "ato-risk-engine 1.0.0"


def to_ecs(row: pd.Series) -> dict:
    return {
        "@timestamp": row["login_ts"].strftime("%Y-%m-%dT%H:%M:%S+07:00"),
        "event": {
            "kind": "alert",
            "category": ["authentication"],
            "type": ["start"],
            "outcome": "success",
            "action": ACTIONS[row["tier"]],
            "severity": 73 if row["tier"] == "HIGH" else 47,
        },
        "user": {"name": row["user_id"]},
        "source": {
            "ip": row["ip"],
            "geo": {"country_iso_code": row["country"], "region_name": row["region"],
                    "city_name": row["city"]},
            "as": {"number": int(row["asn"]), "organization": {"name": row["asn_name"]}},
        },
        "user_agent": {"original": row["user_agent"], "device": {"name": row["device_type"]}},
        "risk": {"calculated_score": float(row["risk_score"]),
                 "calculated_level": row["tier"].lower(),
                 "calculated_score_norm": round(float(row["p_ato"]), 6)},
        "rule": {"name": "Account takeover risk", "ruleset": ENGINE_VERSION,
                 "reference": "docs/alert_playbook.md"},
        "ato": {
            "reasons": list(row["reasons"]),
            "signals": {"ml_probability": round(float(row["p_ml"]), 6),
                        "anomaly_percentile": round(float(row["anomaly_pct"]), 4),
                        "rule_points": int(row["rule_points"])},
            "device_id": row["device_id"],
        },
    }


def write_jsonl(alerts: pd.DataFrame, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for _, r in alerts.iterrows():
            fh.write(json.dumps(to_ecs(r), ensure_ascii=False) + "\n")
    return len(alerts)
