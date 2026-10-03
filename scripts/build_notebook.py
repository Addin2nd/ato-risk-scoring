"""Build and execute notebooks/01_walkthrough.ipynb without a Jupyter install.

Cells are defined below, executed in one namespace, and their outputs
(stdout, last-expression value, matplotlib figures) are embedded so the notebook
renders fully on GitHub. Run after scripts/run_pipeline.py.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
("md", """# Account takeover (ATO) detection & login risk scoring — walkthrough

This notebook walks through the data, the behavioural signals, the model comparison
and one alert investigation. The heavy lifting lives in `src/ato/`; run
`python scripts/run_pipeline.py` first.

> **Data note.** The login telemetry is **synthetic** (schema of the public RBA dataset),
> generated with known attack campaigns so every number here is reproducible.
> Results describe this simulation, not a production system."""),
("code", """import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from ato.features import build_features, FEATURES
from ato.rules import rule_hits, RULES
from ato.report import _style, SERIES, TIER, INK2
_style()
pd.set_option("display.width", 140, "display.max_columns", 20)

events = pd.read_csv(ROOT / "data/login_events.csv.gz", parse_dates=["login_ts"])
metrics = json.loads((ROOT / "reports/metrics.json").read_text())
print(f"{len(events):,} login events | {events.user_id.nunique():,} users | "
      f"{events.login_ts.min():%d %b} – {events.login_ts.max():%d %b %Y}")
events.head(3).T"""),
("md", """## 1. What is in the log?

Most traffic is ordinary. The interesting part is the **legitimate noise** — travel,
VPNs, new phones, typos — because that is what generates false positives."""),
("code", """summary = (events.groupby("scenario")
           .agg(events=("user_id", "size"),
                success_rate=("login_successful", "mean"),
                takeovers=("is_account_takeover", "sum"))
           .sort_values("events", ascending=False))
summary["success_rate"] = (100 * summary["success_rate"]).round(1)
summary"""),
("md", """## 2. Point-in-time features

Each login is compared only with the user's **earlier** successful logins
(and the IP's/ASN's earlier attempts) — exactly what a real-time engine would know.
A unit test (`tests/test_pipeline.py::test_features_are_point_in_time`) proves that adding
future events never changes a past feature."""),
("code", """feat = build_features(events)
scored = feat[(feat.login_successful == 1) & (feat.login_ts >= feat.login_ts.min().normalize() + pd.Timedelta(days=14))]
cols = ["new_device", "new_country", "new_asn", "dist_first_km", "geo_velocity_kmh",
        "hour_deviation", "user_fails_24h", "is_hosting_asn", "country_global_share"]
cmp = scored.groupby("is_account_takeover")[cols].mean().T
cmp.columns = ["legit (mean)", "takeover (mean)"]
cmp.round(3)"""),
("md", """## 3. Why simple rules are not enough

Classic SIEM rules fire on *legitimate* behaviour too. Impossible-travel looks great on paper,
but VPN users hopping between Jakarta and an Amsterdam exit node, or travellers alternating
between roaming and hotel Wi-Fi, trip it constantly."""),
("code", """rc = pd.DataFrame(metrics["rule_coverage_test"]).set_index("rule")
rc[["points", "fires_per_day", "precision_pct", "ato_coverage_pct"]]"""),
("code", """hits = rule_hits(scored)
noise = (pd.concat([scored[["scenario", "is_account_takeover"]], hits["R01_IMPOSSIBLE_TRAVEL"]], axis=1)
         .query("R01_IMPOSSIBLE_TRAVEL"))
noise.groupby("scenario").size().sort_values(ascending=False).rename("impossible-travel hits")"""),
("md", """## 4. Model comparison on the unseen test period

All four detectors go through the **same** policy machinery: thresholds are picked on the
validation weeks by minimising expected cost, then frozen and applied to the test weeks."""),
("code", """rows = {k: {"PR-AUC": v["pr_auc"], "ATO intercepted %": v["ato_intercepted_pct"],
            "ATO blocked %": v["ato_blocked_pct"], "legit MFA %": v["legit_mfa_rate_pct"],
            "legit blocked %": v["legit_blocked_rate_pct"], "HIGH alerts/day": v["high_alerts_per_day"],
            "HIGH precision %": v["high_alert_precision_pct"],
            "cost (Rp jt)": round(v["expected_cost_idr"] / 1e6, 1)}
        for k, v in metrics["test_comparison"].items()}
pd.DataFrame(rows).T"""),
("code", """print("Stacker weights (logistic regression on out-of-fold signals):")
for k, v in metrics["stacker_weights"].items():
    print(f"  {k:18s} {v:+.3f}")"""),
("md", """## 5. Who gets challenged? Legit noise vs attacks

The score should push genuine-but-unusual users (travel, VPN, new phone) into **step-up MFA**,
not into a block — while takeovers land in the block tier."""),
("code", """st = pd.read_csv(ROOT / "reports/scored_test.csv.gz")
order = ["normal", "new_device", "travel_domestic", "travel_abroad", "vpn",
         "targeted_ato", "credential_stuffing", "brute_force"]
mix = (st.groupby("scenario")["tier"].value_counts(normalize=True).unstack(fill_value=0)
         .reindex(index=order, columns=["LOW", "MEDIUM", "HIGH"], fill_value=0) * 100)

fig, ax = plt.subplots(figsize=(9, 4))
left = np.zeros(len(mix))
for tier in ["HIGH", "MEDIUM", "LOW"]:
    ax.barh(mix.index, mix[tier], left=left, color=TIER[tier], height=0.55,
            edgecolor="#fcfcfb", linewidth=2, label=tier)
    left += mix[tier].to_numpy()
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", colors=INK2, length=0)
ax.set_xlabel("% of successful logins in test period")
ax.set_title("Risk tier by scenario (top 5 legit, bottom 3 attacks)")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.show()
mix.round(1)"""),
("md", """## 6. Alert investigation (mini post-incident review)

Take one HIGH alert emitted in ECS format and reconstruct the account's timeline —
the evidence an analyst (or auditor) would attach to the case."""),
("code", """alerts = [json.loads(l) for l in open(ROOT / "reports/alerts_test.jsonl")]
high = [a for a in alerts if a["risk"]["calculated_level"] == "high"]
case = next(a for a in high if "R02_NEW_COUNTRY_NEW_DEVICE" in a["ato"]["reasons"])
print(json.dumps(case, indent=2)[:1800])"""),
("code", """uid = case["user"]["name"]
t = pd.Timestamp(case["@timestamp"][:19])
tl = (events[(events.user_id == uid) & events.login_ts.between(t - pd.Timedelta(days=4), t + pd.Timedelta(days=2))]
      [["login_ts", "login_successful", "city", "country", "asn_name", "browser", "os", "device_id",
        "is_account_takeover"]])
tl"""),
("md", """**Reading the timeline:** the account's history is a stable device on Indonesian networks;
the flagged session uses an unseen device from a different country, which the engine blocked and
sent to the SOC with reason codes. Follow-up per `docs/alert_playbook.md`: confirm with the
customer, revoke sessions, reset credentials, and add the infrastructure to the blocklist.

## 7. Limitations & next steps
- Synthetic data: attackers here are simpler than real adversaries; validate on the real RBA dataset
  (similar schema; needs a geo lookup and an ASN-type list — see README) and on production logs
  before trusting any number.
- Labels in reality arrive late and incomplete (chargebacks, customer reports) → monitor with delayed
  labels and keep the unsupervised layer as a safety net.
- Friction and score distributions drift as behaviour changes → weekly monitoring (PSI) and recalibration.
- Next: device fingerprint reputation, graph features (accounts sharing devices/IPs), Kibana dashboard."""),
]


def _render(value) -> dict | None:
    if value is None:
        return None
    data = {"text/plain": repr(value)}
    if hasattr(value, "to_html"):
        data["text/html"] = value.to_html()
    return {"output_type": "execute_result", "data": data, "metadata": {}, "execution_count": None}


def run() -> Path:
    os.chdir(ROOT / "notebooks")
    ns: dict = {"__name__": "__main__"}
    plt.show = lambda *a, **k: None  # figures are collected after each cell
    cells = []
    count = 0
    for kind, src in CELLS:
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
            continue
        count += 1
        outputs = []
        tree = ast.parse(src)
        last_expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, f"<cell {count}>", "exec"), ns)
            value = eval(compile(ast.Expression(last_expr.value), f"<cell {count}>", "eval"), ns) \
                if last_expr else None
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                                     "text/plain": "<Figure>"}})
            plt.close(fig)
        res = _render(value)
        if res:
            res["execution_count"] = count
            outputs.append(res)
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": count,
                      "source": src.splitlines(True), "outputs": outputs})
        print(f"cell {count} ok", file=sys.stderr)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}}}
    path = ROOT / "notebooks" / "01_walkthrough.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    return path


if __name__ == "__main__":
    print(run())
