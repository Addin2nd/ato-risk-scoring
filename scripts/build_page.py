"""Render docs/index.html (portfolio case-study page) from the template + reports/.

Every number on the page is read from reports/metrics.json or reports/scored_test.csv.gz,
and the figures are embedded as data URIs, so the page is one self-contained file that
works on GitHub Pages or any static host.

    python scripts/build_page.py --repo https://github.com/<you>/ato-risk-scoring
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "https://github.com/your-username/ato-risk-scoring"

SCALE_ROWS = [  # (scenario, label, is_attack)
    ("normal", "Normal login", False),
    ("new_device", "New phone", False),
    ("travel_domestic", "Domestic trip", False),
    ("travel_abroad", "Trip abroad", False),
    ("vpn", "VPN user", False),
    ("credential_stuffing", "Credential stuffing", True),
    ("targeted_ato", "Phished takeover", True),
]

REASON_TEXT = {
    "R01_IMPOSSIBLE_TRAVEL": "Travel speed above 900 km/h since the previous login",
    "R02_NEW_COUNTRY_NEW_DEVICE": "First login from this country, on a device never seen before",
    "R07_NEW_DEVICE_NEW_NETWORK": "Unseen device on an unseen network (ASN)",
    "R08_RARE_COUNTRY": "Country makes up under 0.1% of platform logins",
    "R06_HOSTING_ASN": "Source is a hosting or VPN provider",
    "R05_IP_SPRAY": "Source IP tried 5+ accounts in 24h, mostly failing",
    "R03_AUTOMATION_UA": "Scripted or headless user-agent",
    "R04_FAILED_BURST_USER": "5+ failed logins on this account in the last hour",
    "R09_ODD_HOUR": "More than 6h from the user's usual login hour",
}


def fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def highlight_json(doc: dict) -> str:
    text = html.escape(json.dumps(doc, indent=2, ensure_ascii=False))
    text = re.sub(r'(&quot;[^&]*?&quot;)(\s*:)', r'<span class="k">\1</span>\2', text)
    text = re.sub(r'(:\s*)(&quot;.*?&quot;)', r'\1<span class="s">\2</span>', text)
    return text


def scale_rows(st: pd.DataFrame, t_mfa: float, t_block: float) -> str:
    bands = (f'<div class="band b-low" style="left:0;width:{t_mfa}%"></div>'
             f'<div class="band b-med" style="left:{t_mfa}%;width:{t_block - t_mfa}%"></div>'
             f'<div class="band b-high" style="left:{t_block}%;right:0"></div>')
    out = []
    first_attack = True
    for i, (sc, label, attack) in enumerate(SCALE_ROWS):
        s = st.loc[st["scenario"] == sc, "risk_score"]
        if s.empty:
            continue
        p10, p50, p90 = s.quantile([0.1, 0.5, 0.9]).round(1).tolist()
        if attack and first_attack:
            out.append('<div class="rows-sep"></div>')
            first_attack = False
        val_cls = "val left" if p50 > 70 else "val"
        val_pos = p10 if p50 > 70 else p90
        band_html = bands
        if i == 0:
            band_html += (f'<span class="band-label" style="left:6px">allow</span>'
                          f'<span class="band-label" style="left:calc({t_mfa}% + 6px)">step-up MFA</span>'
                          f'<span class="band-label" style="right:6px">block</span>')
        out.append(
            f'<div class="row-name{" attack" if attack else ""}">{label}<small>n = {len(s):,}</small></div>'
            f'<div class="row-track" title="{label}: median {p50}, 10th–90th pct {p10}–{p90}">'
            f'<div class="bands">{band_html}</div>'
            f'<div class="rng" style="left:{p10}%;width:{max(p90 - p10, 0.6)}%"></div>'
            f'<div class="mid" style="left:{p50}%"></div>'
            f'<span class="{val_cls}" style="left:{val_pos}%">{p50:g}</span></div>')
    return "\n".join(out)


def build(repo: str) -> Path:
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    st = pd.read_csv(ROOT / "reports/scored_test.csv.gz")
    comp = m["test_comparison"]
    hy, ru = comp["Hybrid (stacked)"], comp["Rules only"]
    th = m["chosen_thresholds"]
    rules = {r["rule"]: r for r in m["rule_coverage_test"]}
    rob = m["robustness_unseen_attack"]
    ipd = m["ip_detector_test"]

    rows = []
    for name, c in comp.items():
        ci = c["pr_auc_ci95"]
        cls = ' class="final"' if name.startswith("Hybrid") else ""
        prec = "–" if c["high_alerts_per_day"] == 0 else f'{c["high_alert_precision_pct"]:.0f}%'
        rows.append(
            f'<tr{cls}><td>{name}</td>'
            f'<td>{c["pr_auc"]:.2f}</td><td class="ci">{ci[0]:.2f}–{ci[1]:.2f}</td>'
            f'<td>{c["ato_intercepted_pct"]:.1f}%</td><td>{c["legit_mfa_rate_pct"]:.2f}%</td>'
            f'<td>{c["high_alerts_per_day"]:.1f}</td>'
            f'<td>{prec}</td>'
            f'<td>Rp {c["expected_cost_idr"] / 1e6:,.0f} jt</td></tr>')

    alerts = [json.loads(line) for line in open(ROOT / "reports/alerts_test.jsonl")]
    case = next(a for a in alerts if a["risk"]["calculated_level"] == "high"
                and "R02_NEW_COUNTRY_NEW_DEVICE" in a["ato"]["reasons"])
    slim = {k: case[k] for k in ["@timestamp", "event", "user", "source", "risk", "ato"]}
    slim["event"] = {k: case["event"][k] for k in ["kind", "category", "action"]}
    slim["ato"] = {"reasons": case["ato"]["reasons"], "signals": case["ato"]["signals"]}
    reason_rows = "\n".join(f'<div class="reason"><code>{r}</code><span>{REASON_TEXT.get(r, r)}</span></div>'
                            for r in case["ato"]["reasons"])

    nb = st[(st["is_account_takeover"] == 1) & (st["tier"] != "HIGH")]

    values = {
        "repo_url": repo.rstrip("/"),
        "events": f'{m["dataset"]["events"]:,}', "users": f'{m["dataset"]["users"]:,}',
        "t_mfa": f'{th["mfa"]:g}', "t_block": f'{th["block"]:g}',
        "intercepted": fmt(hy["ato_intercepted_pct"]), "blocked": fmt(hy["ato_blocked_pct"]),
        "intercepted_ci": f'{hy["ato_intercepted_pct_ci95"][0]:.0f}–{hy["ato_intercepted_pct_ci95"][1]:.0f}%',
        "legit_blocked": fmt(hy["legit_blocked_rate_pct"], 3), "legit_mfa": fmt(hy["legit_mfa_rate_pct"]),
        "alerts_day": fmt(hy["high_alerts_per_day"]), "alert_precision": fmt(hy["high_alert_precision_pct"], 0),
        "cost_cut": f'{100 * (1 - hy["expected_cost_idr"] / ru["expected_cost_idr"]):.0f}',
        "cost_hybrid": f'{hy["expected_cost_idr"] / 1e6:,.0f}', "cost_rules": f'{ru["expected_cost_idr"] / 1e6:,.0f}',
        "cost_none": f'{hy["cost_no_control_idr"] / 1e6:,.0f}',
        "test_ato": str(m["splits"]["test"]["ato"]), "test_rows": f'{m["splits"]["test"]["rows"]:,}',
        "comparison_rows": "\n".join(rows),
        "r01_prec": fmt(rules["R01_IMPOSSIBLE_TRAVEL"]["precision_pct"], 0),
        "r01_day": fmt(rules["R01_IMPOSSIBLE_TRAVEL"]["fires_per_day"], 0),
        "r09_day": fmt(rules["R09_ODD_HOUR"]["fires_per_day"], 0),
        "r09_prec": fmt(rules["R09_ODD_HOUR"]["precision_pct"]),
        "rob_hybrid": fmt(rob["Hybrid (stacked)"], 0), "rob_gbm": fmt(rob["Gradient boosting only"]),
        "rob_iso": fmt(rob["Isolation Forest only"]), "rob_n": str(rob["novel_ato_events"]),
        "ip_hit": f'{ipd["attack_attempts_blocked"] + ipd["attack_attempts_captcha"]:,}',
        "ip_total": f'{ipd["attack_attempts_total"]:,}', "ip_legit_blocked": str(ipd["legit_logins_blocked"]),
        "nonblocked_phrase": (f"all {len(nb)} takeovers that were not blocked"
                              if (nb["country"] == "ID").all()
                              else f'{int((nb["country"] == "ID").sum())} of the {len(nb)} takeovers that were not blocked'),
        "alert_json": highlight_json(slim), "reason_rows": reason_rows,
        "scale_rows": scale_rows(st, th["mfa"], th["block"]),
    }
    page = (ROOT / "docs/page_template.html").read_text()

    def img(match):
        data = base64.b64encode((ROOT / "reports/figures" / match.group(1)).read_bytes()).decode()
        return f"data:image/png;base64,{data}"

    page = re.sub(r"\{\{img:([\w.]+)\}\}", img, page)
    page = re.sub(r"\{\{(\w+)\}\}", lambda mm: values[mm.group(1)], page)
    assert "{{" not in page, "unfilled placeholder"
    out = ROOT / "docs/index.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="your GitHub repo URL")
    print(build(ap.parse_args().repo))
