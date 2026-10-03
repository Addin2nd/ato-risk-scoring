"""Static dashboard figures (PNG) for the README / portfolio page.

Palette: a validated, colour-blind-safe categorical order (blue, orange, aqua, yellow),
an ordinal blue ramp for risk tiers, recessive hairline grids, text in ink colours.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .model import prob_to_score  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TIER = {"LOW": "#86b6ef", "MEDIUM": "#2a78d6", "HIGH": "#104281"}
SPLIT_SHADE = {"train": "#f3f2ee", "valid": "#ebeae4", "test": "#e1e0d9"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 12, "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def _save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def fig_daily(reports: Path, metrics: dict, out: Path):
    d = pd.read_csv(reports / "daily_volume.csv", parse_dates=["day"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35})
    for ax in (a1, a2):
        for name, sp in metrics["splits"].items():
            ax.axvspan(pd.Timestamp(sp["from"]), pd.Timestamp(sp["to"]) + pd.Timedelta(days=1),
                       color=SPLIT_SHADE[name], lw=0, zorder=0)
    a1.plot(d["day"], d["logins"], color=SERIES[0])
    a1.set_title("Successful logins per day", pad=22)
    _subtitle(a1, "Weekly rhythm of normal users. Shaded: train | validation | test periods "
                  "(first 14 days = baseline warm-up)")
    a1.set_ylim(0, d["logins"].max() * 1.15)
    for name, sp in metrics["splits"].items():
        mid = pd.Timestamp(sp["from"]) + (pd.Timestamp(sp["to"]) - pd.Timestamp(sp["from"])) / 2
        a1.text(mid, d["logins"].max() * 1.06, name, ha="center", color=INK2, fontsize=9)
    a2.bar(d["day"], d["attack_attempts"], width=0.8, color=SERIES[1])
    a2.set_title("Attack login attempts per day", pad=8)
    a2.set_ylim(0, d["attack_attempts"].max() * 1.12)
    top = d.nlargest(1, "attack_attempts").iloc[0]
    a2.annotate(f"credential-stuffing campaign\n{int(top.attack_attempts):,} attempts",
                (top.day, top.attack_attempts), xytext=(25, -10), textcoords="offset points",
                color=INK2, fontsize=9, va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for ax in (a1, a2):
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="x", visible=False)
    _save(fig, out / "daily_volume.png")


def fig_score_distribution(scored: pd.DataFrame, th: dict, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bins = np.arange(0, 102.5, 2.5)
    legit = scored.loc[scored["is_account_takeover"] == 0, "risk_score"]
    ato = scored.loc[scored["is_account_takeover"] == 1, "risk_score"]
    ax.hist(legit, bins=bins, color=SERIES[0], alpha=0.9, label=f"Legitimate logins ({len(legit):,})",
            edgecolor=SURFACE, linewidth=1)
    ax.hist(ato, bins=bins, color=SERIES[1], alpha=0.95, label=f"Account takeovers ({len(ato):,})",
            edgecolor=SURFACE, linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(0.8, len(legit) * 3)
    for x, lab in [(th["mfa"], f"step-up MFA ≥ {th['mfa']:g}"), (th["block"], f"block ≥ {th['block']:g}")]:
        ax.axvline(x, color=INK2, lw=1)
        right = x > 75
        ax.text(x - 0.8 if right else x + 0.8, len(legit) * 1.6, lab, color=INK2, fontsize=9, va="top",
                ha="right" if right else "left")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk score (each +25 points ≈ 10× higher takeover likelihood)")
    ax.set_ylabel("Logins (log scale)")
    ax.set_title("Risk score separates takeovers from normal traffic", pad=22)
    _subtitle(ax, "Test period, successful logins only")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    _save(fig, out / "score_distribution.png")


def fig_pr_curves(scored: pd.DataFrame, metrics: dict, out: Path):
    y = scored["is_account_takeover"].to_numpy()
    variants = {
        "Rules only": scored["rule_points"].to_numpy(float),
        "Isolation Forest only": scored["anomaly_pct"].to_numpy(),
        "Gradient boosting only": prob_to_score(scored["p_ml"].to_numpy()),
        "Hybrid (stacked)": scored["risk_score"].to_numpy(),
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for (name, s), color in zip(variants.items(), SERIES):
        p, r, _ = precision_recall_curve(y, s)
        auc = metrics["test_comparison"][name]["pr_auc"]
        ax.plot(r, p, color=color, label=f"{name}  (PR-AUC {auc:.2f})", drawstyle="steps-post")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall (share of takeovers caught)")
    ax.set_ylabel("Precision (share of flags that are real)")
    ax.set_title("Precision-recall on the unseen test period", pad=22)
    _subtitle(ax, f"{int(y.sum())} takeovers among {len(y):,} successful logins (0.2% base rate)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9.5)
    _save(fig, out / "pr_curves.png")


def fig_cost(metrics: dict, out: Path):
    comp = metrics["test_comparison"]
    names = ["No control"] + list(comp)
    vals = [comp["Rules only"]["cost_no_control_idr"]] + [comp[n]["expected_cost_idr"] for n in comp]
    vals = np.array(vals) / 1e6
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ypos = np.arange(len(names))[::-1]
    colors = [AXIS] + [SERIES[0]] * (len(names) - 2) + ["#104281"]
    ax.barh(ypos, vals, height=0.5, color=colors)
    for yv, v in zip(ypos, vals):
        ax.text(v + vals.max() * 0.01, yv, f"Rp {v:,.0f} jt", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(ypos, names)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.set_xlim(0, vals.max() * 1.18)
    ax.set_xlabel("Expected cost over the 3-week test period (IDR millions)")
    ax.set_title("Expected cost: fraud losses + customer friction + analyst time", pad=22)
    _subtitle(ax, "Same cost model and threshold search for every detector; thresholds set on validation")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "cost_comparison.png")


def fig_scenarios(metrics: dict, out: Path):
    sc = pd.DataFrame(metrics["test_by_scenario"]).T[["HIGH", "MEDIUM", "LOW"]]
    names = {"credential_stuffing": "Credential stuffing", "targeted_ato": "Targeted ATO (phished)",
             "brute_force": "Brute force"}
    sc.index = [names.get(i, i) for i in sc.index]
    sc = sc.loc[sc.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = np.zeros(len(sc))
    labels = {"HIGH": "Blocked (HIGH)", "MEDIUM": "Step-up MFA (MEDIUM)", "LOW": "Missed (LOW)"}
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        ax.barh(sc.index, sc[tier], left=left, height=0.5, color=TIER[tier], label=labels[tier],
                edgecolor=SURFACE, linewidth=2)
        left += sc[tier].to_numpy()
    for i, total in enumerate(left):
        ax.text(total + 0.8, i, f"{int(total)}", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, left.max() * 1.1)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.set_xlabel("Takeover logins in test period")
    ax.set_title("What happened to each takeover, by attack type", pad=22)
    _subtitle(ax, "Hybrid engine with cost-optimised thresholds")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", visible=False)
    _save(fig, out / "outcomes_by_scenario.png")


def fig_importance(reports: Path, out: Path, top: int = 12):
    imp = pd.read_csv(reports / "feature_importance.csv").head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(imp["feature"], imp["importance"], height=0.55, color=SERIES[0])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("What the supervised model relies on", pad=22)
    _subtitle(ax, "Permutation importance on the validation period")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "feature_importance.png")


def fig_cost_surface(reports: Path, th: dict, out: Path):
    cs = pd.read_csv(reports / "cost_surface_valid.csv.gz")
    cs = cs[(cs["t_mfa"] % 2.5 == 0) & (cs["t_block"] % 2.5 == 0)]
    piv = cs.pivot(index="t_block", columns="t_mfa", values="cost") / 1e6
    fig, ax = plt.subplots(figsize=(8, 5.4))
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    mesh = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap=cmap,
                         norm=LogNorm(vmin=np.nanmin(piv.values), vmax=np.nanmax(piv.values)), shading="nearest")
    ax.plot(th["mfa"], th["block"], "o", ms=9, color=SERIES[1], mec=SURFACE, mew=2)
    ax.annotate(f"chosen: MFA ≥ {th['mfa']:g}, block ≥ {th['block']:g}", (th["mfa"], th["block"]),
                xytext=(12, -16), textcoords="offset points", color=INK, fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none", alpha=0.9))
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Expected cost on validation (IDR millions, log)", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if round(v) in (200, 300, 500) else ""))
    ax.set_xlabel("Step-up MFA threshold")
    ax.set_ylabel("Block threshold")
    ax.set_title("Choosing thresholds = choosing a risk tolerance", pad=22)
    _subtitle(ax, "Lighter = cheaper. Blank: MFA ≥ block, or more than 3% of legit logins challenged")
    ax.grid(False)
    _save(fig, out / "threshold_cost_surface.png")


def build_all(root: Path) -> list[Path]:
    _style()
    reports, out = root / "reports", root / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((reports / "metrics.json").read_text())
    scored = pd.read_csv(reports / "scored_test.csv.gz", parse_dates=["login_ts"])
    th = metrics["chosen_thresholds"]
    fig_daily(reports, metrics, out)
    fig_score_distribution(scored, th, out)
    fig_pr_curves(scored, metrics, out)
    fig_cost(metrics, out)
    fig_scenarios(metrics, out)
    fig_importance(reports, out)
    fig_cost_surface(reports, th, out)
    return sorted(out.glob("*.png"))
