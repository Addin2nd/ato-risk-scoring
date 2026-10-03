import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ato.alerts import to_ecs  # noqa: E402
from ato.features import FEATURES, _window_sum, build_features  # noqa: E402
from ato.generate import GenConfig, generate  # noqa: E402
from ato.policy import CostModel, Thresholds, assign_tier, expected_cost, optimise_thresholds  # noqa: E402
from ato.rules import reason_codes, rule_hits  # noqa: E402


@pytest.fixture(scope="module")
def small_events():
    ev, _ = generate(GenConfig(n_users=150, days=40, n_stuffing_campaigns=1,
                               n_targeted_victims=6, n_bruteforce_victims=2, seed=1))
    return ev


def test_generator_schema_and_labels(small_events):
    ev = small_events
    assert ev["login_ts"].is_monotonic_increasing
    # an account takeover is always a successful login from attacker infrastructure
    ato = ev[ev["is_account_takeover"] == 1]
    assert len(ato) > 0
    assert (ato["login_successful"] == 1).all() and (ato["is_attack_ip"] == 1).all()


def test_window_sum_matches_brute_force():
    rng = np.random.default_rng(0)
    g = np.sort(rng.integers(0, 5, 300))
    t = np.concatenate([np.sort(rng.integers(0, 10_000, (g == k).sum())) for k in range(5)])
    v = rng.integers(0, 2, 300).astype(float)
    got = _window_sum(g, t, v, 500)
    want = [v[:i][(g[:i] == g[i]) & (t[:i] >= t[i] - 500)].sum() for i in range(300)]
    np.testing.assert_allclose(got, want)


def test_features_are_point_in_time(small_events):
    """Adding future events must not change features of past events (no leakage)."""
    ev = small_events
    cutoff = ev["login_ts"].quantile(0.6)
    full = build_features(ev)
    past = build_features(ev[ev["login_ts"] < cutoff])
    a = full[full["login_ts"] < cutoff][FEATURES].reset_index(drop=True)
    b = past[FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-9)


def test_new_device_flag(small_events):
    f = build_features(small_events)
    first_success = f[f["login_successful"] == 1].groupby(["user_id", "device_id"]).head(1)
    assert (first_success["new_device"] == 1).all()


def test_tiers_and_cost():
    cm = CostModel()
    y = np.array([0, 0, 1, 1])
    s = np.array([10.0, 50.0, 50.0, 95.0])
    th = Thresholds(mfa=40, block=90)
    assert list(assign_tier(s, th)) == ["LOW", "MEDIUM", "MEDIUM", "HIGH"]
    want = cm.mfa_on_legit + cm.missed_ato * cm.mfa_bypass_rate + cm.soc_review
    assert expected_cost(y, s, th, cm) == pytest.approx(want)


def test_threshold_search_respects_friction_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.01).astype(int)
    s = np.clip(rng.normal(20, 10, 5000) + 60 * y, 0, 100)
    th, res = optimise_thresholds(y, s, CostModel(), max_mfa_rate=0.02)
    assert th.mfa < th.block
    assert np.mean((s >= th.mfa) & (y == 0)) <= 0.02 + 1e-9


def test_ecs_alert_shape(small_events):
    f = build_features(small_events)
    row = f[f["is_account_takeover"] == 1].head(1).copy()
    row["reasons"] = reason_codes(row, rule_hits(row))
    row = row.assign(tier="HIGH", risk_score=93.2, p_ato=0.42, p_ml=0.6, anomaly_pct=0.99, rule_points=55)
    doc = to_ecs(row.iloc[0])
    assert doc["event"]["category"] == ["authentication"]
    assert doc["risk"]["calculated_level"] == "high"
    assert doc["event"]["action"] == "block_session_and_reset_password"
    assert isinstance(doc["ato"]["reasons"], list) and doc["ato"]["reasons"]
