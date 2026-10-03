"""Hybrid login risk engine: rules + Isolation Forest + gradient boosting -> 0-100 score.

Layers
------
1. Rules            : known attack patterns, transparent (rules.py)
2. Isolation Forest : unsupervised, label-free -> catches *novel* behaviour
3. Gradient boosting: supervised on confirmed ATO cases (labels from fraud ops / chargebacks)
4. Stacker          : logistic regression that blends the three signals into a probability.
                      Fitted on out-of-fold, time-ordered predictions to avoid leakage.

Score scale
-----------
risk_score = 25 * (log10(p) + 4), clipped to [0, 100]
  ->  every +25 points = 10x more likely to be an account takeover
      0 = p <= 0.01%,  25 = 0.1%,  50 = 1%,  75 = 10%,  100 = 100%
A log scale spreads out the tiny probabilities where almost all traffic lives,
and is easy to explain to non-technical stakeholders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURES
from .rules import rule_hits, rule_points

TARGET = "is_account_takeover"

ISO_FEATURES = [
    "new_country", "new_city", "new_asn", "new_device", "new_platform",
    "dist_prev_km", "geo_velocity_kmh", "dist_first_km", "hour_deviation",
    "user_fails_1h", "ip_fail_ratio_24h", "ip_distinct_users_24h",
    "is_hosting_asn", "is_foreign", "ua_automation", "country_global_share",
]


def _gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15, early_stopping=False,
        l2_regularization=1.0, min_samples_leaf=40, random_state=7,
    )


def prob_to_score(p: np.ndarray) -> np.ndarray:
    return np.clip(25 * (np.log10(np.clip(p, 1e-12, 1)) + 4), 0, 100)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class RiskEngine:
    gbm: HistGradientBoostingClassifier = field(default_factory=_gbm)
    iso: IsolationForest = field(default_factory=lambda: IsolationForest(n_estimators=300, random_state=7))
    stacker: LogisticRegression = field(default_factory=lambda: LogisticRegression(C=1.0))
    _iso_ref: np.ndarray | None = None

    # ---- component signals ---------------------------------------------------------------
    def _anomaly_pct(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation Forest score as a percentile of the training distribution (0-1)."""
        raw = -self.iso.score_samples(X[ISO_FEATURES])
        return np.searchsorted(self._iso_ref, raw, side="right") / len(self._iso_ref)

    def _stack_inputs(self, p_ml, anomaly, rules) -> np.ndarray:
        return np.column_stack([_logit(p_ml), anomaly, rules / 100.0])

    # ---- training ------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame, n_splits: int = 4) -> "RiskEngine":
        train = train.sort_values("login_ts")
        y = train[TARGET].to_numpy()

        self.iso.fit(train[ISO_FEATURES])
        self._iso_ref = np.sort(-self.iso.score_samples(train[ISO_FEATURES]))

        # Out-of-fold GBM predictions in time order -> honest inputs for the stacker
        oof = np.full(len(train), np.nan)
        for fit_idx, pred_idx in TimeSeriesSplit(n_splits=n_splits).split(train):
            m = _gbm().fit(train.iloc[fit_idx][FEATURES], y[fit_idx])
            oof[pred_idx] = m.predict_proba(train.iloc[pred_idx][FEATURES])[:, 1]
        mask = ~np.isnan(oof)
        Z = self._stack_inputs(oof[mask], self._anomaly_pct(train[mask]),
                               rule_points(rule_hits(train[mask])).to_numpy())
        self.stacker.fit(Z, y[mask])

        self.gbm.fit(train[FEATURES], y)
        return self

    # ---- inference -----------------------------------------------------------------------
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        hits = rule_hits(df)
        pts = rule_points(hits).to_numpy()
        p_ml = self.gbm.predict_proba(df[FEATURES])[:, 1]
        anomaly = self._anomaly_pct(df)
        p = self.stacker.predict_proba(self._stack_inputs(p_ml, anomaly, pts))[:, 1]
        return pd.DataFrame({
            "p_ml": p_ml,
            "anomaly_pct": anomaly,
            "rule_points": pts,
            "p_ato": p,
            "risk_score": prob_to_score(p).round(1),
        }, index=df.index)

    def stacker_weights(self) -> dict[str, float]:
        names = ["ml_logit", "anomaly_pct", "rule_points/100"]
        w = dict(zip(names, self.stacker.coef_[0].round(3).tolist()))
        w["intercept"] = round(float(self.stacker.intercept_[0]), 3)
        return w
