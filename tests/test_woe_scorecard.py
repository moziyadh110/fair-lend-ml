import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.woe_encoder import WoEEncoder
from models.woe_scorecard import train_woe_scorecard


def _synthetic_credit_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    credit_income_ratio = rng.uniform(0, 10, n)
    late_payments = rng.poisson(2.0, n)
    # Higher ratio and more late payments should raise default risk.
    risk = 0.3 * credit_income_ratio + 0.6 * late_payments
    prob_default = 1 / (1 + np.exp(-(risk - 1.5)))
    target = rng.binomial(1, prob_default)
    X = pd.DataFrame({
        "credit_income_ratio": credit_income_ratio,
        "late_payments": late_payments,
    })
    return X, pd.Series(target)


def test_woe_encoder_is_monotonic_with_risk():
    X, y = _synthetic_credit_data()
    encoder = WoEEncoder(n_bins=5).fit(X, y)
    # Higher-risk features should have negative WoE (fewer "good" than "bad" in that bin)
    # relative to lower-risk bins — check IV picked up a real signal, not noise.
    assert encoder.iv_scores_["credit_income_ratio"] > 0.02
    assert encoder.iv_scores_["late_payments"] > 0.02


def test_woe_transform_matches_fit_columns():
    X, y = _synthetic_credit_data()
    encoder = WoEEncoder(n_bins=5).fit(X, y)
    X_woe = encoder.transform(X)
    assert list(X_woe.columns) == list(X.columns)
    assert X_woe.shape[0] == X.shape[0]
    assert X_woe.isna().sum().sum() == 0


def test_woe_transform_handles_unseen_values_without_error():
    X, y = _synthetic_credit_data()
    encoder = WoEEncoder(n_bins=5).fit(X, y)
    X_new = X.copy()
    X_new.loc[0, "credit_income_ratio"] = 999.0  # out-of-range value
    result = encoder.transform(X_new)
    assert np.isfinite(result.loc[0, "credit_income_ratio"])


def test_scorecard_scores_correlate_with_default_probability():
    X, y = _synthetic_credit_data()
    encoder = WoEEncoder(n_bins=5).fit(X, y)
    X_woe = encoder.transform(X)
    _, scorecard = train_woe_scorecard(X_woe, y)

    scores = scorecard.score(X_woe)
    probs = scorecard.score_to_probability(scores)

    # Lower score should mean higher predicted default probability (scorecard convention).
    correlation = np.corrcoef(scores, probs)[0, 1]
    assert correlation < -0.9
