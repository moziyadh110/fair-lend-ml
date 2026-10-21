import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explainability.fairness import (
    FairEnsembleModel,
    apply_group_thresholds,
    calibrate_group_thresholds,
    exponentiated_gradient_predict_proba,
    full_fairness_audit,
    retrain_with_fairness_constraints,
)


def _biased_predictions(n=2000, seed=0):
    """Group B gets systematically higher predicted default (and denial) rates
    than Group A, with no difference in the true label — a textbook unfair model."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["M", "F"], size=n)
    age = rng.uniform(20, 70, size=n)
    y_true = rng.integers(0, 2, n)

    y_pred = y_true.copy()
    female_mask = gender == "F"
    # Force extra denials (y_pred=1) for group F regardless of true label.
    flip_idx = np.where(female_mask & (y_true == 0))[0][: int(female_mask.sum() * 0.3)]
    y_pred[flip_idx] = 1

    y_pred_proba = y_pred.astype(float) * 0.7 + rng.uniform(0, 0.1, n)
    return y_true, y_pred, y_pred_proba, gender, age


def _fair_predictions(n=20000, seed=0):
    # Larger n than _biased_predictions: dp_age compares 4 age-bin groups
    # (vs. 2 for gender), so it needs more samples before random noise
    # reliably stays under the 0.05 threshold.
    rng = np.random.default_rng(seed)
    gender = rng.choice(["M", "F"], size=n)
    age = rng.uniform(20, 70, size=n)
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    y_pred_proba = y_pred.astype(float) * 0.7 + rng.uniform(0, 0.1, n)
    return y_true, y_pred, y_pred_proba, gender, age


def test_audit_flags_biased_model():
    y_true, y_pred, y_pred_proba, gender, age = _biased_predictions()
    results = full_fairness_audit(y_true, y_pred, y_pred_proba, gender, age)
    assert results["dp_gender"] > 0.05
    assert results["demographic_parity_pass"] is False


def test_audit_passes_fair_model():
    y_true, y_pred, y_pred_proba, gender, age = _fair_predictions()
    results = full_fairness_audit(y_true, y_pred, y_pred_proba, gender, age)
    assert results["dp_gender"] < 0.05
    assert results["demographic_parity_pass"] is True


def test_four_fifths_rule_computed_per_group():
    y_true, y_pred, y_pred_proba, gender, age = _biased_predictions()
    results = full_fairness_audit(y_true, y_pred, y_pred_proba, gender, age)
    assert "M" in results["approval_rates_by_group"]
    assert "F" in results["approval_rates_by_group"]
    assert 0 <= results["four_fifths_ratio"] <= 1


def _age_biased_proba(n=20000, seed=0):
    """Predicted probability rises steeply with youth — mirrors the real
    finding in this project (18-30 gets denied far more than 60+)."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["M", "F"], size=n)
    age = rng.uniform(18, 90, size=n)
    y_true = rng.integers(0, 2, n)
    proba = np.clip(0.9 - (age - 18) / 90 + rng.normal(0, 0.05, n), 0, 1)
    return y_true, proba, gender, age


def test_calibrate_group_thresholds_equalizes_denial_rates():
    y_true, proba, gender, age = _age_biased_proba()
    raw_pred = (proba >= 0.5).astype(int)
    before = full_fairness_audit(y_true, raw_pred, proba, gender, age)
    assert before["dp_age"] > 0.05  # confirms the synthetic data reproduces a real age gap

    thresholds = calibrate_group_thresholds(proba, age)
    calibrated_pred = apply_group_thresholds(proba, age, thresholds)
    after = full_fairness_audit(y_true, calibrated_pred, proba, gender, age)

    # Parity is only required below 62; 62+ is deliberately favored (Reg B).
    assert after["dp_age_under_62"] < before["dp_age_under_62"]
    assert after["dp_age_under_62"] < 0.05
    assert after["demographic_parity_pass_age"] is True


def test_calibrated_thresholds_are_deterministic():
    """Same input, same output every time — no randomized decision boundary."""
    y_true, proba, gender, age = _age_biased_proba()
    thresholds = calibrate_group_thresholds(proba, age)
    pred1 = apply_group_thresholds(proba, age, thresholds)
    pred2 = apply_group_thresholds(proba, age, thresholds)
    assert np.array_equal(pred1, pred2)


def test_unseen_group_falls_back_to_default_threshold():
    thresholds = {"45-61": 0.7}
    proba = np.array([0.6, 0.8])
    age = np.array([25, 25])
    pred = apply_group_thresholds(proba, age, thresholds, default_threshold=0.5)
    assert list(pred) == [1, 1]  # both >= default 0.5, group not in thresholds dict


def test_exponentiated_gradient_predict_proba_is_deterministic():
    """.predict() on a fitted ExponentiatedGradient is randomized by Fairlearn's
    own design (see docstring) — the deterministic weighted-average alternative
    this project deploys must actually be deterministic across repeated calls."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    y_true, proba, gender, age = _age_biased_proba(n=2000)
    X = pd.DataFrame({"proba_feature": proba})  # a feature correlated with the label, for a fast-fitting toy model
    mitigated = retrain_with_fairness_constraints(
        X, y_true, sensitive_features=gender, estimator=LogisticRegression(max_iter=200), max_iter=5,
    )
    p1 = exponentiated_gradient_predict_proba(mitigated, X)
    p2 = exponentiated_gradient_predict_proba(mitigated, X)
    assert np.array_equal(p1, p2)
    assert np.all((p1 >= 0) & (p1 <= 1))


def test_fairness_constrained_retraining_reduces_demographic_parity_gap():
    """The whole point: a model trained WITH the constraint should be fairer
    on its own than the same architecture trained without it, at a flat
    0.5 threshold — no post-hoc correction needed."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    n = 4000
    rng = np.random.default_rng(0)
    gender = rng.choice(["M", "F"], size=n)
    # A feature that's both predictive of the label AND correlated with gender —
    # an unconstrained model will lean on it and inherit the gender disparity.
    biased_feature = rng.normal(size=n) + np.where(gender == "F", 1.5, 0)
    y_true = (biased_feature + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    X = pd.DataFrame({"biased_feature": biased_feature})

    unconstrained = LogisticRegression(max_iter=200).fit(X, y_true)
    unconstrained_proba = unconstrained.predict_proba(X)[:, 1]
    unconstrained_pred = (unconstrained_proba >= 0.5).astype(int)
    dp_unconstrained = full_fairness_audit(
        y_true, unconstrained_pred, unconstrained_proba, gender, rng.uniform(25, 60, n)
    )["dp_gender"]

    mitigated = retrain_with_fairness_constraints(
        X, y_true, sensitive_features=gender, estimator=LogisticRegression(max_iter=200), max_iter=15,
    )
    mitigated_proba = exponentiated_gradient_predict_proba(mitigated, X)
    mitigated_pred = (mitigated_proba >= 0.5).astype(int)
    dp_mitigated = full_fairness_audit(
        y_true, mitigated_pred, mitigated_proba, gender, rng.uniform(25, 60, n)
    )["dp_gender"]

    assert dp_mitigated < dp_unconstrained


def test_fair_ensemble_model_predict_proba_matches_weighted_average():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    y_true, proba, gender, age = _age_biased_proba(n=2000)
    X = pd.DataFrame({"proba_feature": proba})
    mitigated = retrain_with_fairness_constraints(
        X, y_true, sensitive_features=gender, estimator=LogisticRegression(max_iter=200), max_iter=5,
    )
    wrapped = FairEnsembleModel.from_fitted(mitigated)

    expected = exponentiated_gradient_predict_proba(mitigated, X)
    actual = wrapped.predict_proba(X)[:, 1]
    assert np.allclose(expected, actual)
    assert wrapped.dominant_predictor is not None


def test_fair_ensemble_model_is_picklable():
    import pickle

    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    y_true, proba, gender, age = _age_biased_proba(n=2000)
    X = pd.DataFrame({"proba_feature": proba})
    mitigated = retrain_with_fairness_constraints(
        X, y_true, sensitive_features=gender, estimator=LogisticRegression(max_iter=200), max_iter=5,
    )
    wrapped = FairEnsembleModel.from_fitted(mitigated)

    restored = pickle.loads(pickle.dumps(wrapped))
    assert np.allclose(wrapped.predict_proba(X), restored.predict_proba(X))


def test_fallback_threshold_is_most_lenient():
    """DENY is proba >= threshold, so the most lenient cutoff is the highest.
    Guards against the old bug where this took the minimum (the strictest)."""
    from explainability.fairness import fallback_threshold

    thresholds = {"18-29": 0.7, "30-44": 0.4, "62+": 0.6}
    assert fallback_threshold(thresholds) == 0.7


def test_fallback_threshold_falls_back_when_empty():
    from explainability.fairness import fallback_threshold

    assert fallback_threshold({}, fallback=0.55) == 0.55


def test_apply_group_thresholds_defaults_to_lenient_fallback_for_unseen_group():
    thresholds = {"30-44": 0.8, "45-61": 0.3}
    proba = np.array([0.5])  # denied at 0.3, approved at the lenient 0.8 fallback
    age = np.array([25])  # 18-29 isn't in the dict
    pred = apply_group_thresholds(proba, age, thresholds)  # no default_threshold passed
    assert pred[0] == 0


def test_age_bins_put_62_year_olds_in_the_elderly_group():
    from explainability.fairness import age_bin_labels

    labels = age_bin_labels([18, 29.9, 30, 61.9, 62, 100]).astype(str).tolist()
    assert labels == ["18-29", "18-29", "30-44", "45-61", "62+", "62+"]


def test_applicants_62_plus_always_get_the_most_favorable_cutoff():
    """Reg B: an applicant 62 or older can't be treated less favorably because
    of age. Equal-rate calibration alone would give 62+ the strictest cutoff,
    since the model scores older applicants as lower risk."""
    _, proba, _, age = _age_biased_proba()
    thresholds = calibrate_group_thresholds(proba, age)
    assert thresholds["62+"] == max(thresholds.values())


def test_62_plus_applicant_approved_whenever_any_younger_applicant_with_same_score_is():
    _, proba, _, age = _age_biased_proba()
    thresholds = calibrate_group_thresholds(proba, age)
    for score in np.linspace(0, 1, 101):
        elderly_denied = apply_group_thresholds([score], [70], thresholds)[0]
        younger_denied = [apply_group_thresholds([score], [a], thresholds)[0] for a in (22, 35, 50)]
        assert elderly_denied <= min(younger_denied)


def test_age_gate_fails_when_62_plus_is_approved_less_than_other_groups():
    rng = np.random.default_rng(0)
    n = 20000
    age = rng.uniform(18, 90, size=n)
    y_true = rng.integers(0, 2, n)
    y_pred = np.zeros(n, dtype=int)
    elderly = age >= 62
    y_pred[elderly & (rng.random(n) < 0.2)] = 1  # deny 20% of 62+, nobody else
    gender = rng.choice(["M", "F"], size=n)
    results = full_fairness_audit(y_true, y_pred, y_pred.astype(float), gender, age)
    assert results["dp_age_under_62"] == 0
    assert results["elderly_not_disadvantaged"] is False
    assert results["demographic_parity_pass_age"] is False


def test_age_gate_allows_favoring_62_plus():
    rng = np.random.default_rng(0)
    n = 20000
    age = rng.uniform(18, 90, size=n)
    y_true = rng.integers(0, 2, n)
    y_pred = (rng.random(n) < 0.3).astype(int)
    y_pred[age >= 62] = 0  # approve every 62+ applicant
    gender = rng.choice(["M", "F"], size=n)
    results = full_fairness_audit(y_true, y_pred, y_pred.astype(float), gender, age)
    assert results["dp_age"] > 0.05  # the all-ages gap is large on purpose...
    assert results["demographic_parity_pass_age"] is True  # ...and still passes
