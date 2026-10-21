"""Fairlearn-based fairness audit and mitigation. This is the module that
decides whether a model is even allowed to ship — see PROGRESS.md Tradeoff 1
(AUC vs. fairness) for why a model that fails this gate has zero deployment
value regardless of its AUC.
"""
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DEMOGRAPHIC_PARITY_THRESHOLD = 0.05
FOUR_FIFTHS_THRESHOLD = 0.80
# Left-closed bins so a 62-year-old lands in "62+": Reg B (12 CFR 1002.6(b)(2))
# protects applicants aged 62 or older from being treated less favorably
# because of their age, and the last bin has to line up with that cutoff.
AGE_BIN_EDGES = [18, 30, 45, 62, np.inf]
AGE_BIN_LABELS = ["18-29", "30-44", "45-61", "62+"]
ELDERLY_AGE_BIN = "62+"


def age_bin_labels(age) -> pd.Series:
    """Single source of truth for age binning — used by the audit, the
    threshold calibrator, and serving/app.py, so they can never drift apart."""
    return pd.cut(pd.Series(age), bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, right=False)


def full_fairness_audit(y_true, y_pred, y_pred_proba, gender, age) -> dict:
    """Complete fairness audit across gender and age-group protected classes."""
    y_true, y_pred, y_pred_proba = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_pred_proba)
    gender = np.asarray(gender)

    results = {}
    results["dp_gender"] = demographic_parity_difference(y_true, y_pred, sensitive_features=gender)
    results["eo_gender"] = equalized_odds_difference(y_true, y_pred, sensitive_features=gender)

    age_bins = age_bin_labels(age).values
    results["dp_age"] = demographic_parity_difference(y_true, y_pred, sensitive_features=age_bins)

    mf = MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=y_pred_proba, sensitive_features=gender)
    results["auc_disparity_gender"] = mf.difference()

    # Four-fifths rule: min group approval rate / max group approval rate >= 0.80
    approval_rates = {}
    for group in np.unique(gender):
        mask = gender == group
        approval_rates[str(group)] = float((y_pred[mask] == 0).mean())

    min_rate, max_rate = min(approval_rates.values()), max(approval_rates.values())
    results["four_fifths_ratio"] = min_rate / max_rate if max_rate > 0 else 0.0
    results["four_fifths_pass"] = bool(results["four_fifths_ratio"] >= FOUR_FIFTHS_THRESHOLD)
    results["approval_rates_by_group"] = approval_rates

    # Age approval rates too — age is an ECOA-protected class same as gender,
    # and a gate that only checks gender gives a false "all clear" if age
    # disparity is the one that's actually large. See PROGRESS.md Stage 14.
    age_approval_rates = {}
    for group in pd.Series(age_bins).dropna().unique():
        mask = age_bins == group
        age_approval_rates[str(group)] = float((y_pred[mask] == 0).mean())
    results["approval_rates_by_age_group"] = age_approval_rates

    # The age gate is Reg B-aware: approval rates must match across the
    # under-62 groups, and the 62+ group must not be approved less often than
    # any of them. Reg B lets a creditor favor applicants 62 or older, and the
    # calibrated thresholds do exactly that, so `dp_age` across all four bins
    # is still reported but would penalize the favoring the law permits.
    # "Not approved less often" allows 2 standard errors of sampling noise:
    # a strict >= fails a perfectly age-blind model about half the time. The
    # hard guarantee is the cutoff itself — calibrate_group_thresholds gives
    # 62+ the most lenient one — this is the outcome-level check on top.
    under_62 = age_bins != ELDERLY_AGE_BIN
    results["dp_age_under_62"] = demographic_parity_difference(
        y_true[under_62], y_pred[under_62], sensitive_features=age_bins[under_62]
    )
    elderly_rate = age_approval_rates.get(ELDERLY_AGE_BIN)
    other_rates = [v for k, v in age_approval_rates.items() if k != ELDERLY_AGE_BIN]
    if elderly_rate is None or not other_rates:
        results["elderly_not_disadvantaged"] = True
    else:
        n_elderly = int((age_bins == ELDERLY_AGE_BIN).sum())
        std_err = np.sqrt(elderly_rate * (1 - elderly_rate) / n_elderly)
        results["elderly_not_disadvantaged"] = bool(elderly_rate >= max(other_rates) - 2 * std_err)

    results["demographic_parity_pass_gender"] = bool(results["dp_gender"] <= DEMOGRAPHIC_PARITY_THRESHOLD)
    results["demographic_parity_pass_age"] = bool(
        results["dp_age_under_62"] <= DEMOGRAPHIC_PARITY_THRESHOLD and results["elderly_not_disadvantaged"]
    )
    results["demographic_parity_pass"] = bool(
        results["demographic_parity_pass_gender"] and results["demographic_parity_pass_age"]
    )

    return results


def retrain_with_fairness_constraints(X_train, y_train, sensitive_features, estimator=None,
                                       constraint: str = "dp", max_iter: int = 50):
    """Retrain with fairness constraints baked into the model itself, instead
    of correcting the decision after the fact (see `calibrate_group_thresholds`
    below, and PROGRESS.md Stage 15 for why that post-hoc approach exists too).

    `estimator` defaults to LogisticRegression if not given, but any estimator
    with `fit(X, y, sample_weight)` / `predict_proba(X)` works — pass the same
    model family as the champion (e.g. CatBoost) to keep this comparable.
    """
    constraint_obj = DemographicParity() if constraint == "dp" else EqualizedOdds()
    estimator = estimator if estimator is not None else LogisticRegression(max_iter=1000)

    mitigated = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraint_obj,
        max_iter=max_iter,
    )
    mitigated.fit(X_train, y_train, sensitive_features=sensitive_features)
    return mitigated


def _weighted_ensemble_proba(predictors, weights, X) -> np.ndarray:
    weighted_proba = np.zeros(len(X))
    for predictor, weight in zip(predictors, weights):
        if weight == 0:
            continue
        weighted_proba += weight * predictor.predict_proba(X)[:, 1]
    return weighted_proba


def exponentiated_gradient_predict_proba(mitigated: ExponentiatedGradient, X) -> np.ndarray:
    """Deterministic probability from a fitted ExponentiatedGradient.

    `.predict()` on this object is intentionally randomized (Fairlearn's own
    docs: "repeatedly calling predict with the same feature data may yield
    different output") — the reduction algorithm produces a *mixture* of
    classifiers (`.predictors_`, `.weights_`) and satisfies the fairness
    constraint only in expectation over random draws from that mixture. That
    randomization is the same problem this project already avoided for
    post-hoc thresholding (see `optimize_thresholds` above) — deploying it
    here would reintroduce the exact flaw that was engineered around.

    Instead, this computes the weighted-average predicted probability across
    the ensemble (weights sum to 1), which is deterministic: same input,
    same output, every time.
    """
    return _weighted_ensemble_proba(mitigated.predictors_, mitigated.weights_, X)


class FairEnsembleModel:
    """Picklable, sklearn-`predict_proba`-shaped wrapper around a fitted
    ExponentiatedGradient's mixture of sub-estimators, using the deterministic
    weighted average (see `exponentiated_gradient_predict_proba`) instead of
    the library's randomized `.predict()`.

    Exists so `serving/app.py` and `serving/batch_predict.py` don't need to
    know or care whether the deployed champion is a single CatBoost model or
    a fairness-constrained ensemble — both expose `.predict_proba(X)` the
    same way. `dominant_predictor` (the single highest-weight sub-model) is
    used for SHAP explanations, since SHAP's TreeExplainer needs one concrete
    tree model, not a weighted mixture — see PROGRESS.md Stage 16 for why
    that's a reasonable approximation rather than a compromise being hidden.
    """

    def __init__(self, predictors: list, weights: np.ndarray):
        self.predictors = list(predictors)
        self.weights = np.asarray(weights)

    @classmethod
    def from_fitted(cls, mitigated: ExponentiatedGradient) -> "FairEnsembleModel":
        # Drop zero-weight predictors — they never contribute to predict_proba
        # and just bloat the pickled artifact.
        keep = [(p, w) for p, w in zip(mitigated.predictors_, mitigated.weights_) if w > 0]
        predictors, weights = zip(*keep)
        return cls(predictors, np.array(weights))

    def predict_proba(self, X) -> np.ndarray:
        p1 = _weighted_ensemble_proba(self.predictors, self.weights, X)
        return np.column_stack([1 - p1, p1])

    @property
    def dominant_predictor(self):
        return self.predictors[int(np.argmax(self.weights))]


def optimize_thresholds(estimator, X, y, sensitive_features, constraints: str = "demographic_parity"):
    """Post-hoc per-group threshold adjustment via Fairlearn's ThresholdOptimizer.

    NOT used for the deployed model: ThresholdOptimizer's default behavior
    interpolates between two thresholds using a randomized decision for some
    applicants near the boundary, to hit an exact fairness target. That means
    two identical applicants in the same group can get different outcomes by
    chance — a real problem for a lending decision, not just an implementation
    detail. Kept here as the audit/validation tool that proves mitigation is
    possible (see PROGRESS.md Stage 14) — `calibrate_group_thresholds` below
    is the deterministic version actually wired into serving.
    """
    optimizer = ThresholdOptimizer(estimator=estimator, constraints=constraints, prefit=True)
    optimizer.fit(X, y, sensitive_features=sensitive_features)
    return optimizer


def calibrate_group_thresholds(y_pred_proba, age, target_positive_rate: float | None = None) -> dict:
    """Deterministic per-age-group probability threshold for the DENY
    decision, calibrated so every group's denial rate matches the overall
    portfolio's denial rate at the given target — equalizing approval rates
    without any randomized/interpolated decision.

    Keyed on age group only. Gender used to be part of the key, but setting
    a different cutoff by sex is disparate treatment under Reg B however
    good the intent, so gender is now audited but never decides anything.

    Returns {age_bin: threshold}. A group's threshold is the quantile of
    that group's predicted probabilities corresponding to
    `target_positive_rate` — e.g. if the target denial rate is 20%, the
    threshold is set at each group's own 80th percentile of predicted risk,
    so ~20% of that group is denied, same as every other group.

    One override: the 62+ group always gets the most lenient (highest)
    threshold of any group. Equal-rate calibration alone would give it the
    strictest cutoff, since the model scores older applicants as lower risk,
    and Reg B doesn't allow an applicant 62 or older to be treated less
    favorably because of age. This costs some approval-rate parity on
    purpose; the audit reports the gap honestly rather than hiding it.
    """
    y_pred_proba = np.asarray(y_pred_proba)
    age_bins = age_bin_labels(age).astype(str).values

    if target_positive_rate is None:
        target_positive_rate = float((y_pred_proba >= 0.5).mean())

    thresholds = {}
    for age_bin in AGE_BIN_LABELS:
        group_proba = y_pred_proba[age_bins == age_bin]
        if len(group_proba) == 0:
            continue
        thresholds[age_bin] = float(np.quantile(group_proba, 1 - target_positive_rate))

    if ELDERLY_AGE_BIN in thresholds:
        thresholds[ELDERLY_AGE_BIN] = max(thresholds.values())

    return thresholds


def fallback_threshold(thresholds: dict, fallback: float = 0.5) -> float:
    """The DENY threshold for an applicant whose age group has no calibrated
    threshold (e.g. a group too small to appear in the calibration data).

    Deliberately NOT the flat/raw model's threshold — that threshold is
    exactly the one that produced a 0.408 age demographic-parity gap, so
    falling back to it would quietly reintroduce the bias the calibration
    exists to fix. Instead, use the MOST LENIENT threshold among all
    calibrated groups: an applicant DENY is `proba >= threshold`, so the
    most lenient cutoff is the HIGHEST one. (An earlier version took the
    lowest here, which is the strictest — the opposite of what it claimed.)
    """
    if not thresholds:
        return fallback
    return max(thresholds.values())


def apply_group_thresholds(y_pred_proba, age, thresholds: dict, default_threshold: float | None = None) -> np.ndarray:
    """Apply per-age-group thresholds from calibrate_group_thresholds to
    produce binary decisions. Falls back to `default_threshold` for any age
    group not seen during calibration — or, if not given, to
    `fallback_threshold(thresholds)` (see its docstring for why that's the
    safer default than a flat threshold)."""
    y_pred_proba = np.asarray(y_pred_proba)
    age_bins = age_bin_labels(age).astype(str).values
    fallback = default_threshold if default_threshold is not None else fallback_threshold(thresholds)

    pred = np.zeros(len(y_pred_proba), dtype=int)
    for i in range(len(y_pred_proba)):
        t = thresholds.get(age_bins[i], fallback)
        pred[i] = int(y_pred_proba[i] >= t)
    return pred
