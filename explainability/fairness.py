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
AGE_BIN_EDGES = [18, 30, 45, 60, 100]
AGE_BIN_LABELS = ["18-30", "30-45", "45-60", "60+"]


def age_bin_labels(age) -> pd.Series:
    """Single source of truth for age binning — used by the audit, the
    threshold calibrator, and serving/app.py, so they can never drift apart."""
    return pd.cut(pd.Series(age), bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)


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

    results["demographic_parity_pass_gender"] = bool(results["dp_gender"] <= DEMOGRAPHIC_PARITY_THRESHOLD)
    results["demographic_parity_pass_age"] = bool(results["dp_age"] <= DEMOGRAPHIC_PARITY_THRESHOLD)
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


def calibrate_group_thresholds(y_pred_proba, gender, age, target_positive_rate: float | None = None) -> dict:
    """Deterministic per-(gender, age-group) probability threshold for the
    DENY decision, calibrated so every group's denial rate matches the
    overall portfolio's denial rate at the given target — equalizing
    approval rates without any randomized/interpolated decision.

    Returns {(gender, age_bin): threshold}. A group's threshold is the
    quantile of that group's predicted probabilities corresponding to
    `target_positive_rate` — e.g. if the target denial rate is 20%, the
    threshold is set at each group's own 80th percentile of predicted risk,
    so exactly ~20% of that group is denied, same as every other group.
    """
    y_pred_proba = np.asarray(y_pred_proba)
    gender = np.asarray(gender)
    age_bins = age_bin_labels(age).astype(str).values

    if target_positive_rate is None:
        target_positive_rate = float((y_pred_proba >= 0.5).mean())

    thresholds = {}
    groups = pd.DataFrame({"gender": gender, "age_bin": age_bins}).drop_duplicates()
    for _, row in groups.iterrows():
        mask = (gender == row["gender"]) & (age_bins == row["age_bin"])
        group_proba = y_pred_proba[mask]
        if len(group_proba) == 0:
            continue
        threshold = float(np.quantile(group_proba, 1 - target_positive_rate))
        thresholds[(row["gender"], row["age_bin"])] = threshold

    return thresholds


def non_disclosure_threshold(thresholds: dict, fallback: float = 0.5) -> float:
    """The DENY threshold for an applicant whose protected-attribute group is
    unknown (declined to disclose gender, or a combination never seen during
    calibration).

    Deliberately NOT the flat/raw model's threshold — that threshold is
    exactly the one Stage 14/15 showed produces a 0.456 age demographic-
    parity gap, so falling back to it would mean non-disclosure quietly
    reintroduces the bias the calibration exists to fix. Instead, use the
    MOST LENIENT (lowest) threshold among all calibrated groups: an applicant
    who doesn't disclose is judged at least as favorably as the best-treated
    disclosed group, so choosing not to disclose can only help, never hurt,
    relative to the fairness-corrected population. This is a "benefit of the
    doubt" policy, not a mathematical fairness guarantee — see PROGRESS.md
    for the real tradeoff this makes (denial rate rises slightly for the
    disclosed population to keep the portfolio-wide rate on target, since
    non-disclosed applicants are, on average, judged more leniently).
    """
    if not thresholds:
        return fallback
    return min(thresholds.values())


def apply_group_thresholds(y_pred_proba, gender, age, thresholds: dict, default_threshold: float | None = None) -> np.ndarray:
    """Apply per-group thresholds from calibrate_group_thresholds to produce
    binary decisions. Falls back to `default_threshold` for any (gender,
    age_bin) combination not seen during calibration — or, if not given, to
    `non_disclosure_threshold(thresholds)` (see its docstring for why that's
    the safer default than a flat threshold)."""
    y_pred_proba = np.asarray(y_pred_proba)
    gender = np.asarray(gender)
    age_bins = age_bin_labels(age).astype(str).values
    fallback = default_threshold if default_threshold is not None else non_disclosure_threshold(thresholds)

    pred = np.zeros(len(y_pred_proba), dtype=int)
    for i in range(len(y_pred_proba)):
        t = thresholds.get((gender[i], age_bins[i]), fallback)
        pred[i] = int(y_pred_proba[i] >= t)
    return pred
