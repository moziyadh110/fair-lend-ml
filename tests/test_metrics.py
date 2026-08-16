import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.credit_metrics import (
    full_evaluation_report,
    gini_coefficient,
    ks_statistic,
    lift_chart,
    population_stability_index,
)


def _perfect_separation_data(n=1000):
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, n)
    # Predicted probability perfectly tracks the label plus a little noise.
    proba = y * 0.8 + rng.uniform(0, 0.2, n)
    return y, proba


def test_ks_and_gini_are_high_for_well_separated_predictions():
    y, proba = _perfect_separation_data()
    assert ks_statistic(y, proba) > 0.5
    assert gini_coefficient(y, proba) > 0.5


def test_gini_is_zero_for_random_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    proba = rng.uniform(0, 1, 5000)
    assert abs(gini_coefficient(y, proba)) < 0.1


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=5000)
    assert population_stability_index(sample, sample) < 0.01


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    expected = rng.normal(loc=0, size=5000)
    actual = rng.normal(loc=3, size=5000)
    assert population_stability_index(expected, actual) > 0.25


def test_lift_top_decile_exceeds_one_for_informative_model():
    y, proba = _perfect_separation_data()
    lift = lift_chart(y, proba)
    assert lift.iloc[-1] > 1.0


def test_full_evaluation_report_has_expected_keys():
    y, proba = _perfect_separation_data()
    report = full_evaluation_report(y, proba, model_name="test")
    assert set(report) == {"model", "auc_roc", "ks_statistic", "gini", "lift_top_decile"}
