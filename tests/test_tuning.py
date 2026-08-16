import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.tuning.optuna_tuner import tune
from models.tuning.stability_analysis import stability_check


def _synthetic_classification_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    logits = 1.5 * X["f1"] - 1.0 * X["f2"]
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logits))))
    return X, y


def test_optuna_tuner_finds_a_valid_param_set():
    X, y = _synthetic_classification_data()
    study = tune(X, y, n_trials=3, n_splits=2)
    assert study.best_value > 0.5  # better than random on a signal-bearing dataset
    assert "max_depth" in study.best_params


def test_stability_check_reports_low_variance_on_easy_data():
    X, y = _synthetic_classification_data(n=800)
    split = 600
    result = stability_check(
        {"max_depth": 3, "n_estimators": 50},
        X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:],
        n_seeds=3,
    )
    assert 0 <= result["auc_std"] < 0.1
    assert result["auc_mean"] > 0.5
