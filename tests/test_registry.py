import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import (
    _deserialize_group_thresholds,
    _serialize_group_thresholds,
    load_champion,
    register_champion,
)


@pytest.fixture(autouse=True)
def isolated_mlflow_registry(tmp_path):
    """Without this, these tests register dummy models under the SAME
    `credit_default` model name and `champion` alias as the real production
    registration (models/train_champion.py) — a real bug caught by exactly
    this: running the test suite silently overwrote the real CatBoost
    champion with a test LogisticRegression dummy, which then broke
    serving/app.py at startup. Every test in this file gets its own
    throwaway tracking store instead."""
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'test_mlflow.db'}")
    yield
    mlflow.set_tracking_uri(original_uri)


def test_group_thresholds_round_trip_through_json_serialization():
    thresholds = {("F", "18-30"): 0.7, ("M", "60+"): 0.4}
    records = _serialize_group_thresholds(thresholds)
    restored = _deserialize_group_thresholds(records)
    assert restored == thresholds


def test_register_and_load_champion_round_trip():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    y = (X["f1"] + X["f2"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    feature_cols = ["f1", "f2"]
    group_thresholds = {("F", "18-30"): 0.55, ("M", "30-45"): 0.6}

    version = register_champion(
        model, feature_cols, group_thresholds, default_threshold=0.5,
        metrics={"auc_roc": 0.99}, cleared_for_production=True,
    )
    assert version is not None

    loaded_model, metadata = load_champion()
    assert metadata["feature_cols"] == feature_cols
    assert metadata["group_thresholds"] == group_thresholds
    assert metadata["default_threshold"] == 0.5
    assert np.allclose(loaded_model.predict_proba(X), model.predict_proba(X))


def test_low_auc_model_is_registered_but_not_promoted():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=100)})
    y = rng.integers(0, 2, 100)
    model = LogisticRegression().fit(X, y)

    version = register_champion(
        model, ["f1"], {}, default_threshold=0.5,
        metrics={"auc_roc": 0.51}, cleared_for_production=True,
    )
    assert version is None


def test_not_cleared_for_production_blocks_promotion_regardless_of_auc():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    y = (X["f1"] + X["f2"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    version = register_champion(
        model, ["f1", "f2"], {}, default_threshold=0.5,
        metrics={"auc_roc": 0.95}, cleared_for_production=False,
    )
    assert version is None
