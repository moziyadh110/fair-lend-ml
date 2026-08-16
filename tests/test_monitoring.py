import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.alerting import evaluate_alerts
from monitoring.drift_detector import compute_all_psi, retrain_recommended
import numpy as np
import pandas as pd


def test_evaluate_alerts_fires_on_breach():
    fired = evaluate_alerts({"auc_rolling_30d": 0.60, "api_error_rate": 0.001})
    names = {a["name"] for a in fired}
    assert "AUC degradation" in names
    assert "Error rate" not in names


def test_evaluate_alerts_ignores_missing_metrics():
    fired = evaluate_alerts({"unrelated_metric": 1.0})
    assert fired == []


def test_psi_and_retrain_recommendation_on_shifted_data():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"f1": rng.normal(size=2000), "f2": rng.normal(size=2000)})
    current = pd.DataFrame({"f1": rng.normal(loc=3, size=2000), "f2": rng.normal(loc=3, size=2000)})
    psi_results = compute_all_psi(reference, current, ["f1", "f2"])
    assert all(r["status"] == "retrain" for r in psi_results.values())
    assert retrain_recommended(psi_results) is True


def test_no_retrain_recommendation_on_stable_data():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"f1": rng.normal(size=2000)})
    current = pd.DataFrame({"f1": rng.normal(size=2000)})
    psi_results = compute_all_psi(reference, current, ["f1"])
    assert retrain_recommended(psi_results) is False
