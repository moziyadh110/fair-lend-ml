import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.alerting import evaluate_alerts, format_alert_message, send_alerts
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


SAMPLE_ALERT = {
    "name": "AUC degradation",
    "metric": "auc_rolling_30d",
    "threshold": 0.65,
    "direction": "below",
    "action": "trigger_retrain",
    "current_value": 0.60,
}


def test_format_alert_message_includes_key_fields():
    msg = format_alert_message(SAMPLE_ALERT)
    assert "AUC degradation" in msg
    assert "auc_rolling_30d" in msg
    assert "0.6000" in msg
    assert "trigger_retrain" in msg


def test_send_alerts_noops_without_webhook_configured():
    sent = send_alerts([SAMPLE_ALERT], webhook_url=None)
    assert sent == 0


def test_send_alerts_posts_to_webhook():
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()

    with patch("monitoring.alerting.requests.post", return_value=fake_response) as mock_post:
        sent = send_alerts([SAMPLE_ALERT, SAMPLE_ALERT], webhook_url="https://hooks.slack.test/fake")

    assert sent == 2
    assert mock_post.call_count == 2
    assert mock_post.call_args.args[0] == "https://hooks.slack.test/fake"
    assert "AUC degradation" in mock_post.call_args.kwargs["json"]["text"]
