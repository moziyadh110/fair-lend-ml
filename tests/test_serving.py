import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "champion_model.joblib").exists(),
    reason="Run models/train_champion.py first to produce artifacts/champion_model.joblib",
)

AUTH_HEADERS = {"X-API-Key": os.environ.get("API_KEY", "")}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serving.app import app
    with TestClient(app) as c:
        yield c


VALID_APPLICATION = {
    "amt_income_total": 150000,
    "amt_credit": 500000,
    "amt_annuity": 25000,
    "amt_goods_price": 450000,
    "age_years": 35,
    "years_employed": 5,
    "owns_car": 1,
    "owns_realty": 1,
    "cnt_children": 0,
    "bureau_loan_count": 2,
    "active_loans": 1,
    "closed_loans": 1,
    "avg_days_overdue": 0,
    "max_days_overdue": 0,
    "bureau_debt_ratio": 0.2,
    "total_bureau_credit": 100000,
    "prev_app_count": 1,
    "prev_approved": 1,
    "prev_refused": 0,
    "credit_application_ratio": 1.0,
    "late_payments": 0,
    "avg_days_late": 0,
    "payment_volatility": 0,
    "total_installments": 12,
    "avg_cc_utilization": 0.1,
    "max_cc_utilization": 0.2,
    "avg_days_past_due_cc": 0,
    "num_credit_cards": 1,
    "pos_late_payments": 0,
    "avg_pos_days_past_due": 0,
}


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["champion_loaded"] is True
    assert body["fairness_thresholds_active"] is True  # calibrated thresholds from train_champion.py


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=VALID_APPLICATION, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["default_probability"] <= 1
    assert 300 <= body["credit_score"] <= 850
    assert body["decision"] in {"AUTO_APPROVE", "APPROVE", "MANUAL_REVIEW", "DENY"}


def test_predict_rejects_absurd_credit_amount(client):
    bad_app = dict(VALID_APPLICATION, amt_credit=VALID_APPLICATION["amt_income_total"] * 200)
    resp = client.post("/predict", json=bad_app, headers=AUTH_HEADERS)
    assert resp.status_code == 422


def test_predict_is_cached_on_repeat_request(client):
    r1 = client.post("/predict", json=VALID_APPLICATION, headers=AUTH_HEADERS)
    r2 = client.post("/predict", json=VALID_APPLICATION, headers=AUTH_HEADERS)
    assert r1.json()["application_id"] == r2.json()["application_id"]  # cache hit returns identical response


def test_high_risk_application_gets_adverse_action_reasons(client):
    risky_app = dict(
        VALID_APPLICATION,
        amt_income_total=50000,
        amt_credit=400000,
        bureau_debt_ratio=0.9,
        avg_cc_utilization=0.95,
        late_payments=10,
        prev_refused=3,
    )
    resp = client.post("/predict", json=risky_app, headers=AUTH_HEADERS)
    body = resp.json()
    if body["decision"] in ("DENY", "MANUAL_REVIEW"):
        assert len(body["adverse_action_reasons"]) > 0


def test_deny_thresholds_differ_by_age_group_proving_mitigation_is_live(client):
    """The calibrated per-group thresholds are only worth having if they're
    actually different across groups — this is the check that the fairness
    mitigation from models/train_champion.py is live in this process, not
    just present in a report nobody reads. `client` fixture forces app
    startup (loads GROUP_THRESHOLDS) even though the HTTP client isn't used."""
    import serving.app as serving_app

    assert len(serving_app.GROUP_THRESHOLDS) > 0
    # 22 vs 50, not vs 65: 62+ deliberately shares the most lenient cutoff,
    # which on real data is the youngest group's.
    young_threshold = serving_app._deny_threshold(22)
    middle_threshold = serving_app._deny_threshold(50)
    assert young_threshold != middle_threshold


def test_62_plus_gets_the_most_lenient_live_cutoff(client):
    import serving.app as serving_app

    elderly = serving_app._deny_threshold(65)
    assert all(elderly >= serving_app._deny_threshold(a) for a in (22, 35, 50))


def test_gender_never_changes_the_decision(client):
    """Gender is neither a model input nor a threshold key. Sending it (or
    not) must not change a single field of the decision."""
    bodies = []
    for gender in ("M", "F", None):
        application = dict(VALID_APPLICATION, amt_income_total=150001)  # bypass the response cache
        if gender:
            application["code_gender"] = gender
        resp = client.post("/predict", json=application, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        bodies.append((body["default_probability"], body["decision"], body["risk_tier"]))
    assert bodies[0] == bodies[1] == bodies[2]


def test_batch_and_realtime_thresholds_agree(client, monkeypatch):
    import pandas as pd
    import serving.app as serving_app
    from serving.batch_predict import _deny_thresholds_per_row

    thresholds = {"18-29": 0.6, "30-44": 0.5, "45-61": 0.45, "62+": 0.6}
    monkeypatch.setattr(serving_app, "GROUP_THRESHOLDS", thresholds)
    ages = [22, 35, 50, 62, 80]
    batch = _deny_thresholds_per_row(pd.Series(ages), thresholds, default_threshold=0.5)
    assert list(batch) == [serving_app._deny_threshold(a) for a in ages]


def test_predict_without_api_key_is_rejected(client):
    resp = client.post("/predict", json=VALID_APPLICATION)  # no headers
    assert resp.status_code == 401


def test_predict_with_wrong_api_key_is_rejected(client):
    resp = client.post("/predict", json=VALID_APPLICATION, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_health_does_not_require_api_key(client):
    resp = client.get("/health")  # no auth header, and none required by design
    assert resp.status_code == 200


def test_fairness_report_requires_api_key(client):
    resp = client.get("/fairness-report")
    assert resp.status_code == 401
    resp = client.get("/fairness-report", headers=AUTH_HEADERS)
    assert resp.status_code == 200
