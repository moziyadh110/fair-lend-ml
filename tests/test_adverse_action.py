import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explainability.adverse_action import generate_adverse_action_reasons


def test_only_positive_shap_features_are_returned():
    shap_values = [0.3, -0.1, 0.05, -0.2, 0.4]
    feature_names = ["credit_income_ratio", "age_years", "late_payment_rate", "owns_car", "bureau_debt_ratio"]
    reasons = generate_adverse_action_reasons(shap_values, feature_names, top_n=4)
    assert all(r["impact_score"] > 0 for r in reasons)
    assert len(reasons) == 3  # only 3 features had positive SHAP


def test_ranked_by_impact_descending():
    shap_values = [0.1, 0.5, 0.2]
    feature_names = ["a", "b", "c"]
    reasons = generate_adverse_action_reasons(shap_values, feature_names, top_n=4)
    impacts = [r["impact_score"] for r in reasons]
    assert impacts == sorted(impacts, reverse=True)


def test_unknown_feature_gets_generic_reason():
    reasons = generate_adverse_action_reasons([0.5], ["some_unmapped_feature"], top_n=1)
    assert "some unmapped feature" in reasons[0]["reason"]


def test_respects_top_n():
    shap_values = [0.5, 0.4, 0.3, 0.2, 0.1]
    feature_names = [f"f{i}" for i in range(5)]
    reasons = generate_adverse_action_reasons(shap_values, feature_names, top_n=2)
    assert len(reasons) == 2
