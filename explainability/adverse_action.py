"""ECOA requires specific, individualized denial reasons for every rejected
applicant. This turns per-prediction SHAP values into those reason codes.
"""
REASON_CODE_MAP = {
    "credit_income_ratio": "Loan amount is high relative to income",
    "late_payment_rate": "History of late payments on previous loans",
    "avg_cc_utilization": "High credit card utilization ratio",
    "bureau_debt_ratio": "High outstanding debt relative to available credit",
    "max_days_overdue": "History of extended payment delinquency",
    "bureau_loan_count": "Number of existing credit obligations",
    "avg_days_late": "Average delay in payment beyond due date",
    "active_loans": "Number of currently active loans",
    "payment_volatility": "Inconsistent payment amounts across loan history",
    "years_employed": "Length of current employment",
    "age_years": "Limited credit history length",
    "prev_refused": "Previous loan applications were declined",
    "exposure_at_risk": "Total exposure across active credit lines",
}


def generate_adverse_action_reasons(shap_values, feature_names, top_n: int = 4) -> list[dict]:
    """Top N features pushing the prediction toward default (positive SHAP),
    translated into human-readable regulatory reason codes."""
    feature_impacts = sorted(zip(feature_names, shap_values), key=lambda x: x[1], reverse=True)
    risk_drivers = [(name, val) for name, val in feature_impacts if val > 0][:top_n]

    reasons = []
    for rank, (feature, impact) in enumerate(risk_drivers, 1):
        readable = REASON_CODE_MAP.get(feature, f"Risk factor: {feature.replace('_', ' ')}")
        reasons.append({
            "rank": rank,
            "reason_code": f"RC{rank:02d}",
            "reason": readable,
            "feature": feature,
            "impact_score": round(float(impact), 4),
        })
    return reasons
