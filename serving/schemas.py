"""Pydantic request/response models for the prediction API."""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    # Optional, and NOT used as a model input feature — collected only to apply
    # the fairness-calibrated per-group decision threshold (see
    # explainability.fairness.calibrate_group_thresholds). Applicants who don't
    # disclose fall back to the flat default threshold. Collecting protected-
    # class data for fairness monitoring/remediation, while excluding it from
    # the underlying prediction, is standard practice under ECOA/Reg B.
    code_gender: Optional[str] = Field(None, pattern="^[MF]$")

    amt_income_total: float = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    amt_annuity: float = Field(..., gt=0)
    amt_goods_price: Optional[float] = Field(None, gt=0)
    age_years: float = Field(..., ge=18, le=100)
    years_employed: Optional[float] = Field(None, ge=0)
    owns_car: int = Field(..., ge=0, le=1)
    owns_realty: int = Field(..., ge=0, le=1)
    cnt_children: int = Field(0, ge=0, le=20)
    bureau_loan_count: int = Field(0, ge=0)
    active_loans: int = Field(0, ge=0)
    closed_loans: int = Field(0, ge=0)
    avg_days_overdue: float = Field(0, ge=0)
    max_days_overdue: float = Field(0, ge=0)
    bureau_debt_ratio: float = Field(0, ge=0)
    total_bureau_credit: float = Field(0, ge=0)
    prev_app_count: int = Field(0, ge=0)
    prev_approved: int = Field(0, ge=0)
    prev_refused: int = Field(0, ge=0)
    credit_application_ratio: float = Field(0, ge=0)
    late_payments: int = Field(0, ge=0)
    avg_days_late: float = Field(0, ge=0)
    payment_volatility: float = Field(0, ge=0)
    total_installments: int = Field(1, ge=0)
    avg_cc_utilization: float = Field(0, ge=0, le=5)
    max_cc_utilization: float = Field(0, ge=0, le=5)
    avg_days_past_due_cc: float = Field(0, ge=0)
    num_credit_cards: int = Field(0, ge=0)
    pos_late_payments: int = Field(0, ge=0)
    avg_pos_days_past_due: float = Field(0, ge=0)

    @field_validator("amt_credit")
    @classmethod
    def credit_within_reason(cls, v, info):
        income = info.data.get("amt_income_total", 0)
        if income and v > income * 100:
            raise ValueError("Credit amount exceeds 100x income — likely data error")
        return v


class PredictionResponse(BaseModel):
    application_id: str
    default_probability: float
    credit_score: int
    risk_tier: str
    decision: str
    adverse_action_reasons: list[dict]
    credit_income_ratio: float
    model_version: str
    prediction_timestamp: str
