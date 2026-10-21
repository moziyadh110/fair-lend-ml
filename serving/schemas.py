"""Pydantic request/response models for the prediction API."""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    # No gender field: it isn't a model input and no longer picks the decision
    # threshold either (a sex-based cutoff is disparate treatment under Reg B).
    # Gender fairness is audited offline in models/train_champion.py. A client
    # that still sends `code_gender` is fine — Pydantic ignores unknown fields.
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
