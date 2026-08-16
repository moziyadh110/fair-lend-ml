"""Train the WoE logistic regression scorecard on the real feature table and
report its credit-risk metrics. This is the incumbent baseline every
challenger model in Step 4 has to beat.

Usage: python models/train_baseline.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.credit_metrics import full_evaluation_report
from features.woe_encoder import WoEEncoder
from models.woe_scorecard import train_woe_scorecard

load_dotenv()

FEATURE_COLS = [
    "age_years", "years_employed", "amt_income_total", "amt_credit", "amt_annuity",
    "credit_income_ratio", "annuity_income_ratio", "goods_to_credit_ratio",
    "owns_car", "owns_realty", "cnt_children",
    "bureau_loan_count", "active_loans", "closed_loans", "avg_days_overdue",
    "max_days_overdue", "bureau_debt_ratio", "total_bureau_credit",
    "prev_app_count", "prev_approved", "prev_refused", "credit_application_ratio",
    "late_payments", "avg_days_late", "payment_volatility", "total_installments",
    "avg_cc_utilization", "max_cc_utilization", "avg_days_past_due_cc", "num_credit_cards",
    "pos_late_payments", "avg_pos_days_past_due",
    "exposure_at_risk", "late_payment_rate", "high_util_x_high_leverage", "many_loans_x_young",
]


def load_features(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM features.fct_credit_features", engine)


def main():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    df = load_features(engine)

    X, y = df[FEATURE_COLS], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    encoder = WoEEncoder(n_bins=10).fit(X_train, y_train)
    X_train_woe, X_test_woe = encoder.transform(X_train), encoder.transform(X_test)

    print("Information Value per feature (>= 0.02 considered useful):")
    for name, iv in sorted(encoder.iv_scores_.items(), key=lambda kv: -kv[1]):
        flag = "" if iv >= 0.02 else "  (weak/drop)"
        print(f"  {name:35s} IV={iv:.4f}{flag}")

    _, scorecard = train_woe_scorecard(X_train_woe, y_train)
    scores = scorecard.score(X_test_woe)
    probs = scorecard.score_to_probability(scores)

    report = full_evaluation_report(y_test, probs, model_name="woe_logistic_baseline")
    print("\nBaseline scorecard performance (holdout):")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
