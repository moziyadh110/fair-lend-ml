"""Sanity checks on the dbt feature table, run after every `dbt run` and before training.

Plain pandas/assert checks instead of the full Great Expectations framework:
this project only needs a fixed set of range/null/shape checks on one table,
which a GX DataContext (suites, checkpoints, docs sites) is heavyweight for.
Swap in GX if this grows into many tables/checkpoints with a shared team dashboard.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


class ValidationError(Exception):
    pass


def load_feature_table(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM features.fct_credit_features", engine)


def validate_feature_table(df: pd.DataFrame) -> list[str]:
    """Return a list of failure messages; empty list means all checks passed."""
    failures = []

    def check(condition: bool, message: str):
        if not condition:
            failures.append(message)

    check(df["sk_id_curr"].notna().all(), "sk_id_curr has nulls")
    check(df["target"].notna().all(), "target has nulls")

    age = df["age_years"].dropna()
    check(age.between(18, 100).all(), "age_years has values outside [18, 100]")

    cir = df["credit_income_ratio"].dropna()
    check(cir.between(0, 200).all(), "credit_income_ratio has values outside [0, 200]")

    lpr = df["late_payment_rate"].dropna()
    check(lpr.between(0, 1).all(), "late_payment_rate has values outside [0, 1]")

    target_share = df["target"].mean()
    check(0.01 <= target_share <= 0.99, f"target proportion {target_share:.4f} outside [0.01, 0.99]")

    check(250_000 <= len(df) <= 350_000, f"row count {len(df)} outside [250000, 350000]")

    return failures


def main():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    df = load_feature_table(engine)
    failures = validate_feature_table(df)

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)

    print(f"All checks passed on {len(df)} rows.")


if __name__ == "__main__":
    main()
