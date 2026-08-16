import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_engineering.validation.expectations import validate_feature_table


def _valid_row(**overrides):
    row = {
        "sk_id_curr": 1,
        "target": 0,
        "age_years": 40.0,
        "credit_income_ratio": 3.0,
        "late_payment_rate": 0.1,
    }
    row.update(overrides)
    return row


def test_passes_on_valid_data():
    df = pd.DataFrame([_valid_row(sk_id_curr=i, target=i % 10 == 0) for i in range(1, 300_000)])
    assert validate_feature_table(df) == []


def test_flags_out_of_range_age():
    df = pd.DataFrame([_valid_row(age_years=150.0)] * 300_000)
    failures = validate_feature_table(df)
    assert any("age_years" in f for f in failures)


def test_flags_null_target():
    df = pd.DataFrame([_valid_row()] * 300_000)
    df.loc[0, "target"] = None
    failures = validate_feature_table(df)
    assert any("target has nulls" in f for f in failures)
