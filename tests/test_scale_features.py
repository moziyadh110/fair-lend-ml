import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.dask_features import _rolling_volatility
from features.spark_features import _slope_udf


def test_slope_udf_detects_declining_payments():
    df = pd.DataFrame({
        "sk_id_curr": [1, 1, 1, 1],
        "days_instalment": [-120, -90, -60, -30],
        "amt_payment": [1000, 800, 500, 200],
    })
    result = _slope_udf(df)
    assert result["sk_id_curr"].iloc[0] == 1
    # amt_payment falls (1000->200) as days_instalment rises toward 0 (-120->-30),
    # i.e. payments decline as the due date approaches -> negative slope.
    assert result["payment_trend_slope"].iloc[0] < 0


def test_slope_udf_handles_too_few_points():
    df = pd.DataFrame({"sk_id_curr": [2, 2], "days_instalment": [-30, -10], "amt_payment": [100, 200]})
    result = _slope_udf(df)
    assert result["payment_trend_slope"].iloc[0] == 0.0


def test_rolling_volatility_detects_erratic_utilization():
    stable = pd.DataFrame({
        "months_balance": range(6),
        "amt_balance": [500] * 6,
        "amt_credit_limit_actual": [1000] * 6,
    })
    erratic = pd.DataFrame({
        "months_balance": range(6),
        "amt_balance": [100, 900, 50, 950, 100, 900],
        "amt_credit_limit_actual": [1000] * 6,
    })
    assert _rolling_volatility(erratic) > _rolling_volatility(stable)


def test_rolling_volatility_handles_too_few_rows():
    df = pd.DataFrame({"months_balance": [0, 1], "amt_balance": [100, 200], "amt_credit_limit_actual": [1000, 1000]})
    assert _rolling_volatility(df) == 0.0
