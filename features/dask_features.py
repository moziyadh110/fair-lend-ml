"""Complementary scale-supplement feature to spark_features.py: per-applicant
rolling volatility of credit card balance utilization. dbt/Postgres window
functions can compute STDDEV over a fixed window, but a per-group *rolling*
statistic (3-month rolling std, then take the max) needs either a recursive
CTE or exactly the groupby-apply pattern Dask makes natural on out-of-core
data — credit_card_balance.csv is 424MB, larger than comfortably fits a
single pandas groupby-apply in memory alongside everything else running.

Usage: python features/dask_features.py
"""
import os
import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_card_balance.csv"


def _rolling_volatility(group: pd.DataFrame) -> float:
    """Max 3-month rolling std of balance/credit-limit utilization, sorted by
    month. A spike here (utilization suddenly became erratic) is a different
    signal than the average utilization level already captured in dbt."""
    group = group.sort_values("months_balance")
    utilization = (group["amt_balance"] / group["amt_credit_limit_actual"].replace(0, pd.NA)).fillna(0)
    if len(utilization) < 3:
        return 0.0
    rolling_std = utilization.rolling(window=3).std()
    return float(rolling_std.max(skipna=True) or 0.0)


def compute_utilization_volatility(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    ddf = dd.read_csv(
        csv_path,
        usecols=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"],
        dtype={"AMT_CREDIT_LIMIT_ACTUAL": "float64"},
    )
    ddf.columns = [c.lower() for c in ddf.columns]

    result = (
        ddf.groupby("sk_id_curr")
        .apply(_rolling_volatility, meta=("utilization_volatility", "f8"))
        .compute()
    )
    return result.reset_index().rename(columns={0: "utilization_volatility"})


def main():
    result = compute_utilization_volatility()
    print(f"Computed utilization_volatility for {len(result)} applicants")

    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    result.to_sql("utilization_volatility_features", engine, schema="features", if_exists="replace", index=False)
    print("Loaded features.utilization_volatility_features")


if __name__ == "__main__":
    main()
