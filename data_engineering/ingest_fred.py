"""Pull macroeconomic indicators from FRED (St. Louis Fed) into Postgres —
demonstrates the external-data integration pattern the plan calls for
(credit risk correlates with the economic cycle, not just applicant-level
features). Uses `requests` directly against FRED's REST API rather than the
`fredapi` package: fredapi is a thin wrapper over the same GET endpoint, and
`requests` is already a project dependency — no reason to add a second HTTP
client for one API call shape.

Usage: python data_engineering/ingest_fred.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Indicators plausibly correlated with default risk: unemployment, mortgage
# rates, the federal funds rate (credit availability), and inflation.
SERIES = {
    "UNRATE": "unemployment_rate",
    "MORTGAGE30US": "mortgage_rate_30yr",
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
}


def fetch_series(series_id: str, api_key: str) -> pd.DataFrame:
    resp = requests.get(
        FRED_BASE_URL,
        params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    observations = resp.json()["observations"]

    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing values
    df["series_id"] = series_id
    return df.dropna(subset=["value"])


def build_engine():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url)


def main():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not set in .env — skipping.")
        return

    engine = build_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS raw")

    frames = []
    for series_id, name in SERIES.items():
        df = fetch_series(series_id, api_key)
        df["indicator_name"] = name
        frames.append(df)
        print(f"Fetched {series_id} ({name}): {len(df)} observations")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_sql("economic_indicators", engine, schema="raw", if_exists="replace", index=False)
    print(f"Loaded raw.economic_indicators: {len(combined)} rows across {len(SERIES)} series")


if __name__ == "__main__":
    main()
