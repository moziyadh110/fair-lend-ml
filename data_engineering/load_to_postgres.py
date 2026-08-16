"""Load the raw Home Credit CSVs into Postgres, one table per file, schema `raw`.

Uses COPY (via psycopg2) for the actual data transfer instead of row-by-row
INSERTs — pandas.to_sql's default executemany is too slow for the largest
files here (installments_payments.csv alone is 723MB / ~13.6M rows).
pandas is still used to infer table schema from each CSV's dtypes.

Usage: python data_engineering/load_to_postgres.py
"""
import io
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNKSIZE = 200_000

TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}


def build_engine():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url)


def copy_chunk(raw_conn, table_name: str, chunk: pd.DataFrame) -> None:
    buf = io.StringIO()
    chunk.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with raw_conn.cursor() as cur:
        cur.copy_expert(f"COPY raw.{table_name} FROM STDIN WITH (FORMAT csv, NULL '')", buf)
    raw_conn.commit()


def load_table(engine, table_name: str, filename: str) -> int:
    path = DATA_DIR / filename
    row_count = 0
    raw_conn = engine.raw_connection()
    try:
        for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNKSIZE)):
            chunk.columns = [c.lower() for c in chunk.columns]
            if i == 0:
                # Let pandas create the table schema from dtypes, then truncate —
                # COPY needs a pre-existing table to stream into.
                chunk.head(0).to_sql(table_name, engine, schema="raw", if_exists="replace", index=False)
            copy_chunk(raw_conn, table_name, chunk)
            row_count += len(chunk)
    finally:
        raw_conn.close()
    return row_count


def main():
    engine = build_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS raw")

    for table_name, filename in TABLES.items():
        count = load_table(engine, table_name, filename)
        print(f"Loaded raw.{table_name}: {count} rows", flush=True)


if __name__ == "__main__":
    main()
