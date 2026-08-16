"""Smoke test for the Postgres load step. Requires `docker compose up -d postgres` running.
Run: pytest tests/test_pipeline.py
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_engineering.load_to_postgres import build_engine, load_table


def test_load_smallest_table_round_trips():
    engine = build_engine()
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS raw")
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")

    count = load_table(engine, "application_test", "application_test.csv")
    assert count > 0

    with engine.connect() as conn:
        db_count = conn.execute(text("SELECT COUNT(*) FROM raw.application_test")).scalar()
    assert db_count == count
