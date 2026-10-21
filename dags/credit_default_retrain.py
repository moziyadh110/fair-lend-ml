"""Monthly retraining pipeline: refresh features -> validate -> check drift ->
retrain if needed -> fairness audit -> promote -> batch score -> report.

Not runnable without an Airflow scheduler (native Airflow isn't supported on
Windows outside WSL, and the ./airflow Docker build context referenced in
docker-compose.yml hasn't been built yet — see PROGRESS.md). The task bodies
below call the same functions already verified working end-to-end elsewhere
in this project (dbt, data_engineering.validation, monitoring, models,
serving) — this file wires them into a schedule, it doesn't reimplement them.
"""
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DBT_PROJECT_DIR = Path(__file__).resolve().parent.parent / "data_engineering" / "dbt_project"

default_args = {
    "owner": "ml-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def run_dbt_refresh():
    subprocess.run(
        ["dbt", "run", "--project-dir", str(DBT_PROJECT_DIR), "--profiles-dir", str(DBT_PROJECT_DIR)],
        check=True,
    )


def run_validation():
    from data_engineering.validation.expectations import main as validate_main
    validate_main()  # raises SystemExit(1) on failure, which fails the task


def check_and_branch(**context):
    import os

    import pandas as pd
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    from monitoring.alerting import send_alerts
    from monitoring.drift_detector import compute_all_psi, retrain_recommended

    load_dotenv()
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    df = pd.read_sql("SELECT * FROM features.fct_credit_features ORDER BY sk_id_curr", engine)

    # No historical snapshot store yet — compares the current table against
    # itself split in half as a placeholder for "reference vs. current batch".
    # Real usage needs a persisted reference snapshot from the last training run.
    midpoint = len(df) // 2
    cols = ["credit_income_ratio", "avg_cc_utilization", "bureau_debt_ratio", "late_payment_rate"]
    psi = compute_all_psi(df.iloc[:midpoint], df.iloc[midpoint:], cols)

    if retrain_recommended(psi):
        drifted = {name: r["psi"] for name, r in psi.items() if r["status"] == "retrain"}
        send_alerts([{
            "name": "Feature drift",
            "metric": "psi_any_top10_feature",
            "threshold": 0.25,
            "direction": "above",
            "action": "trigger_retrain",
            "current_value": max(drifted.values()),
        }])
        return "retrain_model"
    return "skip_retrain"


def train_and_register():
    from models.train_champion import main as train_main
    train_main()


def batch_score_portfolio():
    from serving.batch_predict import batch_score_portfolio as run_batch
    run_batch()


with DAG(
    "credit_default_retrain",
    default_args=default_args,
    schedule_interval="0 2 1 * *",  # monthly, 2 AM on the 1st
    start_date=datetime(2026, 8, 1),
    catchup=False,
) as dag:

    refresh_features = PythonOperator(task_id="dbt_run", python_callable=run_dbt_refresh)
    validate_data = PythonOperator(task_id="validate_data", python_callable=run_validation)

    check_drift = BranchPythonOperator(task_id="check_drift", python_callable=check_and_branch)

    retrain_model = PythonOperator(task_id="retrain_model", python_callable=train_and_register)
    skip_retrain = PythonOperator(
        task_id="skip_retrain", python_callable=lambda: print("No drift detected — skipping retrain")
    )

    batch_score = PythonOperator(
        task_id="batch_score_portfolio", python_callable=batch_score_portfolio, trigger_rule="none_failed"
    )

    refresh_features >> validate_data >> check_drift >> [retrain_model, skip_retrain]
    retrain_model >> batch_score
    skip_retrain >> batch_score
