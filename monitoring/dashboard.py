"""Streamlit monitoring dashboard: model performance, fairness audit, data
drift, and pipeline health. Data-fetching kept in plain functions (below)
separate from the `st.*` calls so they're testable without a Streamlit
runtime — see tests/test_dashboard_data.py.

Run: streamlit run monitoring/dashboard.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def get_engine():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url)


def load_fairness_report() -> dict:
    path = ARTIFACTS_DIR / "fairness_report.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_batch_scores(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM serving.batch_scores", engine)


def load_comparison_results() -> pd.DataFrame:
    path = Path(__file__).resolve().parent.parent / "experiments" / "comparison_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def risk_tier_breakdown(scores_df: pd.DataFrame) -> pd.Series:
    return scores_df["risk_tier"].value_counts()


def render():
    st.set_page_config(page_title="CreditShield Model Monitor", layout="wide")
    st.title("CreditShield Model Monitor")

    engine = get_engine()
    tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Fairness Audit", "Data Drift", "Pipeline Health"])

    with tab1:
        comparison = load_comparison_results()
        if comparison.empty:
            st.info("Run experiments/run_comparison.py to populate this tab.")
        else:
            st.dataframe(comparison.sort_values("roc_auc", ascending=False))

        try:
            scores = load_batch_scores(engine)
            st.subheader("Risk tier breakdown (latest batch score)")
            st.bar_chart(risk_tier_breakdown(scores))
            st.subheader("Predicted default probability distribution")
            st.bar_chart(pd.cut(scores["default_probability"], bins=20).value_counts().sort_index())
        except Exception:
            st.info("Run serving/batch_predict.py to populate portfolio-level charts.")

    with tab2:
        report = load_fairness_report()
        if not report:
            st.info("Run models/train_champion.py to populate this tab.")
        else:
            st.metric("Deployed model", report["deployed_model"])
            st.metric("Cleared for production", str(report["cleared_for_production"]))

            st.subheader("Candidate comparison (flat 0.5 threshold, no post-hoc correction)")
            candidates_df = pd.DataFrame(report["candidates"]).T
            st.dataframe(candidates_df[["auc", "dp_gender", "dp_age", "demographic_parity_pass"]])

            audit = report["audit_deployed_final"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Demographic parity gap (gender)", f"{audit['dp_gender']:.4f}",
                    delta=f"threshold {report['demographic_parity_threshold']}",
                    delta_color="normal" if audit["demographic_parity_pass_gender"] else "inverse",
                )
                st.metric("Equalized odds gap (gender)", f"{audit['eo_gender']:.4f}")
            with col2:
                st.metric(
                    "Demographic parity gap (age)", f"{audit['dp_age']:.4f}",
                    delta=f"threshold {report['demographic_parity_threshold']}",
                    delta_color="normal" if audit["demographic_parity_pass_age"] else "inverse",
                )
                st.metric("Four-fifths ratio (gender)", f"{audit['four_fifths_ratio']:.3f}",
                           delta="PASS" if audit["four_fifths_pass"] else "FAIL")

            st.subheader("Approval rate by age group (deployed model, final decision)")
            st.bar_chart(pd.Series(report["approval_rates_by_age_group"]))

            if report["mitigation_applied"]:
                st.write("Post-hoc per-group threshold calibration was applied on top of the "
                         "deployed model (defense in depth). Deployed:", report["mitigation_deployed"])

    with tab3:
        st.info("Drift comparisons require a persisted reference snapshot from the last "
                "training run — see monitoring/drift_detector.py and the open decision "
                "in PROGRESS.md about wiring this into the Airflow DAG.")

    with tab4:
        st.write("dbt models: staging, intermediate, features — see `dbt run` output.")
        st.write("Data validation: `data_engineering/validation/expectations.py`")
        st.write("Airflow DAG: `dags/credit_default_retrain.py` (see PROGRESS.md for run status)")


if __name__ == "__main__":
    render()
