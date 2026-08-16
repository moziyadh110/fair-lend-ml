"""Evidently HTML reports for stakeholders + per-feature PSI for automated
retrain-trigger checks (Evidently's presets are qualitative; PSI gives the
numeric threshold the Airflow DAG branches on).
"""
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from evaluation.credit_metrics import population_stability_index


def generate_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str):
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference_df, current_data=current_df)
    result.save_html(output_path)
    return result


def compute_all_psi(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    psi_results = {}
    for col in feature_cols:
        psi = population_stability_index(reference_df[col].dropna(), current_df[col].dropna())
        psi_results[col] = {
            "psi": psi,
            "status": "stable" if psi < 0.10 else "investigate" if psi < 0.25 else "retrain",
        }
    return psi_results


def retrain_recommended(psi_results: dict, share_threshold: float = 0.3) -> bool:
    """Flag retraining if more than `share_threshold` of features have drifted (PSI >= 0.25)."""
    if not psi_results:
        return False
    drifted = sum(1 for r in psi_results.values() if r["psi"] >= 0.25)
    return (drifted / len(psi_results)) > share_threshold
