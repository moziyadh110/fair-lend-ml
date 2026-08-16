"""Monthly portfolio-review batch scoring — reuses the same champion model
artifact the real-time API loads, so real-time and batch predictions never
diverge.

Usage: python serving/batch_predict.py
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explainability.fairness import age_bin_labels, non_disclosure_threshold

load_dotenv()
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _deny_thresholds_per_row(gender: pd.Series, age: pd.Series, group_thresholds: dict,
                              default_threshold: float) -> np.ndarray:
    """Same per-group DENY cutoff logic as serving/app.py._deny_threshold,
    vectorized for a whole portfolio — kept in sync deliberately since batch
    and real-time predictions must use the same fairness-calibrated decision,
    not just the same model. Rows with a (gender, age_bin) combination not
    seen during calibration (e.g. the excluded XNA-gender rows) fall back to
    the most lenient calibrated threshold, not the flat default — same
    non-disclosure policy as serving/app.py, for the same reason."""
    age_bins = age_bin_labels(age).astype(str)
    fallback = non_disclosure_threshold(group_thresholds, fallback=default_threshold)
    thresholds = np.full(len(gender), fallback)
    for i in range(len(gender)):
        t = group_thresholds.get((gender.iloc[i], age_bins.iloc[i]), fallback)
        thresholds[i] = max(0.25, t)
    return thresholds


def batch_score_portfolio(output_table: str = "batch_scores") -> int:
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    features_df = pd.read_sql("SELECT * FROM features.fct_credit_features", engine)

    bundle = joblib.load(ARTIFACTS_DIR / "champion_model.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    group_thresholds = bundle.get("group_thresholds", {})
    default_threshold = bundle.get("default_threshold", 0.5)

    X = features_df.reindex(columns=feature_cols, fill_value=0)
    probabilities = model.predict_proba(X)[:, 1]

    deny_thresholds = _deny_thresholds_per_row(
        features_df["_protected_gender"], features_df["_protected_age"],
        group_thresholds, default_threshold,
    )

    risk_tier = np.where(
        probabilities < 0.10, "low",
        np.where(probabilities < 0.25, "medium",
                 np.where(probabilities < deny_thresholds, "high", "very_high")),
    )

    results = pd.DataFrame({
        "sk_id_curr": features_df["sk_id_curr"],
        "default_probability": probabilities,
        "risk_tier": risk_tier,
        "scored_at": pd.Timestamp.now(),
        "model_version": "2.0.0",
    })

    results.to_sql(output_table, engine, schema="serving", if_exists="replace", index=False)
    return len(results)


if __name__ == "__main__":
    with create_engine(
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    ).begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS serving")

    count = batch_score_portfolio()
    print(f"Scored {count} applicants -> serving.batch_scores")
