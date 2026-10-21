"""Step 4 entrypoint: train the model zoo (raw features, no WoE encoding —
tree models don't need it) on the same split as the Step 3 baseline, and
report the same metrics for direct comparison.

Usage: python experiments/run_comparison.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.compare import ModelExperimentRunner
from experiments.model_zoo import CANDIDATE_MODELS
from models.train_baseline import FEATURE_COLS

load_dotenv()


def main():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    df = pd.read_sql("SELECT * FROM features.fct_credit_features ORDER BY sk_id_curr", engine)

    # A handful of rows have code_gender='XNA' (Home Credit's placeholder for
    # undisclosed gender, not a real category) — same fix as models/train_champion.py
    # (see PROGRESS.md Stage 11). Without this, the DP-gender gaps below are
    # inflated by a 4-row statistical outlier group, same failure mode already
    # found and fixed for the deployed champion, now applied here too (item #5
    # in the improvement list — this table was flagged as still using the
    # unfiltered, worse-than-real numbers).
    df = df[df["_protected_gender"] != "XNA"].reset_index(drop=True)

    X, y = df[FEATURE_COLS], df["target"]
    gender = df["_protected_gender"]

    X_train, X_test, y_train, y_test, _, gender_test = train_test_split(
        X, y, gender, test_size=0.2, stratify=y, random_state=42
    )

    # Tree models handle raw numeric features and NaNs natively — no WoE needed here.
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    runner = ModelExperimentRunner(X_train, X_test, y_train, y_test, gender_test)
    results = runner.run_all_experiments(CANDIDATE_MODELS)

    print("\n=== Model comparison (holdout, vs. WoE baseline AUC=0.701, Gini=0.402) ===")
    print(results.sort_values("roc_auc", ascending=False).to_string(index=False))

    results.to_csv("experiments/comparison_results.csv", index=False)
    print("\nSaved to experiments/comparison_results.csv")


if __name__ == "__main__":
    main()
