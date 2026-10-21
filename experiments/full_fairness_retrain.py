"""Item #1 from the improvement list: does a FULL-budget ExponentiatedGradient
run (matching the champion's own CatBoost settings — 500 iterations, depth 6
— and Fairlearn's default 50 rounds, instead of the reduced 150/10 used for
local iteration in models/train_champion.py) clear the fairness gate on its
own, without the post-hoc threshold layer?

Standalone experiment, not wired into the deployed champion — run this,
look at the result, then decide whether to update models/train_champion.py's
settings based on real numbers, same discipline as Stage 16.

Usage: python experiments/full_fairness_retrain.py
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.credit_metrics import full_evaluation_report
from explainability.fairness import FairEnsembleModel, full_fairness_audit, retrain_with_fairness_constraints
from models.train_baseline import FEATURE_COLS

load_dotenv()


def main():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    df = pd.read_sql("SELECT * FROM features.fct_credit_features ORDER BY sk_id_curr", engine)
    df = df[df["_protected_gender"] != "XNA"].reset_index(drop=True)

    X, y = df[FEATURE_COLS].fillna(0), df["target"]
    gender, age = df["_protected_gender"], df["_protected_age"]

    X_train, X_test, y_train, y_test, gender_train, gender_test, age_train, age_test = train_test_split(
        X, y, gender, age, test_size=0.2, stratify=y, random_state=42
    )

    def age_bin(s):
        return pd.cut(s, bins=[18, 30, 45, 60, 100], labels=["18-30", "30-45", "45-60", "60+"])

    sensitive_train = pd.DataFrame({"gender": gender_train.values, "age_bin": age_bin(age_train).values})

    full_catboost = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1,
        auto_class_weights="Balanced", verbose=0, random_seed=42,
    )

    print("Training FULL-budget fairness-constrained CatBoost "
          "(500 iterations x up to 50 ExponentiatedGradient rounds — this will take a while)...")
    start = time.time()
    mitigated = retrain_with_fairness_constraints(
        X_train, y_train, sensitive_features=sensitive_train,
        estimator=full_catboost, constraint="dp", max_iter=50,
    )
    elapsed = time.time() - start
    print(f"Training took {elapsed / 60:.1f} minutes, {len(mitigated.predictors_)} sub-estimators fitted "
          f"({sum(1 for w in mitigated.weights_ if w > 0)} with nonzero weight).")

    fair_model = FairEnsembleModel.from_fitted(mitigated)
    fair_proba_test = fair_model.predict_proba(X_test)[:, 1]
    fair_pred_test = (fair_proba_test >= 0.5).astype(int)

    fair_auc = roc_auc_score(y_test, fair_proba_test)
    audit_fair = full_fairness_audit(y_test, fair_pred_test, fair_proba_test, gender_test, age_test)

    print("\nFull-budget fairness-constrained CatBoost — holdout performance:")
    print(full_evaluation_report(y_test, fair_proba_test, model_name="catboost_fair_full_budget"))
    print(f"Fairness (flat 0.5 threshold, no post-hoc correction): "
          f"dp_gender={audit_fair['dp_gender']:.4f} dp_age={audit_fair['dp_age']:.4f} "
          f"(pass={audit_fair['demographic_parity_pass']})")

    # Compare against the raw champion's own numbers (from the last train_champion.py run)
    raw_auc = 0.7244007520247014
    print(f"\nAUC comparison — raw champion: {raw_auc:.4f}, full-budget constrained: {fair_auc:.4f} "
          f"(cost: {raw_auc - fair_auc:.4f})")

    result = {
        "auc": fair_auc,
        "training_minutes": elapsed / 60,
        "n_predictors": len(mitigated.predictors_),
        "n_nonzero_predictors": sum(1 for w in mitigated.weights_ if w > 0),
        **{k: v for k, v in audit_fair.items() if not k.startswith("approval_rates")},
        "auc_cost_vs_raw_champion": raw_auc - fair_auc,
    }
    Path("experiments/full_fairness_retrain_result.json").write_text(json.dumps(result, indent=2))
    print("\nSaved experiments/full_fairness_retrain_result.json")


if __name__ == "__main__":
    main()
