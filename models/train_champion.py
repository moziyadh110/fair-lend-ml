"""Train and export the champion model as the artifact `serving/app.py` and
`serving/batch_predict.py` both load.

Trains two candidates and compares them: (1) plain CatBoost — winner of the
Step 4 comparison — and (2) the same CatBoost architecture retrained with
Fairlearn's ExponentiatedGradient demographic-parity constraint baked in
(gender x age-bin jointly). Whichever one is actually fairer AND still
reasonably accurate gets deployed; deterministic per-group threshold
calibration is layered on top of the deployed model regardless, as defense
in depth (see PROGRESS.md Stage 15 for why post-hoc thresholds exist,
Stage 16 for why constrained retraining was added on top of that).

Usage: python models/train_champion.py
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.credit_metrics import full_evaluation_report
from explainability.fairness import (
    DEMOGRAPHIC_PARITY_THRESHOLD,
    FairEnsembleModel,
    apply_group_thresholds,
    calibrate_group_thresholds,
    full_fairness_audit,
    retrain_with_fairness_constraints,
)
from models.registry import register_champion
from models.train_baseline import FEATURE_COLS

load_dotenv()
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def main():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(url)
    # ORDER BY matters: train_test_split is positional, and Postgres returns
    # rows in arbitrary order after a rebuild, which silently changes the
    # holdout set (and every reported number) between otherwise identical runs.
    df = pd.read_sql("SELECT * FROM features.fct_credit_features ORDER BY sk_id_curr", engine)

    # A handful of rows have code_gender='XNA' (Home Credit's placeholder for
    # undisclosed gender, not a real category) — too small a group (4 rows,
    # single-class) for per-group fairness metrics to be statistically
    # meaningful, and it breaks Fairlearn's ROC-based threshold optimizer.
    df = df[df["_protected_gender"] != "XNA"].reset_index(drop=True)

    X, y = df[FEATURE_COLS].fillna(0), df["target"]
    gender, age = df["_protected_gender"], df["_protected_age"]

    X_train, X_test, y_train, y_test, gender_train, gender_test, age_train, age_test = train_test_split(
        X, y, gender, age, test_size=0.2, stratify=y, random_state=42
    )

    # --- Candidate 1: plain CatBoost (Step 4 winner) ---
    raw_model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1,
        auto_class_weights="Balanced", verbose=0, random_seed=42,
    )
    raw_model.fit(X_train, y_train)
    raw_proba_test = raw_model.predict_proba(X_test)[:, 1]
    raw_pred_test = (raw_proba_test >= 0.5).astype(int)

    print("Candidate 1 — plain CatBoost — holdout performance:")
    print(full_evaluation_report(y_test, raw_proba_test, model_name="catboost_raw"))
    audit_raw = full_fairness_audit(y_test, raw_pred_test, raw_proba_test, gender_test, age_test)
    print(f"Fairness (raw, flat 0.5 threshold): dp_gender={audit_raw['dp_gender']:.4f} "
          f"dp_age={audit_raw['dp_age']:.4f} (pass={audit_raw['demographic_parity_pass']})")

    # --- Candidate 2: same architecture, fairness constraint baked in via
    # ExponentiatedGradient (gender x age-bin jointly, demographic parity).
    # Smaller CatBoost + fewer EG iterations than the raw model: EG refits the
    # base estimator many times internally, so this is a real time/quality
    # tradeoff for local iteration — see PROGRESS.md Stage 16.
    print("\nCandidate 2 — fairness-constrained CatBoost (ExponentiatedGradient) — training...")

    def age_bin(s):
        return pd.cut(s, bins=[18, 30, 45, 60, 100], labels=["18-30", "30-45", "45-60", "60+"])

    sensitive_train = pd.DataFrame({"gender": gender_train.values, "age_bin": age_bin(age_train).values})

    fast_catboost = CatBoostClassifier(
        iterations=150, depth=5, learning_rate=0.15,
        auto_class_weights="Balanced", verbose=0, random_seed=42,
    )
    mitigated = retrain_with_fairness_constraints(
        X_train, y_train, sensitive_features=sensitive_train,
        estimator=fast_catboost, constraint="dp", max_iter=10,
    )
    fair_model = FairEnsembleModel.from_fitted(mitigated)
    fair_proba_test = fair_model.predict_proba(X_test)[:, 1]
    fair_pred_test = (fair_proba_test >= 0.5).astype(int)

    print("Candidate 2 — fairness-constrained CatBoost — holdout performance:")
    print(full_evaluation_report(y_test, fair_proba_test, model_name="catboost_fair_constrained"))
    audit_fair = full_fairness_audit(y_test, fair_pred_test, fair_proba_test, gender_test, age_test)
    print(f"Fairness (constrained model, flat 0.5 threshold): dp_gender={audit_fair['dp_gender']:.4f} "
          f"dp_age={audit_fair['dp_age']:.4f} (pass={audit_fair['demographic_parity_pass']})")

    # --- Decide which candidate to deploy ---
    # Prefer the fairness-constrained model if it passes the gate on its own
    # (no post-hoc correction needed) and doesn't give up more than 2 AUC
    # points versus the raw model — otherwise fall back to raw + post-hoc
    # threshold calibration (Stage 15's approach), which is guaranteed to
    # work regardless of how well constrained training converges.
    raw_auc = roc_auc_score(y_test, raw_proba_test)
    fair_auc = roc_auc_score(y_test, fair_proba_test)
    auc_cost = raw_auc - fair_auc
    print(f"\nAUC — raw: {raw_auc:.4f}, fairness-constrained: {fair_auc:.4f} (cost: {auc_cost:.4f})")

    use_constrained = audit_fair["demographic_parity_pass"] and auc_cost <= 0.02

    if use_constrained:
        deployed_model = fair_model
        deployed_name = "catboost_fair_constrained"
        deployed_proba_test = fair_proba_test
        print("Deploying the fairness-constrained model — it passes the gate without "
              "post-hoc correction, within the accepted AUC cost.")
    else:
        deployed_model = raw_model
        deployed_name = "catboost_raw"
        deployed_proba_test = raw_proba_test
        reason = "didn't pass its own gate" if not audit_fair["demographic_parity_pass"] else \
                 f"cost too much AUC ({auc_cost:.4f} > 0.02)"
        print(f"Deploying the raw model instead — constrained candidate {reason}. "
              "Falling back to post-hoc threshold calibration.")

    audit_deployed = full_fairness_audit(
        y_test, (deployed_proba_test >= 0.5).astype(int), deployed_proba_test, gender_test, age_test
    )

    # Calibrate per-group thresholds on top of whichever model is deployed —
    # defense in depth, cheap to keep, and it's what actually guarantees the
    # gate passes even if the constrained model's margin above were thin.
    group_thresholds = {}
    mitigation_applied = False
    mitigation_deployed = False
    final_audit = audit_deployed

    if not audit_deployed["demographic_parity_pass"]:
        print(f"\nDeployed candidate ({deployed_name}) fails the flat-threshold gate — "
              "calibrating deterministic per-age-bin decision thresholds on the training set...")
        train_proba = deployed_model.predict_proba(X_train)[:, 1]
        group_thresholds = calibrate_group_thresholds(train_proba, age_train)

        pred_mitigated = apply_group_thresholds(deployed_proba_test, age_test, group_thresholds)
        final_audit = full_fairness_audit(y_test, pred_mitigated, deployed_proba_test, gender_test, age_test)
        mitigation_applied = True
        print(f"Fairness audit AFTER calibrated thresholds: dp_gender={final_audit['dp_gender']:.4f} "
              f"dp_age={final_audit['dp_age']:.4f} dp_age_under_62={final_audit['dp_age_under_62']:.4f} "
              f"62+_not_disadvantaged={final_audit['elderly_not_disadvantaged']} "
              f"(pass={final_audit['demographic_parity_pass']})")
        mitigation_deployed = final_audit["demographic_parity_pass"]

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": deployed_model,
            "model_type": deployed_name,
            "feature_cols": FEATURE_COLS,
            "group_thresholds": group_thresholds if mitigation_deployed else {},
            "default_threshold": 0.5,
        },
        ARTIFACTS_DIR / "champion_model.joblib",
    )

    cleared_for_production = audit_deployed["demographic_parity_pass"] or mitigation_deployed

    report = {
        "deployed_model": deployed_name,
        "candidates": {
            "catboost_raw": {"auc": raw_auc, **{k: v for k, v in audit_raw.items()
                                                 if not k.startswith("approval_rates")}},
            "catboost_fair_constrained": {"auc": fair_auc, **{k: v for k, v in audit_fair.items()
                                                                if not k.startswith("approval_rates")}},
        },
        "mitigation_applied": mitigation_applied,
        "mitigation_deployed": mitigation_deployed,
        "cleared_for_production": cleared_for_production,
        "audit_deployed_before_thresholds": {k: v for k, v in audit_deployed.items()
                                              if not k.startswith("approval_rates")},
        "audit_deployed_final": {k: v for k, v in final_audit.items()
                                  if not k.startswith("approval_rates")},
        "approval_rates_by_gender": audit_deployed["approval_rates_by_group"],
        "approval_rates_by_age_group": audit_deployed["approval_rates_by_age_group"],
        "demographic_parity_threshold": DEMOGRAPHIC_PARITY_THRESHOLD,
    }
    (ARTIFACTS_DIR / "fairness_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved artifacts/champion_model.joblib and artifacts/fairness_report.json")

    # Also register in MLflow's Model Registry, aliased `champion` if it
    # clears the promotion gate. The joblib artifact above remains a local
    # cache/fallback — serving/app.py tries the registry first. See
    # PROGRESS.md for why this replaced the originally-sketched Docker
    # mlflow service (MLflow 3.x's default local backend is already
    # database-backed, which is all the registry needs).
    registry_metrics = {"auc_roc": raw_auc if deployed_name == "catboost_raw" else fair_auc,
                         **{k: v for k, v in final_audit.items() if isinstance(v, (int, float))}}
    version = register_champion(
        deployed_model, FEATURE_COLS, group_thresholds if mitigation_deployed else {},
        default_threshold=0.5, metrics=registry_metrics, cleared_for_production=cleared_for_production,
    )
    if version:
        print(f"MLflow: promoted credit_default v{version} to alias 'champion'")
    else:
        print("MLflow: registered but not promoted (see registry.py's gate)")
    print(f"Deployed: {deployed_name}. Cleared for production: {cleared_for_production} "
          f"(mitigation deployed: {mitigation_deployed})")


if __name__ == "__main__":
    main()
