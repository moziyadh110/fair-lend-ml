"""Run every candidate model through an identical train/test evaluation and
report the same credit-risk + fairness metrics for each, so the comparison
table is apples-to-apples with the Step 3 WoE baseline (same split, same
random_state).

MLflow tracking uses the local file store (./mlruns) rather than the
Postgres-backed server from the plan — no `mlflow` Docker service has been
built yet, and a local file store is a drop-in swap later (just change the
tracking URI), so it's not worth blocking model comparison on writing and
standing up that container first. See PROGRESS.md.
"""
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.credit_metrics import gini_coefficient, ks_statistic


class ModelExperimentRunner:
    """Fits each candidate once on a fixed train/test split and scores it —
    a single split (not k-fold) to keep this fast enough for local iteration
    while still directly comparable to the Step 3 baseline's holdout numbers.
    """

    def __init__(self, X_train, X_test, y_train, y_test, gender_test):
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.gender_test = gender_test

    def run_all_experiments(self, models_dict: dict) -> pd.DataFrame:
        mlflow.set_experiment("credit_default_model_comparison")
        results = []

        for model_name, model in models_dict.items():
            with mlflow.start_run(run_name=model_name):
                start = time.time()
                model.fit(self.X_train, self.y_train)
                train_time = time.time() - start

                start = time.time()
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                inference_ms = (time.time() - start) / len(self.X_test) * 1000

                y_pred = (y_pred_proba >= 0.5).astype(int)

                metrics = {
                    "model": model_name,
                    "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
                    "pr_auc": average_precision_score(self.y_test, y_pred_proba),
                    "ks": ks_statistic(self.y_test, y_pred_proba),
                    "gini": gini_coefficient(self.y_test, y_pred_proba),
                    "f1": f1_score(self.y_test, y_pred),
                    "precision": precision_score(self.y_test, y_pred),
                    "recall": recall_score(self.y_test, y_pred),
                    "dp_gender": demographic_parity_difference(
                        self.y_test, y_pred, sensitive_features=self.gender_test
                    ),
                    "eo_gender": equalized_odds_difference(
                        self.y_test, y_pred, sensitive_features=self.gender_test
                    ),
                    "train_time_seconds": train_time,
                    "inference_ms_per_sample": inference_ms,
                }

                for key, value in metrics.items():
                    if key != "model":
                        mlflow.log_metric(key, value)
                mlflow.log_param("model_type", model_name)

                results.append(metrics)
                print(f"{model_name}: AUC={metrics['roc_auc']:.4f} "
                      f"Gini={metrics['gini']:.4f} KS={metrics['ks']:.4f} "
                      f"DP_gap={metrics['dp_gender']:.4f}")

        return pd.DataFrame(results)
