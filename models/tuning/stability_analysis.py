"""Multi-seed stability check. Production models must be stable, not lucky —
a hyperparameter set that only wins because of a favorable random seed will
regress the moment it's retrained on new data.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from evaluation.credit_metrics import gini_coefficient, ks_statistic


def stability_check(params: dict, X_train, y_train, X_val, y_val, n_seeds: int = 5) -> dict:
    results = []
    for seed in range(n_seeds):
        model = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != "random_state"},
                                   random_state=seed, eval_metric="auc", n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        results.append({
            "seed": seed,
            "auc": roc_auc_score(y_val, preds),
            "ks": ks_statistic(y_val, preds),
            "gini": gini_coefficient(y_val, preds),
        })

    results_df = pd.DataFrame(results)
    return {
        "auc_mean": results_df["auc"].mean(),
        "auc_std": results_df["auc"].std(),
        "ks_mean": results_df["ks"].mean(),
        "ks_std": results_df["ks"].std(),
        "is_stable": bool(results_df["auc"].std() < 0.01),
    }
