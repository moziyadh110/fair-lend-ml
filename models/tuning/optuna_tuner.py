"""Bayesian hyperparameter tuning for the XGBoost challenger via Optuna,
cross-validated so the objective isn't just fit to one lucky split.
"""
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def objective(trial, X, y, n_splits: int = 3):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 15),
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = xgb.XGBClassifier(**params, eval_metric="auc", random_state=42, n_jobs=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], preds))

    return float(np.mean(scores))


def tune(X, y, n_trials: int = 30, n_splits: int = 3) -> optuna.Study:
    """n_trials/n_splits kept small for local iteration speed (30 trials x 3-fold
    ~ 90 XGBoost fits). Bump both up for a final production tuning run."""
    study = optuna.create_study(direction="maximize", study_name="credit_risk_xgb")
    study.optimize(lambda trial: objective(trial, X, y, n_splits), n_trials=n_trials)
    return study
