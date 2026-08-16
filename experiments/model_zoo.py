"""Candidate models for the Step 4 comparison. Each one is scored against the
same holdout with the same features — no per-model feature engineering, so
differences in the comparison table reflect the algorithm, not data leakage
between candidates.
"""
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CANDIDATE_MODELS = {
    "xgboost": xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
        eval_metric="auc", random_state=42, n_jobs=-1,
    ),
    "lightgbm": lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
        random_state=42, n_jobs=-1, verbose=-1,
    ),
    "catboost": CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1,
        auto_class_weights="Balanced",
        verbose=0, random_seed=42,
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="saga", penalty="elasticnet", l1_ratio=0.5,
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    ),
}
