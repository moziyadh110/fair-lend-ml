"""H2O AutoML as a third model family — used here for its leaderboard
comparison against the CatBoost champion, not as the deployed model (see
PROGRESS.md: MOJO export, the plan's intended production serving format, is
no longer free in current H2O-3 OSS — it now requires the paid "H2O-3
Secure" tier). Exports via `h2o.save_model` instead, which is still free and
sufficient for loading the model back in Python.

`max_runtime_secs` defaults to 5 minutes here, not the plan's 1 hour, so this
is usable for local iteration. Bump it up for a final production run.
"""
import h2o
from h2o.automl import H2OAutoML


def run_automl(train_df, feature_cols, target_col="target",
                max_runtime_secs: int = 300, max_models: int = 10,
                model_path: str = "models/h2o/"):
    h2o.init()
    train_h2o = h2o.H2OFrame(train_df[feature_cols + [target_col]])
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        seed=42,
        sort_metric="AUC",
        exclude_algos=["DeepLearning"],  # skip neural nets for interpretability
    )
    aml.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    leaderboard = aml.leaderboard.as_data_frame()
    saved_path = h2o.save_model(model=aml.leader, path=model_path, force=True)

    return aml, leaderboard, saved_path
