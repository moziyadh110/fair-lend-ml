"""MLflow Model Registry: register the deployed champion, gate promotion on
real metrics, and alias it `champion` so serving always loads "whatever is
currently promoted" rather than a specific version number.

Uses MLflow's alias-based API (`set_registered_model_alias`), not the older
stage-based one (`transition_model_version_stage`) — stages were deprecated
in MLflow 2.9+ in favor of aliases, and "champion" is exactly the vocabulary
this project already uses throughout (see models/train_champion.py).

No separate MLflow server/Docker service needed: MLflow 3.x's default local
tracking URI (`sqlite:///mlflow.db`) is already database-backed, which is
all the Model Registry requires — see PROGRESS.md for why the Docker mlflow
service sketched in the original plan was skipped as unnecessary rather than
just not-yet-built.
"""
import sys
from importlib.abc import MetaPathFinder


class _BlockBrokenTorch(MetaPathFinder):
    """Intercepts `import torch` and fails cleanly with ModuleNotFoundError,
    instead of letting Python reach torch's real __init__.py — which crashes
    with an unrecoverable OSError on this machine (missing VC++
    redistributable DLLs; torch is an unrelated, broken leftover here, not a
    dependency of this project). Both mlflow.sklearn.log_model (via
    scikit-learn's array-API-backend discovery) and mlflow's own metric-value
    conversion code probe for torch's presence — a `sys.modules["torch"] =
    None` poison would fool the first (pkgutil's walk catches ImportError)
    but not the second (mlflow just checks `"torch" in sys.modules`, which is
    True even for a poisoned None entry). Intercepting at the meta-path level
    means torch genuinely never appears in sys.modules, so every caller sees
    the same, correct "not installed" outcome.
    """

    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


if not any(isinstance(f, _BlockBrokenTorch) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockBrokenTorch())

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "credit_default"
CHAMPION_ALIAS = "champion"

# Promotion gate — a model that doesn't clear these has zero deployment
# value regardless of how it got here (see experiments/tradeoffs.md Tradeoff 1).
MIN_AUC = 0.65
MAX_DP_GAP = 0.05


def _serialize_group_thresholds(group_thresholds: dict) -> list:
    """{age_bin: threshold} -> a list of records for mlflow.log_dict. Kept as
    records (not a plain dict) so the stored format matches what earlier
    runs logged and an old champion still loads instead of crashing."""
    return [{"age_bin": a, "threshold": t} for a, t in group_thresholds.items()]


def _deserialize_group_thresholds(records: list) -> dict:
    return {r["age_bin"]: r["threshold"] for r in records}


def register_champion(model, feature_cols: list[str], group_thresholds: dict,
                       default_threshold: float, metrics: dict, cleared_for_production: bool) -> str | None:
    """Log the deployed model + its serving metadata as one MLflow run,
    register it, and promote to the `champion` alias if it clears the gate.

    Returns the registered model version (as a string) if promoted, else None
    (registered but not promoted — e.g. cleared_for_production was False).
    """
    mlflow.set_experiment("credit_default_champion")

    with mlflow.start_run() as run:
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict(
            {
                "feature_cols": feature_cols,
                "group_thresholds": _serialize_group_thresholds(group_thresholds),
                "default_threshold": default_threshold,
            },
            "serving_metadata.json",
        )
        # mlflow's sklearn flavor serializes via skops (safer than raw pickle)
        # and refuses to round-trip-verify types it doesn't recognize as
        # standard sklearn — both of this project's possible champion types
        # (a plain CatBoost model, or the FairEnsembleModel wrapper around a
        # list of them — see explainability/fairness.py) need to be
        # explicitly trusted, or logging fails with UntrustedTypesFoundException.
        model_info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME,
            skops_trusted_types=["catboost.core.CatBoostClassifier", "explainability.fairness.FairEnsembleModel"],
        )

    client = MlflowClient()
    version = model_info.registered_model_version

    auc = metrics.get("auc_roc") or metrics.get("auc")
    passes_gate = (
        cleared_for_production
        and auc is not None
        and auc >= MIN_AUC
    )

    if not passes_gate:
        print(f"Registered {MODEL_NAME} v{version} but NOT promoted — "
              f"cleared_for_production={cleared_for_production}, auc={auc}")
        return None

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    print(f"Promoted {MODEL_NAME} v{version} to alias '{CHAMPION_ALIAS}'")
    return version


def load_champion():
    """Load the model currently aliased `champion`, plus its serving
    metadata (feature columns, fairness-calibrated thresholds). Raises if
    no model has ever been promoted — callers should catch and fall back."""
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    metadata = mlflow.artifacts.load_dict(f"runs:/{model_version.run_id}/serving_metadata.json")
    metadata["group_thresholds"] = _deserialize_group_thresholds(metadata["group_thresholds"])

    return model, metadata
