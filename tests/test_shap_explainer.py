from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "champion_model.joblib").exists(),
    reason="Run models/train_champion.py first to produce artifacts/champion_model.joblib",
)


def test_explainer_produces_one_shap_value_per_feature():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from explainability.shap_explainer import build_explainer, explain_single_prediction

    bundle = joblib.load(ARTIFACTS_DIR / "champion_model.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    row = pd.DataFrame([{c: 0.0 for c in feature_cols}])
    explainer = build_explainer(model)
    result = explain_single_prediction(explainer, row)

    assert len(result["shap_values"]) == len(feature_cols)
    assert np.all(np.isfinite(result["shap_values"]))
