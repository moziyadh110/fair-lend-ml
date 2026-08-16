"""Global + local SHAP explainability for tree-based challenger models."""
import shap


def build_explainer(model):
    return shap.TreeExplainer(model)


def global_summary(explainer, X, feature_names=None):
    """Returns SHAP values for the whole dataset — feed into shap.summary_plot
    for the global "what drives defaults" view."""
    return explainer.shap_values(X)


def explain_single_prediction(explainer, X_row) -> dict:
    """Local explanation for one applicant — the basis for adverse action reasons."""
    shap_values = explainer.shap_values(X_row)
    values = shap_values[0] if shap_values.ndim == 2 else shap_values
    feature_names = list(X_row.columns) if hasattr(X_row, "columns") else None
    return {"shap_values": values, "feature_names": feature_names}
