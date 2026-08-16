"""Credit-risk-specific evaluation metrics — what a credit risk team actually
reports internally, on top of generic ML metrics like AUC/F1.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true, y_pred_proba) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    default and non-default distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true, y_pred_proba) -> float:
    """Gini = 2 * AUC - 1. Industry-standard discrimination-power metric."""
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1


def population_stability_index(expected, actual, n_bins: int = 10) -> float:
    """PSI measures distribution shift between two samples of the same score.
    < 0.10: no significant change. 0.10-0.25: moderate shift, investigate.
    > 0.25: significant shift, retrain."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def lift_chart(y_true, y_pred_proba, n_bins: int = 10) -> pd.Series:
    """Decile lift: how much better than random at each decile of predicted risk."""
    df = pd.DataFrame({"true": np.asarray(y_true), "pred": np.asarray(y_pred_proba)})
    df["decile"] = pd.qcut(df["pred"], q=n_bins, labels=False, duplicates="drop")
    return df.groupby("decile")["true"].mean() / df["true"].mean()


def full_evaluation_report(y_true, y_pred_proba, model_name: str = "model") -> dict:
    """Run every credit-risk-specific metric and return a flat dict."""
    return {
        "model": model_name,
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "ks_statistic": ks_statistic(y_true, y_pred_proba),
        "gini": gini_coefficient(y_true, y_pred_proba),
        "lift_top_decile": float(lift_chart(y_true, y_pred_proba).iloc[-1]),
    }
