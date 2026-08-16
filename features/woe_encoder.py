"""Weight of Evidence encoding — the credit-scoring industry standard for
turning raw features into monotonic, interpretable inputs for a logistic
scorecard.

Resolves the fit/transform gap in the original plan sketch (that version's
`fit()` only handled one feature per call and had no `transform()`) with a
standard sklearn-style `fit(X, y)` / `transform(X)` pair operating on a
whole DataFrame at once.
"""
import numpy as np
import pandas as pd


class WoEEncoder:
    """Bins each column (quantile bins for numeric, categories as-is for
    object/category dtype), computes WoE per bin from the fit data, then
    maps any input back through those same bin edges/values at transform time.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.woe_maps_: dict[str, dict] = {}
        self.iv_scores_: dict[str, float] = {}

    def _bin_column(self, x: pd.Series, edges=None):
        # Low-cardinality numeric columns (flags, small counts) get treated as
        # discrete categories — quantile-binning a column with few unique values
        # collapses to one giant bin (duplicate edges get dropped), silently
        # destroying real signal (e.g. a binary flag would always end up as a
        # single bin covering 100% of rows).
        if pd.api.types.is_numeric_dtype(x) and x.nunique(dropna=True) > self.n_bins:
            if edges is None:
                _, edges = pd.qcut(x, q=self.n_bins, duplicates="drop", retbins=True)
                edges = np.asarray(edges)
                edges[0], edges[-1] = -np.inf, np.inf
            return pd.cut(x, bins=edges, include_lowest=True), edges
        return x.astype("object").fillna("__missing__"), None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        y = y.astype(int)
        for col in X.columns:
            binned, edges = self._bin_column(X[col])
            if edges is not None:
                self.bin_edges_[col] = edges

            crosstab = pd.crosstab(binned, y)
            crosstab = crosstab.reindex(columns=[0, 1], fill_value=0)
            crosstab.columns = ["good", "bad"]

            pct_good = (crosstab["good"] + 0.5) / (crosstab["good"].sum() + 0.5 * len(crosstab))
            pct_bad = (crosstab["bad"] + 0.5) / (crosstab["bad"].sum() + 0.5 * len(crosstab))

            woe = np.log(pct_good / pct_bad)
            iv = ((pct_good - pct_bad) * woe).sum()

            self.woe_maps_[col] = woe.to_dict()
            self.iv_scores_[col] = float(iv)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            edges = self.bin_edges_.get(col)
            binned, _ = self._bin_column(X[col], edges=edges)
            woe_map = self.woe_maps_[col]
            default = np.mean(list(woe_map.values())) if woe_map else 0.0
            out[col] = binned.map(woe_map).astype(float).fillna(default)
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_selection(self, threshold: float = 0.02) -> dict[str, float]:
        """IV < 0.02: not useful. 0.02-0.1: weak. 0.1-0.3: medium.
        0.3-0.5: strong. > 0.5: suspiciously good, check for leakage."""
        return {k: v for k, v in self.iv_scores_.items() if v >= threshold}
