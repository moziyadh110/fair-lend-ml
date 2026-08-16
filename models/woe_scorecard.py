"""Convert a WoE-encoded logistic regression into a points-based credit
scorecard — the methodology the credit industry has used since 1988. This is
the incumbent baseline every challenger model (Step 4) must beat.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class CreditScorecard:
    def __init__(self, base_score: int = 600, pdo: int = 20, target_odds: int = 50):
        self.base_score = base_score
        self.pdo = pdo  # points to double the odds
        self.target_odds = target_odds
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(target_odds)
        self.points_: dict[str, float] = {}
        self.intercept_points_: float = 0.0

    def fit(self, lr_model: LogisticRegression, feature_names: list[str]) -> "CreditScorecard":
        self.points_ = {
            name: -(coef * self.factor)
            for name, coef in zip(feature_names, lr_model.coef_[0])
        }
        self.intercept_points_ = -(lr_model.intercept_[0] * self.factor) + self.offset
        return self

    def score(self, X_woe: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(X_woe), self.intercept_points_)
        for name in X_woe.columns:
            if name in self.points_:
                scores = scores + X_woe[name].values * self.points_[name]
        return scores

    def score_to_probability(self, score: np.ndarray) -> np.ndarray:
        odds = np.exp((score - self.offset) / self.factor)
        return 1 / (1 + odds)


def train_woe_scorecard(X_woe: pd.DataFrame, y: pd.Series) -> tuple[LogisticRegression, CreditScorecard]:
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_woe, y)
    scorecard = CreditScorecard().fit(lr, list(X_woe.columns))
    return lr, scorecard
