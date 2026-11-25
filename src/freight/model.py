"""Additive model: gradient boosting for load pricing + linear market/time term.

    log(rate) = f(load features) + b0 + b1*log(market) + b2*days + b3*daily_quote

f is a HistGradientBoostingRegressor; the linear part is ordinary least
squares. The two are fitted by backfitting (alternating on each other's
residuals). Trees cannot extrapolate a time trend, which matters because the
validation period (Nov-Dec) lies after the training period (Jan-Oct).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .features import LINEAR_FEATURES, TREE_FEATURES

DEFAULT_TREE_PARAMS = dict(
    max_iter=800,
    learning_rate=0.05,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
)


class AdditiveRateModel:
    def __init__(
        self,
        tree_features: list[str] = TREE_FEATURES,
        linear_features: list[str] = LINEAR_FEATURES,
        n_backfit_rounds: int = 3,
        random_state: int = 0,
        **tree_params,
    ):
        self.tree_features = list(tree_features)
        self.linear_features = list(linear_features)
        self.n_backfit_rounds = n_backfit_rounds
        self.random_state = random_state
        self.tree_params = {**DEFAULT_TREE_PARAMS, **tree_params}

    def _new_tree(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            categorical_features=[c == "equipment" for c in self.tree_features],
            random_state=self.random_state,
            **self.tree_params,
        )

    def fit(self, X: pd.DataFrame, rate: pd.Series | np.ndarray) -> "AdditiveRateModel":
        y = np.log(np.asarray(rate, dtype=float))
        linear_part = np.zeros(len(y))
        for _ in range(self.n_backfit_rounds):
            self.tree_ = self._new_tree().fit(X[self.tree_features], y - linear_part)
            tree_part = self.tree_.predict(X[self.tree_features])
            if self.linear_features:
                self.linear_ = LinearRegression().fit(X[self.linear_features], y - tree_part)
                linear_part = self.linear_.predict(X[self.linear_features])
            else:
                self.linear_ = None
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.tree_.predict(X[self.tree_features])
        if self.linear_ is not None:
            pred = pred + self.linear_.predict(X[self.linear_features])
        return pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))

    def linear_coefficients(self) -> dict[str, float]:
        if self.linear_ is None:
            return {}
        coefs = dict(zip(self.linear_features, self.linear_.coef_.round(6)))
        coefs["intercept"] = round(float(self.linear_.intercept_), 6)
        return coefs
