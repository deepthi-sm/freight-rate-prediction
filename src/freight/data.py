"""Loading and cleaning of the raw load files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TARGET = "posted_rate"

# A clean load sits within roughly +/-20% of a simple lane-level model.
# The corrupted labels are 2.2x-5x too high or too low (|log ratio| > 0.78),
# so 0.5 separates the two groups with a wide margin.
LABEL_ERROR_LOG_THRESHOLD = 0.5


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    return train, valid


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fix feature-level data quality issues. Safe to apply to any split."""
    out = df.copy()
    # ~0.6% of weights are negative with the same magnitude distribution as
    # the positive ones, i.e. a sign error. Missing weights stay NaN; the
    # gradient-boosting model handles missing values natively.
    out["weight"] = out["weight"].abs()
    return out


def flag_label_errors(train: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.Series:
    """Flag rows whose posted_rate is inconsistent with the rest of the data.

    Uses out-of-fold predictions from a simple model on load attributes only
    (no time features), so a row is never judged by a model that saw it.
    """
    X = pd.DataFrame(
        {
            "log_distance": np.log(train["distance"]),
            "equipment": train["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2}),
            "weight": train["weight"].abs(),
            "quote_signal": train["quote_signal"],
        }
    )
    y = np.log(train[TARGET].to_numpy())
    oof = np.zeros(len(train))
    for fit_idx, pred_idx in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        model = HistGradientBoostingRegressor(
            loss="absolute_error",  # robust to the very errors we are looking for
            max_iter=400,
            categorical_features=[False, True, False, False],
            random_state=seed,
        )
        model.fit(X.iloc[fit_idx], y[fit_idx])
        oof[pred_idx] = model.predict(X.iloc[pred_idx])
    return pd.Series(np.abs(y - oof) > LABEL_ERROR_LOG_THRESHOLD, index=train.index, name="label_error")
