"""Validation schemes and metrics."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

ModelFactory = Callable[[], object]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    return {
        "mape_pct": float(np.mean(np.abs(err) / actual) * 100),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "n": int(len(actual)),
    }


def rolling_origin_folds(dates: pd.Series, first_cutoff_month: int = 4, last_cutoff_month: int = 8, horizon_months: int = 2):
    """Expanding-window folds: train on months <= k, test on the next `horizon` months.

    Mirrors the real task, where Jan-Oct is used to predict Nov-Dec.
    """
    month = dates.dt.month
    for k in range(first_cutoff_month, last_cutoff_month + 1):
        train_mask = month <= k
        test_mask = (month > k) & (month <= k + horizon_months)
        label = f"train Jan-{_mon(k)} -> test {_mon(k + 1)}-{_mon(k + horizon_months)}"
        yield label, train_mask.to_numpy(), test_mask.to_numpy()


def city_holdout_folds(df: pd.DataFrame, n_cities: int = 8, n_repeats: int = 3, seed: int = 0):
    """Hold out every load touching a random set of cities.

    The validation file contains 8 cities that never occur in training, so
    the model must price unseen origins/destinations from coordinates.
    """
    rng = np.random.default_rng(seed)
    cities = np.array(sorted(set(df["pickup"]) | set(df["delivery"])))
    for r in range(n_repeats):
        held = rng.choice(cities, n_cities, replace=False)
        test_mask = (df["pickup"].isin(held) | df["delivery"].isin(held)).to_numpy()
        yield f"repeat {r + 1}: {', '.join(sorted(held))}", ~test_mask, test_mask


def run_cv(
    folds,
    X: pd.DataFrame,
    rate: pd.Series,
    label_error: pd.Series,
    make_model: ModelFactory,
) -> pd.DataFrame:
    """Fit on clean training rows of each fold; score on clean and on all test rows."""
    rate = rate.to_numpy()
    bad = label_error.to_numpy()
    rows = []
    for label, train_mask, test_mask in folds:
        fit_mask = train_mask & ~bad
        model = make_model().fit(X[fit_mask], rate[fit_mask])
        pred = model.predict(X[test_mask])
        clean = ~bad[test_mask]
        m_clean = metrics(rate[test_mask][clean], pred[clean])
        m_all = metrics(rate[test_mask], pred)
        rows.append(
            {
                "fold": label,
                "n_test": m_clean["n"],
                "mape_pct": m_clean["mape_pct"],
                "mae": m_clean["mae"],
                "rmse": m_clean["rmse"],
                "mape_pct_incl_label_errors": m_all["mape_pct"],
                "mae_incl_label_errors": m_all["mae"],
            }
        )
    return pd.DataFrame(rows)


def _mon(m: int) -> str:
    return pd.Timestamp(2025, m, 1).strftime("%b")
