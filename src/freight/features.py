"""Feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
TREND_ORIGIN = pd.Timestamp("2025-01-01")

# Load-level features, modelled non-linearly by gradient boosting.
TREE_FEATURES = [
    "log_distance",
    "equipment",
    "weight",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]
# Market / time features, modelled linearly so they extrapolate to future dates.
LINEAR_FEATURES = ["log_daily_market_index", "days_since_start", "daily_quote_signal"]


def daily_market_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Average market_index and quote_signal per date.

    Row-level market_index is the day's market level plus small noise
    (within-day std ~0.025) and is missing for ~0.8% of rows, so the daily
    mean is a cleaner signal. Only feature columns are used (no labels), so
    validation-period dates can be included.
    """
    allrows = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    return allrows.groupby("date").agg(
        daily_market_index=("market_index", "mean"),
        daily_quote_signal=("quote_signal", "mean"),
    )


def build_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    missing_dates = set(df["date"]) - set(daily.index)
    if missing_dates:
        raise ValueError(f"no daily market data for {len(missing_dates)} date(s)")
    X = pd.DataFrame(index=df.index)
    X["log_distance"] = np.log(df["distance"])
    X["equipment"] = df["equipment"].map(EQUIPMENT_CODES)
    if X["equipment"].isna().any():
        raise ValueError("unknown equipment type")
    X["weight"] = df["weight"]
    X["quote_signal"] = df["quote_signal"]
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[col] = df[col]
    day = daily.reindex(df["date"]).set_index(df.index)
    X["log_daily_market_index"] = np.log(day["daily_market_index"])
    X["days_since_start"] = (df["date"] - TREND_ORIGIN).dt.days.astype(float)
    X["daily_quote_signal"] = day["daily_quote_signal"]
    return X
