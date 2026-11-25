"""End-to-end pipeline: clean -> validate -> fit final model -> write predictions.

Usage:
    python train.py

Outputs:
    validation_predictions.csv            final predictions for data/validation.csv
    data/december_chart_inputs.csv        predicted_rate column filled in
    reports/metrics.json, reports/*.csv   validation results
    reports/figures/*.png                 figures used in the report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from freight.data import DATA_DIR, TARGET, clean_features, flag_label_errors, load_raw  # noqa: E402
from freight.evaluate import city_holdout_folds, metrics, rolling_origin_folds, run_cv  # noqa: E402
from freight.features import LINEAR_FEATURES, TREE_FEATURES, build_features, daily_market_table  # noqa: E402
from freight.model import AdditiveRateModel  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


class RatePerMileBaseline:
    """Median rate-per-mile by equipment and distance band, times distance."""

    BANDS = [0, 150, 300, 500, 800, 1200, 1800, 2500, np.inf]

    def fit(self, X, rate):
        key = self._key(X)
        rpm = np.asarray(rate) / np.exp(X["log_distance"].to_numpy())
        self.table_ = pd.Series(rpm).groupby(key.to_numpy()).median()
        self.fallback_ = float(np.median(rpm))
        return self

    def predict(self, X):
        rpm = self._key(X).map(self.table_).fillna(self.fallback_).to_numpy()
        return rpm * np.exp(X["log_distance"].to_numpy())

    def _key(self, X):
        band = pd.cut(np.exp(X["log_distance"]), self.BANDS, labels=False)
        return X["equipment"].astype(int).astype(str) + "_" + band.astype(str)


CANDIDATES = {
    "Baseline: median rate/mile by equipment x distance band": lambda: RatePerMileBaseline(),
    "GBM, load features only (no coords, no time)": lambda: AdditiveRateModel(
        tree_features=["log_distance", "equipment", "weight", "quote_signal"], linear_features=[]
    ),
    "GBM, load features + market_index in trees": lambda: AdditiveRateModel(
        tree_features=["log_distance", "equipment", "weight", "quote_signal", "log_daily_market_index"],
        linear_features=[],
    ),
    "GBM + coords (no time)": lambda: AdditiveRateModel(linear_features=[]),
    "Additive: GBM (no coords) + linear market/trend": lambda: AdditiveRateModel(
        tree_features=["log_distance", "equipment", "weight", "quote_signal"]
    ),
    "Additive: GBM + coords + linear market/trend (final)": lambda: AdditiveRateModel(),
}
FINAL = "Additive: GBM + coords + linear market/trend (final)"


def explore(train: pd.DataFrame, label_error: pd.Series, daily: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """Data-quality summary and figures."""
    raw_train, raw_valid = load_raw()
    unseen = sorted((set(valid["pickup"]) | set(valid["delivery"])) - (set(train["pickup"]) | set(train["delivery"])))
    rpm = train[TARGET] / train["distance"]
    summary = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
        "validation_dates": [str(valid["date"].min().date()), str(valid["date"].max().date())],
        "label_errors_flagged": int(label_error.sum()),
        "label_errors_pct": round(float(label_error.mean() * 100), 2),
        "label_errors_high": int((label_error & (rpm > rpm[~label_error].median())).sum()),
        "label_errors_low": int((label_error & (rpm <= rpm[~label_error].median())).sum()),
        "negative_weight_train": int((raw_train["weight"] < 0).sum()),
        "negative_weight_validation": int((raw_valid["weight"] < 0).sum()),
        "missing_weight_train": int(raw_train["weight"].isna().sum()),
        "missing_weight_validation": int(raw_valid["weight"].isna().sum()),
        "missing_market_index_train": int(raw_train["market_index"].isna().sum()),
        "missing_market_index_validation": int(raw_valid["market_index"].isna().sum()),
        "duplicate_load_ids": int(raw_train["load_id"].duplicated().sum() + raw_valid["load_id"].duplicated().sum()),
        "cities_only_in_validation": unseen,
        "validation_rows_touching_unseen_city": int((valid["pickup"].isin(unseen) | valid["delivery"].isin(unseen)).sum()),
    }

    FIGURES.mkdir(parents=True, exist_ok=True)
    # 1. Rate per mile: clean vs flagged labels
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    bins = np.linspace(np.log(rpm).min(), np.log(rpm).max(), 80)
    ax.hist(np.log(rpm[~label_error]), bins=bins, color="#064A56", alpha=0.85, label="kept")
    ax.hist(np.log(rpm[label_error]), bins=bins, color="#D1495B", alpha=0.9, label=f"flagged label error ({int(label_error.sum())})")
    ax.set_yscale("log")
    ax.set_xlabel("log(posted_rate / distance)")
    ax.set_ylabel("loads (log scale)")
    ax.set_title("Rate per mile: corrupted labels form separate clusters", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "rate_per_mile_label_errors.png")
    plt.close(fig)

    # 2. Daily market index across train and validation periods
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    split = pd.Timestamp("2025-11-01")
    ax.plot(daily.index[daily.index < split], daily["daily_market_index"][daily.index < split], color="#064A56", lw=1, label="train period")
    ax.plot(daily.index[daily.index >= split], daily["daily_market_index"][daily.index >= split], color="#E39B2D", lw=1, label="validation period")
    ax.set_ylabel("daily mean market_index")
    ax.set_title("Market index: weekly cycle, seasonal swing, lower level in Nov-Dec", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_market_index.png")
    plt.close(fig)
    return summary


def plot_time_effect(model: AdditiveRateModel, X: pd.DataFrame, rate: pd.Series, dates: pd.Series) -> None:
    """Daily average of log(rate) minus the tree part, against the fitted linear term."""
    partial = np.log(rate.to_numpy()) - model.tree_.predict(X[model.tree_features])
    fitted = model.linear_.predict(X[model.linear_features])
    df = pd.DataFrame({"date": dates.to_numpy(), "partial": partial, "fitted": fitted}).groupby("date").mean()
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    ax.plot(df.index, (np.exp(df["partial"]) - 1) * 100, color="#9DAFB3", lw=1, label="observed daily level")
    ax.plot(df.index, (np.exp(df["fitted"]) - 1) * 100, color="#064A56", lw=1.6, label="linear market + trend term")
    ax.set_ylabel("rate level vs. baseline (%)")
    ax.set_title("Daily rate level is explained by market index + a steady upward trend", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "time_effect.png")
    plt.close(fig)


def main() -> None:
    train_raw, valid_raw = load_raw()
    train = clean_features(train_raw)
    valid = clean_features(valid_raw)
    daily = daily_market_table(train, valid)

    print("Flagging label errors ...")
    label_error = flag_label_errors(train_raw)
    summary = explore(train, label_error, daily, valid)
    print(json.dumps(summary, indent=2))

    X = build_features(train, daily)
    rate = train[TARGET]

    print("\nRolling-origin validation (train on past months, test on next 2):")
    comparison = []
    fold_tables = {}
    for name, factory in CANDIDATES.items():
        res = run_cv(rolling_origin_folds(train["date"]), X, rate, label_error, factory)
        fold_tables[name] = res
        comparison.append({"model": name, "mape_pct": res["mape_pct"].mean(), "mae": res["mae"].mean(), "rmse": res["rmse"].mean()})
        print(f"  {name:<58s} MAPE {res['mape_pct'].mean():.2f}%  MAE ${res['mae'].mean():.1f}")
    comparison = pd.DataFrame(comparison)
    final_folds = fold_tables[FINAL]

    print("\nUnseen-city holdout for the final model:")
    city = run_cv(city_holdout_folds(train), X, rate, label_error, CANDIDATES[FINAL])
    city_nocoord = run_cv(
        city_holdout_folds(train), X, rate, label_error, CANDIDATES["Additive: GBM (no coords) + linear market/trend"]
    )
    city["mape_pct_without_coords"] = city_nocoord["mape_pct"].to_numpy()
    print(city[["fold", "n_test", "mape_pct", "mape_pct_without_coords"]].to_string(index=False))

    print("\nFitting final model on all clean training rows ...")
    keep = ~label_error.to_numpy()
    final = AdditiveRateModel().fit(X[keep], rate[keep])
    in_sample = metrics(rate[keep], final.predict(X[keep]))
    plot_time_effect(final, X[keep], rate[keep], train["date"][keep])

    # Validation predictions, in template order
    X_valid = build_features(valid, daily)
    pred = pd.Series(final.predict(X_valid), index=valid["load_id"])
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    template["predicted_rate"] = template["load_id"].map(pred).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(template):,} rows)")

    december = predict_december(final, train, daily)

    REPORTS.mkdir(exist_ok=True)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    final_folds.to_csv(REPORTS / "rolling_origin_folds.csv", index=False)
    city.to_csv(REPORTS / "city_holdout_folds.csv", index=False)
    out = {
        "data_summary": summary,
        "final_model": {
            "tree_features": TREE_FEATURES,
            "linear_features": LINEAR_FEATURES,
            "linear_coefficients": final.linear_coefficients(),
            "trained_on_rows": int(keep.sum()),
            "in_sample": in_sample,
        },
        "rolling_origin_cv": {
            "mean_mape_pct": float(final_folds["mape_pct"].mean()),
            "mean_mae": float(final_folds["mae"].mean()),
            "mean_rmse": float(final_folds["rmse"].mean()),
            "mean_mape_pct_incl_label_errors": float(final_folds["mape_pct_incl_label_errors"].mean()),
        },
        "city_holdout": {
            "mean_mape_pct": float(city["mape_pct"].mean()),
            "mean_mape_pct_without_coords": float(city["mape_pct_without_coords"].mean()),
        },
        "validation_predictions": {
            "mean": float(template["predicted_rate"].mean()),
            "median": float(template["predicted_rate"].median()),
            "min": float(template["predicted_rate"].min()),
            "max": float(template["predicted_rate"].max()),
        },
        "december": {"min": float(december.min()), "max": float(december.max()), "mean": float(december.mean())},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nFinal model rolling-origin MAPE: {out['rolling_origin_cv']['mean_mape_pct']:.2f}%")


def predict_december(model: AdditiveRateModel, train: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Fill data/december_chart_inputs.csv.

    The chart file only has lane, distance, equipment, weight and date. The
    model also needs coordinates (looked up from the city table) and the
    day's market_index / quote_signal, which come from the daily averages of
    validation.csv (it covers every December date).
    """
    path = DATA_DIR / "december_chart_inputs.csv"
    chart = pd.read_csv(path)
    coords = pd.concat(
        [
            train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
            train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        ]
    ).drop_duplicates("city").set_index("city")

    rows = chart.copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows["pickup_lat"] = rows["pickup"].map(coords["lat"])
    rows["pickup_lon"] = rows["pickup"].map(coords["lon"])
    rows["delivery_lat"] = rows["delivery"].map(coords["lat"])
    rows["delivery_lon"] = rows["delivery"].map(coords["lon"])
    rows["quote_signal"] = rows["date"].map(daily["daily_quote_signal"])
    rows["market_index"] = rows["date"].map(daily["daily_market_index"])
    if rows[["pickup_lat", "delivery_lat", "quote_signal", "market_index"]].isna().any().any():
        raise ValueError("missing lookup data for December chart inputs")

    pred = model.predict(build_features(rows, daily))
    chart["predicted_rate"] = np.round(pred, 2)
    chart.to_csv(path, index=False)
    print(f"Filled {path.relative_to(ROOT)} (range ${pred.min():.0f}-${pred.max():.0f})")
    return chart["predicted_rate"]


if __name__ == "__main__":
    main()
