# Freight Rate Prediction

Predicts `posted_rate` for 12,000 freight loads (Nov–Dec 2025) from 48,000 labelled loads (Jan–Oct 2025).

**Result:** 2.06% MAPE / $48 MAE on a rolling-origin time-based validation (5 folds), against 4.52% for a rate-per-mile baseline.

- Final predictions: [`validation_predictions.csv`](validation_predictions.csv)
- Report: [`reports/report.pdf`](reports/report.pdf)
- December chart: [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png)

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
# 1. Clean data, run validation, fit the final model, write predictions
#    -> validation_predictions.csv, data/december_chart_inputs.csv, reports/
python train.py

# 2. Validate the outputs and generate the December chart (provided scorer)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`train.py` takes about 3–4 minutes on a laptop and is deterministic (fixed random seeds).

## Approach

**Data quality**

| Issue | Treatment |
|---|---|
| 677 corrupted labels (1.4%), rates 2.2–5x too high or too low | Flagged from out-of-fold residuals of a robust model; excluded from training |
| Negative weights (sign errors) | Absolute value |
| Missing weight | Left missing; handled natively by the gradient-boosting model |
| Missing or noisy `market_index` | Replaced by the daily mean market index |
| 8 cities appear only in validation | No city-ID features; locations enter through latitude/longitude |

**Validation.** The validation loads all come after the training period, so a random split would be optimistic. Models are compared with a rolling-origin split: train on Jan..k, test on the next two months, for k = Apr..Aug. A city-holdout test checks generalisation to unseen cities.

**Model.** An additive model on `log(rate)`:

```
log(rate) = GBM(log distance, equipment, weight, quote_signal, pickup/delivery lat-lon)
          + linear(log daily market index, days since 1 Jan, daily mean quote_signal)
```

The gradient-boosting part (scikit-learn `HistGradientBoostingRegressor`) captures non-linear load pricing. The linear part captures the market level and the upward trend, so they extrapolate to future dates, which trees cannot do. The two parts are fitted by backfitting.

| Model (rolling-origin CV) | MAPE | MAE |
|---|---|---|
| Median rate/mile by equipment x distance band | 4.52% | $104 |
| GBM, load features only | 2.81% | $64 |
| GBM, load features + market index in trees | 3.29% | $75 |
| GBM + coordinates | 2.65% | $62 |
| Additive: GBM + linear market/trend | 2.59% | $58 |
| **Additive: GBM + coordinates + linear market/trend (final)** | **2.06%** | **$48** |

## Project structure

```
train.py                 pipeline entry point
score.py                 provided scorer (unchanged)
src/freight/
    data.py              loading, feature cleaning, label-error detection
    features.py          feature engineering, daily market table
    model.py             AdditiveRateModel (GBM + linear market/trend term)
    evaluate.py          rolling-origin and city-holdout validation, metrics
data/                    provided input files (december_chart_inputs.csv is filled by train.py)
reports/                 metrics.json, CV tables, figures, report.pdf
scorer_results/          chart produced by score.py
```
