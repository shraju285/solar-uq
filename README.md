# solar-uq

Uncertainty-Aware Deep Learning for Short-Term Solar Power Forecasting with an Interactive Streamlit Dashboard.

Final-year BSc Computer Science dissertation project (module 6CS007, University of Wolverhampton). Research question, full scope, constraints and decision log: see `docs/decisions.md` and `stage0/` in the project files.

**Status: Stage 1 (repository and environment) — no data downloaded, no models trained yet.**

## What this is

Short-term (≤2 hour) solar PV power forecasting for one array, comparing persistence, ARIMA and an LSTM point forecaster against a quantile-regression LSTM that produces calibrated prediction intervals. The emphasis is on whether the uncertainty estimates are reliable (calibrated) and useful (sharp), evaluated with a leakage-safe chronological pipeline, and presented in a minimal Streamlit dashboard.

## Repository structure

```
configs/            Central YAML config — every script reads settings from here
data/
  raw/               Downloaded data, untouched (git-ignored)
  interim/           Cleaned/resampled (git-ignored)
  processed/         Windowed, scaled, split-ready (git-ignored)
src/solaruq/
  data/              Loading and cleaning
  features/          Feature engineering, scaling, windowing
  models/            Persistence, ARIMA, LSTM, quantile LSTM
  uncertainty/       CQR (stretch goal, Stage 7)
  evaluation/         Metrics shared by every model
  utils/             Config loading and small shared helpers
scripts/             Command-line entry points (train, evaluate, ...)
notebooks/           01_eda.ipynb, 02_results_analysis.ipynb only
results/runs/         One folder per experiment run: config, metrics, predictions, plots
models/               Saved model weights (git-ignored by default)
dashboard/            Streamlit app (Stage 10)
tests/                pytest test suite
docs/
  decisions.md         Decision log — what was decided, why, what would change it
  methodology.md       Dissertation-ready write-up per stage, filled in as we go
  experiment_log.md    One entry per experiment run
```

## Prediction file schema

Every model writes predictions in the same format, so the evaluation module and the dashboard never contain model-specific logic:

```
timestamp, horizon, y_true, y_hat, [q05, q10, q25, q50, q75, q90, q95]
```

Quantile columns are present only for models that produce them (persistence and ARIMA point forecasts omit them; the quantile LSTM includes all seven).

## Setup

```bash
cd solar-uq
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                  # installs the solaruq package in editable mode
pytest                            # should pass — Stage 1 has one smoke test
```

## Data

No raw data is committed to this repository unless NIST's licence explicitly permits it and we decide to do so (see `data/README.md`). Download instructions are added in Stage 2.
