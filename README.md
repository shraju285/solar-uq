# solar-uq

Uncertainty-Aware Deep Learning for Short-Term Solar Power Forecasting with an Interactive Streamlit Dashboard.

Final-year BSc Computer Science dissertation project (module 6CS007, University of Wolverhampton). Research question, full scope, constraints and decision log: see `docs/decisions.md` and `stage0/` in the project files.

**Status: Stages 0-9 complete.** Data audited, preprocessed and windowed; persistence, ARIMA, LSTM and quantile-regression LSTM baselines trained; split-conformal calibration applied; final consolidated results produced (`docs/stage8_final_report.md`); and a minimal Streamlit dashboard built on top of the frozen results (`dashboard/`, see below). Results-chapter write-up (Stage 10) has not started yet. Full decision history: `docs/decisions.md`.

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
  uncertainty/       Split-conformal calibration (Stage 7; see docs/decisions.md
                     for why this is per-horizon/per-quantile calibration, not
                     canonical CQR, despite the module's working name)
  evaluation/         Metrics shared by every model
  utils/             Config loading and small shared helpers
scripts/             Command-line entry points (train, evaluate, ...)
notebooks/           01_eda.ipynb, 02_results_analysis.ipynb only
results/runs/         One folder per experiment run: config, metrics, predictions, plots
models/               Saved model weights (git-ignored by default)
dashboard/            Streamlit app (Stage 9) -- presentation layer only, reads
                     the frozen results/runs/ files, trains nothing
tests/                pytest test suite
docs/
  decisions.md         Decision log — what was decided, why, what would change it
  methodology.md       Dissertation-ready write-up per stage, filled in as we go
  experiment_log.md    One entry per experiment run
```

## Prediction and results files

Each `results/runs/<run_id>/` folder is a complete, self-contained record of one experiment: the metrics (`metrics.json`), and, where relevant, the raw per-window predictions as a NumPy `.npz` array (`y_true`, `y_pred_quantiles`, `origin_time`, `quantiles`). This is the actual on-disk format produced by Stages 4-8 (see `docs/decisions.md` for how each stage's `metrics.json` is structured); the dashboard (`dashboard/`) reads these files directly and contains no model-specific logic of its own.

## Setup

```bash
cd solar-uq
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                  # installs the solaruq package in editable mode
pytest                            # should pass from a clean checkout -- the test
                                   # suite uses synthetic data only, not the
                                   # results below, which are git-ignored
```

## Data

No raw data is committed to this repository unless NIST's licence explicitly permits it and we decide to do so (see `data/README.md`). Download instructions are in `docs/data_audit.md`.

## Reproducing the pipeline results

`results/runs/`, `models/` and `data/{raw,interim,processed}` are git-ignored (see `.gitignore`) -- a fresh clone has the code but not the trained models, predictions or metrics. To regenerate everything the dashboard needs, run, in order, from the repository root with the venv above active:

```bash
python scripts/audit_nist_data.py          # Stage 2: dataset audit (needs data/raw/, see docs/data_audit.md)
python scripts/build_stage3_dataset.py     # Stage 3: preprocessing, windowing, split
python scripts/run_stage4_baselines.py     # Stage 4: persistence + ARIMA
python scripts/train_stage5_lstm.py        # Stage 5: point-forecast LSTM
python scripts/train_stage6_quantile_lstm.py   # Stage 6: quantile-regression LSTM
python scripts/run_stage7_cqr_calibration.py   # Stage 7: split-conformal calibration
python scripts/run_stage8_consolidation.py # Stage 8: consolidated results (needs Stages 4-7 above)
```

Each script reads `configs/config.yaml` and writes its own `results/runs/<run_id>/`; nothing here needs a GPU (CPU-only training times are logged in `docs/experiment_log.md`, under a few minutes per model on this project's dataset size).

## Dashboard (Stage 9)

A minimal Streamlit dashboard presents the results above -- a selectable test-period forecast plot, summary metrics, a per-horizon breakdown, a full model-comparison table, and a plain-language explanation of what the calibrated interval does and doesn't guarantee. It is read-only: it loads `results/runs/stage7_cqr/` and `results/runs/stage8_final/` (both produced above) and never trains, retrains or recalibrates anything -- see `dashboard/README.md` for how it relates to the rest of the pipeline.

```bash
# from the repository root, with the venv above active and the results
# regenerated as above (at least through Stage 8)
streamlit run dashboard/app.py
```

If a required results file is missing, the dashboard shows which pipeline script to run rather than crashing.
