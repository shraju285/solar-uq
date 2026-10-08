# Experiment log

*One entry per experiment run, appended chronologically. Stub created at Stage 1 — nothing to log yet, no modelling has started.*

Each entry (from Stage 4 onward) should record: date, git commit hash, config used, result folder path (`results/runs/<run_id>/`), headline metrics, and one line on what was learned or what went wrong. This is the raw material for the dissertation's results chapter and for the "evidence of project management" the handbook requires.

## 2026-10-08 — Stage 4: persistence + ARIMA baselines

Commit `e13ff0d`. Config: `configs/config.yaml` (ARIMA order selected at runtime, not config-fixed). Results: `results/runs/stage4_baselines/metrics.json`.
Headline (test): persistence MAE 13.41 / RMSE 25.86 kW; ARIMA(2,0,0) MAE 16.48 / RMSE 39.77 kW.
Learned: persistence beats ARIMA on this dataset — ARIMA(2,0,0) is undifferenced, so it extrapolates away from the last value on high-variance lookback windows instead of staying anchored, inflating RMSE far more than MAE. Not a bug; a real, explained finding.

## 2026-10-08 — Stage 5: direct multi-step LSTM point forecast

Commit `23548a9`. Config: `configs/config.yaml` `models.lstm` (hidden_size 64, num_layers 1, dropout 0.1, Adam 1e-3, batch 256, seed 42). Results: `results/runs/stage5_lstm/`.
Headline (test): MAE 8.71 / RMSE 15.81 kW — beats both Stage 4 baselines on every split and (beyond the first horizon step) every horizon. Best epoch 10/15 (early-stopped), 146.8s total training time, 20,168 parameters.
Learned: a single small LSTM layer is enough to clearly beat both baselines on this problem; persistence still edges it out at the very first (15-min) horizon step, where the most recent reading is extremely informative.
