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

## 2026-10-08 — Stage 6: quantile-regression LSTM (P10-P90)

Commit `9922496`. Config: `configs/config.yaml` `models.quantile_lstm` (same encoder hyperparameters as Stage 5, quantiles q05..q95). Results: `results/runs/stage6_quantile_lstm/`.
Headline (test): P50 MAE 8.85 / RMSE 16.28 kW; P10-P90 coverage 77.2% (nominal 80%), width 30.14 kW, pinball loss 2.758, interval score 45.16. Best epoch 17/22 (early-stopped), 262.6s training time, 24,440 parameters.
Learned: the P50 head costs a little point-forecast accuracy relative to Stage 5's dedicated LSTM (expected, from jointly optimizing 7 quantiles). Pooled calibration looks reasonable, but per-horizon coverage and interval width are noticeably uneven rather than smoothly widening with horizon, and the training loss curve shows some non-monotonic oscillation before settling — a genuine limitation, not an evaluation artefact; see `docs/decisions.md` (D-2026-10-08, Stage 6 entry) for the full discussion and a scoped follow-up idea.

## 2026-10-08 — Stage 7: split-conformal calibration of the Stage 6 quantile LSTM

Commit `06d236c`. Config: `configs/config.yaml` `uncertainty.cqr` (method `split_conformal_per_horizon_per_quantile`, calibration source = the existing validation split). Results: `results/runs/stage7_cqr/`. No retraining; Stage 5/6 untouched.
Headline (test, before -> after calibration): P10-P90 coverage 77.2% -> 80.0% overall; per-horizon coverage 65-90% -> 77.5-84.4% (much more even); interval width 30.14 -> 27.22 kW (down, not up); P50 MAE 8.85 -> 8.38 kW; pinball loss (all quantiles) 2.758 -> 2.589.
Learned: a per-horizon, per-quantile split-conformal correction fixed Stage 6's uneven calibration without the width/coverage trade-off you'd normally expect — but horizon step 5 is still the worst-calibrated and widest step even after correction (84.4% coverage, 47.8 kW width), so the underlying per-horizon irregularity in the trained quantile head is reduced, not eliminated. Also: correcting all 7 quantiles independently causes real crossing — 33.8% of (window, horizon) pairs needed re-sorting to restore order, a genuine cost of this extension of CQR worth flagging if revisited.

## 2026-10-08 — Stage 8: final evaluation and consolidated results

Commit: see below (log updated after commit, same two-step pattern as Stages 6-7). No new run/config — reads `results/runs/{stage4_baselines,stage5_lstm,stage6_quantile_lstm,stage7_cqr}/metrics.json` only. Results: `results/runs/stage8_final/` (consolidated_results.json + 5 figures); full write-up: `docs/stage8_final_report.md`.
Headline (test): LSTM beats persistence (8.71 vs 13.41 kW MAE) and ARIMA (vs 16.48 kW) at every horizon from 30 min onward. Quantile LSTM calibrated: MAE 8.38 kW, P10-P90 coverage 80.0% (vs 77.2% uncalibrated), width 27.22 kW (vs 30.14 kW uncalibrated) — calibration improved coverage AND reduced width here, not a trade-off.
Learned: consolidating across stages makes the horizon-5 anomaly and the point-vs-probabilistic trade-off visually obvious in a way no single stage's report showed on its own. No significance test was run (deliberate scope decision, documented in `docs/decisions.md`) — all comparisons are magnitudes, not significance claims.
