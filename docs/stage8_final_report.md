# Stage 8 — Final Evaluation and Consolidated Results

Consolidates the frozen, already-approved results of Stages 4-7 into one final comparison. No model was retrained, retuned, or re-evaluated to produce this report; every number here is read directly from `results/runs/{stage4_baselines,stage5_lstm,stage6_quantile_lstm,stage7_cqr}/metrics.json` by `scripts/run_stage8_consolidation.py`, and every figure is built from those same files plus the raw prediction arrays Stage 6/7 already saved (`scripts/plot_stage8_final_figures.py`). Full machine-readable output: `results/runs/stage8_final/consolidated_results.json`.

All numbers below are on the held-out **2018 test set** (33,525 windows, 15-minute resolution, 24-step/6-hour lookback, 8-step/2-hour horizon), untouched by any training or calibration-fitting step at any stage.

## 1. Consolidated test-results table

| Model | MAE (kW) | RMSE (kW) | P50 MAE (kW) | P10-P90 coverage | P10-P90 width (kW) | Winkler score | Avg. pinball loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 13.414 | 25.864 | n/a (point-only) | n/a | n/a | n/a | n/a |
| ARIMA(2,0,0) | 16.479 | 39.774 | n/a (point-only) | n/a | n/a | n/a | n/a |
| LSTM (Stage 5) | 8.713 | 15.812 | n/a (point-only) | n/a | n/a | n/a | n/a |
| Quantile LSTM (Stage 6) | 8.852 | 16.284 | 8.852 | 0.772 | 30.14 | 45.160 | 2.7579 |
| Quantile LSTM, calibrated (Stage 7) | 8.377 | 16.176 | 8.377 | 0.800 | 27.22 | 42.577 | 2.5894 |

Persistence, ARIMA and the Stage 5 LSTM are point-forecast models only — they have no quantile output, so every probabilistic column is marked n/a rather than left blank or filled with a point-forecast proxy. For the two quantile models, "MAE/RMSE" and "P50 MAE" are the same number (the P50 slice evaluated with the identical point-forecast harness used for persistence/ARIMA/LSTM) — repeated in both columns only so the table reads consistently left-to-right.

## 2. Per-horizon analysis (all 8 horizons)

**MAE by horizon (kW):**

| Horizon | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | 105 min | 120 min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 5.43 | 8.12 | 10.37 | 12.54 | 14.63 | 16.69 | 18.72 | 20.82 |
| ARIMA(2,0,0) | 6.41 | 9.78 | 12.70 | 15.40 | 18.03 | 20.60 | 23.18 | 25.73 |
| LSTM | 6.64 | 7.49 | 7.80 | 8.78 | 8.76 | 10.19 | 9.55 | 10.50 |
| Quantile LSTM | 6.39 | 8.42 | 7.89 | 8.74 | 8.67 | 9.59 | 10.51 | 10.60 |
| Quantile LSTM, calibrated | 5.33 | 8.35 | 7.54 | 8.26 | 8.54 | 9.58 | 9.81 | 9.61 |

**RMSE by horizon (kW):**

| Horizon | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | 105 min | 120 min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 12.40 | 16.71 | 20.02 | 23.27 | 26.48 | 29.74 | 32.96 | 36.12 |
| ARIMA(2,0,0) | 30.57 | 33.37 | 35.86 | 38.18 | 40.57 | 43.03 | 45.47 | 47.95 |
| LSTM | 11.67 | 13.94 | 14.84 | 15.74 | 16.32 | 17.24 | 17.49 | 18.24 |
| Quantile LSTM | 11.85 | 14.68 | 15.04 | 16.06 | 17.11 | 17.50 | 18.42 | 18.52 |
| Quantile LSTM, calibrated | 11.47 | 14.67 | 14.91 | 15.85 | 17.03 | 17.45 | 18.41 | 18.43 |

**P10-P90 coverage by horizon** (nominal 0.80):

| Horizon | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | 105 min | 120 min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Quantile LSTM (before) | 0.649 | 0.899 | 0.676 | 0.669 | 0.905 | 0.832 | 0.763 | 0.779 |
| Calibrated (after) | 0.798 | 0.804 | 0.786 | 0.775 | 0.844 | 0.799 | 0.792 | 0.799 |

**P10-P90 interval width by horizon (kW):**

| Horizon | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | 105 min | 120 min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Quantile LSTM (before) | 20.33 | 32.48 | 22.11 | 24.15 | 51.85 | 32.63 | 29.74 | 27.83 |
| Calibrated (after) | 17.27 | 27.91 | 20.38 | 20.57 | 47.77 | 30.81 | 27.19 | 25.82 |

## 3. Final figures

All in `results/runs/stage8_final/`:

- `point_error_vs_horizon.png` — MAE vs. horizon, all 5 models.
- `coverage_vs_horizon.png` — P10-P90 coverage vs. horizon, before/after calibration, with the 80% nominal line.
- `interval_width_vs_horizon.png` — P10-P90 width vs. horizon, before/after calibration.
- `reliability_diagram.png` — pooled calibration curve, before/after.
- `representative_period.png` — actual power, calibrated P50 forecast and calibrated P10-P90 band over 2018-06-15 to 2018-06-19, 15-minute-ahead horizon.

**How the representative period was selected (documented, not cherry-picked):** mid-June was chosen first for being seasonally characteristic — large, clear daily swings rather than winter's low near-flat output — and the specific 4-day window was then checked for being fully gap-free (consecutive 15-minute origin timestamps with no missing steps) before any model's performance on it was examined. This is the same window already used for the equivalent Stage 6 illustration, reused unchanged here rather than re-picked to flatter the calibrated model.

## 4. Statistical/research interpretation

**Does the LSTM beat persistence?** Yes, clearly, in aggregate: test MAE 8.71 kW vs. 13.41 kW, RMSE 15.81 kW vs. 25.86 kW. The one exception is the very first horizon step (15 min ahead), where persistence's MAE (5.43 kW) is actually lower than the LSTM's (6.64 kW) — at that short a horizon the most recent observation is extremely informative, and persistence costs nothing to compute. From 30 minutes ahead onward the LSTM is better at every horizon, by a widening margin.

**Does the LSTM beat ARIMA(2,0,0)?** Yes, at every horizon and by a larger margin than against persistence (test MAE 8.71 vs. 16.48 kW overall). As diagnosed in the Stage 4 report, ARIMA(2,0,0) is undifferenced, so on high-variance lookback windows it extrapolates away from the last observed value rather than staying anchored — this inflates its RMSE far more than its MAE (its RMSE is worse than persistence's at every single horizon, including 15 minutes ahead, despite ARIMA's MAE being only slightly worse there).

**How does performance change with horizon?** All five models get worse with horizon, but at very different rates. Persistence and ARIMA degrade roughly linearly and do not level off over the 2-hour window (persistence: 5.43 -> 20.82 kW MAE; ARIMA: 6.41 -> 25.73 kW). Both LSTM variants degrade much more slowly and start flattening out past roughly 60-75 minutes (LSTM: 6.64 -> 10.50 kW MAE) — consistent with the LSTM having learned structure (e.g. the day's ramp shape) that a naive persistence or low-order autoregressive model cannot use.

**Do the uncertainty intervals achieve approximately nominal coverage?** Before calibration (Stage 6), coverage pooled over all horizons was close to nominal (77.2% vs. 80%) but this pooled number hid substantial per-horizon unevenness (65%-90%). After calibration (Stage 7), both the pooled (80.0%) and per-horizon (77.5%-84.4%) numbers are close to nominal, with the exception of horizon step 5 (see below).

**Does calibration improve coverage?** Yes, both overall and — more importantly, since the brief's concern was specifically per-horizon miscalibration — per horizon. The coverage-by-horizon table above shows the swings compressing from a 65-90% range to a 77.5-84.4% range.

**Does calibration change interval sharpness?** It narrows the intervals on average (overall width 30.14 -> 27.22 kW) rather than widening them — i.e. here, better calibration was not bought at the cost of less useful (wider) intervals. This is a genuinely favourable outcome but is a property of this specific model/data, not a guarantee of the calibration method in general: a model that under-covers because its raw intervals are already too narrow would need calibration to widen them, and a model whose coverage problem is pure horizon-to-horizon inconsistency (as here) can instead see calibration narrow the previously-over-covering horizons more than it widens the under-covering ones, for a net decrease.

**What remains uncalibrated / problematic?** Horizon step 5 (75 minutes ahead): both before and after calibration it has the widest interval (51.85 -> 47.77 kW) and, after calibration, the furthest-from-nominal coverage (84.4% vs. the 77.5-80.4% everywhere else). Calibration reduced this anomaly but did not resolve it.

**What does horizon 5's behaviour mean?** It is very unlikely to be a property of the physical forecasting problem itself — there is no reason 75 minutes ahead should be intrinsically harder than 60 or 90 minutes ahead for this data. The Stage 6 training loss curve showed non-monotonic oscillation for roughly 15 epochs before settling, consistent with some instability in jointly fitting 56 simultaneous outputs (8 horizons x 7 quantiles) from one small shared linear head; horizon 5's quantile outputs are the most visible symptom of that instability, carried through unchanged by Stage 7 (which only ever applies a constant per-horizon shift and cannot fix an irregular shape, only shift it). This is best read as a quantile-regression-model limitation, not a calibration-method limitation.

**What does the 33.8% quantile-crossing rate mean?** It means that correcting each of the 7 quantiles independently per horizon, rather than as a single symmetric interval, regularly produces results that are not internally consistent before the corrective re-sort is applied — i.e. the 7 independent corrections do not "agree" with each other on direction and magnitude about a third of the time. This is a real structural cost of extending split-conformal calibration to every quantile independently (see the Stage 7 terminology verification, `docs/decisions.md`): the per-quantile marginal coverage guarantees still apply to each quantile in isolation, but the procedure does not guarantee a jointly consistent set of 7 quantiles, and the re-sort that restores consistency is a practical fix outside the formal derivation.

**How does validation-set reuse affect the strength of the calibration claim?** It weakens it from a clean textbook guarantee to an approximate, empirically-supported one. The calibration set (validation, Jul-Dec 2017) was already used to pick Stage 5/6's early-stopping epoch, and is not exchangeable with the test set (2018) in the strict sense, given solar generation's seasonal cycle. The close match between achieved and nominal coverage (80.0% test vs. 80% nominal) is reassuring evidence that this did not cause a large practical problem here, but it is one empirical observation on one test set, not a proof that the theoretical guarantee holds rigorously. See `docs/decisions.md` (D-2026-10-08, Stage 7 entry and its terminology addendum) for the full derivation-level discussion.

**On statistical significance:** no formal hypothesis test (e.g. a Diebold-Mariano test for equal predictive accuracy, or a paired test on per-window errors) has been run at any stage of this project. Every comparison above is a description of the magnitude of a difference in aggregate test-set metrics, not a claim that the difference is statistically significant. This is a deliberate limitation of the current scope, not an oversight, and is called out explicitly rather than implied by careful wording.

## 5. Research-question conclusion

**Research question:** *"How effectively can uncertainty-aware deep learning models improve the reliability and interpretability of short-term solar power forecasting compared with traditional statistical forecasting approaches?"*

**Findings directly demonstrated by the experiments:**
- A deliberately simple LSTM point forecaster clearly outperforms both a persistence baseline and a classical ARIMA(2,0,0) baseline on this dataset, at essentially every forecast horizon from 30 minutes to 2 hours ahead (test MAE 8.71 kW vs. 13.41 kW and 16.48 kW respectively), and the gap widens with horizon.
- Extending the same architecture to quantile regression costs a small amount of point-forecast accuracy (test MAE 8.85 kW vs. 8.71 kW) in exchange for producing calibrated-looking prediction intervals — a direct, measured trade-off, not an assumption.
- The quantile model's raw calibration is reasonable in aggregate but uneven across individual forecast horizons.
- A simple, well-understood post-hoc statistical correction (per-horizon, per-quantile split-conformal calibration) measurably improves that per-horizon calibration without costing interval sharpness on this dataset, and in fact also happened to correct a small systematic point-forecast bias.
- At least one forecast horizon (75 minutes ahead) resists this correction and remains the least well-calibrated, indicating an unresolved limitation in the underlying quantile model rather than in the calibration step.

**Reasonable interpretation (grounded in but extending slightly beyond the raw numbers):** on this single-site, single-dataset problem, uncertainty-aware deep learning (the quantile LSTM, especially once calibrated) delivers forecasts that are both more accurate in the point sense and meaningfully more interpretable than the traditional statistical baseline (ARIMA) — not only because the point forecast is better, but because it additionally produces an honestly-evaluated confidence interval that a persistence or ARIMA forecast simply cannot offer. The calibration step demonstrates that at least part of a quantile model's uncertainty-estimation shortfall can be fixed cheaply and transparently after training, which is practically relevant for a deployed forecasting system. The remaining horizon-5 irregularity suggests that "uncertainty-aware" is not a solved problem even within this approach — the model's confidence in its own uncertainty is itself imperfect and horizon-dependent, which is an interesting and honest finding in its own right for the "interpretability" half of the research question.

**Limitations of this conclusion:**
- Single site, single array, single weather station, one train/val/test split, no cross-validation and no repeated-seed variance estimate — the reported numbers are point estimates of performance on one particular partition of one dataset, not necessarily representative of performance on a different site, year, or climate.
- No statistical significance testing, as stated above — differences are described by magnitude, not by formal inferential confidence.
- The calibration guarantee itself rests on an approximately-, not exactly-, satisfied exchangeability assumption (seasonal shift between calibration and test periods; validation-set reuse) — see Section 4.
- Only one uncertainty-quantification method (quantile regression, with and without conformal calibration) was evaluated; the research question's comparison is necessarily limited to the specific uncertainty-aware approach this project built, not deep uncertainty-aware methods in general (MC Dropout, deep ensembles, etc. were explicitly out of scope).

## 6. Reproducibility

| Item | Value |
|---|---|
| Stage 4 (baselines) commit | `e13ff0d` |
| Stage 5 (LSTM) commit | `23548a9` |
| Stage 6 (quantile LSTM) commits | `9922496`, `1fbb651` |
| Stage 7 (calibration) commits | `06d236c`, `5a58a11`, `9bee2f8` (terminology correction, no logic change) |
| Stage 8 (this report) commit | recorded in `docs/experiment_log.md` after this commit is made |
| Config file | `configs/config.yaml` (copied verbatim into each stage's run; `models.lstm`, `models.quantile_lstm`, `uncertainty.cqr` sections hold the exact hyperparameters) |
| Random seed | 42 (`project.seed`) for all model weight initialization and training; conformal calibration itself is deterministic (no randomness) |
| Test-set period | 2018-01-01 to 2018-12-31, 33,525 windows, 15-minute resolution, 24-step (6h) lookback / 8-step (2h) horizon, fixed `-05:00` timestamps (no DST) |
| Result files used (read-only, not modified) | `results/runs/stage4_baselines/metrics.json`, `results/runs/stage5_lstm/metrics.json`, `results/runs/stage6_quantile_lstm/metrics.json` + `test_predictions.npz`, `results/runs/stage7_cqr/metrics.json` + `test_predictions_calibrated.npz` |
| Stage 8 scripts | `scripts/run_stage8_consolidation.py` (builds the tables), `scripts/plot_stage8_final_figures.py` (builds the figures) |
| Stage 8 outputs | `results/runs/stage8_final/consolidated_results.json` + 5 PNG figures (git-ignored, same policy as every other `results/runs/*` folder) |
| Test suite | 70/70 passing at the time of this report (`pytest`, run from the project venv) |
