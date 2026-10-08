# Methodology

*Filled in as each stage completes. Stub created at Stage 1.*

This will document, for each pipeline stage: the method used, why it was chosen over the alternatives, and what could go wrong with it (leakage risk, assumptions, limitations). See `docs/decisions.md` for the decisions themselves; this file is where the reasoning becomes dissertation-ready prose.

## Contents (to be filled)

- [ ] Dataset and audit findings (Stage 2)
- [ ] Preprocessing, feature engineering, leakage-safe windowing (Stage 3)
- [ ] Baselines: persistence, smart persistence, ARIMA (Stage 4)
- [x] LSTM point forecast (Stage 5)
- [x] Quantile-regression LSTM (Stage 6)
- [ ] CQR, if built (Stage 7)
- [ ] Final evaluation methodology (Stage 8)
- [ ] Dashboard design (Stage 10)

## LSTM point forecast (Stage 5)

A single-layer LSTM (hidden size 64) encodes the 24-step (6-hour) lookback — the target's own history plus the six observed WS_1 weather variables plus solar position. Its final hidden state is concatenated with the flattened 8-step horizon's solar position (the only input knowable in advance over the 2-hour forecast horizon) and passed through one linear layer that outputs all 8 horizon values directly, in a single forward pass — a direct multi-step forecast, not a recursive one, so forecast errors at step *k* cannot compound into step *k+1*'s input. The target is standardized (z-score, fit on the training split only) purely for training stability; every reported metric is in physical kW after inverting that transform. Trained with Adam (early stopping on validation loss, seed 42, CPU-only), it converged in 146.8s and comfortably outperformed both baselines on the held-out 2018 test set (MAE 8.71 kW vs. persistence's 13.41 kW and ARIMA(2,0,0)'s 16.48 kW). Full configuration and results: `docs/decisions.md` (D-2026-10-08, Stage 5 entry) and `results/runs/stage5_lstm/`.

## Quantile-regression LSTM (Stage 6)

Stage 6 reuses Stage 5's encoder unchanged and widens only the output head, so the only variable between the two models is the loss/output structure, not the network size — the point being to isolate what quantile regression itself costs or gains relative to a dedicated point forecaster. The head predicts all 7 locked quantiles (`q05`...`q95`, including the required P10/P50/P90) for all 8 horizon steps in one pass (56 outputs), trained jointly with the pinball (quantile) loss. Quantile crossing — a known failure mode of independently-trained multi-quantile heads, where a higher quantile's prediction can land below a lower one's — is prevented structurally rather than left to the data: each quantile above the lowest is built as the previous one plus a non-negative increment, so the output is non-decreasing by construction, not merely in practice.

The P50 slice, evaluated with the same harness as persistence/ARIMA/Stage 5, is close to but slightly behind the dedicated Stage 5 LSTM (test MAE 8.85 kW vs. 8.71 kW) — an expected, small cost of optimizing 7 objectives jointly rather than one. Pooled across all horizons, the model's calibration is reasonably close to nominal (test P10-P90 coverage 77.2% against an 80% target; the full reliability diagram tracks the diagonal closely). **Per forecast horizon, however, coverage and interval width are visibly uneven rather than smoothly widening with horizon** — e.g. test coverage swings between 65% and 90% across the 8 steps, and interval width spikes sharply at horizon step 5. The training curve shows both train and validation pinball loss oscillating for several epochs before settling, consistent with some instability in jointly fitting 56 outputs from one small shared head. This is reported as a genuine, unresolved limitation of the current configuration (see `docs/decisions.md`, D-2026-10-08 Stage 6 entry, for the full discussion and a scoped follow-up), not smoothed over to present a cleaner result. Full configuration, metrics and plots: `results/runs/stage6_quantile_lstm/`.
