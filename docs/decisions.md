# Decisions log

One entry per decision that shapes the project. Newest first. This is the record you (and any examiner) can point to for "why did you choose X" — every entry says what was decided, why, and what would change if it turned out wrong.

---

## D-2026-10-08 — Stage 7: split-conformal calibration of the Stage 6 quantile LSTM

**Status:** implemented and approved by the user 2026-10-08. Terminology verified against the code 2026-10-08 (see addendum below) before Stage 8 began — no implementation change, only how it's described.

**Terminology addendum (post-approval verification, no code changed):** the user asked for a precise mathematical review of this stage before Stage 8. Confirmed facts, read directly from `src/solaruq/uncertainty/conformal.py`: the conformity score is the signed one-sided residual r_i = y_i - q_hat_{tau,h}(x_i), computed independently per (horizon, quantile) pair — not the two-sided max-based nonconformity score E_i = max(q_lo - y, y - q_hi) that canonical CQR (Romano, Patterson & Candes, 2019) defines for a single interval. The correct name for this implementation is therefore **per-horizon, per-quantile split-conformal calibration**, not "CQR" — it reuses CQR's finite-sample order-statistic formula but applies it one-sidedly to every quantile independently rather than symmetrically to one interval. Point 3 and the module docstring below use "CQR" loosely as the inspiring reference; `docs/methodology.md`'s Stage 7 section has been corrected to use the precise term and spell out this distinction for the dissertation. Coverage-claim precision: each individually calibrated quantile carries a standard split-conformal marginal guarantee; the reported P10-P90 interval's ~80% coverage follows only as a conservative union-bound consequence of the two one-sided guarantees, not the tighter bound canonical CQR's symmetric construction gives directly, and the post-hoc crossing re-sort (point 5 below) is outside the derivation entirely. The exchangeability assumption underlying all of it is only approximate: the calibration period (Jul-Dec 2017) and test period (2018) are not exchangeable in the strict sense given solar generation's seasonal cycle, and reusing the validation split (point 2 below) can mildly bias corrections to be less conservative than theory assumes.

**Context:** Stage 6 was approved and frozen as an immutable benchmark. Its calibration was pooled-reasonable (test P10-P90 coverage 77.2% vs 80% nominal) but uneven per horizon (coverage ranging 65%-90% across the 8 steps, an irregular width spike at horizon step 5). The brief asked whether this can be improved with a simple, scientifically defensible post-hoc method, without retraining, without touching the test set, and without hiding the horizon-5 behaviour.

**1. Inspecting the current split structure (required before choosing anything):** the project only has train/val/test (`configs/config.yaml` `split:` — no `calibration_end`). Checking `scripts/train_stage6_quantile_lstm.py`, Stage 5 and Stage 6 both already used the *entire* validation split (16,843 windows) for every epoch's validation loss / early-stopping decision — there is no leftover, untouched subset of validation. So the current structure does **not** support a textbook-strict, fully independent third split without either (a) touching the locked Stage 3 split boundaries, or (b) rerunning Stage 5/6 with a smaller validation set — both excluded by this brief (preserve Stage 5/6, don't change the test set, don't rerun Stage 3's locked methodology without a genuine defect).

**2. Decision: reuse the full validation split as the calibration set, documented as a deliberate, acceptable trade-off, not an oversight.** Validation was used only to pick an early-stopping *epoch* — never to fit the model's weights (those came from train only). Split-conformal calibration's theoretical requirement is that the calibration set wasn't used to fit the predictor's parameters, and that it's exchangeable with the evaluation set; it does not require the calibration set to be untouched by model-selection decisions. With a large calibration set (16,843 windows) the residual bias this reuse could introduce is expected to be small. This is recorded here explicitly, instead of silently treating it as textbook-clean, so it can be revisited if challenged in the viva.

**3. Calibration method chosen: split-conformal quantile calibration, applied independently per (horizon step, quantile).** For each of the 8 horizon steps and each of the 7 quantiles, fit one additive correction from calibration-set residuals using the standard finite-sample order-statistic formula (Vovk, Gammerman & Shafer 2005; the two-sided interval case is Romano, Patterson & Candes 2019, "Conformalized Quantile Regression" — this extends the same idea to every quantile independently, since the brief asks for pinball loss on all 7, not just the P10/P90 pair): `c = k`-th smallest calibration residual, `k = ceil((n+1) * tau)`. Calibrated prediction = raw Stage 6 quantile + its correction. Chosen over alternatives because: it's pure arithmetic on top of already-frozen predictions (no new neural architecture, as instructed), it's distribution-free (no assumption about the residual distribution's shape), and it directly targets the diagnosed problem (per-horizon miscalibration) by correcting each horizon separately rather than with one global shift. Implemented in `src/solaruq/uncertainty/conformal.py` — the module the README already reserved for "CQR (stretch goal, Stage 7)" since Stage 0.

**4. Data used to fit calibration:** the validation split only (`data/processed/stage3/val.npz`), run through the frozen Stage 6 model (`models/stage6_quantile_lstm.pt`) in eval-mode inference — no gradient updates, no retraining. The test set is read back byte-for-byte from Stage 6's own saved `test_predictions.npz`, never recomputed, and a runtime assertion in `scripts/run_stage7_cqr_calibration.py` checks the reused array still reproduces Stage 6's originally reported P50 MAE before anything else runs, so any accidental drift would fail loudly rather than silently change "before" numbers.

**5. An honest side effect, reported rather than hidden: corrected quantiles can cross.** Correcting each of the 7 quantiles independently per horizon gives no guarantee they stay in order (e.g. a corrected P75 landing below a corrected P50). This happened in **33.8% of all (window, horizon) evaluations** on the test set — a real, substantial rate, not a rare edge case. The fix applied is the simplest defensible one: re-sort the 7 corrected values back into ascending order per (window, horizon) after correction. This is reported as a genuine limitation of applying CQR per-quantile independently (a known issue in the conformal literature), not swept under the rug.

**Real-data result (`results/runs/stage7_cqr/metrics.json`), test set, calibration fit on validation only:** P10-P90 coverage improved from 77.2% to **80.0%** (almost exactly nominal) and, importantly, per-horizon coverage became far more even (65%-90% before -> 77.5%-84.4% after) **without the width getting worse on average** (overall interval width actually *decreased*, 30.14 -> 27.22 kW) — calibration was not bought at the cost of systematically wider intervals here. The P50 point forecast also improved (MAE 8.85 -> 8.38 kW), because the calibration set happened to reveal a consistent median bias that also existed in the test period; this is a genuine, data-grounded result, not guaranteed to hold if the relationship between the 2017-H2 calibration period and the 2018 test period were less stable. **Horizon step 5 remains the worst-calibrated and widest step after calibration** (84.4% coverage, 47.8 kW width, both still the most extreme of the 8 steps) — deliberately not smoothed away; this is the clearest unresolved finding carried into the Stage 7 report.

---

## D-2026-10-08 — Stage 6: quantile-regression LSTM

**Status:** implemented, approved by the user 2026-10-08.

**Why quantile regression (vs. e.g. MC Dropout or deep ensembles):** fixed at Stage 0 (D-2026-09-27) — a single well-defined loss (pinball loss) that is easier to fully explain and defend in the viva than a sampling-based method, and it maps directly onto the `[q05, q10, q25, q50, q75, q90, q95]` prediction-file schema the project already committed to in the README. Not revisited here; this entry only covers what Stage 5 left open.

**Quantiles selected:** the full set already locked in `configs/config.yaml` at Stage 0 — `[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`. This satisfies Stage 6's explicit minimum (P10/P50/P90) while giving a proper calibration/reliability diagram (needs more than 3 points) at no extra implementation cost — the model predicts all of them in one forward pass regardless of how many there are.

| Decision | Choice | Why |
|---|---|---|
| Architecture | Identical encoder to Stage 5's `LSTMPointForecaster` (1-layer LSTM, hidden_size=64, concatenated with flattened horizon-known solar position), only the head changes: `Linear(88, horizon_steps * n_quantiles)` instead of `Linear(88, horizon_steps)` | The brief asked to "keep the architecture as close as reasonably possible to Stage 5 so the comparison is scientifically meaningful" — isolating the effect of quantile regression from the effect of a differently-sized network |
| Non-crossing quantiles | Enforced by construction: the head's raw output is reshaped to (horizon_steps, n_quantiles), the lowest quantile is used as-is, and each subsequent quantile is the previous one plus `softplus(raw)` (always >= 0), via `cumsum` | Independently pinball-trained quantiles have no guarantee of coming out in the right order (crossing). Enforcing it structurally is simpler to explain and test than a post-hoc sorting fix or a crossing-penalty loss term, and keeps the "no architectural changes without a specific technical reason" rule satisfied — this is a one-line change to the head, not a new architecture |
| Loss | Mean pinball loss across every (window, horizon step, quantile) triple, summed into one scalar for one shared backward pass | Standard for joint multi-quantile regression; a single loss keeps training as close to Stage 5's single-MSE loop as possible |
| Target scaling | Same standardize-train-only-fit-then-invert approach as Stage 5 (`src/solaruq/utils/scaling.py`, reused, not reimplemented). Inversion is valid for quantiles because it's an increasing affine map (`y*std + mean`, std > 0), which commutes with quantiles | Keeps Stage 5 and Stage 6 directly comparable in physical kW without duplicating scaling logic |
| Hyperparameters (lr, batch size, epochs, patience) | Identical to Stage 5's: Adam 1e-3, batch 256, max 40 epochs, patience 5, seed 42 | Same reasoning as the architecture choice — isolate the effect of the loss/output structure, not retune two things at once (also satisfies "no extensive hyperparameter search") |
| P10-P90 as the reported interval | `coverage_interval: [0.10, 0.90]` added to `configs/config.yaml` | The brief's explicit ask; an 80% nominal interval is also the conventional default in the solar-forecasting UQ literature the proposal cites |
| Interval (Winkler) score, in addition to coverage and width | Added, for the P10-P90 interval only | Justified because coverage and width alone can hide a bad trade-off (e.g. a deceptively "sharp" but undercovering interval) — the Winkler score (Gneiting & Raftery, 2007) is a single proper scoring rule combining both, standard in the UQ literature, and trivial to compute once coverage/width exist |

**Real-data result and an honest calibration limitation (see `results/runs/stage6_quantile_lstm/`):** P50 test MAE/RMSE (8.85/16.28 kW) is close to but slightly worse than Stage 5's dedicated point LSTM (8.71/15.81 kW) — expected, since this model is optimizing 7 quantiles jointly rather than one point forecast. Overall P10-P90 coverage (77.2% test, nominal 80%) and the pooled (all-horizons) calibration plot both look reasonably close to nominal. **However, per-horizon coverage and interval width are visibly uneven, not smoothly degrading with horizon as expected** (`coverage_by_horizon.png`, `interval_width_by_horizon.png`): e.g. test coverage swings from 65% (horizon step 1) to 90% (step 2) to 67% (steps 3-4) back up to 90% (step 5); interval width spikes to ~52 kW at horizon step 5 against a ~20-30 kW range elsewhere. The training loss curve (`loss_curve.png`) shows this isn't a plotting artefact — both train and validation pinball loss bounce non-monotonically for several epochs before settling, consistent with a somewhat unstable joint optimization across 56 simultaneous outputs (8 horizons x 7 quantiles) from one small shared encoder and one linear head. This is reported as-is, not smoothed over or re-run with a cherry-picked seed, per the brief's explicit instruction not to overclaim calibration.

**What would change if revisited:** the most likely fix for the uneven per-horizon calibration is a slightly larger/more stable head (e.g. a small per-horizon linear layer instead of one shared 56-wide layer) or more training epochs with a lower learning rate — not a different model family. Out of scope for Stage 6 as briefed (no architecture changes without a specific technical reason, no extensive hyperparameter search); flagged here as a concrete, scoped follow-up rather than silently patched.

**Deferred, not a Stage 6 gap:** the README's `[q05, q10, ..., q95]` prediction-file schema (timestamp/horizon/y_true/y_hat/quantile columns written to a file) is not implemented by either Stage 5 or Stage 6 — both write `metrics.json` summaries only, matching the Stage 4 convention. Building the actual prediction-file export is a dashboard-integration concern, naturally belonging to Stage 10 (or whenever Stage 8/9 first needs it), not introduced here to avoid scope creep.

---

## D-2026-10-08 — Stage 5: LSTM point-forecast architecture and training choices

**Status:** implemented, approved by the user 2026-10-08.

The user's Stage 5 brief fixed the model family (direct multi-step LSTM, no recursion), the inputs (lookback target + weather + solar position; horizon-known solar position), the split usage (train/val/test), and explicitly forbade GRU/Transformer/attention/ensembles/extensive hyperparameter search. It did not fix the following, so they were decided here:

| Decision | Choice | Why |
|---|---|---|
| How `X_horizon_known` (future solar position) enters the model | Flattened (8×3=24 values) and concatenated with the LSTM's final hidden state, then passed through one `nn.Linear` head to all 8 outputs | Keeps the model a single small recurrent layer + linear head, not a second sequence model over the horizon — stays within "deliberately simple"; the horizon's solar position is deterministic (no data to flow through a second LSTM), so concatenation is sufficient |
| Target scaling | Standardize (z-score) `y`, fit on train split only, inverted before computing any metric | Stage 3 deliberately left the target in physical kW for persistence/ARIMA/evaluation; LSTMs train more stably on standardized targets. Done entirely inside Stage 5's own training script — does not touch Stage 3's saved `.npz` files or its feature scaler |
| `hidden_size` / `num_layers` / `dropout` | 64 / 1 / 0.1 | Smallest architecture likely to beat persistence given ~85.7k training windows; single layer keeps parameter count and CPU training time low (this container has no GPU — confirmed `torch.cuda.is_available() == False`, 4 CPU threads) |
| Optimizer / learning rate / batch size | Adam, 1e-3, 256 | Standard defaults for a small LSTM regression task; not tuned (explicit "no extensive hyperparameter search") |
| Early stopping | Monitor validation loss, patience 5 epochs, restore best-epoch weights, cap at 40 epochs | Directly satisfies "validation data for early stopping/model selection" |
| Reproducibility | `torch.manual_seed(project.seed)` (42, same seed as the rest of the project) before model init and training | Satisfies the explicit test requirement for deterministic/reproducible configuration |

**What would change if wrong:** if the LSTM underperforms persistence even after this, the first thing to revisit is `hidden_size`/`num_layers` (still within "simple"), not adding recursion, attention, or extra models — those remain out of scope per the user's brief.

---

## D-2026-10-08 — Stage 2 Task 2: file-level NIST audit complete (Canopy + WS_1, 2015–2018)

**Status:** audit complete; full detail in `docs/data_audit.md` §6–13. **All 4 decisions below confirmed and locked by the user, 2026-10-08.** Stage 3 may proceed on this basis.

**Resolved:**
- **Timestamp convention: fixed EST, no DST** — confirmed directly from raw file text (every July sample still shows `-05:00`, which rules out DST-observing local time). The data dictionary's "Local Solar Time" label was wrong, or described an earlier/different version of the dataset; the 2017 publication was right.
- **Coverage:** both tables have a file for all 1,461 expected days (2015–2018), zero duplicate timestamps, near-perfect continuity (WS_1: 0 gaps; Canopy: 6 gaps totalling 8h13m, two known incidents — a 4-night Oct 2018 outage and three single-minute July 2018 drops).
- **Cross-instrument consistency:** Canopy ↔ WS_1 alignment and pyranometer correlation (r=0.9845) both check out.
- **The apparent 2017–2018 Canopy power "drop"** (raw annual mean nearly halves) is a data-quality artifact, not real degradation: a `-999.0` error code in `InvPAC_kW_Avg`/`InvOpStatus_Avg` became ~10x more frequent from 2017 onward (clustered in specific months, always in Canopy, never in WS_1). Once excluded, Canopy's weather-normalised output is flat across all 4 years.

**Decisions — LOCKED 2026-10-08:**

| # | Decision | Locked value | Why it wasn't just assumed |
|---|---|---|---|
| 1 | Which array is the project's single PV site | **Canopy** | "One PV site/array" was decided at Stage 0, but *which* array was never written down as its own decision until now. |
| 2 | Canopy nameplate capacity for any capacity-normalised metrics | **243 kW** (2017 publication figure). The PDR record's vaguer "73–217 kW" is documented as a known source discrepancy — 217 kW is directly contradicted by the audit's own observed max of 260.2 kW. | Don't want to pick a capacity number without the user seeing the conflict — feeds into any normalised error metric reported in the dissertation. |
| 3 | Cleaning policy for Canopy's `-999.0` sentinel in `InvPAC_kW_Avg`/`InvOpStatus_Avg` | **Treated as missing/error (NaN)**, not a literal reading, before any statistics, scaling, or labels are computed. | Stage 3 implementation choice, not something to silently bake into the audit script. |
| 4 | Known Canopy gaps (4-night Oct 2018 outage, 3 single-minute July 2018 drops) | **Handled under the existing general gap policy** (excluded, like any other gap sized against lookback+horizon) — no special-casing. | Confirms this is a plain application of an existing rule, not a new one. |

**Still open from Stage 0, unaffected by this audit:**
- Submission deadline (you're still verifying the official Canvas date).
- Ethics screening status of the Project Registration Form.

---

## D-2026-09-27 — Working research design, locked in at Stage 0

**Status:** accepted working defaults. Dataset choice remains conditional on the Stage 2 NIST audit (see D-dataset below); everything downstream of dataset/resolution/horizon is provisional until that audit completes.

### Locked / working decisions

| Decision | Value | Why |
|---|---|---|
| Resolution | 15 minutes | Matches solar-forecasting literature conventions; keeps window/horizon step counts small enough to fully explain. |
| Forecast horizon | Up to 2 hours (8 × 15-min steps) | Long enough that persistence/ARIMA visibly degrade and uncertainty matters; short enough that no real future weather forecast input is needed — only knowable-in-advance features (solar position, clear-sky output). |
| Historical lookback | ~6 hours (24 × 15-min steps) | Captures the day's ramp shape and a preceding cloud episode without making the model large or hard to explain. Lowest-risk parameter to revisit later — it's a data-loader setting, not an architecture choice. |
| Core point-forecast models | Persistence, ARIMA, LSTM | Fixed project scope; supervisor-approved in the proposal. |
| Core uncertainty method | Quantile-regression LSTM, pinball loss | Produces the q05–q95 columns the prediction-file schema commits to; single well-defined loss, easier to fully understand and defend than sampling-based methods (MC Dropout, ensembles — explicitly out of scope). |
| Evaluation — point | MAE, RMSE | Standard, simple to explain and defend; may be normalised by capacity/clear-sky output once array metadata is confirmed (Stage 2). |
| Evaluation — probabilistic | Empirical coverage vs nominal, interval width/sharpness, calibration across quantiles | Directly operationalises "reliability" in the research question; standard pairing in the UQ literature the proposal already cites. |
| Data split | Chronological train / validation / test, with gaps sized to lookback + horizon | Mandatory under the leakage rules — solar power has strong daily/seasonal structure, so a shuffled split would leak near-identical days across the boundary. |
| Dashboard | Minimal Streamlit presentation layer, reads standardised prediction files only | Agreed across the proposal, the brainstorming notes, and the user's brief; not a place for model-specific logic; explicitly told not to spend significant time on UI. |

### Dataset (conditional, not fully locked)

- **Working default: NIST** Campus PV Arrays and Weather Station Data Sets, official NIST source only.
- **Fallback: NREL** (named in the original proposal), used only if the Stage 2 audit shows NIST is unsuitable (insufficient history, missing key variables, restrictive licence, etc.).
- **Rule:** if the audit finds a serious problem with NIST, work stops and the evidence is shown before switching — no silent fallback.
- **Stage 2 Task 1, documentation-level audit (2026-10-03): provisionally suitable, nothing disqualifying found.** Headline facts: DOI 10.18434/M3S67G, 3 arrays (Canopy 243kW, Ground 271kW, Roof 73kW per the 2017 paper), rich weather-station variable set including precomputed solar position, 1-second/1-minute native sampling (downsamples cleanly to our 15-min design), permissive licence with mandatory attribution. Date range resolved from NIST's own PDR record (`ark:/88434/mds2-2167`): 2015–2018, 4 years.
- **Stage 2 Task 2, file-level audit (2026-10-08): dataset confirmed suitable, no longer conditional.** Full detail: `docs/data_audit.md` §6–13, decisions needed listed in D-2026-10-08 above. Timestamp conflict resolved (fixed EST, no DST). Two real but well-isolated data-quality issues found (a `-999` sentinel code in Canopy's `InvPAC_kW_Avg`/`InvOpStatus_Avg`, worse from 2017 on; a faulty anemometer specific to Canopy, irrelevant since wind comes from WS_1). No evidence of real equipment degradation or clipping. **Dataset choice (NIST, Canopy array, WS_1 weather station) can now be treated as settled**, pending your sign-off on the 4 items in D-2026-10-08.

### Stretch goals (cut first under time pressure)

| Item | Condition to build it |
|---|---|
| Smart persistence baseline | Only if the Stage 2/3 audit shows it adds meaningful value (e.g. clear correlation with a `pvlib` clear-sky model) at low implementation cost. |
| Conformalised Quantile Regression (CQR) | Only after the core project (through Stage 8) is complete and time remains. Requires adding a calibration split, which is otherwise not created. |

### Administrative items still requiring confirmation (not assumed)

1. **Submission deadline** — not in any supplied document; user is verifying the official Canvas date. Work continues on everything that doesn't depend on it.
2. **Ethics screening status** — whether the ethics section of the Project Registration Form has already been reviewed by the School ethics committee, and the outcome. Not assumed; user is confirming.
3. Milestone progress against the handbook's M2–M5 (has the Literature Review section, M2, been submitted?) — not yet confirmed.

### Superseded / background

- Full reasoning, alternatives and consequences for each decision above: `stage0/stage0_orientation_v2.md` (Project files → stage0), §3.
- Conflicts between the proposal, supervisor feedback, ethics form and handbook: same document, §2.
- v1 orientation (research question, objectives, artefact, full risk table): `stage0/stage0_orientation.md`.
