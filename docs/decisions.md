# Decisions log

One entry per decision that shapes the project. Newest first. This is the record you (and any examiner) can point to for "why did you choose X" — every entry says what was decided, why, and what would change if it turned out wrong.

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
- **Stage 2 documentation-level audit (2026-10-03): provisionally suitable, nothing disqualifying found.** Full detail: `docs/data_audit.md`. Headline facts: DOI 10.18434/M3S67G, 3 arrays (Canopy 243kW, Ground 271kW, Roof 73kW), rich weather-station variable set including precomputed solar position, 1-second/1-minute native sampling (downsamples cleanly to our 15-min design), permissive licence with mandatory attribution (satisfies Category 0's "legal, publicly available data"). **Two unresolved conflicts, not yet verified against actual files:** (1) timestamp convention — data dictionary says Local Solar Time, the peer-reviewed publication says fixed Eastern Standard Time, no DST; (2) date range — the publication says 2015–2016 (2 years), the data.gov catalog metadata says 2015–2018 (4 years). **File-level verification (real date range, timestamp check, exact array location/tilt for pvlib, file format, actual missing/duplicate/clipping behaviour) is blocked** — this cloud container's network policy denies `pvdata.nist.gov`. See the Stage 2 report in-thread for the options to unblock this.

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
