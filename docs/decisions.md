# Decisions log

One entry per decision that shapes the project. Newest first. This is the record you (and any examiner) can point to for "why did you choose X" — every entry says what was decided, why, and what would change if it turned out wrong.

---

## D-2026-10-08 — Stage 2 Task 2: file-level NIST audit complete (Canopy + WS_1, 2015–2018)

**Status:** audit complete; full detail in `docs/data_audit.md` §6–13. This resolves the dataset's remaining open conflict and finds two real (but manageable) data-quality issues. **No decision here is final until you confirm the items in "Needs your decision" below** — none of them have been assumed.

**Resolved:**
- **Timestamp convention: fixed EST, no DST** — confirmed directly from raw file text (every July sample still shows `-05:00`, which rules out DST-observing local time). The data dictionary's "Local Solar Time" label was wrong, or described an earlier/different version of the dataset; the 2017 publication was right.
- **Coverage:** both tables have a file for all 1,461 expected days (2015–2018), zero duplicate timestamps, near-perfect continuity (WS_1: 0 gaps; Canopy: 6 gaps totalling 8h13m, two known incidents — a 4-night Oct 2018 outage and three single-minute July 2018 drops).
- **Cross-instrument consistency:** Canopy ↔ WS_1 alignment and pyranometer correlation (r=0.9845) both check out.
- **The apparent 2017–2018 Canopy power "drop"** (raw annual mean nearly halves) is a data-quality artifact, not real degradation: a `-999.0` error code in `InvPAC_kW_Avg`/`InvOpStatus_Avg` became ~10x more frequent from 2017 onward (clustered in specific months, always in Canopy, never in WS_1). Once excluded, Canopy's weather-normalised output is flat across all 4 years.

**Needs your decision before Stage 3:**

| # | Decision | Recommendation | Why it's not just assumed |
|---|---|---|---|
| 1 | **Which array is the project's single PV site** — Task 2 only audited Canopy (not Ground or Roof), which I've taken as your working choice since you specifically scoped the audit that way. | Lock in **Canopy**. | "One PV site/array" was decided at Stage 0, but *which* array was never written down as its own decision until now. |
| 2 | **Canopy nameplate capacity to use for any capacity-normalised metrics in Stage 3** — PDR record says "73–217 kW" for the three arrays combined/vaguely; the 2017 publication says 243 kW for Canopy specifically; the audit's own observed max is 260.2 kW (already above 217 kW, consistent with ~243 kW + normal AC headroom). | Use **243 kW** (2017 publication), noting 217 kW is contradicted by the data itself. | Don't want to pick a capacity number for you without you seeing the conflict — this feeds into any normalised error metric you report in the dissertation. |
| 3 | **Cleaning policy for Canopy's `-999.0` sentinel** in `InvPAC_kW_Avg`/`InvOpStatus_Avg`. | Treat as missing (NaN) at the start of Stage 3 cleaning, before any scaling/statistics — then decide imputation vs. exclusion per-window like any other gap. | This is a Stage 3 implementation choice, not something to silently bake into the audit script. |
| 4 | **Known Canopy gaps** (4-night Oct 2018 outage, 3 single-minute July 2018 drops) — whether to simply exclude them as gaps when building lookback/horizon windows. | Yes — treat exactly like any other gap under the existing leakage rule ("gaps between blocks sized to lookback+horizon"); no special-casing needed given how small/rare they are. | Confirming this is a trivial application of an existing rule, not a new one, before Stage 3 starts. |

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
