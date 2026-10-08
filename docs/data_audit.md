# Stage 2 — Dataset audit: NIST Campus PV Arrays and Weather Station Data Sets

Status: **Task 1 (documentation-level audit) and Task 2 (file-level audit of Canopy + WS_1, 2015–2018) both complete.** Dataset is judged suitable — see §9 for the verdict and §10 for decisions still needed before Stage 3. No preprocessing has started; raw files are untouched and not committed to git.

## 1. Identity, source, licence

- **Dataset:** "NIST Campus Photovoltaic (PV) Arrays and Weather Station Data Sets," Boyd M, Chen T, Dougherty B (2017).
- **DOI (official, citable):** https://doi.org/10.18434/M3S67G
- **Portal (official download):** https://pvdata.nist.gov/
- **Supporting publication (peer-reviewed, most authoritative source on the data itself):** Boyd, M. "Performance Data from the National Institute of Standards and Technology Photovoltaic Arrays and Weather Station," *J. Res. NIST* 122:40 (2017). http://nvlpubs.nist.gov/nistpubs/jres/122/jres.122.040.pdf
- **Licence:** NIST's standard open data/software licence (https://www.nist.gov/open/license). Non-SRD NIST data is not subject to U.S. copyright (17 U.S.C. §105); free to use, redistribute and create derivative works. **Attribution is mandatory** — cite the publication above, and note any modifications made.
- **Ethics relevance:** this is institutional sensor data (PV arrays, weather instruments) — no human participants, questionnaires, tissue or animals, and the data is legally public. This satisfies the Ethics form's Category 0 wording ("no human participants ... uses only legal, publicly available data") **on the data side**. It doesn't replace your confirmation of what the School ethics committee already decided about your Project Registration Form (still an open item from Stage 0).

## 2. What's actually in it

**Three PV arrays** on the NIST campus, ~587 kW combined (~1,600 modules):

| Array | Rated capacity | Mounting |
|---|---|---|
| Canopy | 243 kW | East/west-facing canopies over a parking lot |
| Ground | 271 kW | Tilted ground-mounted racks, open field |
| Roof | 73 kW | Tilted, weighted racks on a building roof |

Each array has its own inverter, power-meter and combiner-box instrumentation (fault codes, per-string DC current/power, AC voltage/current/power/frequency per phase). A separate **weather station** measures a large set of radiometric and meteorological variables, including several pyranometers (GHI/POA at different tilts), a pyrheliometer, pyrgeometer, UV sensors, wind (at two heights), temperature, humidity, rain, snow depth — and, usefully, **precomputed solar position**: `SolarZenith_deg`, `SolarAzFromSouth_deg`, `SolarTime_hr`, `Declination_deg`, `AirMass`. These are exactly the "knowable in advance" deterministic features the leakage rules call for (Stage 0, rule 8) — they depend only on time and location, not on future weather.

**Likely target variable:** `InvPAC_kW` (AC real power at the inverter) per array — directly comparable across arrays and the natural "y" for a power forecast. `ShuntPDC_kW_Avg(1–7)` gives per-string DC detail if ever needed.

## 3. Sampling, timestamps — one conflict to resolve

- **Sampling:** most of the 360+ measurements are recorded every **1 second** (RTDs/thermocouples every 10 s), then saved as **1-minute averages** (some as 1-min min/max). Our 15-minute design is a **downsample** from this — straightforward, no upsampling or interpolation needed for the core signal.
- **Timestamp — conflict, not yet resolved:**
  - The data dictionary (NIST documentation page) labels the `TIMESTAMP` column **"LST"**, which it expands as *Local Solar Time*.
  - The peer-reviewed publication states timestamps are in **"Eastern Standard Time (no daylight saving time)"**.
  - These are not the same thing — Local Solar Time drifts continuously with the sun's position and the site's longitude; EST is a fixed civil-clock offset (UTC−5) that doesn't observe DST. I can't resolve this from documentation alone; it needs checking against an actual file's timestamp column (e.g. does solar noon fall near 12:00, which would suggest solar time, or does it drift by the ~15–20 minutes you'd expect from the equation of time if it's really EST?). **If it genuinely is fixed EST with no DST, that's good news for Stage 3** — no duplicate/missing-hour DST transitions to handle, one of the more common timestamp bugs in this kind of pipeline.

## 4. Date range — RESOLVED (2026-10-03)

- **Resolved using NIST's own authoritative PDR record** (`ark:/88434/mds2-2167` at data.nist.gov — the dataset's official metadata record, not a third-party harvest): *"one-minute averaged values and one-second instantaneous values for 2015 through 2018."* **The real range is 4 years (2015–2018)**, not the 2-year figure.
- The 2017 publication's "2015–2016" simply predates this later extension — the PDR record's own modification date (2019-12-31) is consistent with the dataset having grown after that paper was written. Not a real conflict once the two documents' dates are accounted for.
- **Known quality incidents from the 2017 publication** (useful for the cleaning policy either way, though only covering 2015-2016): a 33-day Ground Array module removal/reinstallation, and an inverter arcing event with repairs spanning **25 Aug – 9 Sep 2015**. Any incidents in 2017-2018 are not yet documented anywhere I've found — worth checking for once real files are available.
- **New minor conflict spotted while resolving this:** the PDR record describes the three arrays as spanning **"73 kW to 217 kW"**; the 2017 publication gives Ground as **271 kW**. Not resolved — flagged rather than guessed. Doesn't block anything yet (array capacity matters for Stage 3 normalisation, not for the suitability verdict).

4 years of history is comfortable for a chronological train/validation/test split across multiple seasons — this removes what was the main data-adequacy risk.

## 5. How file-level verification was actually done

The portal/Box download path described below turned out not to be needed. **This blocker is now superseded**, not resolved through it: on 2026-10-08 the user supplied the actual data directly — 8 zip archives (one per array/table per... actually one per table per multi-year span) covering Canopy and WS_1, 2015–2018, as 1-minute-averaged daily CSV files. Those are the files §6 onward audits. The portal investigation below is kept for the record in case Ground or Roof data is ever needed later and has to go through the same path.

This cloud container's network policy denies direct access to `pvdata.nist.gov` *and* `api.box.com` (confirmed: the outbound proxy returns a policy-denial 403 on connection to both). The portal is also a JavaScript application, so even if permitted, a plain fetch wouldn't show real file listings — browsing it properly needs a normal browser. The project currently has no custom cloud environment configured that could allow these hosts.

The user found (2026-10-03) that the portal's own "Bulk Download" button is additionally broken by what looks like a genuine site bug: the page's Content Security Policy blocks its own JavaScript from calling `api.box.com` (the backend that actually serves files), so the browser console shows a CSP refusal + failed fetch for everyone, not just automated tools. NIST's official metadata record has no file-level listing to fall back on either.

## 6. Task 2 — file-level audit scope and coverage

**Inputs:** the user's 8 uploaded zip archives, extracted read-only into `data/raw/Canopy/` and `data/raw/WS_1/` (git-ignored, never committed). Each extracts to one CSV per calendar day, path pattern `<table>/<year>/<month>/onemin-<table>-<date>.csv`. Duplicate-named uploads were checked byte-identical via `sha256sum` before being treated as redundant.

| Table | Days expected (2015–2018) | Days found | Rows | Columns |
|---|---|---|---|---|
| Canopy | 1,461 | 1,461 (0 missing) | 2,103,347 | 102 |
| WS_1 | 1,461 | 1,461 (0 missing) | 2,103,840 | 49 |

Both tables have a file for every single calendar day across all 4 years — no missing days at the file level. Within days, Canopy has 7 days with fewer than 1,440 rows (minute-level gaps, not missing files — see §7); WS_1 has none.

Audit script: `scripts/audit_nist_data.py` (read-only on `data/raw/`; writes only to `results/audits/nist_canopy_ws1/`). Outputs: `summary.json` (every number below), plus `canopy_reduced.parquet` / `ws1_reduced.parquet` (the per-minute values for the columns actually checked in detail, for anyone re-plotting without re-parsing 2,922 CSVs).

## 7. Timestamps, duplicates, continuity — the LST-vs-EST conflict is RESOLVED

**Timestamp convention: fixed EST, no DST — confirmed directly from the raw files.** Reading the literal first-data-line text (before pandas parses it) for 15 January and 15 July of every year, both tables, the UTC offset is **`-05:00` in every single sample, including every July sample**:

```
2015-01-15 00:00:00-05:00   2015-07-15 00:00:00-05:00
2016-01-15 00:00:00-05:00   2016-07-15 00:00:00-05:00
2017-01-15 00:00:00-05:00   2017-07-15 00:00:00-05:00
2018-01-15 00:00:00-05:00   2018-07-15 00:00:00-05:00
```

If this were DST-observing local time, every July offset would read `-04:00`. It doesn't, in any of the 4 years, for either table. This directly contradicts the data dictionary's "Local Solar Time" label and confirms the 2017 publication's claim of fixed Eastern Standard Time. **This conflict is closed**: use fixed UTC−5 for every timestamp in the dataset, with no DST handling needed in Stage 3 — one less class of bug (no duplicate/missing hour at DST transitions) to worry about.

**Duplicate timestamps: zero**, in both tables, across all 4.2M combined rows.

**Continuity (gaps away from exactly 1 minute):**

| Table | Non-1-minute steps | Total missing time |
|---|---|---|
| Canopy | 6 | 8h 13m |
| WS_1 | 0 | 0 |

Canopy's 6 gaps are two distinct, genuine incidents, not random noise:
- **Three isolated 1-minute drops**, each losing exactly one minute: midnight boundaries of 2018-07-08, 07-10, 07-11.
- **A 4-night partial outage, 26–29 October 2018**, each night losing roughly 1.5–3 hours overnight (e.g. 2018-10-28 23:30 → 2018-10-29 02:31, a 3h01m gap). Four consecutive days with reduced row counts (1,322 / 1,294 / 1,365 / 1,289 rows instead of 1,440).

WS_1 recorded straight through both incidents with no gap at all — it's a separate logger, so this is a Canopy-logger-specific fault, not a site-wide outage. This also explains the one WS_1-vs-Canopy asymmetry in §8: WS_1 has 493 timestamps Canopy doesn't (493 minutes ≈ the 8h13m above), i.e. WS_1 simply kept running while Canopy's logger was down — a consistency check that *passes*, not a WS_1 problem.

## 8. Canopy ↔ WS_1 alignment

| | |
|---|---|
| Canopy rows | 2,103,347 |
| WS_1 rows | 2,103,840 |
| Matched timestamps | 2,103,347 (100% of Canopy) |
| Canopy-only timestamps | 0 |
| WS_1-only timestamps | 493 (= Canopy's own gap minutes, §7) |
| Correlation, Canopy's `Pyra1_Wm2_Avg` vs WS_1's `Pyra1_Wm2_Avg` (two independent, co-located pyranometers) | **0.9845** |

Every Canopy timestamp has a matching WS_1 row, and the two independent irradiance sensors agree to r=0.9845. This is strong evidence both loggers' clocks and timestamp handling are sound and mutually consistent — a real risk for a two-source dataset like this, and it checks out.

## 9. Missing values

Across almost every column in both tables there's a small, uniform background rate — **0.251% of rows in Canopy (5,271 rows), 0.007% in WS_1 (156 rows)** — affecting dozens of unrelated columns identically. That pattern (same row count, many columns at once) points to whole-scan datalogger dropouts, not individual sensor faults, and is small enough not to be a concern.

Two columns are **genuinely** elevated, for different and explainable reasons:

- **Canopy's own `WindSpeedAve_ms` / `WindSpeed_ms_Max` / `WindRef_V_Min`: ~19.1–19.5% NaN.** This looks like a real, long-running fault in Canopy's own anemometer, distinct from the uniform background rate above. It doesn't affect our modelling plan: wind comes from **WS_1**, where the equivalent column (`WindSpeedAve_ms`) sits at the uniform 0.007% baseline, i.e. clean.
- **WS_1's `AirMass_Avg`: ~45.1% NaN.** This is physically expected, not a defect — air mass is undefined once the sun is at or below the horizon, and roughly half the day is night/twilight. Not a column our design needs anyway (solar position comes from the already-clean `SolarZenith_deg_Avg` / `SolarAzFromSouth_deg_Avg` / `SolarTime_hr`, which sit at the uniform baseline).

Also present in Canopy, but handled separately because it isn't a NaN — see §10: a literal **`-999.0`** sentinel/error code recorded in `InvPAC_kW_Avg` and `InvOpStatus_Avg`.

## 10. The apparent 2017–2018 power drop — investigated, and it's a data-quality artifact, not real degradation

The single biggest thing this audit turned up. The raw annual mean of Canopy's `InvPAC_kW_Avg` (power column, including night) looks like a serious decline:

| Year | Raw annual mean InvPAC (kW) | Annual mean irradiance, Pyra1 (W/m²) |
|---|---|---|
| 2015 | 32.80 | 167.4 |
| 2016 | 34.86 | 170.7 |
| 2017 | 18.78 | 166.3 |
| 2018 | 16.75 | 157.0 |

Power roughly halves from 2016 to 2017 while irradiance barely moves — on its face this looks like the array lost half its output. **It didn't.** The cause is a `-999.0` sentinel/error code that `InvPAC_kW_Avg` (and the correlated `InvOpStatus_Avg`) record instead of a real reading, and that code's frequency rose sharply:

| Year | Minutes recorded as `-999` | % of year |
|---|---|---|
| 2015 | 760 | 0.14% |
| 2016 | 141 | 0.03% |
| 2017 | 7,408 | 1.41% |
| 2018 | 7,984 | 1.52% |

Each `-999` minute subtracts roughly 999 kW·min from that year's sum, so ~7,500 extra sentinel minutes is enough on its own to explain the entire "drop" (7,408 × 999 ÷ 525,600 ≈ 14.1 kW, almost exactly the 2016→2017 gap). Excluding these rows confirms it directly:

| Year | Mean InvPAC, sentinel rows excluded (kW) | Daytime-only InvPAC ÷ Pyra1 ratio, sentinel rows excluded |
|---|---|---|
| 2015 | 34.29 | 0.2028 |
| 2016 | 35.13 | 0.2040 |
| 2017 | 33.36 | 0.2004 |
| 2018 | 32.53 | 0.2041 |

Once the sentinel rows are removed, both the annual mean and the weather-normalised output ratio are flat across all 4 years — and that ratio stays flat (0.19–0.21) within every irradiance band I checked (100–300, 300–600, 600–900, 900+ W/m²), in every year. **There is no evidence of real panel/inverter degradation.**

The sentinel occurrences are also not random noise — they cluster into specific months (707 in June 2015; 3,358 in June 2017; 4,001 in October 2017; 1,997 in Feb 2018, 2,217 in June 2018, 3,640 in October 2018), and they occur **only** in Canopy, never in any WS_1 column. This points to a recurring logger/inverter-communication fault specific to the Canopy array, worse and more frequent from 2017 onward, rather than anything affecting the PV output itself.

**Implication for Stage 3 (flagged as a decision below):** `-999.0` in Canopy's `InvPAC_kW_Avg`/`InvOpStatus_Avg` must be treated as a missing-value sentinel during cleaning, before any statistics, scaling, or labels are computed from it — left as-is, it's a massive spurious outlier (and would wreck any mean/scaler fit on the raw column).

## 11. Night-time behaviour

- **Canopy `InvPAC_kW_Avg`** (night = hour<5 or hour≥21, coarse): once the `-999` sentinel is accounted for (§10), the real night-time range is small negative (a touch of self-consumption/standby draw) to ~2.1 kW — ordinary. The raw `night_min` of −999 is the sentinel, not a real reading.
- **WS_1 `Pyra1_Wm2_Avg`** at night: mean −5.3, range −9.9 to +20.3 W/m² — a small sensor offset/noise band around zero, completely normal for a pyranometer in the dark. (One caveat on the script itself: its "nonzero > 1kW" check was written with power columns in mind and isn't a meaningful threshold for a W/m² irradiance column — on this column it trivially flags nearly every night-time row, which is not itself informative. The mean/min/max figures above are the useful numbers here.)

## 12. Clipping / flat-lining, and the array-capacity conflict

Across all 2.1M Canopy rows, the observed maximum of `InvPAC_kW_Avg` is **260.2 kW**. The top 20 highest readings span 256.3–260.2 kW with no repeated/flat value, and only **14 rows total** (out of 2.1M) sit within 1% of that maximum. That's not what a hard inverter ceiling looks like — a clipped array would show many readings pinned at the same ceiling value for sustained stretches. **No evidence of clipping.**

This also bears on the array-capacity conflict noted in §4 (PDR record's vague "73–217 kW" range vs the 2017 publication's 243 kW for Canopy): an AC output of 260.2 kW observed directly in the data is **already above** the PDR's 217 kW upper bound, which makes that figure implausible for Canopy specifically. It's consistent with — slightly above, which is normal for AC output vs DC-STC nameplate — the publication's 243 kW figure. Weak but real evidence favouring 243 kW over 217 kW; not something to treat as fully settled without a response from the NIST data steward, but enough to proceed on.

## 13. Suitability verdict

**NIST (Canopy + WS_1, 2015–2018) is suitable for the 15-minute / 2-hour-horizon / 6-hour-lookback design**, and this is no longer provisional:

- 4 full years, zero missing days, chronological split across multiple seasons is comfortable.
- Timestamp convention resolved (fixed EST, no DST) — removes a real source of silent bugs.
- The two real quality issues found (Canopy's `-999` sentinel codes, Canopy's own faulty anemometer) are both well-understood, well-isolated to specific columns, and straightforward to handle in a cleaning policy — neither touches WS_1, which is the source for all the weather/solar-position features the design needs.
- Cross-instrument consistency (Canopy ↔ WS_1 alignment and pyranometer correlation) passes convincingly.
- No evidence of real equipment degradation or clipping that would compromise power statistics.

See `docs/decisions.md` (entry D-2026-10-08) for the decisions this still leaves for the user before Stage 3 starts.
