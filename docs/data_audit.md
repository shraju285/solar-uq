# Stage 2 — Dataset audit: NIST Campus PV Arrays and Weather Station Data Sets

Status: **documentation-level audit complete; file-level verification blocked** (see §5 — this container's network policy denies the actual data portal). Nothing downloaded yet; no preprocessing has started.

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

## 5. What's blocking file-level verification

This cloud container's network policy denies direct access to `pvdata.nist.gov` *and* `api.box.com` (confirmed: the outbound proxy returns a policy-denial 403 on connection to both). The portal is also a JavaScript application, so even if permitted, a plain fetch wouldn't show real file listings — browsing it properly needs a normal browser. The project currently has no custom cloud environment configured that could allow these hosts.

The user found (2026-10-03) that the portal's own "Bulk Download" button is additionally broken by what looks like a genuine site bug: the page's Content Security Policy blocks its own JavaScript from calling `api.box.com` (the backend that actually serves files), so the browser console shows a CSP refusal + failed fetch for everyone, not just automated tools. NIST's official metadata record has no file-level listing to fall back on either — its only distribution is the DOI link back to the same portal, so there's no public mapping from a Box file ID to a specific variable/date selection to check it against. Next step is the user trying an alternate (non-bulk) export, or contacting the NIST data steward if it's a real site bug.

**Resolved from official metadata, no file access needed:**
- ✅ Date range: 2015–2018 (§4)

**Still not verified, and can't be from here:**
- The LST-vs-EST timestamp question, against an actual file
- Exact array GPS coordinates, tilt and azimuth (needed for `pvlib` clear-sky modelling in Stage 3 — not found in what I could fetch of the documentation)
- File structure/format (one file per array per year? per month? CSV? how large?) and the practical shape of a download
- Real missing-value/duplicate-timestamp/clipping/flat-line behaviour — only the publication's own summary (>99% availability for 2015-2016, two named incidents) is available so far, not a direct check
- The 73-217kW vs 271kW array-capacity conflict (§4)

## 6. Verdict so far

**Provisionally suitable, pending the file-level checks above.** Nothing found rules NIST out: licence is permissive, the variable set is rich and well-documented, sampling resolution supports the 15-min design, and the known outages are the kind of thing a cleaning policy is meant to handle, not a reason to abandon the dataset. The two conflicts (timestamp convention, date range) are exactly the sort of thing the audit is meant to catch before they become silent bugs — they don't look like red flags, just unresolved facts.

**This is not yet a pass/fail verdict** — see decisions.md for the three options to resolve §5.
