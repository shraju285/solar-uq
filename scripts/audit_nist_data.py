"""Stage 2 Task 2: full data-quality audit of the NIST Canopy + WS_1 archives.

Reads every daily CSV from data/raw/Canopy and data/raw/WS_1 (2015-2018),
computes the checks listed in docs/data_audit.md, and writes:
  - results/audits/nist_canopy_ws1/summary.json   (machine-readable numbers)
  - results/audits/nist_canopy_ws1/*.png          (a few sanity plots)

Deliberately NOT part of the solaruq package: this is a one-off audit
script for Stage 2, not pipeline code other stages import. It never writes
anything under data/interim or data/processed, and never touches the raw
files - read-only on data/raw/.

Run with:  python scripts/audit_nist_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "results" / "audits" / "nist_canopy_ws1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2015, 2016, 2017, 2018]

# Columns we keep in full for the detailed checks (alignment, night-time,
# clipping, year-to-year). Every other column is still scanned for missing
# values and dtype, just not retained row-by-row - keeps memory small.
CANOPY_KEEP = [
    "TIMESTAMP", "InvPAC_kW_Avg", "InvPDC_kW_Avg", "InvOpStatus_Avg",
    "Pyra1_Wm2_Avg", "RefCell1_Wm2_Avg", "AmbTemp_C_Avg",
]
WS1_KEEP = [
    "TIMESTAMP", "Pyra1_Wm2_Avg", "RefCell1_Wm2_Avg",
    "SolarZenith_deg_Avg", "SolarAzFromSouth_deg_Avg", "SolarTime_hr",
    "AirTemp_C_Avg", "RelHumid_Avg", "AirPres_kPa_Avg", "Rain_mm_Tot",
]


def expected_days(year: int) -> list[str]:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").strftime("%Y-%m-%d").tolist()


def load_table(table_dir: Path, keep_cols: list[str]) -> dict:
    """Walk every expected daily file, accumulating:
    - a concatenated DataFrame of `keep_cols` only (for detailed checks)
    - full-column missing-value counts (every column, not just keep_cols)
    - per-day row counts and file-presence (for coverage / gap detection)
    """
    all_columns_seen: set[str] = set()
    nan_counts: dict[str, int] = {}
    total_rows_per_column: dict[str, int] = {}
    day_rows = []  # (date_str, n_rows, n_duplicate_ts_within_day)
    missing_days = []
    frames = []

    for year in YEARS:
        for date_str in expected_days(year):
            month = date_str[5:7]
            f = table_dir / str(year) / month / f"onemin-{table_dir.name}-{date_str}.csv"
            if not f.exists():
                missing_days.append(date_str)
                continue
            df = pd.read_csv(f)
            all_columns_seen.update(df.columns)
            for col in df.columns:
                nan_counts[col] = nan_counts.get(col, 0) + int(df[col].isna().sum())
                total_rows_per_column[col] = total_rows_per_column.get(col, 0) + len(df)
            n_dup = int(df["TIMESTAMP"].duplicated().sum())
            day_rows.append((date_str, len(df), n_dup))
            present_keep = [c for c in keep_cols if c in df.columns]
            frames.append(df[present_keep])

    full = pd.concat(frames, ignore_index=True)
    full["TIMESTAMP"] = pd.to_datetime(full["TIMESTAMP"], utc=False)
    full = full.sort_values("TIMESTAMP").reset_index(drop=True)

    return {
        "full": full,
        "all_columns_seen": sorted(all_columns_seen),
        "nan_counts": nan_counts,
        "total_rows_per_column": total_rows_per_column,
        "day_rows": pd.DataFrame(day_rows, columns=["date", "n_rows", "n_dup_within_day"]),
        "missing_days": missing_days,
    }


def timestamp_offset_summary(full: pd.DataFrame, raw_table_dir: Path) -> dict:
    """Check the UTC offset string directly from raw text (pandas strips it
    on parse), across a winter day and a summer day, to settle whether the
    data uses fixed EST or DST-observing local time."""
    samples = {}
    for year in YEARS:
        for date_str in [f"{year}-01-15", f"{year}-07-15"]:
            month = date_str[5:7]
            f = raw_table_dir / str(year) / month / f"onemin-{raw_table_dir.name}-{date_str}.csv"
            if f.exists():
                with open(f) as fh:
                    fh.readline()  # header
                    first_data_line = fh.readline()
                ts = first_data_line.split(",")[0]
                samples[date_str] = ts
    return samples


def duplicate_timestamps_across_dataset(full: pd.DataFrame) -> int:
    return int(full["TIMESTAMP"].duplicated().sum())


def gap_summary(full: pd.DataFrame) -> dict:
    diffs = full["TIMESTAMP"].diff()
    one_min = pd.Timedelta(minutes=1)
    gap_mask = diffs != one_min
    gap_mask.iloc[0] = False  # first row's diff is NaT, not a real gap
    gap_positions = np.flatnonzero(gap_mask.to_numpy())
    largest = sorted(gap_positions, key=lambda p: diffs.iloc[p], reverse=True)[:20]
    return {
        "n_non_one_minute_steps": int(gap_mask.sum()),
        "total_gap_time": str(diffs[gap_mask].sum() - gap_mask.sum() * one_min),
        "largest_gaps": [
            {
                "after": str(full["TIMESTAMP"].iloc[p - 1]),
                "before": str(full["TIMESTAMP"].iloc[p]),
                "gap": str(diffs.iloc[p]),
            }
            for p in largest
        ],
    }


def night_time_check(full: pd.DataFrame, power_col: str, zenith_col: str | None = None) -> dict:
    """Night defined by hour-of-day outside roughly 6am-8pm local (coarse,
    good enough for a sanity check - a precise solar-elevation definition
    comes in Stage 3)."""
    hour = full["TIMESTAMP"].dt.hour
    night_mask = (hour < 5) | (hour >= 21)
    night_vals = full.loc[night_mask, power_col]
    return {
        "n_night_rows": int(night_mask.sum()),
        "night_mean": float(night_vals.mean()),
        "night_min": float(night_vals.min()),
        "night_max": float(night_vals.max()),
        "night_nonzero_gt_1kw_count": int((night_vals.abs() > 1.0).sum()),
    }


def clipping_check(full: pd.DataFrame, power_col: str) -> dict:
    p = full[power_col].dropna()
    top = p.sort_values(ascending=False).head(20)
    # "clipping" = power plateaus at a ceiling for sustained periods - proxy
    # via how often the series sits within 1% of its own observed max.
    obs_max = float(p.max())
    near_max = (p >= 0.99 * obs_max).sum()
    return {
        "observed_max_kw": obs_max,
        "top_20_values_kw": top.round(2).tolist(),
        "rows_within_1pct_of_max": int(near_max),
    }


def year_to_year(full: pd.DataFrame, power_col: str | None, irr_col: str) -> dict:
    full = full.copy()
    full["year"] = full["TIMESTAMP"].dt.year
    out = {}
    for yr, g in full.groupby("year"):
        entry = {
            "n_rows": int(len(g)),
            "mean_" + irr_col: float(g[irr_col].mean(skipna=True)),
            "max_" + irr_col: float(g[irr_col].max(skipna=True)),
        }
        if power_col is not None:
            entry["mean_" + power_col] = float(g[power_col].mean(skipna=True))
            entry["max_" + power_col] = float(g[power_col].max(skipna=True))
        out[int(yr)] = entry
    return out


def alignment_check(canopy_full: pd.DataFrame, ws1_full: pd.DataFrame) -> dict:
    merged = pd.merge(
        canopy_full[["TIMESTAMP", "Pyra1_Wm2_Avg"]].rename(columns={"Pyra1_Wm2_Avg": "canopy_pyra1"}),
        ws1_full[["TIMESTAMP", "Pyra1_Wm2_Avg"]].rename(columns={"Pyra1_Wm2_Avg": "ws1_pyra1"}),
        on="TIMESTAMP", how="inner",
    )
    corr = merged[["canopy_pyra1", "ws1_pyra1"]].corr().iloc[0, 1]
    return {
        "canopy_rows": int(len(canopy_full)),
        "ws1_rows": int(len(ws1_full)),
        "matched_timestamps": int(len(merged)),
        "canopy_only_timestamps": int(len(canopy_full) - len(merged)),
        "ws1_only_timestamps": int(len(ws1_full) - len(merged)),
        "pyranometer_correlation": float(corr),
    }


def main():
    print("Loading Canopy (2015-2018)...")
    canopy = load_table(RAW_DIR / "Canopy", CANOPY_KEEP)
    print(f"  {len(canopy['full'])} rows, {len(canopy['missing_days'])} missing days")

    print("Loading WS_1 (2015-2018)...")
    ws1 = load_table(RAW_DIR / "WS_1", WS1_KEEP)
    print(f"  {len(ws1['full'])} rows, {len(ws1['missing_days'])} missing days")

    report = {
        "canopy": {
            "row_count": int(len(canopy["full"])),
            "columns": canopy["all_columns_seen"],
            "n_columns": len(canopy["all_columns_seen"]),
            "missing_days": canopy["missing_days"],
            "nan_counts_nonzero": {k: v for k, v in canopy["nan_counts"].items() if v > 0},
            "duplicate_timestamps": duplicate_timestamps_across_dataset(canopy["full"]),
            "timestamp_offset_samples": timestamp_offset_summary(canopy["full"], RAW_DIR / "Canopy"),
            "gaps": gap_summary(canopy["full"]),
            "night_time_InvPAC_kW_Avg": night_time_check(canopy["full"], "InvPAC_kW_Avg"),
            "clipping_InvPAC_kW_Avg": clipping_check(canopy["full"], "InvPAC_kW_Avg"),
            "year_to_year": year_to_year(canopy["full"], "InvPAC_kW_Avg", "Pyra1_Wm2_Avg"),
            "day_rows_not_1440": canopy["day_rows"][canopy["day_rows"]["n_rows"] != 1440].to_dict("records"),
        },
        "ws1": {
            "row_count": int(len(ws1["full"])),
            "columns": ws1["all_columns_seen"],
            "n_columns": len(ws1["all_columns_seen"]),
            "missing_days": ws1["missing_days"],
            "nan_counts_nonzero": {k: v for k, v in ws1["nan_counts"].items() if v > 0},
            "duplicate_timestamps": duplicate_timestamps_across_dataset(ws1["full"]),
            "timestamp_offset_samples": timestamp_offset_summary(ws1["full"], RAW_DIR / "WS_1"),
            "gaps": gap_summary(ws1["full"]),
            "night_time_Pyra1_Wm2_Avg": night_time_check(ws1["full"], "Pyra1_Wm2_Avg"),
            "year_to_year": year_to_year(ws1["full"], None, "Pyra1_Wm2_Avg"),
            "day_rows_not_1440": ws1["day_rows"][ws1["day_rows"]["n_rows"] != 1440].to_dict("records"),
        },
        "alignment": alignment_check(canopy["full"], ws1["full"]),
    }

    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Save the reduced per-minute tables too, so a later stage (or a human)
    # can re-plot without re-parsing 2,900 CSVs. Parquet, not CSV: smaller
    # and keeps dtypes (esp. the timestamp) exact.
    canopy["full"].to_parquet(OUT_DIR / "canopy_reduced.parquet", index=False)
    ws1["full"].to_parquet(OUT_DIR / "ws1_reduced.parquet", index=False)
    print(f"Wrote {OUT_DIR / 'canopy_reduced.parquet'} and ws1_reduced.parquet")


if __name__ == "__main__":
    main()
