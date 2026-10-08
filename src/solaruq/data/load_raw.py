"""Load the Canopy target and WS_1 feature columns from data/raw/ into one
minute-resolution table.

This is read-only on data/raw/ (never writes there) and only touches the
columns Stage 3 actually needs — not every column in the raw files (see
docs/data_audit.md for the full schema).

Two things it's deliberately careful about, both from docs/decisions.md
(D-2026-10-08):
  - Canopy's `-999.0` sentinel/error code (in InvPAC_kW_Avg and
    InvOpStatus_Avg) is masked to NaN, not kept as a literal reading.
  - Every expected minute in [start_date, end_date] is present in the
    output, even where no raw file/row exists — missing minutes become NaN
    rows rather than being silently dropped. Gap detection downstream
    (resample.py) depends on this: a dropped row and a NaN row look
    different unless the index is complete.
"""

from pathlib import Path

import pandas as pd

# Full span of the audited dataset (docs/data_audit.md). Callers needing a
# narrower range (mainly tests) pass start_date/end_date explicitly instead
# of editing these.
DATASET_START = "2015-01-01"
DATASET_END = "2018-12-31"

# -999 is recorded as a literal float, but not always exactly -999.0 (the
# audit found values clustered tightly around it, e.g. -998.8 mean for
# flagged rows) — so this is a threshold, not an exact match. No real
# InvPAC/InvOpStatus reading comes close to -500 in either direction.
SENTINEL_THRESHOLD = -500.0

CANOPY_TARGET_COL = "InvPAC_kW_Avg"
CANOPY_STATUS_COL = "InvOpStatus_Avg"

WS1_FEATURE_COLS = [
    "Pyra1_Wm2_Avg",
    "RefCell1_Wm2_Avg",
    "AirTemp_C_Avg",
    "RelHumid_Avg",
    "AirPres_kPa_Avg",
    "Rain_mm_Tot",
    "SolarZenith_deg_Avg",
    "SolarAzFromSouth_deg_Avg",
    "SolarTime_hr",
]


def _expected_days(start_date: str, end_date: str) -> list[str]:
    return pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()


def _load_table(table_dir: Path, keep_cols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Walk every expected daily file under table_dir and concatenate the
    requested columns (plus TIMESTAMP). A missing file is simply skipped —
    the caller reindexes onto a complete minute grid afterwards, so a
    missing day shows up as NaN rows, not as an absence the rest of the
    pipeline has to special-case."""
    frames = []
    for date_str in _expected_days(start_date, end_date):
        year, month = date_str[:4], date_str[5:7]
        f = table_dir / year / month / f"onemin-{table_dir.name}-{date_str}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["TIMESTAMP", *keep_cols])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=keep_cols, index=pd.DatetimeIndex([], name="TIMESTAMP"))

    full = pd.concat(frames, ignore_index=True)
    full["TIMESTAMP"] = pd.to_datetime(full["TIMESTAMP"])
    full = full.drop_duplicates(subset="TIMESTAMP").set_index("TIMESTAMP").sort_index()
    return full


def _complete_minute_index(start_date: str, end_date: str, timezone_offset: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{start_date} 00:00:00{timezone_offset}")
    end = pd.Timestamp(f"{end_date} 23:59:00{timezone_offset}")
    return pd.date_range(start, end, freq="min")


def load_canopy_target(
    raw_dir: Path,
    timezone_offset: str = "-05:00",
    start_date: str = DATASET_START,
    end_date: str = DATASET_END,
) -> pd.DataFrame:
    """Canopy's target column, reindexed onto a complete minute grid, with
    the -999 sentinel masked to NaN. Returns a DataFrame with one column,
    `InvPAC_kW_Avg`, indexed by TIMESTAMP."""
    df = _load_table(raw_dir / "Canopy", [CANOPY_TARGET_COL, CANOPY_STATUS_COL], start_date, end_date)
    df = df.reindex(_complete_minute_index(start_date, end_date, timezone_offset))
    sentinel = (df[CANOPY_TARGET_COL] <= SENTINEL_THRESHOLD) | (
        df[CANOPY_STATUS_COL] <= SENTINEL_THRESHOLD
    )
    df.loc[sentinel, CANOPY_TARGET_COL] = pd.NA
    return df[[CANOPY_TARGET_COL]].astype(float)


def load_ws1_features(
    raw_dir: Path,
    timezone_offset: str = "-05:00",
    start_date: str = DATASET_START,
    end_date: str = DATASET_END,
) -> pd.DataFrame:
    """WS_1's weather + solar-position feature columns, reindexed onto a
    complete minute grid. WS_1 had zero -999-style sentinels in the Stage 2
    audit, so no masking is applied here."""
    df = _load_table(raw_dir / "WS_1", WS1_FEATURE_COLS, start_date, end_date)
    df = df.reindex(_complete_minute_index(start_date, end_date, timezone_offset))
    return df.astype(float)


def load_minute_table(
    raw_dir: Path,
    timezone_offset: str = "-05:00",
    start_date: str = DATASET_START,
    end_date: str = DATASET_END,
) -> pd.DataFrame:
    """Canopy's target joined with WS_1's features on a single complete
    minute index. This is the one function the rest of Stage 3 calls."""
    target = load_canopy_target(raw_dir, timezone_offset, start_date, end_date)
    features = load_ws1_features(raw_dir, timezone_offset, start_date, end_date)
    return target.join(features, how="left")
