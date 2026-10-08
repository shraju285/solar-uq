"""Tests for load_raw.py, using small synthetic CSV fixtures rather than the
real data/raw/ files.

This is deliberate, not a shortcut: data/raw/ is git-ignored (see
.gitignore and docs/decisions.md — raw data is never committed), so a test
suite that depended on the real files would fail on a fresh checkout or in
CI. These fixtures reproduce the real schema (docs/data_audit.md §2) on a
handful of rows.
"""

import pandas as pd
import pytest

from solaruq.data.load_raw import (
    CANOPY_STATUS_COL,
    CANOPY_TARGET_COL,
    WS1_FEATURE_COLS,
    load_canopy_target,
    load_minute_table,
    load_ws1_features,
)


def _write_csv(path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"

    # Canopy, 2015-01-01: 4 of the day's minutes, including one -999 sentinel
    # in the target column and one in the status column (both must mask).
    _write_csv(
        raw / "Canopy" / "2015" / "01" / "onemin-Canopy-2015-01-01.csv",
        [
            {"TIMESTAMP": "2015-01-01 00:00:00-05:00", CANOPY_TARGET_COL: 10.0, CANOPY_STATUS_COL: 5.0},
            {"TIMESTAMP": "2015-01-01 00:01:00-05:00", CANOPY_TARGET_COL: -999.0, CANOPY_STATUS_COL: 5.0},
            {"TIMESTAMP": "2015-01-01 00:02:00-05:00", CANOPY_TARGET_COL: 12.0, CANOPY_STATUS_COL: -999.0},
            {"TIMESTAMP": "2015-01-01 00:03:00-05:00", CANOPY_TARGET_COL: 15.0, CANOPY_STATUS_COL: 5.0},
        ],
    )
    # 2015-01-02 deliberately has no Canopy file at all -> every minute of
    # that day should come back as NaN after reindexing.

    # WS_1, 2015-01-01: same 4 minutes, plain values, no sentinel masking.
    _write_csv(
        raw / "WS_1" / "2015" / "01" / "onemin-WS_1-2015-01-01.csv",
        [
            {"TIMESTAMP": f"2015-01-01 00:0{i}:00-05:00", **{c: float(i) for c in WS1_FEATURE_COLS}}
            for i in range(4)
        ],
    )
    return raw


def test_canopy_target_masks_both_sentinel_columns(raw_dir):
    df = load_canopy_target(raw_dir, start_date="2015-01-01", end_date="2015-01-02")

    assert df.loc["2015-01-01 00:00:00-05:00", CANOPY_TARGET_COL] == 10.0
    assert pd.isna(df.loc["2015-01-01 00:01:00-05:00", CANOPY_TARGET_COL])  # target was -999
    assert pd.isna(df.loc["2015-01-01 00:02:00-05:00", CANOPY_TARGET_COL])  # status was -999
    assert df.loc["2015-01-01 00:03:00-05:00", CANOPY_TARGET_COL] == 15.0


def test_canopy_target_reindexes_missing_minutes_and_days_to_nan(raw_dir):
    df = load_canopy_target(raw_dir, start_date="2015-01-01", end_date="2015-01-02")

    # A minute of 2015-01-01 with no row in the file at all.
    assert pd.isna(df.loc["2015-01-01 00:10:00-05:00", CANOPY_TARGET_COL])
    # The entire missing day.
    assert pd.isna(df.loc["2015-01-02 12:00:00-05:00", CANOPY_TARGET_COL])
    # Complete minute grid: 2 days = 2880 rows, none dropped.
    assert len(df) == 2880


def test_ws1_features_load_without_masking(raw_dir):
    df = load_ws1_features(raw_dir, start_date="2015-01-01", end_date="2015-01-02")

    assert df.loc["2015-01-01 00:02:00-05:00", "Pyra1_Wm2_Avg"] == 2.0
    assert pd.isna(df.loc["2015-01-01 00:10:00-05:00", "Pyra1_Wm2_Avg"])
    assert len(df) == 2880


def test_load_minute_table_joins_target_and_features(raw_dir):
    df = load_minute_table(raw_dir, start_date="2015-01-01", end_date="2015-01-02")

    assert CANOPY_TARGET_COL in df.columns
    assert all(c in df.columns for c in WS1_FEATURE_COLS)
    row = df.loc["2015-01-01 00:00:00-05:00"]
    assert row[CANOPY_TARGET_COL] == 10.0
    assert row["Pyra1_Wm2_Avg"] == 0.0
