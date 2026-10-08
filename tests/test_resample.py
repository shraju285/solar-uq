"""Tests for the 15-minute aggregation + completeness/gap logic.

Mirrors the worked example from the Stage 3 design discussion: three
15-minute bins, one full, one at 87% completeness (passes the 80%
threshold), one at 40% (becomes a gap).
"""

import numpy as np
import pandas as pd

from solaruq.data.resample import resample_to_15min


def _minute_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="min")


def test_full_bin_is_mean_of_all_minutes():
    idx = _minute_index("2018-01-01 14:00:00-05:00", 15)
    df = pd.DataFrame({"x": np.arange(15, dtype=float)}, index=idx)

    out = resample_to_15min(df, resolution_minutes=15, completeness_threshold=0.8)

    assert len(out) == 1
    assert out["completeness"].iloc[0] == 1.0
    assert not out["is_gap"].iloc[0]
    assert out["x"].iloc[0] == np.arange(15).mean()


def test_bin_above_threshold_uses_mean_of_present_minutes():
    idx = _minute_index("2018-01-01 14:15:00-05:00", 15)
    values = np.arange(15, dtype=float)
    values[5] = np.nan
    values[9] = np.nan
    df = pd.DataFrame({"x": values}, index=idx)

    out = resample_to_15min(df, resolution_minutes=15, completeness_threshold=0.8)

    assert len(out) == 1
    assert out["completeness"].iloc[0] == 13 / 15
    assert not out["is_gap"].iloc[0]
    expected_mean = np.delete(values, [5, 9]).mean()
    assert np.isclose(out["x"].iloc[0], expected_mean)


def test_bin_below_threshold_becomes_a_gap():
    idx = _minute_index("2018-01-01 14:30:00-05:00", 15)
    values = np.full(15, np.nan)
    values[:6] = np.arange(6, dtype=float)  # only 6 of 15 minutes present (40%)
    df = pd.DataFrame({"x": values}, index=idx)

    out = resample_to_15min(df, resolution_minutes=15, completeness_threshold=0.8)

    assert len(out) == 1
    assert out["completeness"].iloc[0] == 6 / 15
    assert out["is_gap"].iloc[0]
    assert pd.isna(out["x"].iloc[0])


def test_completeness_requires_every_column_present():
    # A minute only counts if ALL columns are present — a minute with one
    # column missing shouldn't contribute to either column's mean.
    idx = _minute_index("2018-01-01 15:00:00-05:00", 15)
    x = np.arange(15, dtype=float)
    y = np.arange(15, dtype=float) * 2
    y[0] = np.nan  # first minute has y missing, x present
    df = pd.DataFrame({"x": x, "y": y}, index=idx)

    out = resample_to_15min(df, resolution_minutes=15, completeness_threshold=0.8)

    assert out["completeness"].iloc[0] == 14 / 15
    assert np.isclose(out["x"].iloc[0], x[1:].mean())
    assert np.isclose(out["y"].iloc[0], y[1:].mean())


def test_three_bins_match_the_worked_example():
    idx = _minute_index("2018-01-01 14:00:00-05:00", 45)
    values = np.arange(45, dtype=float)
    # Bin 2 (14:15-14:30): drop 2 minutes -> 13/15 = 87%, passes.
    values[20] = np.nan
    values[24] = np.nan
    # Bin 3 (14:30-14:45): drop 9 minutes -> 6/15 = 40%, gap.
    values[33:42] = np.nan
    df = pd.DataFrame({"x": values}, index=idx)

    out = resample_to_15min(df, resolution_minutes=15, completeness_threshold=0.8)

    assert len(out) == 3
    assert list(out["is_gap"]) == [False, False, True]
    assert np.isclose(out["completeness"].iloc[1], 13 / 15)
    assert np.isclose(out["completeness"].iloc[2], 6 / 15)
