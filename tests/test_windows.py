"""Tests for split ranges, the train-only feature scaler, continuous-block
detection, and the sliding window builder — in particular, that no window
can cross a split boundary (the main leakage control this module is
responsible for)."""

import numpy as np
import pandas as pd
import pytest

from solaruq.features.windows import (
    HORIZON_KNOWN_COLS,
    LOOKBACK_COLS,
    SCALED_WEATHER_COLS,
    TARGET_COL,
    SplitRange,
    apply_feature_scaler,
    build_windows,
    continuous_blocks,
    fit_feature_scaler,
    split_ranges,
)


def test_continuous_blocks_finds_maximal_gap_free_runs():
    is_gap = pd.Series([False, False, True, False, False, False, True, True, False])
    assert continuous_blocks(is_gap) == [(0, 1), (3, 5), (8, 8)]


def test_continuous_blocks_all_valid_is_one_block():
    is_gap = pd.Series([False] * 5)
    assert continuous_blocks(is_gap) == [(0, 4)]


def test_split_ranges_inserts_gap_buffer_after_each_boundary():
    config = {
        "forecast": {"resolution_minutes": 15},
        "split": {
            "train_end": "2018-01-01T00:00:00-05:00",
            "val_end": "2018-01-02T00:00:00-05:00",
            "test_end": "2018-01-03T00:00:00-05:00",
            "gap_steps": 4,  # 4 * 15min = 1 hour
        },
    }
    data_start = pd.Timestamp("2017-01-01T00:00:00-05:00")
    data_end = pd.Timestamp("2018-01-03T00:00:00-05:00")

    ranges = split_ranges(config, data_start, data_end)

    assert ranges["train"].start == data_start
    assert ranges["train"].end == pd.Timestamp("2018-01-01T00:00:00-05:00")
    assert ranges["val"].start == pd.Timestamp("2018-01-01T01:00:00-05:00")
    assert ranges["val"].end == pd.Timestamp("2018-01-02T00:00:00-05:00")
    assert ranges["test"].start == pd.Timestamp("2018-01-02T01:00:00-05:00")
    assert ranges["test"].end == pd.Timestamp("2018-01-03T00:00:00-05:00")


def test_split_ranges_rejects_test_end_past_available_data():
    config = {
        "forecast": {"resolution_minutes": 15},
        "split": {
            "train_end": "2018-01-01T00:00:00-05:00",
            "val_end": "2018-01-02T00:00:00-05:00",
            "test_end": "2019-01-01T00:00:00-05:00",
            "gap_steps": 4,
        },
    }
    with pytest.raises(ValueError):
        split_ranges(
            config,
            pd.Timestamp("2017-01-01T00:00:00-05:00"),
            pd.Timestamp("2018-01-03T00:00:00-05:00"),
        )


def test_feature_scaler_fits_on_train_rows_only():
    idx = pd.date_range("2018-01-01", periods=6, freq="15min", tz="-05:00")
    df = pd.DataFrame(
        {
            "Pyra1_Wm2_Avg": [0.0, 10.0, 20.0, 1000.0, 1000.0, 1000.0],
            "is_gap": [False, False, False, False, False, False],
        },
        index=idx,
    )
    for col in SCALED_WEATHER_COLS:
        if col not in df.columns:
            df[col] = 0.0

    train_range = SplitRange(idx[0], idx[2])  # only the first 3 rows (0, 10, 20)
    scaler = fit_feature_scaler(df, train_range)

    mean, std = scaler["Pyra1_Wm2_Avg"]
    assert np.isclose(mean, 10.0)  # mean of [0, 10, 20], not the 1000s
    assert not np.isclose(mean, df["Pyra1_Wm2_Avg"].mean())

    scaled = apply_feature_scaler(df, scaler)
    assert np.isclose(scaled["Pyra1_Wm2_Avg"].iloc[0], (0.0 - mean) / std)
    assert np.isclose(scaled["Pyra1_Wm2_Avg"].iloc[3], (1000.0 - mean) / std)  # same transform applied


def test_feature_scaler_excludes_gap_rows_from_training_stats():
    idx = pd.date_range("2018-01-01", periods=4, freq="15min", tz="-05:00")
    df = pd.DataFrame(
        {"Pyra1_Wm2_Avg": [10.0, 10.0, 99999.0, 10.0], "is_gap": [False, False, True, False]},
        index=idx,
    )
    for col in SCALED_WEATHER_COLS:
        if col not in df.columns:
            df[col] = 0.0

    scaler = fit_feature_scaler(df, SplitRange(idx[0], idx[-1]))
    mean, _ = scaler["Pyra1_Wm2_Avg"]
    assert np.isclose(mean, 10.0)  # the gap row's huge value must not pollute the mean


def _make_clean_df(n: int, start: str = "2018-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="15min", tz="-05:00")
    cols = {TARGET_COL: np.arange(n, dtype=float)}
    for col in SCALED_WEATHER_COLS + HORIZON_KNOWN_COLS:
        cols[col] = np.arange(n, dtype=float) * 0.1
    cols["is_gap"] = [False] * n
    return pd.DataFrame(cols, index=idx)


def test_build_windows_shapes_and_values_small_case():
    lookback, horizon = 3, 2
    df = _make_clean_df(n=10)
    ranges = {"train": SplitRange(df.index[0], df.index[-1])}

    out = build_windows(df, lookback, horizon, ranges)["train"]

    # Valid origins: need lookback-1 steps before and horizon steps after.
    # positions 2..7 qualify (0-indexed) -> 6 windows.
    assert out["X_lookback"].shape == (6, lookback, len(LOOKBACK_COLS))
    assert out["X_horizon_known"].shape == (6, horizon, len(HORIZON_KNOWN_COLS))
    assert out["y"].shape == (6, horizon)

    # First window: origin at position 2 -> lookback positions [0,1,2], horizon [3,4].
    assert list(out["y"][0]) == [3.0, 4.0]
    assert list(out["X_lookback"][0][:, LOOKBACK_COLS.index(TARGET_COL)]) == [0.0, 1.0, 2.0]


def test_build_windows_real_lookback_horizon_values():
    lookback, horizon = 24, 8  # the locked Stage 3 values
    df = _make_clean_df(n=40)
    ranges = {"train": SplitRange(df.index[0], df.index[-1])}

    out = build_windows(df, lookback, horizon, ranges)["train"]

    # Valid origins run from position 23 to position 31 inclusive -> 9 windows.
    assert out["X_lookback"].shape == (9, 24, len(LOOKBACK_COLS))
    assert out["y"].shape == (9, 8)


def test_window_never_crosses_a_gap():
    lookback, horizon = 3, 2
    df = _make_clean_df(n=12)
    df.loc[df.index[6], "is_gap"] = True  # plant a single gap step
    ranges = {"train": SplitRange(df.index[0], df.index[-1])}

    out = build_windows(df, lookback, horizon, ranges)["train"]

    # No window's lookback/horizon span (5 steps) may include position 6.
    for origin_time in out["origin_time"]:
        pos = df.index.get_loc(origin_time)
        assert not (pos - lookback + 1 <= 6 <= pos + horizon)


def test_no_window_crosses_a_split_boundary():
    lookback, horizon = 3, 2
    df = _make_clean_df(n=20)
    boundary = df.index[10]
    buffer_steps = lookback + horizon  # satisfies gap_steps >= lookback+horizon
    ranges = {
        "A": SplitRange(df.index[0], boundary),
        "B": SplitRange(df.index[10 + buffer_steps], df.index[-1]),
    }

    out = build_windows(df, lookback, horizon, ranges)

    for split_name, split_range in ranges.items():
        windows = out[split_name]
        for origin_time in windows["origin_time"]:
            pos = df.index.get_loc(origin_time)
            window_start = df.index[pos - lookback + 1]
            window_end = df.index[pos + horizon]
            assert window_start >= split_range.start
            assert window_end <= split_range.end

    # And the two splits' windows never overlap in time at all.
    if len(out["A"]["origin_time"]) and len(out["B"]["origin_time"]):
        assert out["A"]["origin_time"].max() < out["B"]["origin_time"].min()
