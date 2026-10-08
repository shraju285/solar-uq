"""Turn the gap-flagged 15-minute table into fixed-size lookback/horizon
windows, split chronologically into train/validation/test.

Three building blocks, used in this order:
  1. `split_ranges` — where each split starts/ends, with a `gap_steps`
     buffer carved out of the calendar between them (D-2026-10-08) so no
     window can reach from one split into another.
  2. `fit_feature_scaler` / `apply_feature_scaler` — z-score the continuous
     weather columns using train-split statistics only, then apply that
     same transform to validation and test. Solar position (already
     bounded, physically meaningful) and the target (kept in physical kW
     for persistence/ARIMA/evaluation) are never scaled.
  3. `build_windows` — for every split, finds windows whose full
     lookback+horizon span is both (a) inside one continuous gap-free run
     and (b) inside that split's own date range, and assembles them into
     fixed-shape arrays.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_COL = "InvPAC_kW_Avg"
SOLAR_POSITION_COLS = ["SolarZenith_deg_Avg", "SolarAzFromSouth_deg_Avg", "SolarTime_hr"]
SCALED_WEATHER_COLS = [
    "Pyra1_Wm2_Avg",
    "RefCell1_Wm2_Avg",
    "AirTemp_C_Avg",
    "RelHumid_Avg",
    "AirPres_kPa_Avg",
    "Rain_mm_Tot",
]
LOOKBACK_COLS = [TARGET_COL, *SCALED_WEATHER_COLS, *SOLAR_POSITION_COLS]
HORIZON_KNOWN_COLS = SOLAR_POSITION_COLS


@dataclass
class SplitRange:
    start: pd.Timestamp
    end: pd.Timestamp


def split_ranges(config: dict, data_start: pd.Timestamp, data_end: pd.Timestamp) -> dict[str, SplitRange]:
    """Compute train/val/test date ranges from config["split"], inserting a
    gap_steps buffer (excluded from every split) after each boundary."""
    resolution = config["forecast"]["resolution_minutes"]
    buffer = pd.Timedelta(minutes=config["split"]["gap_steps"] * resolution)

    train_end = pd.Timestamp(config["split"]["train_end"])
    val_end = pd.Timestamp(config["split"]["val_end"])
    test_end = pd.Timestamp(config["split"]["test_end"])
    if test_end > data_end:
        raise ValueError(f"configured test_end {test_end} is after the data's last timestamp {data_end}")

    return {
        "train": SplitRange(data_start, train_end),
        "val": SplitRange(train_end + buffer, val_end),
        "test": SplitRange(val_end + buffer, test_end),
    }


def fit_feature_scaler(df: pd.DataFrame, train_range: SplitRange) -> dict[str, tuple[float, float]]:
    """Mean/std of each weather column, computed only from non-gap rows
    inside the training range. Returns {column: (mean, std)}."""
    train_rows = df.loc[train_range.start : train_range.end]
    train_rows = train_rows[~train_rows["is_gap"]]
    return {
        col: (float(train_rows[col].mean()), float(train_rows[col].std()))
        for col in SCALED_WEATHER_COLS
    }


def apply_feature_scaler(df: pd.DataFrame, scaler: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Apply a scaler fitted by fit_feature_scaler to every row (train,
    validation, and test alike — the statistics came from train only, but
    the same fixed transform is applied everywhere)."""
    out = df.copy()
    for col, (mean, std) in scaler.items():
        out[col] = (out[col] - mean) / std
    return out


def continuous_blocks(is_gap: pd.Series) -> list[tuple[int, int]]:
    """Maximal (start_pos, end_pos) inclusive positional runs where is_gap
    is False. Positional, not time-based — assumes df.index is a complete,
    regularly-spaced grid (true for resample_to_15min's output)."""
    valid = (~is_gap).to_numpy()
    blocks = []
    start = None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(valid) - 1))
    return blocks


def _valid_origins(is_gap: pd.Series, lookback_steps: int, horizon_steps: int) -> np.ndarray:
    """Positions p such that every step from p-lookback_steps+1 to
    p+horizon_steps (inclusive) is gap-free. Vectorised via a rolling sum
    rather than slicing every candidate window."""
    window_len = lookback_steps + horizon_steps
    valid = (~is_gap).to_numpy().astype(int)
    rolling_valid_count = pd.Series(valid).rolling(window_len).sum().to_numpy()
    # rolling(window_len) labels its result at the window's last position;
    # that last position is p + horizon_steps, so p = window_end - horizon_steps.
    window_ends = np.flatnonzero(rolling_valid_count == window_len)
    return window_ends - horizon_steps


def build_windows(
    df: pd.DataFrame,
    lookback_steps: int,
    horizon_steps: int,
    ranges: dict[str, SplitRange],
) -> dict[str, dict[str, np.ndarray]]:
    """Build lookback/horizon windows for every split in `ranges`.

    Returns {split_name: {"X_lookback", "X_horizon_known", "y", "origin_time"}}
    where X_lookback has shape (n, lookback_steps, len(LOOKBACK_COLS)),
    X_horizon_known has shape (n, horizon_steps, len(HORIZON_KNOWN_COLS)),
    y has shape (n, horizon_steps), and origin_time has shape (n,).
    """
    origins = _valid_origins(df["is_gap"], lookback_steps, horizon_steps)
    index = df.index
    lookback_values = df[LOOKBACK_COLS].to_numpy()
    horizon_known_values = df[HORIZON_KNOWN_COLS].to_numpy()
    target_values = df[TARGET_COL].to_numpy()

    out: dict[str, dict[str, np.ndarray]] = {}
    for name, split in ranges.items():
        lookback_starts = index[origins - lookback_steps + 1]
        horizon_ends = index[origins + horizon_steps]
        in_split = (lookback_starts >= split.start) & (horizon_ends <= split.end)
        split_origins = origins[in_split]

        X_lookback = np.stack(
            [lookback_values[p - lookback_steps + 1 : p + 1] for p in split_origins]
        ) if len(split_origins) else np.empty((0, lookback_steps, len(LOOKBACK_COLS)))
        X_horizon_known = np.stack(
            [horizon_known_values[p + 1 : p + horizon_steps + 1] for p in split_origins]
        ) if len(split_origins) else np.empty((0, horizon_steps, len(HORIZON_KNOWN_COLS)))
        y = np.stack(
            [target_values[p + 1 : p + horizon_steps + 1] for p in split_origins]
        ) if len(split_origins) else np.empty((0, horizon_steps))

        out[name] = {
            "X_lookback": X_lookback,
            "X_horizon_known": X_horizon_known,
            "y": y,
            "origin_time": index[split_origins].to_numpy(),
        }
    return out
