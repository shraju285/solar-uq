"""Stage 3: build the preprocessed, windowed, split dataset from data/raw/.

Pipeline: load_raw -> resample_to_15min -> fit/apply train-only feature
scaler -> build_windows. Writes:
  - data/interim/canopy_ws1_15min.parquet   (the resampled 15-min table)
  - data/processed/stage3/{train,val,test}.npz
  - data/processed/stage3/feature_scaler.json

Prints every number asked for in the Stage 3 report: raw rows processed,
15-min bins produced, bins/percentage removed by the completeness rule,
continuous blocks, windows per split, and an explicit boundary-crossing
check.

Read-only on data/raw/. Everything it writes goes under data/interim/ and
data/processed/, both git-ignored (see .gitignore) — never commits any
data, only this script and the modules it calls.

Run with:  python scripts/build_stage3_dataset.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solaruq.data.load_raw import load_minute_table
from solaruq.data.resample import resample_to_15min
from solaruq.features.windows import (
    apply_feature_scaler,
    build_windows,
    continuous_blocks,
    fit_feature_scaler,
    split_ranges,
)
from solaruq.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config()
    raw_dir = REPO_ROOT / config["data"]["raw_dir"]
    tz_offset = config["data"]["timezone_offset"]
    resolution = config["forecast"]["resolution_minutes"]
    lookback = config["forecast"]["lookback_steps"]
    horizon = config["forecast"]["horizon_steps"]
    completeness_threshold = config["forecast"]["completeness_threshold"]

    print("Loading raw Canopy + WS_1 minute data...")
    minute_table = load_minute_table(raw_dir, timezone_offset=tz_offset)
    n_raw_rows = len(minute_table)
    print(f"  {n_raw_rows:,} raw minute rows (2015-01-01 to 2018-12-31, complete grid)")

    print("Resampling to 15-minute bins...")
    resampled = resample_to_15min(minute_table, resolution, completeness_threshold)
    n_bins = len(resampled)
    n_gap_bins = int(resampled["is_gap"].sum())
    print(f"  {n_bins:,} bins produced; {n_gap_bins:,} ({100 * n_gap_bins / n_bins:.3f}%) removed by the {completeness_threshold:.0%} completeness rule")

    blocks = continuous_blocks(resampled["is_gap"])
    print(f"  {len(blocks)} continuous (gap-free) block(s)")

    ranges = split_ranges(config, resampled.index[0], resampled.index[-1])
    scaler = fit_feature_scaler(resampled, ranges["train"])
    scaled = apply_feature_scaler(resampled, scaler)
    print("Fitted feature scaler on train-split rows only:")
    for col, (mean, std) in scaler.items():
        print(f"  {col}: mean={mean:.4f}, std={std:.4f}")

    print("Building windows...")
    windows = build_windows(scaled, lookback, horizon, ranges)

    print("\nBoundary check: confirming no window crosses a split boundary...")
    ok = True
    for name, split in ranges.items():
        w = windows[name]
        if len(w["origin_time"]) == 0:
            continue
        origin_times = pd.DatetimeIndex(w["origin_time"])
        lookback_starts = origin_times - pd.Timedelta(minutes=(lookback - 1) * resolution)
        horizon_ends = origin_times + pd.Timedelta(minutes=horizon * resolution)
        if lookback_starts.min() < split.start or horizon_ends.max() > split.end:
            ok = False
            print(f"  FAIL: {name} has a window outside its split range")
    if ok:
        print("  PASS: every window's full lookback+horizon span lies inside its own split's date range.")

    interim_dir = REPO_ROOT / config["data"]["interim_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)
    scaled.to_parquet(interim_dir / "canopy_ws1_15min.parquet")

    processed_dir = REPO_ROOT / config["data"]["processed_dir"] / "stage3"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, w in windows.items():
        np.savez(
            processed_dir / f"{name}.npz",
            X_lookback=w["X_lookback"],
            X_horizon_known=w["X_horizon_known"],
            y=w["y"],
            origin_time=w["origin_time"],
        )
        print(f"  {name}: {len(w['origin_time']):,} windows -> {processed_dir / f'{name}.npz'}")

    with open(processed_dir / "feature_scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    print(f"\nWrote {interim_dir / 'canopy_ws1_15min.parquet'}")
    print(f"Wrote window files + feature_scaler.json under {processed_dir}")


if __name__ == "__main__":
    main()
