"""Aggregate the 1-minute table from load_raw.py into 15-minute bins.

Each bin takes the mean of whichever raw minutes are present (D-2026-10-08:
`-999`-masked rows count as missing, same as a genuine logger gap — there's
no reason to treat them differently once they're already NaN). A bin built
from too few real minutes isn't trustworthy, so bins below
`completeness_threshold` are marked as gaps and set to NaN outright rather
than silently averaging over a handful of values.

"Complete" is row-wise, not per-column: a raw minute only counts toward a
bin's completeness if every column (target + every feature) is present for
that minute. This matches how the Stage 2 audit found missingness actually
behaves — whole-scan dropouts affecting many columns at once, not
independent per-sensor gaps — and keeps every column in a bin backed by the
same set of raw minutes.
"""

import pandas as pd


def resample_to_15min(
    minute_table: pd.DataFrame,
    resolution_minutes: int = 15,
    completeness_threshold: float = 0.8,
) -> pd.DataFrame:
    """Resample a complete, minute-indexed DataFrame to `resolution_minutes`
    bins. Returns a DataFrame with the same columns, plus `completeness`
    (fraction of raw minutes present in the bin) and `is_gap` (completeness
    below threshold — all value columns are NaN on these rows).
    """
    value_cols = list(minute_table.columns)
    valid_minute = minute_table.notna().all(axis=1)

    freq = f"{resolution_minutes}min"
    resampler = minute_table.where(valid_minute).resample(freq, label="left", closed="left")
    means = resampler.mean()

    completeness = valid_minute.resample(freq, label="left", closed="left").sum() / resolution_minutes
    is_gap = completeness < completeness_threshold

    out = means.copy()
    out.loc[is_gap, value_cols] = pd.NA
    out["completeness"] = completeness
    out["is_gap"] = is_gap
    return out
