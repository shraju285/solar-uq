"""Persistence baseline: forecast every horizon step as the single most
recently observed target value at the forecast origin.

This is deliberately the simplest possible model — the point of a
persistence baseline is to give every other model (ARIMA, later the LSTM)
something trivial to beat. "Most recently observed" means exactly one
number per window: the last entry of its lookback target series. Nothing
else about the window (earlier lookback steps, weather features, horizon
timing) is used.
"""

import numpy as np

from solaruq.features.windows import LOOKBACK_COLS, TARGET_COL

TARGET_IDX = LOOKBACK_COLS.index(TARGET_COL)


def persistence_forecast(X_lookback: np.ndarray, horizon_steps: int) -> np.ndarray:
    """X_lookback: shape (n, lookback_steps, len(LOOKBACK_COLS)).
    Returns y_hat of shape (n, horizon_steps): every horizon step for a
    window repeats that window's last observed target value.
    """
    last_observed = X_lookback[:, -1, TARGET_IDX]
    return np.repeat(last_observed[:, np.newaxis], horizon_steps, axis=1)
