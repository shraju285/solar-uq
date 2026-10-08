"""Shared point-forecast evaluation harness.

One function, used identically by every point-forecast model (persistence,
ARIMA, and later the LSTM point forecast) so results are directly
comparable: same metrics, same horizon breakdown, computed the same way.
"""

import numpy as np


def evaluate_point_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: shape (n_windows, horizon_steps).

    Returns a dict with overall MAE/RMSE (computed over every
    window/horizon-step pair) and MAE/RMSE broken out per horizon step
    (one value per step, averaged over windows) -- e.g. "how good is the
    15-minute-ahead forecast" vs "how good is the 2-hour-ahead forecast".
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    error = y_true - y_pred
    abs_error = np.abs(error)
    sq_error = error**2

    return {
        "n_windows": int(y_true.shape[0]),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(sq_error))),
        "mae_by_horizon": np.mean(abs_error, axis=0).tolist(),
        "rmse_by_horizon": np.sqrt(np.mean(sq_error, axis=0)).tolist(),
    }
