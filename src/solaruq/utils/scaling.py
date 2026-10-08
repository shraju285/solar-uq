"""A small, train-only-fit z-score scaler for the LSTM's target.

Separate from Stage 3's own `feature_scaler.json` (which only ever scales
the weather columns, never the target) -- this one exists purely so the
LSTM trains on a standardized target, and is inverted before any metric is
computed so results stay in physical kW, directly comparable to
persistence/ARIMA.
"""

import numpy as np


def fit_target_scaler(y_train: np.ndarray) -> dict:
    return {"mean": float(y_train.mean()), "std": float(y_train.std())}


def apply_target_scaler(y: np.ndarray, scaler: dict) -> np.ndarray:
    return (y - scaler["mean"]) / scaler["std"]


def invert_target_scaler(y_scaled: np.ndarray, scaler: dict) -> np.ndarray:
    return y_scaled * scaler["std"] + scaler["mean"]
