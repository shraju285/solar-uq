"""Tests for the persistence baseline: forecast alignment (every horizon
step repeats the window's last observed target value) and that it never
looks at anything beyond the lookback window."""

import numpy as np

from solaruq.features.windows import LOOKBACK_COLS, TARGET_COL
from solaruq.models.persistence import persistence_forecast

TARGET_IDX = LOOKBACK_COLS.index(TARGET_COL)


def test_forecast_repeats_last_lookback_value_across_horizon():
    n_windows, lookback_steps, horizon_steps = 3, 4, 5
    X_lookback = np.zeros((n_windows, lookback_steps, len(LOOKBACK_COLS)))
    last_values = [10.0, -2.5, 0.0]
    for i, v in enumerate(last_values):
        X_lookback[i, -1, TARGET_IDX] = v
        X_lookback[i, :-1, TARGET_IDX] = v - 100  # earlier lookback steps must be ignored

    y_hat = persistence_forecast(X_lookback, horizon_steps)

    assert y_hat.shape == (n_windows, horizon_steps)
    for i, v in enumerate(last_values):
        assert np.all(y_hat[i] == v)


def test_only_the_last_lookback_step_is_used():
    # Changing any lookback step except the last must not change the forecast.
    X_lookback = np.random.default_rng(0).normal(size=(5, 24, len(LOOKBACK_COLS)))
    baseline = persistence_forecast(X_lookback.copy(), horizon_steps=8)

    perturbed = X_lookback.copy()
    perturbed[:, :-1, TARGET_IDX] += 1000.0  # perturb everything except the last step
    perturbed_forecast = persistence_forecast(perturbed, horizon_steps=8)

    assert np.array_equal(baseline, perturbed_forecast)


def test_other_feature_columns_are_ignored():
    X_lookback = np.zeros((2, 3, len(LOOKBACK_COLS)))
    X_lookback[:, -1, TARGET_IDX] = 5.0
    X_lookback[:, -1, TARGET_IDX + 1] = 999.0  # a non-target column, should have no effect

    y_hat = persistence_forecast(X_lookback, horizon_steps=2)
    assert np.all(y_hat == 5.0)
