"""Tests for the shared point-forecast evaluation harness."""

import numpy as np
import pytest

from solaruq.evaluation.metrics import evaluate_point_forecast


def test_perfect_forecast_has_zero_error():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = evaluate_point_forecast(y, y.copy())
    assert out["mae"] == 0.0
    assert out["rmse"] == 0.0
    assert out["mae_by_horizon"] == [0.0, 0.0, 0.0]


def test_overall_mae_matches_manual_calculation():
    y_true = np.array([[10.0, 20.0], [30.0, 40.0]])
    y_pred = np.array([[12.0, 18.0], [33.0, 36.0]])
    # abs errors: [2, 2, 3, 4] -> mean = 2.75
    out = evaluate_point_forecast(y_true, y_pred)
    assert out["mae"] == pytest.approx(2.75)
    assert out["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9 + 16) / 4))


def test_mae_by_horizon_averages_over_windows_not_horizon_steps():
    # Horizon step 0 always off by 1, horizon step 1 always off by 3.
    y_true = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    y_pred = np.array([[1.0, 3.0], [11.0, 13.0], [21.0, 23.0]])
    out = evaluate_point_forecast(y_true, y_pred)
    assert out["mae_by_horizon"] == pytest.approx([1.0, 3.0])
    assert out["rmse_by_horizon"] == pytest.approx([1.0, 3.0])
    assert out["mae"] == pytest.approx(2.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_point_forecast(np.zeros((2, 3)), np.zeros((2, 4)))


def test_n_windows_reported():
    y = np.zeros((7, 4))
    out = evaluate_point_forecast(y, y)
    assert out["n_windows"] == 7
