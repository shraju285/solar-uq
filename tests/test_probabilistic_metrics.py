"""Tests for the Stage 6 probabilistic evaluation metrics: coverage,
interval width, interval score, pinball loss, and that the P50 evaluation
is a pure pass-through to the unmodified Stage 4 harness."""

import numpy as np
import pytest

from solaruq.evaluation.probabilistic_metrics import (
    evaluate_p50_point_forecast,
    evaluate_quantile_forecast,
    interval_score,
    pinball_loss_np,
)

QUANTILES = [0.10, 0.50, 0.90]


def test_shape_mismatch_raises():
    y_true = np.zeros((5, 8))
    y_pred = np.zeros((5, 9, 3))
    with pytest.raises(ValueError):
        evaluate_quantile_forecast(y_true, y_pred, QUANTILES)


def test_unsorted_quantiles_raises():
    y_true = np.zeros((5, 8))
    y_pred = np.zeros((5, 8, 3))
    with pytest.raises(ValueError):
        evaluate_quantile_forecast(y_true, y_pred, [0.5, 0.1, 0.9])


def test_coverage_is_one_when_every_true_value_is_inside_the_interval():
    n, h = 10, 4
    y_true = np.full((n, h), 5.0)
    y_pred = np.zeros((n, h, 3))
    y_pred[..., 0] = 0.0   # P10
    y_pred[..., 1] = 5.0   # P50
    y_pred[..., 2] = 10.0  # P90

    result = evaluate_quantile_forecast(y_true, y_pred, QUANTILES, lower_q=0.10, upper_q=0.90)

    assert result["coverage_overall"] == 1.0
    assert all(c == 1.0 for c in result["coverage_by_horizon"])
    assert result["interval_width_overall"] == pytest.approx(10.0)


def test_coverage_is_zero_when_every_true_value_is_outside_the_interval():
    n, h = 10, 4
    y_true = np.full((n, h), 100.0)  # far above the interval
    y_pred = np.zeros((n, h, 3))
    y_pred[..., 0] = 0.0
    y_pred[..., 1] = 5.0
    y_pred[..., 2] = 10.0

    result = evaluate_quantile_forecast(y_true, y_pred, QUANTILES, lower_q=0.10, upper_q=0.90)

    assert result["coverage_overall"] == 0.0
    assert result["interval_score_overall"] > result["interval_width_overall"]  # penalty on top of width


def test_interval_score_matches_manual_calculation():
    y_true = np.array([15.0])
    lower = np.array([0.0])
    upper = np.array([10.0])
    alpha = 0.2  # P10-P90

    score = interval_score(y_true, lower, upper, alpha)

    # width=10, true is above upper by 5: 10 + (2/0.2)*5 = 10 + 50 = 60
    assert score[0] == pytest.approx(60.0)


def test_pinball_loss_np_zero_for_perfect_forecast():
    y_true = np.array([1.0, 2.0, 3.0])
    assert np.all(pinball_loss_np(y_true, y_true, 0.3) == 0.0)


def test_pinball_by_quantile_keys_match_input_quantiles():
    n, h = 6, 8
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(n, h))
    y_pred = np.sort(rng.normal(size=(n, h, len(QUANTILES))), axis=-1)

    result = evaluate_quantile_forecast(y_true, y_pred, QUANTILES, lower_q=0.10, upper_q=0.90)

    assert set(result["pinball_by_quantile"].keys()) == {str(q) for q in QUANTILES}
    assert len(result["pinball_by_horizon"]) == h
    assert len(result["coverage_by_horizon"]) == h
    assert len(result["interval_width_by_horizon"]) == h
    assert len(result["interval_score_by_horizon"]) == h


def test_p50_evaluation_matches_evaluating_the_p50_slice_directly():
    from solaruq.evaluation.metrics import evaluate_point_forecast

    n, h = 7, 8
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=(n, h))
    y_pred = np.sort(rng.normal(size=(n, h, len(QUANTILES))), axis=-1)

    via_wrapper = evaluate_p50_point_forecast(y_true, y_pred, QUANTILES)
    p50_idx = QUANTILES.index(0.50)
    direct = evaluate_point_forecast(y_true, y_pred[..., p50_idx])

    assert via_wrapper == direct


def test_p50_evaluation_requires_a_p50_quantile():
    y_true = np.zeros((3, 8))
    y_pred = np.zeros((3, 8, 2))
    with pytest.raises(ValueError):
        evaluate_p50_point_forecast(y_true, y_pred, [0.1, 0.9])
