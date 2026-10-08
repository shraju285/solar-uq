"""Tests for Stage 7's split-conformal calibration: the order-statistic
correction formula, that correction is fit only from the data it's given
(never a second "test" array), monotonicity restoration after calibration,
and reproducibility (purely deterministic arithmetic, no randomness)."""

import numpy as np
import pytest

from solaruq.uncertainty.conformal import apply_conformal_correction, fit_conformal_correction

QUANTILES = [0.10, 0.50, 0.90]


def test_shape_mismatch_raises():
    y_cal = np.zeros((5, 8))
    y_pred = np.zeros((5, 9, 3))
    with pytest.raises(ValueError):
        fit_conformal_correction(y_cal, y_pred, QUANTILES)


def test_correction_matches_manual_order_statistic_for_a_single_horizon_quantile():
    # n=9 calibration points, tau=0.5 -> k = ceil((9+1)*0.5) = 5 -> 5th smallest residual.
    n, h = 9, 1
    y_cal = np.arange(n, dtype=float).reshape(n, h)  # 0..8
    y_pred = np.zeros((n, h, 1))  # predicts 0 for everyone -> residuals = y_cal itself
    quantiles = [0.5]

    correction = fit_conformal_correction(y_cal, y_pred, quantiles)

    residuals_sorted = np.sort(y_cal[:, 0])  # [0,1,...,8]
    expected = residuals_sorted[5 - 1]  # 5th smallest = index 4 = 4.0
    assert correction.corrections[0, 0] == pytest.approx(expected)
    assert correction.n_calibration == n


def test_correction_only_uses_the_calibration_data_passed_in():
    # Fitting on two different calibration sets must give different corrections
    # when the underlying residual distributions differ -- i.e. there is no
    # hidden dependence on anything but the arrays passed to fit().
    n, h = 20, 2
    rng = np.random.default_rng(0)
    y_pred = np.zeros((n, h, 1))

    y_cal_a = rng.normal(loc=0.0, scale=1.0, size=(n, h))
    y_cal_b = rng.normal(loc=10.0, scale=1.0, size=(n, h))

    correction_a = fit_conformal_correction(y_cal_a, y_pred, [0.5])
    correction_b = fit_conformal_correction(y_cal_b, y_pred, [0.5])

    assert not np.allclose(correction_a.corrections, correction_b.corrections)


def test_apply_correction_shifts_predictions_additively():
    n, h, nq = 5, 2, len(QUANTILES)
    y_pred = np.zeros((n, h, nq))
    y_pred[..., 0], y_pred[..., 1], y_pred[..., 2] = 1.0, 2.0, 3.0

    from solaruq.uncertainty.conformal import ConformalCorrection
    correction = ConformalCorrection(
        quantiles=QUANTILES, horizon_steps=h,
        corrections=np.full((h, nq), 0.5), n_calibration=100,
    )

    calibrated, n_resorted = apply_conformal_correction(y_pred, correction)

    np.testing.assert_allclose(calibrated, y_pred + 0.5)
    assert n_resorted == 0  # still ascending after a uniform shift


def test_apply_correction_restores_monotonicity_when_shifts_cause_crossing():
    from solaruq.uncertainty.conformal import ConformalCorrection

    n, h, nq = 1, 1, 3
    y_pred = np.array([[[1.0, 2.0, 3.0]]])  # ascending: P10=1, P50=2, P90=3
    # A correction that pushes P90 down a lot and P10 up a lot -> crossing.
    correction = ConformalCorrection(
        quantiles=QUANTILES, horizon_steps=h,
        corrections=np.array([[5.0, 0.0, -5.0]]), n_calibration=10,
    )

    calibrated, n_resorted = apply_conformal_correction(y_pred, correction)

    assert n_resorted == 1
    assert np.all(np.diff(calibrated, axis=-1) >= 0)


def test_apply_correction_shape_mismatch_raises():
    from solaruq.uncertainty.conformal import ConformalCorrection

    correction = ConformalCorrection(quantiles=QUANTILES, horizon_steps=8,
                                      corrections=np.zeros((8, 3)), n_calibration=10)
    y_pred = np.zeros((5, 7, 3))  # wrong horizon_steps
    with pytest.raises(ValueError):
        apply_conformal_correction(y_pred, correction)


def test_fit_is_deterministic():
    n, h = 50, 3
    rng = np.random.default_rng(1)
    y_cal = rng.normal(size=(n, h))
    y_pred = rng.normal(size=(n, h, len(QUANTILES)))

    correction_a = fit_conformal_correction(y_cal.copy(), y_pred.copy(), QUANTILES)
    correction_b = fit_conformal_correction(y_cal.copy(), y_pred.copy(), QUANTILES)

    np.testing.assert_array_equal(correction_a.corrections, correction_b.corrections)


def test_zero_residuals_give_zero_correction():
    # A perfect calibration-set forecast (residuals all zero) should need no shift.
    n, h = 10, 2
    y_pred = np.random.default_rng(2).normal(size=(n, h, 1))
    y_cal = y_pred[..., 0]  # perfect match

    correction = fit_conformal_correction(y_cal, y_pred, [0.5])

    np.testing.assert_allclose(correction.corrections, 0.0, atol=1e-10)
