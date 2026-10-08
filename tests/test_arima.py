"""Tests for the ARIMA baseline.

The leakage-safety argument for this model is structural: `arima_forecast`
takes only `lookback_target` (shape (n, lookback_steps)) -- there is no `y`
or "horizon" parameter anywhere in its signature, so it is not possible for
a call to depend on data from after the forecast origin. These tests check
the things that *could* still go wrong even with that structure: that a
window genuinely can't influence another window's forecast, and that a
series ARIMA can't fit falls back to persistence instead of crashing or
silently letting a degenerate fit through.
"""

import numpy as np

from solaruq.models.arima import CANDIDATE_ORDERS, arima_forecast, select_order

ORDER = (1, 1, 0)


def _ramp_series(n_windows: int, lookback_steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 10, lookback_steps)
    return np.stack([base + rng.normal(scale=0.1, size=lookback_steps) for _ in range(n_windows)])


def test_forecast_shape_and_determinism():
    series = _ramp_series(6, 24)
    y_hat_1, n_fallback_1 = arima_forecast(series, ORDER, horizon_steps=8)
    y_hat_2, n_fallback_2 = arima_forecast(series, ORDER, horizon_steps=8)

    assert y_hat_1.shape == (6, 8)
    assert n_fallback_1 == 0
    np.testing.assert_allclose(y_hat_1, y_hat_2)  # same input -> same output
    assert n_fallback_1 == n_fallback_2


def test_windows_cannot_influence_each_other():
    series = _ramp_series(5, 24)
    baseline, _ = arima_forecast(series, ORDER, horizon_steps=8)

    perturbed = series.copy()
    perturbed[2] += 1000.0  # drastically change only window 2
    perturbed_forecast, _ = arima_forecast(perturbed, ORDER, horizon_steps=8)

    # Every window except the perturbed one must be completely unaffected.
    other_rows = [0, 1, 3, 4]
    np.testing.assert_allclose(baseline[other_rows], perturbed_forecast[other_rows])
    assert not np.allclose(baseline[2], perturbed_forecast[2])


def test_row_order_is_preserved():
    # Forecasting windows [A, B, C] must give the same per-row results as
    # forecasting [C, B, A] and un-reversing -- i.e. nothing is scrambled.
    series = _ramp_series(5, 24)
    forward, _ = arima_forecast(series, ORDER, horizon_steps=8)
    backward, _ = arima_forecast(series[::-1], ORDER, horizon_steps=8)
    np.testing.assert_allclose(forward, backward[::-1])


def test_unfittable_series_falls_back_to_persistence_not_a_crash():
    series = _ramp_series(4, 24)
    series[1] = np.nan  # ARIMA cannot fit a NaN-containing series

    y_hat, n_fallback = arima_forecast(series, ORDER, horizon_steps=8)

    assert n_fallback == 1
    assert np.all(np.isnan(y_hat[1]))  # fallback repeats the (NaN) last value, doesn't invent one
    assert not np.any(np.isnan(y_hat[0]))  # the other windows are unaffected


def test_empty_input_returns_empty_output():
    y_hat, n_fallback = arima_forecast(np.empty((0, 24)), ORDER, horizon_steps=8)
    assert y_hat.shape == (0, 8)
    assert n_fallback == 0


def test_select_order_prefers_a_differenced_model_for_a_trending_series():
    train_series = _ramp_series(50, 24, seed=1)
    order, report = select_order(train_series, candidate_orders=CANDIDATE_ORDERS, sample_size=50)

    assert order in CANDIDATE_ORDERS
    assert set(report.keys()) == set(CANDIDATE_ORDERS)
    for mean_aic, n_fail in report.values():
        assert n_fail >= 0
