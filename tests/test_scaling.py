"""Tests for the Stage 5 target scaler: fit-on-train-only semantics and
that apply/invert are exact inverses of each other."""

import numpy as np

from solaruq.utils.scaling import apply_target_scaler, fit_target_scaler, invert_target_scaler


def test_scaler_fit_only_uses_the_given_array():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    scaler = fit_target_scaler(y_train)

    assert scaler["mean"] == y_train.mean()
    assert scaler["std"] == y_train.std()


def test_apply_then_invert_recovers_the_original_values():
    rng = np.random.default_rng(0)
    y_train = rng.normal(loc=50.0, scale=20.0, size=1000)
    y_other = rng.normal(loc=50.0, scale=20.0, size=(30, 8))  # e.g. a val/test split's y
    scaler = fit_target_scaler(y_train)

    y_scaled = apply_target_scaler(y_other, scaler)
    y_recovered = invert_target_scaler(y_scaled, scaler)

    np.testing.assert_allclose(y_recovered, y_other, rtol=1e-10)


def test_scaled_train_target_has_approximately_zero_mean_and_unit_std():
    y_train = np.random.default_rng(1).normal(loc=5.0, scale=2.0, size=10000)
    scaler = fit_target_scaler(y_train)

    y_scaled = apply_target_scaler(y_train, scaler)

    assert abs(y_scaled.mean()) < 1e-8
    assert abs(y_scaled.std() - 1.0) < 1e-8
