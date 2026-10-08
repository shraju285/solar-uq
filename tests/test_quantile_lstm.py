"""Tests for the Stage 6 quantile LSTM: output shape, non-crossing
quantiles, horizon length, seeded-init reproducibility, the pinball loss
itself, and that inverting the target scaler preserves quantile ordering."""

import numpy as np
import pytest
import torch

from solaruq.models.quantile_lstm import QuantileLSTMForecaster, pinball_loss
from solaruq.utils.scaling import apply_target_scaler, fit_target_scaler, invert_target_scaler

N_LOOKBACK_FEATURES = 10
N_HORIZON_KNOWN_FEATURES = 3
LOOKBACK_STEPS = 24
HORIZON_STEPS = 8
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def make_model(quantiles=QUANTILES):
    return QuantileLSTMForecaster(
        n_lookback_features=N_LOOKBACK_FEATURES,
        n_horizon_known_features=N_HORIZON_KNOWN_FEATURES,
        horizon_steps=HORIZON_STEPS,
        quantiles=quantiles,
        hidden_size=16,
        dropout=0.1,
    )


def random_inputs(batch_size=5, lookback_steps=LOOKBACK_STEPS):
    X_lookback = torch.randn(batch_size, lookback_steps, N_LOOKBACK_FEATURES)
    X_horizon_known = torch.randn(batch_size, HORIZON_STEPS, N_HORIZON_KNOWN_FEATURES)
    return X_lookback, X_horizon_known


def test_rejects_unsorted_quantiles():
    with pytest.raises(ValueError):
        make_model(quantiles=[0.5, 0.1, 0.9])


def test_output_shape_is_batch_horizon_quantiles():
    torch.manual_seed(0)
    model = make_model()
    X_lookback, X_horizon_known = random_inputs(batch_size=5)

    out = model(X_lookback, X_horizon_known)

    assert out.shape == (5, HORIZON_STEPS, len(QUANTILES))


def test_output_horizon_length_is_independent_of_lookback_length():
    torch.manual_seed(0)
    model = make_model()
    for lookback_steps in (1, 24, 48):
        X_lookback, X_horizon_known = random_inputs(batch_size=2, lookback_steps=lookback_steps)
        out = model(X_lookback, X_horizon_known)
        assert out.shape == (2, HORIZON_STEPS, len(QUANTILES))


def test_quantiles_are_non_decreasing_along_the_last_axis():
    # Non-crossing is enforced by construction (cumulative softplus), so this
    # must hold for arbitrary random weights and inputs, not just a trained model.
    torch.manual_seed(123)
    model = make_model()
    X_lookback, X_horizon_known = random_inputs(batch_size=20)

    out = model(X_lookback, X_horizon_known).detach().numpy()

    diffs = np.diff(out, axis=-1)
    assert np.all(diffs >= 0)


def test_seeded_initialization_is_reproducible():
    torch.manual_seed(42)
    model_a = make_model()
    torch.manual_seed(42)
    model_b = make_model()

    X_lookback, X_horizon_known = random_inputs(batch_size=3)
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(X_lookback, X_horizon_known)
        out_b = model_b(X_lookback, X_horizon_known)

    assert torch.equal(out_a, out_b)


def test_pinball_loss_is_zero_for_a_perfect_forecast():
    y_true = torch.randn(4, HORIZON_STEPS)
    y_pred = y_true.unsqueeze(-1).expand(-1, -1, len(QUANTILES))

    loss = pinball_loss(y_true, y_pred, QUANTILES)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pinball_loss_matches_manual_calculation_for_a_single_quantile():
    y_true = torch.tensor([[10.0]])
    y_pred = torch.tensor([[[7.0]]])  # under-predicts by 3
    q = 0.9

    loss = pinball_loss(y_true, y_pred, [q])

    # error = 10 - 7 = 3 > 0, so loss = q * error = 0.9 * 3 = 2.7
    assert loss.item() == pytest.approx(2.7, abs=1e-6)


def test_pinball_loss_penalizes_under_and_over_prediction_asymmetrically_by_quantile():
    y_true = torch.tensor([[10.0]])
    under = torch.tensor([[[7.0]]])  # true - pred = 3 (under-prediction)
    over = torch.tensor([[[13.0]]])  # true - pred = -3 (over-prediction)

    # A low quantile (0.1) should penalize over-prediction more than under-prediction.
    loss_under_q10 = pinball_loss(y_true, under, [0.1]).item()
    loss_over_q10 = pinball_loss(y_true, over, [0.1]).item()
    assert loss_over_q10 > loss_under_q10

    # A high quantile (0.9) should penalize under-prediction more than over-prediction.
    loss_under_q90 = pinball_loss(y_true, under, [0.9]).item()
    loss_over_q90 = pinball_loss(y_true, over, [0.9]).item()
    assert loss_under_q90 > loss_over_q90


def test_inverting_the_target_scaler_preserves_quantile_ordering():
    # Scaler inversion is an increasing affine map (y*std + mean, std > 0), so
    # it must preserve the non-crossing order the model already guarantees.
    torch.manual_seed(7)
    model = make_model()
    X_lookback, X_horizon_known = random_inputs(batch_size=10)
    with torch.no_grad():
        scaled_quantiles = model(X_lookback, X_horizon_known).numpy()

    y_train = np.random.default_rng(0).normal(loc=30.0, scale=50.0, size=1000)
    scaler = fit_target_scaler(y_train)

    physical_quantiles = invert_target_scaler(scaled_quantiles, scaler)

    assert np.all(np.diff(physical_quantiles, axis=-1) >= 0)

    # Round-trip sanity: scaling back should recover the original values.
    rescaled = apply_target_scaler(physical_quantiles, scaler)
    np.testing.assert_allclose(rescaled, scaled_quantiles, atol=1e-5)
