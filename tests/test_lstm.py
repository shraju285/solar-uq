"""Tests for the Stage 5 LSTM point forecaster: output tensor shape and
horizon length, seeded-init reproducibility, and that its output is a
drop-in fit for the unmodified Stage 4 evaluation harness."""

import numpy as np
import torch

from solaruq.evaluation.metrics import evaluate_point_forecast
from solaruq.models.lstm import LSTMPointForecaster, count_parameters

N_LOOKBACK_FEATURES = 10
N_HORIZON_KNOWN_FEATURES = 3
LOOKBACK_STEPS = 24
HORIZON_STEPS = 8


def make_model():
    return LSTMPointForecaster(
        n_lookback_features=N_LOOKBACK_FEATURES,
        n_horizon_known_features=N_HORIZON_KNOWN_FEATURES,
        horizon_steps=HORIZON_STEPS,
        hidden_size=16,
        dropout=0.1,
    )


def test_output_shape_matches_batch_and_horizon():
    torch.manual_seed(0)
    model = make_model()
    batch_size = 5
    X_lookback = torch.randn(batch_size, LOOKBACK_STEPS, N_LOOKBACK_FEATURES)
    X_horizon_known = torch.randn(batch_size, HORIZON_STEPS, N_HORIZON_KNOWN_FEATURES)

    out = model(X_lookback, X_horizon_known)

    assert out.shape == (batch_size, HORIZON_STEPS)


def test_output_horizon_length_is_independent_of_lookback_length():
    # The model must directly predict all horizon_steps outputs regardless
    # of how long the lookback sequence it was fed happens to be.
    torch.manual_seed(0)
    model = make_model()
    for lookback_steps in (1, 24, 48):
        X_lookback = torch.randn(2, lookback_steps, N_LOOKBACK_FEATURES)
        X_horizon_known = torch.randn(2, HORIZON_STEPS, N_HORIZON_KNOWN_FEATURES)
        out = model(X_lookback, X_horizon_known)
        assert out.shape == (2, HORIZON_STEPS)


def test_single_forward_pass_produces_the_full_horizon_at_once():
    # "Direct" multi-step: one call to forward() must already contain every
    # horizon step -- there is no per-step recursive loop to find or avoid.
    torch.manual_seed(0)
    model = make_model()
    X_lookback = torch.randn(1, LOOKBACK_STEPS, N_LOOKBACK_FEATURES)
    X_horizon_known = torch.randn(1, HORIZON_STEPS, N_HORIZON_KNOWN_FEATURES)

    out = model(X_lookback, X_horizon_known)

    assert out.shape[-1] == HORIZON_STEPS
    assert torch.isfinite(out).all()


def test_seeded_initialization_is_reproducible():
    torch.manual_seed(42)
    model_a = make_model()
    torch.manual_seed(42)
    model_b = make_model()

    X_lookback = torch.randn(3, LOOKBACK_STEPS, N_LOOKBACK_FEATURES)
    X_horizon_known = torch.randn(3, HORIZON_STEPS, N_HORIZON_KNOWN_FEATURES)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(X_lookback, X_horizon_known)
        out_b = model_b(X_lookback, X_horizon_known)

    assert torch.equal(out_a, out_b)


def test_count_parameters_is_positive_and_matches_manual_sum():
    model = make_model()
    manual_total = sum(p.numel() for p in model.parameters())
    assert count_parameters(model) == manual_total
    assert count_parameters(model) > 0


def test_lstm_output_is_a_drop_in_fit_for_the_stage4_evaluation_harness():
    # Same harness, same shapes, as persistence/ARIMA -- no special-casing
    # for the LSTM's output is allowed.
    torch.manual_seed(0)
    model = make_model()
    model.eval()
    n_windows = 7
    X_lookback = torch.randn(n_windows, LOOKBACK_STEPS, N_LOOKBACK_FEATURES)
    X_horizon_known = torch.randn(n_windows, HORIZON_STEPS, N_HORIZON_KNOWN_FEATURES)
    with torch.no_grad():
        y_hat = model(X_lookback, X_horizon_known).numpy()
    y_true = np.random.default_rng(0).normal(size=(n_windows, HORIZON_STEPS))

    metrics = evaluate_point_forecast(y_true, y_hat)

    assert metrics["n_windows"] == n_windows
    assert len(metrics["mae_by_horizon"]) == HORIZON_STEPS
    assert len(metrics["rmse_by_horizon"]) == HORIZON_STEPS
