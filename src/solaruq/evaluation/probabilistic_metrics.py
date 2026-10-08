"""Probabilistic evaluation for a multi-quantile point forecast (Stage 6).

Kept separate from `metrics.py` (the Stage 4 point-forecast harness, reused
unmodified for every model's P50). This module only evaluates what a plain
point forecast cannot answer: how good the quantiles themselves are.

Metrics:
- Pinball (quantile) loss -- the proper scoring rule quantile regression is
  trained to minimize; reported per quantile, per horizon, and overall.
- Empirical coverage of the P10-P90 interval -- the fraction of true values
  that actually fall inside the predicted interval; compared against the
  nominal 80% the interval claims.
- P10-P90 interval width -- how wide (how "sharp") the interval is. Width
  alone is meaningless without coverage (a trivially wide interval always
  covers), so the two are always reported together.
- Interval (Winkler) score for the P10-P90 interval -- a single proper
  scoring rule combining both coverage and sharpness (Gneiting & Raftery,
  2007): it adds a penalty on top of the interval's width whenever the true
  value falls outside it, scaled by how far outside. Included because
  reporting coverage and width as two separate numbers can hide a bad
  trade-off (e.g. a model could "win" on width while badly undercovering);
  the interval score cannot be gamed that way.
"""

import numpy as np

from solaruq.evaluation.metrics import evaluate_point_forecast


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    """Elementwise pinball loss for a single quantile. y_true, y_pred: same shape."""
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1) * error)


def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    """Winkler/interval score for a (1 - alpha) central interval [lower, upper].
    Elementwise; lower values are better. alpha=0.2 for a P10-P90 (80%) interval.
    """
    width = upper - lower
    below = (lower - y_true) * (y_true < lower)
    above = (y_true - upper) * (y_true > upper)
    return width + (2.0 / alpha) * (below + above)


def evaluate_quantile_forecast(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: list[float],
    lower_q: float = 0.10,
    upper_q: float = 0.90,
) -> dict:
    """y_true: (n_windows, horizon_steps), physical units.
    y_pred_quantiles: (n_windows, horizon_steps, n_quantiles), same units, same
    quantile order as `quantiles` (must be sorted ascending and contain lower_q/upper_q).
    """
    if y_true.shape != y_pred_quantiles.shape[:2]:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred_quantiles {y_pred_quantiles.shape[:2]}")
    if list(quantiles) != sorted(quantiles):
        raise ValueError("quantiles must be sorted ascending")
    if lower_q not in quantiles or upper_q not in quantiles:
        raise ValueError(f"lower_q={lower_q} and upper_q={upper_q} must both be in quantiles={quantiles}")

    result = {"n_windows": int(y_true.shape[0]), "quantiles": list(quantiles)}

    # Pinball loss, per quantile / per horizon / overall.
    per_quantile_losses = []  # (n_quantiles, n_windows, horizon_steps)
    for q in quantiles:
        q_idx = quantiles.index(q)
        per_quantile_losses.append(pinball_loss_np(y_true, y_pred_quantiles[..., q_idx], q))
    per_quantile_losses = np.stack(per_quantile_losses, axis=0)

    result["pinball_by_quantile"] = {str(q): float(per_quantile_losses[i].mean()) for i, q in enumerate(quantiles)}
    result["pinball_by_horizon"] = per_quantile_losses.mean(axis=(0, 1)).tolist()
    result["pinball_overall"] = float(per_quantile_losses.mean())

    # P10-P90 coverage, width, interval score.
    lower_idx, upper_idx = quantiles.index(lower_q), quantiles.index(upper_q)
    lower = y_pred_quantiles[..., lower_idx]
    upper = y_pred_quantiles[..., upper_idx]
    nominal_coverage = upper_q - lower_q
    alpha = 1.0 - nominal_coverage

    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    score = interval_score(y_true, lower, upper, alpha)

    result["nominal_coverage"] = nominal_coverage
    result["coverage_overall"] = float(covered.mean())
    result["coverage_by_horizon"] = covered.mean(axis=0).tolist()
    result["interval_width_overall"] = float(width.mean())
    result["interval_width_by_horizon"] = width.mean(axis=0).tolist()
    result["interval_score_overall"] = float(score.mean())
    result["interval_score_by_horizon"] = score.mean(axis=0).tolist()

    return result


def evaluate_p50_point_forecast(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: list[float]) -> dict:
    """Runs the unmodified Stage 4 harness on the P50 slice, so the LSTM's
    median forecast is scored exactly like persistence/ARIMA/the Stage 5 LSTM.
    """
    if 0.50 not in quantiles:
        raise ValueError("quantiles must include 0.50 to evaluate a P50 point forecast")
    p50_idx = quantiles.index(0.50)
    return evaluate_point_forecast(y_true, y_pred_quantiles[..., p50_idx])
