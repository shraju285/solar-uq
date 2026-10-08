"""Stage 7: split-conformal calibration of Stage 6's quantile forecasts.

Post-hoc, per-(horizon step, quantile) additive correction fit on a
calibration set, applied to the frozen Stage 6 model's quantile
predictions. No retraining, no new neural architecture -- this only
shifts already-predicted quantile values.

Method (standard split-conformal quantile calibration; the per-pair
interval case is Romano, Patterson & Candes 2019, "Conformalized Quantile
Regression" -- this is the direct extension to every quantile the project
already predicts, not just the two bounding one):

For a fixed horizon step h and quantile level tau, with calibration-set
residuals  r_i = y_i - q_hat_{tau,h}(x_i)  for i = 1..n:

    c_{tau,h} = the k-th smallest r_i,  k = ceil((n + 1) * tau), capped at n

This is the finite-sample-correct empirical quantile used by split
conformal prediction (the "(n+1)" and ceiling, rather than a plain
`np.quantile`, are what give the procedure its distribution-free coverage
guarantee under exchangeability between the calibration and evaluation
sets -- see Vovk, Gammerman & Shafer, 2005; Romano et al., 2019, eq. 2.4).

Calibrated prediction:  q_calibrated_{tau,h}(x) = q_hat_{tau,h}(x) + c_{tau,h}

Correcting every quantile independently can reintroduce crossing even
though Stage 6's raw output was non-crossing by construction (the
corrections c_{tau,h} are not themselves guaranteed monotone in tau) --
`calibrate` re-sorts along the quantile axis afterwards and reports how
often re-sorting actually changed anything, rather than assuming it never
happens.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalCorrection:
    quantiles: list[float]
    horizon_steps: int
    corrections: np.ndarray  # shape (horizon_steps, n_quantiles)
    n_calibration: int


def fit_conformal_correction(
    y_cal: np.ndarray, y_pred_quantiles_cal: np.ndarray, quantiles: list[float]
) -> ConformalCorrection:
    """y_cal: (n, horizon_steps). y_pred_quantiles_cal: (n, horizon_steps, n_quantiles).
    Fits one additive correction per (horizon step, quantile), using only
    this calibration set -- never the evaluation/test set.
    """
    if y_cal.shape != y_pred_quantiles_cal.shape[:2]:
        raise ValueError(f"shape mismatch: y_cal {y_cal.shape} vs y_pred_quantiles_cal {y_pred_quantiles_cal.shape[:2]}")

    n, horizon_steps = y_cal.shape
    n_quantiles = len(quantiles)
    corrections = np.zeros((horizon_steps, n_quantiles))

    for h in range(horizon_steps):
        for qi, tau in enumerate(quantiles):
            residuals = np.sort(y_cal[:, h] - y_pred_quantiles_cal[:, h, qi])
            k = min(int(np.ceil((n + 1) * tau)), n)
            k = max(k, 1)
            corrections[h, qi] = residuals[k - 1]  # k-th smallest, 1-indexed

    return ConformalCorrection(quantiles=list(quantiles), horizon_steps=horizon_steps,
                                corrections=corrections, n_calibration=n)


def apply_conformal_correction(
    y_pred_quantiles: np.ndarray, correction: ConformalCorrection
) -> tuple[np.ndarray, int]:
    """Applies a fitted ConformalCorrection to any set of quantile predictions
    (shape (n, horizon_steps, n_quantiles), same horizon_steps/quantiles the
    correction was fit with). Re-sorts along the quantile axis afterwards to
    restore non-crossing.

    Returns (calibrated_predictions, n_windows_resorted) -- the second value
    is how many (window, horizon) pairs were not already in ascending order
    before sorting, i.e. how often correcting each quantile independently
    actually introduced crossing.
    """
    if y_pred_quantiles.shape[1] != correction.horizon_steps or y_pred_quantiles.shape[2] != len(correction.quantiles):
        raise ValueError(
            f"shape mismatch: predictions {y_pred_quantiles.shape[1:]} vs "
            f"correction (horizon_steps={correction.horizon_steps}, n_quantiles={len(correction.quantiles)})"
        )

    calibrated = y_pred_quantiles + correction.corrections[np.newaxis, :, :]
    was_sorted = np.all(np.diff(calibrated, axis=-1) >= 0, axis=-1)
    n_resorted = int((~was_sorted).sum())
    calibrated_sorted = np.sort(calibrated, axis=-1)
    return calibrated_sorted, n_resorted
