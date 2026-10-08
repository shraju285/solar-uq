"""Stage 7: split-conformal calibration of Stage 6's quantile LSTM.

Does not retrain or modify Stage 6 in any way. Loads the frozen Stage 6
model weights (models/stage6_quantile_lstm.pt) purely for a forward-only
inference pass on the validation split (to get calibration-set
predictions Stage 6 never saved) -- the test-set predictions are read
directly from Stage 6's own saved results/runs/stage6_quantile_lstm/test_predictions.npz,
reused byte-for-byte rather than recomputed, so "before calibration" is
provably identical to the approved Stage 6 result.

Calibration set: the full validation split (16,843 windows). Stage 5/6
only ever used it to pick an early-stopping epoch, never to fit model
weights, so it's valid data for this separate, post-hoc calibration step
-- see docs/decisions.md (D-2026-10-08, Stage 7 entry) for why this is
the right call given the project's locked train/val/test structure has
no dedicated calibration split.

Run with:  python scripts/run_stage7_cqr_calibration.py
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from solaruq.evaluation.probabilistic_metrics import evaluate_p50_point_forecast, evaluate_quantile_forecast
from solaruq.models.quantile_lstm import QuantileLSTMForecaster
from solaruq.uncertainty.conformal import apply_conformal_correction, fit_conformal_correction
from solaruq.utils.config import load_config
from solaruq.utils.scaling import apply_target_scaler, invert_target_scaler

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE6_DIR = REPO_ROOT / "results" / "runs" / "stage6_quantile_lstm"


def load_split(name: str) -> dict:
    data = np.load(REPO_ROOT / "data" / "processed" / "stage3" / f"{name}.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}


@torch.no_grad()
def predict_scaled(model, d: dict, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(d["X_lookback"], dtype=torch.float32),
            torch.as_tensor(d["X_horizon_known"], dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = [model(xb, hb).numpy() for xb, hb in loader]
    return np.concatenate(preds, axis=0)


def main():
    config = load_config()
    qcfg = config["models"]["quantile_lstm"]
    horizon_steps = config["forecast"]["horizon_steps"]
    quantiles = qcfg["quantiles"]
    lower_q, upper_q = qcfg["coverage_interval"]

    with open(STAGE6_DIR / "metrics.json") as f:
        stage6_results = json.load(f)
    target_scaler = stage6_results["target_scaler"]
    stage6_test_metrics_recorded = stage6_results["p50_metrics"]["test"]

    # Frozen Stage 6 model -- loaded only for forward-pass inference, never trained.
    model = QuantileLSTMForecaster(
        n_lookback_features=stage6_results["architecture"]["n_lookback_features"],
        n_horizon_known_features=stage6_results["architecture"]["n_horizon_known_features"],
        horizon_steps=horizon_steps,
        quantiles=quantiles,
        hidden_size=qcfg["hidden_size"],
        dropout=qcfg["dropout"],
    )
    model.load_state_dict(torch.load(REPO_ROOT / "models" / "stage6_quantile_lstm.pt"))
    model.eval()

    print("--- Calibration set: validation split (inference only, frozen Stage 6 weights) ---")
    val = load_split("val")
    print(f"val: {len(val['y']):,} windows")
    y_hat_cal_scaled = predict_scaled(model, val, qcfg["batch_size"])
    y_hat_cal = invert_target_scaler(y_hat_cal_scaled, target_scaler)
    y_cal = val["y"]

    print("--- Test predictions: reused byte-for-byte from Stage 6's saved output ---")
    test_data = np.load(STAGE6_DIR / "test_predictions.npz", allow_pickle=True)
    y_test = test_data["y_true"]
    y_hat_test_before = test_data["y_pred_quantiles"]
    saved_quantiles = test_data["quantiles"].tolist()
    assert saved_quantiles == quantiles, "Stage 6's saved quantiles don't match the current config"

    # Sanity check: re-deriving "before" P50 metrics from the reused array must
    # exactly match what Stage 6 itself reported -- proves nothing drifted.
    before_check = evaluate_p50_point_forecast(y_test, y_hat_test_before, quantiles)
    assert math.isclose(before_check["mae"], stage6_test_metrics_recorded["mae"], rel_tol=1e-9), (
        "Reused Stage 6 test predictions do not reproduce Stage 6's own recorded P50 MAE"
    )

    print("\n--- Fitting split-conformal correction (calibration set only) ---")
    correction = fit_conformal_correction(y_cal, y_hat_cal, quantiles)
    print(f"Fit on {correction.n_calibration:,} calibration windows.")
    print("Per-horizon correction at P10/P50/P90 (kW):")
    for h in range(horizon_steps):
        c10 = correction.corrections[h, quantiles.index(0.10)]
        c50 = correction.corrections[h, quantiles.index(0.50)]
        c90 = correction.corrections[h, quantiles.index(0.90)]
        print(f"  horizon {h+1}: P10 {c10:+.3f}, P50 {c50:+.3f}, P90 {c90:+.3f}")

    print("\n--- Applying correction to the (reused, unmodified) test predictions ---")
    y_hat_test_after, n_resorted = apply_conformal_correction(y_hat_test_before, correction)
    n_window_horizon_pairs = len(y_test) * horizon_steps
    print(f"Re-sorted (crossing after correction) in {n_resorted:,} / {n_window_horizon_pairs:,} "
          f"(window, horizon) pairs -- {100*n_resorted/n_window_horizon_pairs:.3f}%")

    results = {
        "method": "split_conformal_per_horizon_per_quantile",
        "calibration_source": "validation_split",
        "n_calibration": correction.n_calibration,
        "corrections_by_horizon_and_quantile": correction.corrections.tolist(),
        "n_resorted_for_monotonicity": n_resorted,
        "n_window_horizon_pairs": n_window_horizon_pairs,
        "fraction_resorted_for_monotonicity": n_resorted / n_window_horizon_pairs,
        "before": {"p50_metrics": {}, "quantile_metrics": {}},
        "after": {"p50_metrics": {}, "quantile_metrics": {}},
    }

    for label, y_hat in (("before", y_hat_test_before), ("after", y_hat_test_after)):
        p50 = evaluate_p50_point_forecast(y_test, y_hat, quantiles)
        qm = evaluate_quantile_forecast(y_test, y_hat, quantiles, lower_q=lower_q, upper_q=upper_q)
        results[label]["p50_metrics"] = p50
        results[label]["quantile_metrics"] = qm
        print(f"\n--- {label} calibration (test) ---")
        print(f"P50: MAE={p50['mae']:.4f} kW, RMSE={p50['rmse']:.4f} kW")
        print(f"P{int(lower_q*100)}-P{int(upper_q*100)}: coverage={qm['coverage_overall']:.3f} "
              f"(nominal {qm['nominal_coverage']:.2f}), width={qm['interval_width_overall']:.4f} kW, "
              f"pinball={qm['pinball_overall']:.5f}, interval_score={qm['interval_score_overall']:.4f}")

    out_dir = REPO_ROOT / "results" / "runs" / "stage7_cqr"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'metrics.json'}")

    np.savez(
        out_dir / "test_predictions_calibrated.npz",
        y_true=y_test,
        y_pred_quantiles_before=y_hat_test_before,
        y_pred_quantiles_after=y_hat_test_after,
        origin_time=test_data["origin_time"],
        quantiles=np.array(quantiles),
    )
    print(f"Wrote {out_dir / 'test_predictions_calibrated.npz'}")


if __name__ == "__main__":
    main()
