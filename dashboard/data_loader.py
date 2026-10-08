"""Read-only access to the frozen Stage 4-8 results, for the Stage 9 dashboard.

Every function here only reads files already written by earlier stages:
results/runs/*/metrics.json, results/runs/stage8_final/consolidated_results.json,
the saved prediction .npz arrays, and configs/config.yaml. Nothing in this
module trains, retrains, or recalibrates a model -- it has no import of
torch, solaruq.models, or any training script, so importing it (or starting
the dashboard) can never trigger model training as a side effect.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solaruq.utils.config import load_config as _load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "runs"


class MissingResultsError(RuntimeError):
    """Raised when a frozen result file the dashboard needs isn't on disk yet."""


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise MissingResultsError(
            f"Missing frozen result file: {path}\n"
            f"Regenerate it by running `python {produced_by}` from the repository "
            "root (see README.md for the full pipeline order), then restart the "
            "dashboard."
        )
    return path


def load_config() -> dict:
    """The project's central config (configs/config.yaml)."""
    return _load_config()


def load_consolidated_results() -> dict:
    """Stage 8's consolidated cross-model table and per-horizon breakdowns."""
    path = _require(
        RESULTS_DIR / "stage8_final" / "consolidated_results.json",
        "scripts/run_stage8_consolidation.py",
    )
    with open(path) as f:
        return json.load(f)


def load_calibrated_test_predictions():
    """The frozen Stage 7 test-set predictions, before and after calibration.

    This reads an array Stage 7 already saved to disk from a one-off,
    already-completed inference+calibration run -- no model is loaded and no
    inference runs here.

    Returns
    -------
    y_true : (n_windows, horizon_steps) ndarray
    y_pred_before : (n_windows, horizon_steps, n_quantiles) ndarray -- Stage 6, uncalibrated
    y_pred_after : (n_windows, horizon_steps, n_quantiles) ndarray -- Stage 7, calibrated
    origin_time : pandas.DatetimeIndex -- the forecast-origin timestamp of each window
    quantiles : list[float]
    """
    path = _require(
        RESULTS_DIR / "stage7_cqr" / "test_predictions_calibrated.npz",
        "scripts/run_stage7_cqr_calibration.py",
    )
    data = np.load(path, allow_pickle=True)
    origin_time = pd.DatetimeIndex(data["origin_time"])
    quantiles = data["quantiles"].tolist()
    return data["y_true"], data["y_pred_quantiles_before"], data["y_pred_quantiles_after"], origin_time, quantiles
