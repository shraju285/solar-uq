"""Stage 9: tests for the dashboard's data-loading layer (dashboard/data_loader.py).

These use small synthetic files written to tmp_path, not the real frozen
results, so the suite stays green on a clean checkout before anyone has run
the pipeline. Two things specifically checked:
  - the loader raises a clear, actionable error when a frozen file is missing
    (rather than starting the dashboard with a confusing traceback), and
  - the dashboard module has no import of torch or any training script, so
    starting it can never retrain or recalibrate a model as a side effect.

Run with:  pytest
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import data_loader  # noqa: E402

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"


def test_dashboard_source_has_no_training_imports():
    for filename in ("data_loader.py", "app.py"):
        lines = (DASHBOARD_DIR / filename).read_text().splitlines()
        import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
        for forbidden in ("torch", "solaruq.models", "train_stage", "run_stage7_cqr_calibration"):
            assert not any(forbidden in line for line in import_lines), (
                f"{filename} must not import {forbidden!r}"
            )


def test_load_consolidated_results_missing_file_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "RESULTS_DIR", tmp_path)
    with pytest.raises(data_loader.MissingResultsError, match="run_stage8_consolidation.py"):
        data_loader.load_consolidated_results()


def test_load_calibrated_test_predictions_missing_file_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "RESULTS_DIR", tmp_path)
    with pytest.raises(data_loader.MissingResultsError, match="run_stage7_cqr_calibration.py"):
        data_loader.load_calibrated_test_predictions()


def test_load_consolidated_results_parses_synthetic_file(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "RESULTS_DIR", tmp_path)
    stage8_dir = tmp_path / "stage8_final"
    stage8_dir.mkdir()
    payload = {"horizon_minutes": [15, 30], "consolidated_table": {"persistence": {"point": {"mae": 1.0}}}}
    (stage8_dir / "consolidated_results.json").write_text(json.dumps(payload))

    results = data_loader.load_consolidated_results()
    assert results["horizon_minutes"] == [15, 30]


def test_load_calibrated_test_predictions_roundtrips_synthetic_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "RESULTS_DIR", tmp_path)
    stage7_dir = tmp_path / "stage7_cqr"
    stage7_dir.mkdir()

    n, horizon_steps, n_quantiles = 5, 2, 3
    y_true = np.random.rand(n, horizon_steps)
    before = np.random.rand(n, horizon_steps, n_quantiles)
    after = before + 1.0
    origin_time = np.array(
        ["2018-01-01 00:00", "2018-01-01 00:15", "2018-01-01 00:30", "2018-01-01 00:45", "2018-01-01 01:00"],
        dtype=object,
    )
    quantiles = np.array([0.10, 0.50, 0.90])
    np.savez(
        stage7_dir / "test_predictions_calibrated.npz",
        y_true=y_true, y_pred_quantiles_before=before, y_pred_quantiles_after=after,
        origin_time=origin_time, quantiles=quantiles,
    )

    loaded_y_true, loaded_before, loaded_after, loaded_origin_time, loaded_quantiles = (
        data_loader.load_calibrated_test_predictions()
    )
    assert loaded_y_true.shape == (n, horizon_steps)
    assert loaded_before.shape == loaded_after.shape == (n, horizon_steps, n_quantiles)
    assert len(loaded_origin_time) == n
    assert loaded_quantiles == [0.10, 0.50, 0.90]


def test_load_config_has_forecast_section():
    config = data_loader.load_config()
    assert "forecast" in config
    assert "horizon_steps" in config["forecast"]
