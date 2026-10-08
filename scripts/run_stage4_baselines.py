"""Stage 4: evaluate the persistence and ARIMA baselines on Stage 3's
windows, using the shared evaluation harness.

Persistence is evaluated on all three splits (it's free -- no fitting at
all). ARIMA is evaluated on validation and test only: it has no learned
parameters that transfer between windows (it's refit independently at
every origin from that window's own 6-hour lookback -- see
src/solaruq/models/arima.py for why that's the leakage control), so a
train-split number wouldn't tell us anything about generalisation, and
skipping it keeps the run time reasonable (training alone would add
~85,000 more independent ARIMA fits for no evaluative benefit).

Run with:  python scripts/run_stage4_baselines.py
"""

import json
import time
from pathlib import Path

import numpy as np

from solaruq.evaluation.metrics import evaluate_point_forecast
from solaruq.models.arima import select_order, arima_forecast
from solaruq.models.persistence import persistence_forecast
from solaruq.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(name: str) -> dict:
    data = np.load(REPO_ROOT / "data" / "processed" / "stage3" / f"{name}.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}


def main():
    config = load_config()
    horizon_steps = config["forecast"]["horizon_steps"]

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    for name, d in splits.items():
        print(f"{name}: {len(d['y']):,} windows")

    results = {"persistence": {}, "arima": {}}

    print("\n--- Persistence ---")
    for name, d in splits.items():
        y_hat = persistence_forecast(d["X_lookback"], horizon_steps)
        results["persistence"][name] = evaluate_point_forecast(d["y"], y_hat)
        print(f"{name}: MAE={results['persistence'][name]['mae']:.4f} kW, RMSE={results['persistence'][name]['rmse']:.4f} kW")

    print("\n--- ARIMA order selection (train sample) ---")
    train_target = splits["train"]["X_lookback"][:, :, 0]
    t0 = time.time()
    order, order_report = select_order(train_target)
    print(f"Selected order {order} in {time.time() - t0:.1f}s. Candidates compared (mean AIC, fit failures):")
    for cand, (mean_aic, n_fail) in order_report.items():
        print(f"  {cand}: mean_aic={mean_aic:.2f}, failures={n_fail}")

    print("\n--- ARIMA (validation + test only, see module docstring) ---")
    for name in ("val", "test"):
        d = splits[name]
        target = d["X_lookback"][:, :, 0]
        t0 = time.time()
        y_hat, n_fallback = arima_forecast(target, order, horizon_steps)
        dt = time.time() - t0
        metrics = evaluate_point_forecast(d["y"], y_hat)
        metrics["n_fallback_to_persistence"] = n_fallback
        results["arima"][name] = metrics
        print(f"{name}: MAE={metrics['mae']:.4f} kW, RMSE={metrics['rmse']:.4f} kW, "
              f"fallbacks={n_fallback}/{len(target)}, {dt:.1f}s ({1000*dt/len(target):.2f} ms/window)")

    results["arima_order"] = list(order)
    out_dir = REPO_ROOT / "results" / "runs" / "stage4_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
