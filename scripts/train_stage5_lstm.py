"""Stage 5: train the direct multi-step LSTM point forecaster on Stage 3's
windows, select the model by validation loss (early stopping), and evaluate
it on train/val/test with the unmodified Stage 4 harness.

Target scaling: the LSTM is trained against a standardized copy of `y`
(fit on the train split only); every reported metric is computed after
inverting back to physical kW, so results are directly comparable to the
persistence/ARIMA numbers in results/runs/stage4_baselines/metrics.json.
This scaler is local to this script -- it does not touch Stage 3's own
feature_scaler.json or any Stage 3/4 file.

Run with:  python scripts/train_stage5_lstm.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from solaruq.evaluation.metrics import evaluate_point_forecast
from solaruq.models.lstm import LSTMPointForecaster, count_parameters
from solaruq.utils.config import load_config
from solaruq.utils.scaling import apply_target_scaler, fit_target_scaler, invert_target_scaler

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split(name: str) -> dict:
    data = np.load(REPO_ROOT / "data" / "processed" / "stage3" / f"{name}.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}


def make_loader(d: dict, y_scaled: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(d["X_lookback"], dtype=torch.float32),
        torch.as_tensor(d["X_horizon_known"], dtype=torch.float32),
        torch.as_tensor(y_scaled, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, d: dict, batch_size: int) -> np.ndarray:
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


@torch.no_grad()
def epoch_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total, n = 0.0, 0
    for X_lb, X_hz, y in loader:
        pred = model(X_lb, X_hz)
        total += loss_fn(pred, y).item() * len(y)
        n += len(y)
    return total / n


def main():
    config = load_config()
    seed = config["project"]["seed"]
    lstm_cfg = config["models"]["lstm"]
    horizon_steps = config["forecast"]["horizon_steps"]

    torch.manual_seed(seed)

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    for name, d in splits.items():
        print(f"{name}: {len(d['y']):,} windows")

    target_scaler = fit_target_scaler(splits["train"]["y"])
    y_scaled = {name: apply_target_scaler(d["y"], target_scaler) for name, d in splits.items()}

    n_lookback_features = splits["train"]["X_lookback"].shape[-1]
    n_horizon_known_features = splits["train"]["X_horizon_known"].shape[-1]

    model = LSTMPointForecaster(
        n_lookback_features=n_lookback_features,
        n_horizon_known_features=n_horizon_known_features,
        horizon_steps=horizon_steps,
        hidden_size=lstm_cfg["hidden_size"],
        dropout=lstm_cfg["dropout"],
    )
    n_params = count_parameters(model)
    print(f"\nModel: {model}\nParameters: {n_params:,}")

    train_loader = make_loader(splits["train"], y_scaled["train"], lstm_cfg["batch_size"], shuffle=True)
    val_loader = make_loader(splits["val"], y_scaled["val"], lstm_cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    t_start = time.time()
    for epoch in range(1, lstm_cfg["max_epochs"] + 1):
        model.train()
        for X_lb, X_hz, y in train_loader:
            optimizer.zero_grad()
            pred = model(X_lb, X_hz)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        train_loss = epoch_loss(model, train_loader, loss_fn)
        val_loss = epoch_loss(model, val_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= lstm_cfg["patience"]:
                print(f"early stopping at epoch {epoch} (no improvement for {lstm_cfg['patience']} epochs)")
                break

    training_time = time.time() - t_start
    model.load_state_dict(best_state)
    print(f"\nBest epoch: {best_epoch}, best val_loss (scaled MSE): {best_val_loss:.5f}")
    print(f"Training time: {training_time:.1f}s")

    results = {
        "architecture": {
            "n_lookback_features": n_lookback_features,
            "n_horizon_known_features": n_horizon_known_features,
            "horizon_steps": horizon_steps,
            "hidden_size": lstm_cfg["hidden_size"],
            "num_layers": lstm_cfg["num_layers"],
            "dropout": lstm_cfg["dropout"],
        },
        "training_config": {
            "optimizer": "Adam",
            "learning_rate": lstm_cfg["learning_rate"],
            "batch_size": lstm_cfg["batch_size"],
            "max_epochs": lstm_cfg["max_epochs"],
            "patience": lstm_cfg["patience"],
            "seed": seed,
            "target_scaling": lstm_cfg["target_scaling"],
        },
        "n_parameters": n_params,
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "best_val_loss_scaled_mse": best_val_loss,
        "target_scaler": target_scaler,
        "loss_history": history,
        "metrics": {},
    }

    print("\n--- LSTM evaluation (physical kW, inverted from standardized target) ---")
    for name, d in splits.items():
        y_hat_scaled = predict(model, d, lstm_cfg["batch_size"])
        y_hat = invert_target_scaler(y_hat_scaled, target_scaler)
        metrics = evaluate_point_forecast(d["y"], y_hat)
        results["metrics"][name] = metrics
        print(f"{name}: MAE={metrics['mae']:.4f} kW, RMSE={metrics['rmse']:.4f} kW")

    out_dir = REPO_ROOT / "results" / "runs" / "stage5_lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'metrics.json'}")

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "stage5_lstm.pt")
    print(f"Wrote {models_dir / 'stage5_lstm.pt'}")


if __name__ == "__main__":
    main()
