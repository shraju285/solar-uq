"""Stage 6: train the quantile-regression LSTM on Stage 3's windows, select
the model by validation pinball loss (early stopping), and evaluate both
its P50 point forecast (Stage 4 harness, exactly like persistence/ARIMA/the
Stage 5 LSTM) and its full quantile output (coverage, interval width,
pinball loss -- src/solaruq/evaluation/probabilistic_metrics.py).

Does not touch Stage 5's files, results, or saved model. Target scaling
mirrors Stage 5: standardized, fit on the train split only, inverted
(exactly, since inversion is an increasing affine map and therefore
commutes with quantiles) before every metric.

Run with:  python scripts/train_stage6_quantile_lstm.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from solaruq.evaluation.probabilistic_metrics import evaluate_p50_point_forecast, evaluate_quantile_forecast
from solaruq.models.lstm import count_parameters
from solaruq.models.quantile_lstm import QuantileLSTMForecaster, pinball_loss
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
def predict(model, d: dict, batch_size: int) -> np.ndarray:
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
def epoch_loss(model, loader: DataLoader, quantiles: list[float]) -> float:
    model.eval()
    total, n = 0.0, 0
    for X_lb, X_hz, y in loader:
        pred = model(X_lb, X_hz)
        total += pinball_loss(y, pred, quantiles).item() * len(y)
        n += len(y)
    return total / n


def main():
    config = load_config()
    seed = config["project"]["seed"]
    qcfg = config["models"]["quantile_lstm"]
    horizon_steps = config["forecast"]["horizon_steps"]
    quantiles = qcfg["quantiles"]
    lower_q, upper_q = qcfg["coverage_interval"]

    torch.manual_seed(seed)

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    for name, d in splits.items():
        print(f"{name}: {len(d['y']):,} windows")

    target_scaler = fit_target_scaler(splits["train"]["y"])
    y_scaled = {name: apply_target_scaler(d["y"], target_scaler) for name, d in splits.items()}

    n_lookback_features = splits["train"]["X_lookback"].shape[-1]
    n_horizon_known_features = splits["train"]["X_horizon_known"].shape[-1]

    model = QuantileLSTMForecaster(
        n_lookback_features=n_lookback_features,
        n_horizon_known_features=n_horizon_known_features,
        horizon_steps=horizon_steps,
        quantiles=quantiles,
        hidden_size=qcfg["hidden_size"],
        dropout=qcfg["dropout"],
    )
    n_params = count_parameters(model)
    print(f"\nModel: {model}\nParameters: {n_params:,}")

    train_loader = make_loader(splits["train"], y_scaled["train"], qcfg["batch_size"], shuffle=True)
    val_loader = make_loader(splits["val"], y_scaled["val"], qcfg["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=qcfg["learning_rate"])

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    t_start = time.time()
    for epoch in range(1, qcfg["max_epochs"] + 1):
        model.train()
        for X_lb, X_hz, y in train_loader:
            optimizer.zero_grad()
            pred = model(X_lb, X_hz)
            loss = pinball_loss(y, pred, quantiles)
            loss.backward()
            optimizer.step()

        train_loss = epoch_loss(model, train_loader, quantiles)
        val_loss = epoch_loss(model, val_loader, quantiles)
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
            if epochs_without_improvement >= qcfg["patience"]:
                print(f"early stopping at epoch {epoch} (no improvement for {qcfg['patience']} epochs)")
                break

    training_time = time.time() - t_start
    model.load_state_dict(best_state)
    print(f"\nBest epoch: {best_epoch}, best val_loss (scaled pinball): {best_val_loss:.5f}")
    print(f"Training time: {training_time:.1f}s")

    results = {
        "architecture": {
            "n_lookback_features": n_lookback_features,
            "n_horizon_known_features": n_horizon_known_features,
            "horizon_steps": horizon_steps,
            "hidden_size": qcfg["hidden_size"],
            "num_layers": qcfg["num_layers"],
            "dropout": qcfg["dropout"],
            "quantiles": quantiles,
        },
        "training_config": {
            "optimizer": "Adam",
            "learning_rate": qcfg["learning_rate"],
            "batch_size": qcfg["batch_size"],
            "max_epochs": qcfg["max_epochs"],
            "patience": qcfg["patience"],
            "seed": seed,
            "target_scaling": qcfg["target_scaling"],
        },
        "n_parameters": n_params,
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "best_val_loss_scaled_pinball": best_val_loss,
        "target_scaler": target_scaler,
        "loss_history": history,
        "p50_metrics": {},
        "quantile_metrics": {},
    }

    print("\n--- Evaluation (physical kW, inverted from standardized target) ---")
    predictions_physical = {}
    for name, d in splits.items():
        y_hat_scaled = predict(model, d, qcfg["batch_size"])
        y_hat = invert_target_scaler(y_hat_scaled, target_scaler)
        predictions_physical[name] = y_hat

        p50_metrics = evaluate_p50_point_forecast(d["y"], y_hat, quantiles)
        results["p50_metrics"][name] = p50_metrics
        print(f"{name} P50: MAE={p50_metrics['mae']:.4f} kW, RMSE={p50_metrics['rmse']:.4f} kW")

        q_metrics = evaluate_quantile_forecast(d["y"], y_hat, quantiles, lower_q=lower_q, upper_q=upper_q)
        results["quantile_metrics"][name] = q_metrics
        print(
            f"{name} P{int(lower_q*100)}-P{int(upper_q*100)}: "
            f"coverage={q_metrics['coverage_overall']:.3f} (nominal {q_metrics['nominal_coverage']:.2f}), "
            f"width={q_metrics['interval_width_overall']:.4f} kW, "
            f"pinball={q_metrics['pinball_overall']:.5f}, "
            f"interval_score={q_metrics['interval_score_overall']:.4f}"
        )

    out_dir = REPO_ROOT / "results" / "runs" / "stage6_quantile_lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'metrics.json'}")

    # Saved for the plotting script: test-split predictions + origin timestamps,
    # so a representative period can be rendered without re-running the model.
    np.savez(
        out_dir / "test_predictions.npz",
        y_true=splits["test"]["y"],
        y_pred_quantiles=predictions_physical["test"],
        origin_time=splits["test"]["origin_time"],
        quantiles=np.array(quantiles),
    )
    print(f"Wrote {out_dir / 'test_predictions.npz'}")

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "stage6_quantile_lstm.pt")
    print(f"Wrote {models_dir / 'stage6_quantile_lstm.pt'}")


if __name__ == "__main__":
    main()
