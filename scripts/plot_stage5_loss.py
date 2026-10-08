"""Stage 5: render the training/validation loss curve saved by
train_stage5_lstm.py (results/runs/stage5_lstm/metrics.json) as a PNG.

Run with:  python scripts/plot_stage5_loss.py
"""

import json
from pathlib import Path

import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics_path = REPO_ROOT / "results" / "runs" / "stage5_lstm" / "metrics.json"
    with open(metrics_path) as f:
        results = json.load(f)

    history = results["loss_history"]
    best_epoch = results["best_epoch"]
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="train loss"))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="validation loss"))
    fig.add_vline(x=best_epoch, line_dash="dash", line_color="gray",
                  annotation_text=f"best epoch ({best_epoch})", annotation_position="top")
    fig.update_layout(
        title="Stage 5 LSTM: training vs validation loss (standardized-target MSE)",
        xaxis_title="epoch",
        yaxis_title="MSE (standardized target)",
        template="plotly_white",
        width=900,
        height=550,
    )

    out_path = metrics_path.parent / "loss_curve.png"
    fig.write_image(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
