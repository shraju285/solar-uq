"""Stage 6: render the quantile LSTM's diagnostic plots from the arrays
train_stage6_quantile_lstm.py saved (results/runs/stage6_quantile_lstm/).

Produces, in the same folder:
- loss_curve.png             train/val pinball loss vs epoch
- actual_vs_p50_band.png     a representative 4-day test period: actual
                              power vs the P50 forecast and P10-P90 band,
                              at the shortest (15-min) and longest (2h)
                              horizon, to show the interval widening
- coverage_by_horizon.png    empirical P10-P90 coverage per horizon step
                              vs the nominal 80%
- interval_width_by_horizon.png   mean P10-P90 width per horizon step
- calibration_plot.png       reliability diagram: nominal vs empirical
                              quantile level, over the whole test set

Run with:  python scripts/plot_stage6_results.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "runs" / "stage6_quantile_lstm"

# A 4-day window confirmed fully contiguous (15-min steps, no gaps) in the
# test split -- picked purely for a clear, representative illustration.
SAMPLE_START = "2018-06-15"
SAMPLE_END = "2018-06-19"


def plot_loss_curve(results: dict):
    history = results["loss_history"]
    best_epoch = results["best_epoch"]
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="train loss"))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="validation loss"))
    fig.add_vline(x=best_epoch, line_dash="dash", line_color="gray",
                  annotation_text=f"best epoch ({best_epoch})", annotation_position="top")
    fig.update_layout(
        title="Stage 6 quantile LSTM: training vs validation pinball loss (standardized target)",
        xaxis_title="epoch", yaxis_title="mean pinball loss (standardized target)",
        template="plotly_white", width=900, height=550,
    )
    fig.write_image(OUT_DIR / "loss_curve.png")
    print(f"Wrote {OUT_DIR / 'loss_curve.png'}")


def plot_actual_vs_p50_band(quantiles: list[float]):
    data = np.load(OUT_DIR / "test_predictions.npz", allow_pickle=True)
    y_true = data["y_true"]  # (n, horizon_steps)
    y_pred_quantiles = data["y_pred_quantiles"]  # (n, horizon_steps, n_quantiles)
    origin_time = pd.DatetimeIndex(data["origin_time"])

    mask = (origin_time >= SAMPLE_START) & (origin_time < SAMPLE_END)
    idx_p10, idx_p50, idx_p90 = quantiles.index(0.10), quantiles.index(0.50), quantiles.index(0.90)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("15-min-ahead forecast (horizon step 1)",
                                          "2-hour-ahead forecast (horizon step 8)"))

    for row, h_idx in [(1, 0), (2, 7)]:
        # The forecast made at each origin for this horizon step verifies at
        # (origin_time + (h_idx+1) * 15 minutes); shift the x-axis accordingly
        # so "actual" and "forecast" lines compare the same real timestamp.
        verify_time = origin_time[mask] + pd.Timedelta(minutes=15 * (h_idx + 1))
        y_t = y_true[mask, h_idx]
        p10 = y_pred_quantiles[mask, h_idx, idx_p10]
        p50 = y_pred_quantiles[mask, h_idx, idx_p50]
        p90 = y_pred_quantiles[mask, h_idx, idx_p90]

        fig.add_trace(go.Scatter(x=verify_time, y=p90, mode="lines", line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=verify_time, y=p10, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(99,110,250,0.2)",
                                  name="P10-P90" if row == 1 else None,
                                  showlegend=(row == 1), hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=verify_time, y=y_t, mode="lines", name="actual" if row == 1 else None,
                                  line=dict(color="black", width=1.5), showlegend=(row == 1)), row=row, col=1)
        fig.add_trace(go.Scatter(x=verify_time, y=p50, mode="lines", name="P50 forecast" if row == 1 else None,
                                  line=dict(color="crimson", width=1.5, dash="dot"), showlegend=(row == 1)), row=row, col=1)

    fig.update_yaxes(title_text="power (kW)")
    fig.update_layout(
        title=f"Actual vs P50 forecast with P10-P90 band, test period {SAMPLE_START} to {SAMPLE_END}",
        template="plotly_white", width=1000, height=700,
    )
    fig.write_image(OUT_DIR / "actual_vs_p50_band.png")
    print(f"Wrote {OUT_DIR / 'actual_vs_p50_band.png'}")


def plot_coverage_and_width_by_horizon(results: dict):
    test_q = results["quantile_metrics"]["test"]
    horizon_steps = list(range(1, len(test_q["coverage_by_horizon"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizon_steps, y=test_q["coverage_by_horizon"], mode="lines+markers",
                              name="empirical coverage"))
    fig.add_hline(y=test_q["nominal_coverage"], line_dash="dash", line_color="gray",
                  annotation_text=f"nominal ({test_q['nominal_coverage']:.0%})")
    fig.update_layout(
        title="Stage 6: P10-P90 empirical coverage by forecast horizon (test)",
        xaxis_title="horizon step (x 15 min)", yaxis_title="coverage", yaxis_range=[0, 1],
        template="plotly_white", width=800, height=500,
    )
    fig.write_image(OUT_DIR / "coverage_by_horizon.png")
    print(f"Wrote {OUT_DIR / 'coverage_by_horizon.png'}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=horizon_steps, y=test_q["interval_width_by_horizon"], mode="lines+markers"))
    fig2.update_layout(
        title="Stage 6: P10-P90 interval width by forecast horizon (test)",
        xaxis_title="horizon step (x 15 min)", yaxis_title="interval width (kW)",
        template="plotly_white", width=800, height=500,
    )
    fig2.write_image(OUT_DIR / "interval_width_by_horizon.png")
    print(f"Wrote {OUT_DIR / 'interval_width_by_horizon.png'}")


def plot_calibration(quantiles: list[float]):
    data = np.load(OUT_DIR / "test_predictions.npz", allow_pickle=True)
    y_true = data["y_true"]
    y_pred_quantiles = data["y_pred_quantiles"]

    empirical = [float((y_true <= y_pred_quantiles[..., i]).mean()) for i in range(len(quantiles))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                              name="perfect calibration"))
    fig.add_trace(go.Scatter(x=quantiles, y=empirical, mode="lines+markers", name="Stage 6 LSTM"))
    fig.update_layout(
        title="Stage 6: calibration (nominal vs empirical quantile level, test, all horizons)",
        xaxis_title="nominal quantile level", yaxis_title="empirical coverage (fraction of y_true <= predicted quantile)",
        xaxis_range=[0, 1], yaxis_range=[0, 1],
        template="plotly_white", width=700, height=600,
    )
    fig.write_image(OUT_DIR / "calibration_plot.png")
    print(f"Wrote {OUT_DIR / 'calibration_plot.png'}")


def main():
    with open(OUT_DIR / "metrics.json") as f:
        results = json.load(f)
    quantiles = results["architecture"]["quantiles"]

    plot_loss_curve(results)
    plot_actual_vs_p50_band(quantiles)
    plot_coverage_and_width_by_horizon(results)
    plot_calibration(quantiles)


if __name__ == "__main__":
    main()
