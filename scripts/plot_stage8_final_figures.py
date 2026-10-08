"""Stage 8: final, publication-oriented figures for the dissertation, built
only from frozen results already saved by Stages 4-7 (results/runs/stage8_final/
consolidated_results.json, plus the raw prediction arrays Stage 6/7 saved).
No retraining, no new inference.

Produces, in results/runs/stage8_final/:
- point_error_vs_horizon.png      MAE vs horizon, all 5 models
- coverage_vs_horizon.png         P10-P90 coverage vs horizon, quantile LSTM
                                   before/after calibration, nominal line
- interval_width_vs_horizon.png   P10-P90 width vs horizon, before/after
- reliability_diagram.png         pooled calibration curve, before/after
- representative_period.png       actual vs point forecast vs P10-P90 band,
                                   one illustrative test period (selection
                                   rule documented in the function below)

Run with:  python scripts/plot_stage8_final_figures.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "runs" / "stage8_final"
STAGE6_DIR = REPO_ROOT / "results" / "runs" / "stage6_quantile_lstm"
STAGE7_DIR = REPO_ROOT / "results" / "runs" / "stage7_cqr"

# Selection rule for the single representative period (documented, not
# cherry-picked post hoc): a mid-summer window was chosen first, for being
# seasonally characteristic (large, clear daily swings rather than winter's
# low near-flat output) -- the SAME window already used in Stage 6's
# illustration, confirmed there to be fully gap-free over 4 consecutive days
# by checking origin_time deltas before ever looking at how any model
# performed on it. Reused here unchanged for Stage 8, not re-picked to flatter
# the calibrated model.
REPRESENTATIVE_START = "2018-06-15"
REPRESENTATIVE_END = "2018-06-19"

MODEL_COLORS = {
    "persistence": "#636EFA",
    "arima_2_0_0": "#EF553B",
    "lstm": "#00CC96",
    "quantile_lstm": "#AB63FA",
    "quantile_lstm_calibrated": "#FFA15A",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "arima_2_0_0": "ARIMA(2,0,0)",
    "lstm": "LSTM (Stage 5)",
    "quantile_lstm": "Quantile LSTM P50 (Stage 6)",
    "quantile_lstm_calibrated": "Quantile LSTM P50, calibrated (Stage 7)",
}


def plot_point_error_vs_horizon(results: dict):
    horizon_minutes = results["horizon_minutes"]
    mae = results["per_horizon"]["mae"]

    fig = go.Figure()
    for model, values in mae.items():
        fig.add_trace(go.Scatter(x=horizon_minutes, y=values, mode="lines+markers",
                                  name=MODEL_LABELS[model], line=dict(color=MODEL_COLORS[model])))
    fig.update_layout(
        title="Point-forecast error (MAE) vs forecast horizon, test set",
        xaxis_title="forecast horizon (minutes)", yaxis_title="MAE (kW)",
        template="plotly_white", width=950, height=600,
    )
    fig.write_image(OUT_DIR / "point_error_vs_horizon.png")
    print(f"Wrote {OUT_DIR / 'point_error_vs_horizon.png'}")


def plot_coverage_vs_horizon(results: dict):
    horizon_minutes = results["horizon_minutes"]
    coverage = results["per_horizon"]["coverage_p10_p90"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizon_minutes, y=coverage["quantile_lstm"], mode="lines+markers",
                              name="Quantile LSTM (Stage 6, before calibration)"))
    fig.add_trace(go.Scatter(x=horizon_minutes, y=coverage["quantile_lstm_calibrated"], mode="lines+markers",
                              name="Calibrated (Stage 7, after)"))
    fig.add_hline(y=0.80, line_dash="dash", line_color="gray", annotation_text="nominal (80%)")
    fig.update_layout(
        title="P10-P90 empirical coverage vs forecast horizon, test set",
        xaxis_title="forecast horizon (minutes)", yaxis_title="coverage", yaxis_range=[0, 1],
        template="plotly_white", width=950, height=600,
    )
    fig.write_image(OUT_DIR / "coverage_vs_horizon.png")
    print(f"Wrote {OUT_DIR / 'coverage_vs_horizon.png'}")


def plot_interval_width_vs_horizon(results: dict):
    horizon_minutes = results["horizon_minutes"]
    width = results["per_horizon"]["interval_width_p10_p90"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizon_minutes, y=width["quantile_lstm"], mode="lines+markers",
                              name="Quantile LSTM (Stage 6, before calibration)"))
    fig.add_trace(go.Scatter(x=horizon_minutes, y=width["quantile_lstm_calibrated"], mode="lines+markers",
                              name="Calibrated (Stage 7, after)"))
    fig.update_layout(
        title="P10-P90 interval width vs forecast horizon, test set",
        xaxis_title="forecast horizon (minutes)", yaxis_title="interval width (kW)",
        template="plotly_white", width=950, height=600,
    )
    fig.write_image(OUT_DIR / "interval_width_vs_horizon.png")
    print(f"Wrote {OUT_DIR / 'interval_width_vs_horizon.png'}")


def plot_reliability_diagram():
    data = np.load(STAGE7_DIR / "test_predictions_calibrated.npz", allow_pickle=True)
    y_true = data["y_true"]
    before = data["y_pred_quantiles_before"]
    after = data["y_pred_quantiles_after"]
    quantiles = data["quantiles"].tolist()

    empirical_before = [float((y_true <= before[..., i]).mean()) for i in range(len(quantiles))]
    empirical_after = [float((y_true <= after[..., i]).mean()) for i in range(len(quantiles))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                              name="perfect calibration"))
    fig.add_trace(go.Scatter(x=quantiles, y=empirical_before, mode="lines+markers",
                              name="Quantile LSTM (Stage 6, before calibration)"))
    fig.add_trace(go.Scatter(x=quantiles, y=empirical_after, mode="lines+markers",
                              name="Calibrated (Stage 7, after)"))
    fig.update_layout(
        title="Calibration (reliability) diagram, test set, all horizons pooled",
        xaxis_title="nominal quantile level", yaxis_title="empirical coverage",
        xaxis_range=[0, 1], yaxis_range=[0, 1],
        template="plotly_white", width=750, height=650,
    )
    fig.write_image(OUT_DIR / "reliability_diagram.png")
    print(f"Wrote {OUT_DIR / 'reliability_diagram.png'}")


def plot_representative_period():
    data = np.load(STAGE7_DIR / "test_predictions_calibrated.npz", allow_pickle=True)
    y_true = data["y_true"]
    after = data["y_pred_quantiles_after"]
    quantiles = data["quantiles"].tolist()
    origin_time = pd.DatetimeIndex(data["origin_time"])

    mask = (origin_time >= REPRESENTATIVE_START) & (origin_time < REPRESENTATIVE_END)
    idx_p10, idx_p50, idx_p90 = quantiles.index(0.10), quantiles.index(0.50), quantiles.index(0.90)

    h_idx = 0  # 15-min-ahead: the most directly actionable horizon for this illustration
    verify_time = origin_time[mask] + pd.Timedelta(minutes=15 * (h_idx + 1))
    y_t = y_true[mask, h_idx]
    p10 = after[mask, h_idx, idx_p10]
    p50 = after[mask, h_idx, idx_p50]
    p90 = after[mask, h_idx, idx_p90]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=verify_time, y=p90, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=verify_time, y=p10, mode="lines", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(255,161,90,0.25)",
                              name="calibrated P10-P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=verify_time, y=y_t, mode="lines", name="actual",
                              line=dict(color="black", width=1.5)))
    fig.add_trace(go.Scatter(x=verify_time, y=p50, mode="lines", name="calibrated P50 forecast",
                              line=dict(color="crimson", width=1.5, dash="dot")))
    fig.update_layout(
        title=dict(
            text=(f"Representative test period, 15-min-ahead forecast ({REPRESENTATIVE_START} to {REPRESENTATIVE_END})"
                  "<br><sup>Period selected for being a gap-free 4-day block, before results were inspected</sup>"),
        ),
        xaxis_title="time", yaxis_title="power (kW)",
        template="plotly_white", width=1050, height=580,
        margin=dict(t=90),
    )
    fig.write_image(OUT_DIR / "representative_period.png")
    print(f"Wrote {OUT_DIR / 'representative_period.png'}")


def main():
    with open(OUT_DIR / "consolidated_results.json") as f:
        results = json.load(f)

    plot_point_error_vs_horizon(results)
    plot_coverage_vs_horizon(results)
    plot_interval_width_vs_horizon(results)
    plot_reliability_diagram()
    plot_representative_period()


if __name__ == "__main__":
    main()
