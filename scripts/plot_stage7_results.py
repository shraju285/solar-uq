"""Stage 7: before/after calibration diagnostic plots, from the arrays
run_stage7_cqr_calibration.py saved (results/runs/stage7_cqr/).

Produces, in the same folder:
- coverage_by_horizon_before_after.png
- interval_width_by_horizon_before_after.png
- calibration_plot_before_after.png   reliability diagram, both curves

Run with:  python scripts/plot_stage7_results.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "runs" / "stage7_cqr"


def plot_coverage_and_width(results: dict):
    before = results["before"]["quantile_metrics"]
    after = results["after"]["quantile_metrics"]
    horizon_steps = list(range(1, len(before["coverage_by_horizon"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizon_steps, y=before["coverage_by_horizon"], mode="lines+markers",
                              name="before calibration (Stage 6)"))
    fig.add_trace(go.Scatter(x=horizon_steps, y=after["coverage_by_horizon"], mode="lines+markers",
                              name="after calibration (Stage 7)"))
    fig.add_hline(y=before["nominal_coverage"], line_dash="dash", line_color="gray",
                  annotation_text=f"nominal ({before['nominal_coverage']:.0%})")
    fig.update_layout(
        title="Stage 7: P10-P90 coverage by horizon, before vs after calibration (test)",
        xaxis_title="horizon step (x 15 min)", yaxis_title="coverage", yaxis_range=[0, 1],
        template="plotly_white", width=850, height=550,
    )
    fig.write_image(OUT_DIR / "coverage_by_horizon_before_after.png")
    print(f"Wrote {OUT_DIR / 'coverage_by_horizon_before_after.png'}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=horizon_steps, y=before["interval_width_by_horizon"], mode="lines+markers",
                               name="before calibration (Stage 6)"))
    fig2.add_trace(go.Scatter(x=horizon_steps, y=after["interval_width_by_horizon"], mode="lines+markers",
                               name="after calibration (Stage 7)"))
    fig2.update_layout(
        title="Stage 7: P10-P90 interval width by horizon, before vs after calibration (test)",
        xaxis_title="horizon step (x 15 min)", yaxis_title="interval width (kW)",
        template="plotly_white", width=850, height=550,
    )
    fig2.write_image(OUT_DIR / "interval_width_by_horizon_before_after.png")
    print(f"Wrote {OUT_DIR / 'interval_width_by_horizon_before_after.png'}")


def plot_calibration(quantiles: list[float]):
    data = np.load(OUT_DIR / "test_predictions_calibrated.npz", allow_pickle=True)
    y_true = data["y_true"]
    before = data["y_pred_quantiles_before"]
    after = data["y_pred_quantiles_after"]

    empirical_before = [float((y_true <= before[..., i]).mean()) for i in range(len(quantiles))]
    empirical_after = [float((y_true <= after[..., i]).mean()) for i in range(len(quantiles))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                              name="perfect calibration"))
    fig.add_trace(go.Scatter(x=quantiles, y=empirical_before, mode="lines+markers",
                              name="before calibration (Stage 6)"))
    fig.add_trace(go.Scatter(x=quantiles, y=empirical_after, mode="lines+markers",
                              name="after calibration (Stage 7)"))
    fig.update_layout(
        title="Stage 7: calibration, before vs after (test, all horizons pooled)",
        xaxis_title="nominal quantile level", yaxis_title="empirical coverage (fraction of y_true <= predicted quantile)",
        xaxis_range=[0, 1], yaxis_range=[0, 1],
        template="plotly_white", width=750, height=650,
    )
    fig.write_image(OUT_DIR / "calibration_plot_before_after.png")
    print(f"Wrote {OUT_DIR / 'calibration_plot_before_after.png'}")


def main():
    with open(OUT_DIR / "metrics.json") as f:
        results = json.load(f)

    import yaml
    with open(REPO_ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    quantiles = config["models"]["quantile_lstm"]["quantiles"]

    plot_coverage_and_width(results)
    plot_calibration(quantiles)


if __name__ == "__main__":
    main()
