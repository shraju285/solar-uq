"""Stage 9: minimal Streamlit dashboard for the solar-uq dissertation artefact.

Presentation layer only. It reads the frozen Stage 4-8 results (results/runs/)
through data_loader.py and never trains, retrains, or recalibrates a model --
see dashboard/README.md for how this relates to the rest of the pipeline.

Run with (from the repository root):
    streamlit run dashboard/app.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    MissingResultsError,
    load_calibrated_test_predictions,
    load_config,
    load_consolidated_results,
)

MODEL_ORDER = ["persistence", "arima_2_0_0", "lstm", "quantile_lstm", "quantile_lstm_calibrated"]
MODEL_LABELS = {
    "persistence": "Persistence",
    "arima_2_0_0": "ARIMA(2,0,0)",
    "lstm": "LSTM (Stage 5)",
    "quantile_lstm": "Quantile LSTM (Stage 6)",
    "quantile_lstm_calibrated": "Quantile LSTM, calibrated (Stage 7)",
}

st.set_page_config(page_title="solar-uq dashboard", layout="wide")

st.title("Uncertainty-Aware Solar Power Forecasting")
st.caption(
    "Final-year BSc Computer Science dissertation artefact (module 6CS007, "
    "University of Wolverhampton). Compares persistence, ARIMA(2,0,0), a "
    "direct multi-step LSTM point forecaster, and a quantile-regression LSTM "
    "with split-conformal calibration, on short-term (15-120 minute) power "
    "forecasting for one PV array (the NIST Canopy array). This page only "
    "reads results already produced by Stages 4-8 -- it performs no training "
    "or calibration itself."
)

try:
    config = load_config()
    results = load_consolidated_results()
    y_true, y_pred_before, y_pred_after, origin_time, quantiles = load_calibrated_test_predictions()
except MissingResultsError as exc:
    st.error(str(exc))
    st.stop()

idx_p10, idx_p50, idx_p90 = quantiles.index(0.10), quantiles.index(0.50), quantiles.index(0.90)
horizon_minutes = results["horizon_minutes"]

# ---------------------------------------------------------------------------
# Headline summary metrics (Stage 8 consolidated test-set numbers)
# ---------------------------------------------------------------------------
st.header("Headline test-set results")
calibrated_entry = results["consolidated_table"]["quantile_lstm_calibrated"]
prob = calibrated_entry["probabilistic"]

cols = st.columns(5)
cols[0].metric("P50 MAE", f"{calibrated_entry['point']['mae']:.2f} kW")
cols[1].metric("P50 RMSE", f"{calibrated_entry['point']['rmse']:.2f} kW")
cols[2].metric(
    "P10-P90 coverage", f"{prob['coverage_p10_p90']:.1%}",
    help="Empirical coverage on the test set; nominal target is 80%.",
)
cols[3].metric("Avg interval width", f"{prob['interval_width_p10_p90']:.2f} kW")
cols[4].metric(
    "Winkler score", f"{prob['winkler_interval_score']:.2f}",
    help="Proper scoring rule combining coverage and sharpness (Gneiting & Raftery, 2007); lower is better.",
)
st.caption(f"Test period: {results['test_period']}. All numbers are for the calibrated quantile LSTM (Stage 7).")

# ---------------------------------------------------------------------------
# Selectable test-period example + time-series plot
# ---------------------------------------------------------------------------
st.header("Example forecast on the test set")
st.write(
    "Pick a window inside the 2018 test period to see the 15-minute-ahead "
    "forecast against what the array actually produced. The shaded band is "
    "the calibrated P10-P90 interval -- see *What the interval means*, below, "
    "before reading it as a simple confidence interval."
)

min_date, max_date = origin_time.min().date(), origin_time.max().date()
default_start = date(2018, 6, 15) if min_date <= date(2018, 6, 15) <= max_date else min_date

col1, col2 = st.columns([2, 1])
with col1:
    period_start = st.date_input(
        "Window start", value=default_start, min_value=min_date, max_value=max_date,
        help="Defaults to a documented, gap-free 4-day block (2018-06-15), the same "
             "illustrative period used in the Stage 6-8 reports.",
    )
with col2:
    window_days = st.slider("Window length (days)", min_value=1, max_value=14, value=4)

period_start_ts = pd.Timestamp(period_start, tz=origin_time.tz)
period_end = period_start_ts + pd.Timedelta(days=window_days)
mask = (origin_time >= period_start_ts) & (origin_time < period_end)

if mask.sum() == 0:
    st.warning(
        "No test-set data falls inside this window -- it may sit in a gap the "
        "Stage 3 preprocessing excluded. Try a different start date or a wider window."
    )
else:
    h_idx = 0  # 15-minute-ahead horizon step: the most directly actionable one to plot
    verify_time = origin_time[mask] + pd.Timedelta(minutes=15 * (h_idx + 1))
    y_t = y_true[mask, h_idx]
    p10 = y_pred_after[mask, h_idx, idx_p10]
    p50 = y_pred_after[mask, h_idx, idx_p50]
    p90 = y_pred_after[mask, h_idx, idx_p90]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=verify_time, y=p90, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=verify_time, y=p10, mode="lines", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(255,161,90,0.3)",
                              name="calibrated P10-P90"))
    fig.add_trace(go.Scatter(x=verify_time, y=y_t, mode="lines", name="actual",
                              line=dict(color="black", width=1.5)))
    fig.add_trace(go.Scatter(x=verify_time, y=p50, mode="lines", name="calibrated P50 forecast",
                              line=dict(color="crimson", width=1.5, dash="dot")))
    fig.update_layout(
        title=f"15-minute-ahead forecast, {period_start} to {period_end.date()}",
        xaxis_title="time", yaxis_title="power (kW)",
        template="plotly_white", height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{int(mask.sum())} forecast origins shown, 15-minute-ahead horizon step only.")

# ---------------------------------------------------------------------------
# Forecast horizon information
# ---------------------------------------------------------------------------
st.header("Forecast horizon")
lookback_hours = config["forecast"]["lookback_steps"] * config["forecast"]["resolution_minutes"] // 60
st.write(
    f"Every forecast is produced in a single pass for all {len(horizon_minutes)} "
    f"horizon steps at once -- {horizon_minutes[0]} to {horizon_minutes[-1]} minutes "
    f"ahead, at {config['forecast']['resolution_minutes']}-minute resolution, from a "
    f"{lookback_hours}-hour lookback window. Accuracy is not uniform across the "
    "horizon; expand below for the full per-horizon breakdown."
)

with st.expander("Per-horizon detail (MAE, P10-P90 coverage, interval width)"):
    mae = results["per_horizon"]["mae"]
    fig_h = go.Figure()
    for model in MODEL_ORDER:
        fig_h.add_trace(go.Scatter(x=horizon_minutes, y=mae[model], mode="lines+markers",
                                    name=MODEL_LABELS[model]))
    fig_h.update_layout(title="MAE vs forecast horizon, all models", xaxis_title="horizon (minutes)",
                         yaxis_title="MAE (kW)", template="plotly_white", height=420)
    st.plotly_chart(fig_h, use_container_width=True)

    coverage = results["per_horizon"]["coverage_p10_p90"]
    width = results["per_horizon"]["interval_width_p10_p90"]
    c1, c2 = st.columns(2)
    with c1:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=horizon_minutes, y=coverage["quantile_lstm"],
                                    mode="lines+markers", name="before calibration"))
        fig_c.add_trace(go.Scatter(x=horizon_minutes, y=coverage["quantile_lstm_calibrated"],
                                    mode="lines+markers", name="after calibration"))
        fig_c.add_hline(y=0.80, line_dash="dash", line_color="gray", annotation_text="nominal (80%)")
        fig_c.update_layout(title="P10-P90 coverage vs horizon", xaxis_title="horizon (minutes)",
                            yaxis_title="coverage", yaxis_range=[0, 1], template="plotly_white", height=380)
        st.plotly_chart(fig_c, use_container_width=True)
    with c2:
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=horizon_minutes, y=width["quantile_lstm"],
                                    mode="lines+markers", name="before calibration"))
        fig_w.add_trace(go.Scatter(x=horizon_minutes, y=width["quantile_lstm_calibrated"],
                                    mode="lines+markers", name="after calibration"))
        fig_w.update_layout(title="P10-P90 interval width vs horizon", xaxis_title="horizon (minutes)",
                            yaxis_title="width (kW)", template="plotly_white", height=380)
        st.plotly_chart(fig_w, use_container_width=True)

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
st.header("Model comparison (test set)")
table = results["consolidated_table"]
rows = []
for model in MODEL_ORDER:
    entry = table[model]
    p = entry["probabilistic"]
    rows.append({
        "Model": MODEL_LABELS[model],
        "MAE (kW)": round(entry["point"]["mae"], 2),
        "RMSE (kW)": round(entry["point"]["rmse"], 2),
        "P10-P90 coverage": f"{p['coverage_p10_p90']:.1%}" if p else "n/a",
        "Avg interval width (kW)": round(p["interval_width_p10_p90"], 2) if p else "n/a",
        "Winkler score": round(p["winkler_interval_score"], 2) if p else "n/a",
        "Avg pinball loss": round(p["avg_pinball_loss"], 2) if p else "n/a",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "\"n/a\" marks metrics that genuinely don't apply: persistence, ARIMA and the "
    "Stage 5 LSTM are point forecasters with no quantile output "
    "(see docs/decisions.md, Stage 8 entry)."
)

# ---------------------------------------------------------------------------
# What the interval means
# ---------------------------------------------------------------------------
st.header("What the P10-P90 interval means")
st.write(
    "The shaded band above is not a generic confidence interval. It comes from "
    "**per-horizon, per-quantile split-conformal calibration** (Stage 7) -- a related "
    "but distinct method from the canonical two-sided Conformalized Quantile "
    "Regression (CQR) of Romano, Patterson & Candes (2019). Each of the 7 forecast "
    "quantiles (P05-P95) is corrected independently, so each one individually carries "
    "a standard split-conformal *marginal* coverage guarantee. The reported ~80% "
    "coverage for the P10-P90 band follows only as a conservative union-bound "
    "consequence of the two one-sided guarantees -- it is not the tighter bound the "
    "symmetric CQR construction would give directly, and the calibration and test "
    "periods are not strictly exchangeable (see *Limitations*). Full derivation: "
    "`docs/methodology.md`; terminology verification: `docs/decisions.md` (Stage 7 entry)."
)

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
st.header("Limitations")
st.markdown(
    "- **Single site, single array.** All results are for one PV array (the NIST "
    "Canopy array) at one location -- nothing here has been tested on another site "
    "or PV technology.\n"
    "- **Fixed test period.** The test set is calendar year 2018 only (a chronological, "
    "non-shuffled split) -- performance on a different year is not evaluated.\n"
    "- **Calibration-set reuse.** Stage 7's calibration reuses the same validation "
    "split Stages 5-6 used for early stopping, and the calibration period (2017 H2) "
    "and test period (2018) are not strictly exchangeable given solar generation's "
    "seasonal cycle -- the coverage claim above is approximate, not rigorous.\n"
    "- **Horizon step 5 (75 minutes ahead) remains the worst-calibrated and widest "
    "step even after calibration** -- reduced, not eliminated.\n"
    "- **33.8% of (window, horizon) pairs needed re-sorting** after independent "
    "per-quantile correction, to restore monotonic quantile order -- a real cost of "
    "this calibration method, not a rare edge case.\n"
    "- **No formal statistical significance test was run** anywhere in this pipeline; "
    "every comparison above is a magnitude, not a significance claim."
)

st.divider()
st.caption(
    "Source: results/runs/{stage4_baselines, stage5_lstm, stage6_quantile_lstm, "
    "stage7_cqr, stage8_final}/. Full write-up: docs/stage8_final_report.md. "
    "This dashboard performs no training, retraining, or recalibration -- see "
    "dashboard/README.md."
)
