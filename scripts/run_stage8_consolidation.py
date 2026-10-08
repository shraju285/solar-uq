"""Stage 8: consolidate the frozen Stages 4-7 results into one final
test-set comparison table and per-horizon breakdown. No retraining, no new
inference beyond re-reading what Stages 4-7 already saved to
results/runs/*/metrics.json -- this script only reads and reshapes.

Run with:  python scripts/run_stage8_consolidation.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "runs"
OUT_DIR = RESULTS_DIR / "stage8_final"


def load(stage_dir: str) -> dict:
    with open(RESULTS_DIR / stage_dir / "metrics.json") as f:
        return json.load(f)


def main():
    stage4 = load("stage4_baselines")
    stage5 = load("stage5_lstm")
    stage6 = load("stage6_quantile_lstm")
    stage7 = load("stage7_cqr")

    horizon_minutes = [15 * (h + 1) for h in range(8)]

    # --- Headline test-set table -------------------------------------------------
    persistence_test = stage4["persistence"]["test"]
    arima_test = stage4["arima"]["test"]
    lstm_test = stage5["metrics"]["test"]
    qlstm_p50_test = stage6["p50_metrics"]["test"]
    qlstm_q_test = stage6["quantile_metrics"]["test"]
    cal_p50_test = stage7["after"]["p50_metrics"]
    cal_q_test = stage7["after"]["quantile_metrics"]

    def point_row(m):
        return {"mae": m["mae"], "rmse": m["rmse"]}

    def prob_row(p50, q):
        return {
            "p50_mae": p50["mae"], "p50_rmse": p50["rmse"],
            "coverage_p10_p90": q["coverage_overall"],
            "interval_width_p10_p90": q["interval_width_overall"],
            "winkler_interval_score": q["interval_score_overall"],
            "avg_pinball_loss": q["pinball_overall"],
        }

    consolidated_table = {
        "persistence": {"point": point_row(persistence_test), "probabilistic": None},
        "arima_2_0_0": {"point": point_row(arima_test), "probabilistic": None},
        "lstm": {"point": point_row(lstm_test), "probabilistic": None},
        "quantile_lstm": {
            "point": {"mae": qlstm_p50_test["mae"], "rmse": qlstm_p50_test["rmse"]},
            "probabilistic": prob_row(qlstm_p50_test, qlstm_q_test),
        },
        "quantile_lstm_calibrated": {
            "point": {"mae": cal_p50_test["mae"], "rmse": cal_p50_test["rmse"]},
            "probabilistic": prob_row(cal_p50_test, cal_q_test),
        },
    }

    # --- Per-horizon table ---------------------------------------------------
    per_horizon = {"horizon_minutes": horizon_minutes}
    per_horizon["mae"] = {
        "persistence": persistence_test["mae_by_horizon"],
        "arima_2_0_0": arima_test["mae_by_horizon"],
        "lstm": lstm_test["mae_by_horizon"],
        "quantile_lstm": qlstm_p50_test["mae_by_horizon"],
        "quantile_lstm_calibrated": cal_p50_test["mae_by_horizon"],
    }
    per_horizon["rmse"] = {
        "persistence": persistence_test["rmse_by_horizon"],
        "arima_2_0_0": arima_test["rmse_by_horizon"],
        "lstm": lstm_test["rmse_by_horizon"],
        "quantile_lstm": qlstm_p50_test["rmse_by_horizon"],
        "quantile_lstm_calibrated": cal_p50_test["rmse_by_horizon"],
    }
    per_horizon["coverage_p10_p90"] = {
        "quantile_lstm": qlstm_q_test["coverage_by_horizon"],
        "quantile_lstm_calibrated": cal_q_test["coverage_by_horizon"],
    }
    per_horizon["interval_width_p10_p90"] = {
        "quantile_lstm": qlstm_q_test["interval_width_by_horizon"],
        "quantile_lstm_calibrated": cal_q_test["interval_width_by_horizon"],
    }

    results = {
        "test_period": "2018-01-01 to 2018-12-31 (33,525 windows, 15-min resolution)",
        "horizon_minutes": horizon_minutes,
        "consolidated_table": consolidated_table,
        "per_horizon": per_horizon,
        "source_files": {
            "stage4": "results/runs/stage4_baselines/metrics.json",
            "stage5": "results/runs/stage5_lstm/metrics.json",
            "stage6": "results/runs/stage6_quantile_lstm/metrics.json (+ test_predictions.npz)",
            "stage7": "results/runs/stage7_cqr/metrics.json (+ test_predictions_calibrated.npz)",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "consolidated_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_DIR / 'consolidated_results.json'}")

    # --- Human-readable console summary --------------------------------------
    print("\n=== Consolidated test-set results ===")
    print(f"{'Model':<26}{'MAE':>8}{'RMSE':>8}{'P50 MAE':>10}{'Cov(80%)':>10}{'Width':>9}{'Winkler':>10}{'Pinball':>10}")
    for name, row in consolidated_table.items():
        point = row["point"]
        prob = row["probabilistic"]
        if prob is None:
            print(f"{name:<26}{point['mae']:>8.3f}{point['rmse']:>8.3f}{'--':>10}{'--':>10}{'--':>9}{'--':>10}{'--':>10}")
        else:
            print(f"{name:<26}{point['mae']:>8.3f}{point['rmse']:>8.3f}{prob['p50_mae']:>10.3f}"
                  f"{prob['coverage_p10_p90']:>10.3f}{prob['interval_width_p10_p90']:>9.2f}"
                  f"{prob['winkler_interval_score']:>10.3f}{prob['avg_pinball_loss']:>10.4f}")


if __name__ == "__main__":
    main()
