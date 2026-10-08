"""Controlled ARIMA baseline: one fixed, pre-chosen (p,d,q) order, refit
independently at every forecast origin from only that origin's own 6-hour
lookback window.

Leakage safety here comes directly from how Stage 3 built the windows, not
from anything new: a window's lookback target series already ends exactly
at its forecast origin and never includes a single horizon-step value
(docs/decisions.md, docs/data_audit.md). Forecasting from that series alone
and discarding it afterwards means no origin's forecast can depend on
anything observed after that origin — not other windows, not later parts of
the same split, and (by construction) never the horizon being predicted.
This is why `arima_forecast` only ever takes the lookback target array as
input: there's no `y` parameter to accidentally leak through.

"Controlled" means the order is chosen once, from a small, explicit
candidate list compared by mean AIC on a train sample (see
`select_order`) — not auto-selected per window, which would make 50,000+
independent modelling decisions no one could explain or defend.

Each window's fit is run sequentially, deliberately not in parallel worker
processes: timed empirically in this container, a multiprocessing pool
made per-window fitting *slower* (CPU appears to be quota-limited here
rather than genuinely multi-core, so spawning workers adds contention
instead of throughput) than a plain loop. A plain loop is also simpler.
"""

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

CANDIDATE_ORDERS = [(1, 0, 0), (2, 0, 0), (1, 1, 0), (2, 1, 0), (1, 1, 1)]


def select_order(
    train_lookback_target: np.ndarray,
    candidate_orders: list[tuple[int, int, int]] = CANDIDATE_ORDERS,
    sample_size: int = 300,
    seed: int = 42,
) -> tuple[tuple[int, int, int], dict]:
    """Compare candidate_orders by mean AIC on a random sample of training
    windows' lookback target series. Returns (best_order, report) where
    report maps each order to (mean_aic, n_fit_failures) for the record."""
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(train_lookback_target), size=min(sample_size, len(train_lookback_target)), replace=False)
    sample = train_lookback_target[sample_idx]

    report = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in candidate_orders:
            aics, n_fail = [], 0
            for series in sample:
                try:
                    aics.append(ARIMA(series, order=order).fit().aic)
                except Exception:
                    n_fail += 1
            report[order] = (float(np.mean(aics)) if aics else float("inf"), n_fail)

    best_order = min(report, key=lambda o: report[o][0])
    return best_order, report


def arima_forecast(
    lookback_target: np.ndarray,
    order: tuple[int, int, int],
    horizon_steps: int,
) -> tuple[np.ndarray, int]:
    """lookback_target: shape (n, lookback_steps) -- the target column only.
    Returns (y_hat of shape (n, horizon_steps), n_fallback) where
    n_fallback counts windows where ARIMA couldn't produce a real forecast
    and persistence was used instead.

    Each window is fit and forecast completely independently of every
    other window -- no state is shared, nothing here ever sees a horizon
    (future) value.
    """
    n = len(lookback_target)
    y_hat = np.empty((n, horizon_steps))
    n_fallback = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, series in enumerate(lookback_target):
            # statsmodels' ARIMA rarely raises on bad input -- a NaN or
            # all-identical series just silently produces a degenerate
            # (NaN or all-zero) forecast rather than an exception. Checked
            # explicitly rather than relying on a try/except that mostly
            # wouldn't trigger: any NaN in the window falls back to
            # persistence instead of letting a bad fit through unnoticed.
            # (Not expected to fire on real Stage 3 windows -- gap
            # filtering already excludes NaN from every built window --
            # this is a defensive check, not something this baseline
            # relies on.)
            if np.isnan(series).any():
                y_hat[i] = series[-1]
                n_fallback += 1
                continue
            try:
                model = ARIMA(series, order=order).fit()
                y_hat[i] = model.forecast(steps=horizon_steps)
            except Exception:
                y_hat[i] = series[-1]
                n_fallback += 1
    return y_hat, n_fallback
