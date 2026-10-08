# dashboard/

The Streamlit app, built at Stage 9. It is a presentation layer only: it reads
the frozen results Stages 4-8 already wrote to `results/runs/` (via
`data_loader.py`) and contains no model-specific logic, no training code and
no import of `torch` or `solaruq.models` -- starting it can never retrain or
recalibrate a model.

## Files

- `data_loader.py` -- all file I/O (JSON metrics, `.npz` predictions, the
  YAML config). Every loader raises `MissingResultsError` with the exact
  pipeline script to run if a frozen result isn't there yet, rather than the
  app crashing with a raw traceback.
- `app.py` -- the Streamlit page itself. No business logic beyond reading
  `data_loader.py`'s return values and plotting them.

## How this relates to the experimental pipeline

| Dashboard section | Source |
| --- | --- |
| Title / description | static |
| Headline summary metrics | `results/runs/stage8_final/consolidated_results.json` (calibrated quantile LSTM, Stage 7) |
| Example forecast (selectable period) | `results/runs/stage7_cqr/test_predictions_calibrated.npz` (Stage 7's calibrated test-set predictions) |
| Forecast horizon info + per-horizon charts | `results/runs/stage8_final/consolidated_results.json` |
| Model comparison table | `results/runs/stage8_final/consolidated_results.json` (all 5 models, Stages 4-7) |
| "What the interval means" / Limitations | static text, kept in sync with `docs/methodology.md` and `docs/decisions.md` (Stage 7 terminology entry) |

## Run it

```bash
# from the repository root, venv active, results regenerated through Stage 8
# (see the top-level README.md "Reproducing the pipeline results" section)
streamlit run dashboard/app.py
```

## Tests

`tests/test_dashboard.py` covers `data_loader.py`'s parsing and error-handling
logic against small synthetic files (so the suite passes on a clean checkout
before the pipeline has been run) and statically checks neither `app.py` nor
`data_loader.py` imports `torch` or any training script.
