# Guide for AI agents (and humans) working on this repo

Read this first. It tells you where every calculation happens and how to change
things without breaking the pipeline or the dashboard.

## Pipeline map

```
data/raw/*.xml ──► ponies/xml_io.py ──► ponies/features.py ──► ponies/model.py ──► ponies/pipeline.py ──► docs/data/*.json ──► docs/index.html
 (1) XML files     parse → "starts"     point-in-time         conditional logit    train/test split,        model.json            (4) dashboard
                    table (1 row per     features (2)          softmax per race     backtest, export         evaluation.json       docs/app.js
                    horse per race)                            (3)                  (ponies/evaluate.py)     predictions.json
```

| Stage | File | Key functions | What happens |
|-------|------|---------------|--------------|
| 1. XML | `ponies/xml_io.py` | `parse_card`, `parse_past_performances`, `load_history` | XML is flattened into a DataFrame with `START_COLUMNS`. The formats are in `DATA_SCHEMA.md`. |
| 2. Features | `ponies/features.py` | `FEATURES` (name → description), `build_features`, `_runner_features` | Each feature for a race on date D uses only starts **before** D. |
| 3. Model | `ponies/model.py` | `ConditionalLogit.fit / predict_proba / contributions` | Standardise → linear score → softmax within the race. Contributions are exact per-feature explanations. |
| 3a. Speed-figure model | `ponies/figure_model.py` | `build_figure_features`, `SpeedFigureModel`, `fit_with_holdout` | Ridge regression for a start's speed figure from the horse's EARLIER starts, then a Monte Carlo race simulation gives win probabilities. Trains on single starts, so it works on past-performance files with no complete fields. Same interface as `ConditionalLogit` (`predict_proba`, `contributions`, `feature_names`). [A-26]–[A-30] |
| 3b. Backtest | `ponies/evaluate.py` | `evaluate` | Log loss vs. uniform and market, Brier score, top-pick rate, ROI, calibration. |
| 3c. Odds | `ponies/odds.py` | `to_decimal`, `implied_probabilities` | Odds-string parsing and removal of the overround. |
| 4. Export | `ponies/pipeline.py` | `run`, `race_cards` | Writes the JSON the dashboard reads. |
| Dashboard | `docs/index.html`, `docs/app.js`, `docs/style.css` | – | Static page with no build step, deployed by `.github/workflows/pages.yml`. |
| In-browser predictor | `ponies/browser.py`, `docs/predict.js` | `load_state`, `predict_card` | The "Enter a race" tab turns the form (or an uploaded XML file) into entries XML, then runs **this same Python package** in the browser via Pyodide, loaded from `docs/data/ponies.zip` together with `history.csv` and `model.json`. Ordinary cards use the exported model. A TrackMaster file trains the speed-figure model in the browser on that file's own PP lines. |
| TrackMaster | `ponies/trackmaster.py` | `parse_trackmaster`, `is_trackmaster` | Reads TrackMaster Plus XML into entries plus past-performance history. Used by `pipeline.run` (from `data/raw/trackmaster/`, git-ignored) and by `browser.predict_card` (file upload). Licensed: never commit these files or publish their PP lines. |
| FormFav (Australia) | `ponies/formfav.py`, `scripts/formfav_collect.py`, `scripts/formfav_train.py`, `scripts/formfav_experiments.py` | `race_rows`, `load_store`, `market_odds` | GitHub Action `.github/workflows/formfav.yml` collects cards, bookmaker prices and results into the Actions cache every 3 hours (never committed: licensed), then trains and runs the walk-forward variant comparison. Logs print summary numbers only. The market is replayed to the scheduled start [A-42]. |
| Config | `ponies/config.py` | – | **Every tunable number lives here**, tagged with an assumption ID. |

## Assumptions

* All judgement calls are listed in **`ASSUMPTIONS.md`** with IDs `A-01`, `A-02` and so on.
* The code carries matching `[A-xx]` tags exactly where each one is applied, so `grep -rn "\[A-05\]" ponies/` finds it.
* `tests/test_assumptions_registry.py` enforces the sync. A new judgement call needs a new ID in both places.

## Commands

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py   # (re)generate SYNTHETIC_ sample XML
python scripts/formfav_experiments.py --store data/raw/formfav   # compare FormFav variants vs the market
python -m ponies.pipeline            # XML -> docs/data/*.json, prints backtest summary
python -m pytest -q                  # tests
python -m http.server -d docs 8000   # view dashboard at http://localhost:8000
```

## Rules for changes

1. **No leakage.** Features may use only data dated strictly before the race. `tests/test_features.py::test_point_in_time_no_leakage` guards this.
2. **Probabilities must sum to 1 per race.** Keep the softmax-per-race structure, or renormalise if you swap in another model.
3. **Judge changes by test-set log loss compared with the market** (`docs/data/evaluation.json`), not by ROI on a small sample. Add a row to the change log in `ASSUMPTIONS.md`.
4. **Adding a feature:** add it to `FEATURES` in `features.py` with a one-line description (the dashboard shows it), compute it in `_runner_features`, and tag any judgement call.
5. **Model choice** is [A-30] (`config.WIN_MODEL`). `pipeline.run` backtests both models on the same held-out races (`evaluation.json`: top-level = conditional logit, `speed_figure_model` = the other, `chosen_model` = the one used).
6. **Swapping the model:** any class with `fit(df)`, `predict_proba(df)`, `contributions(df)`, `to_dict()` and a `weights` attribute works in `pipeline.run`. A tree model can return SHAP-style values from `contributions`.
7. **Dashboard JSON contract:** the fields read by `docs/app.js` are produced in `pipeline.race_cards` and `evaluate.evaluate`. If you rename one, update both sides.
8. **Browser compatibility:** `ponies/` runs in Pyodide (Python 3.12, numpy 2.0, pandas 2.2; the version is pinned in `docs/predict.js`). Use only numpy, pandas and the standard library, avoid syntax newer than 3.12, and keep `tests/test_browser.py` passing. That test checks browser predictions equal pipeline predictions.
9. Keep `docs/data/*.json` committed. The Pages workflow regenerates them, but committing them lets the page work locally and on forks.
