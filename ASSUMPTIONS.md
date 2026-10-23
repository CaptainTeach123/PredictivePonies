# Assumptions registry

Every modelling judgement call has an ID. The same ID appears as a `[A-xx]`
tag in the code at the exact place it is applied, so `grep -rn "A-07" ponies/`
takes you straight there. `tests/test_assumptions_registry.py` fails if the
code and this table drift apart.

**Changing an assumption:** edit the constant in `ponies/config.py` (if it has
one), update the row below, re-run `python -m ponies.pipeline`, and record
the before/after backtest numbers in the *Change log* at the bottom.

| ID | Assumption | Where applied | Config key | Why / risk |
|----|-----------|---------------|------------|------------|
| A-01 | Recent form = the horse's last **3** starts | `features._runner_features` | `RECENT_STARTS` | Balances recency against noise. Could try 4–5 or recency-weighted averages. |
| A-02 | Jockey/trainer win rates use a trailing **365-day** window | `features._WindowCounter.rate` | `CONNECTIONS_WINDOW_DAYS` | Captures current form of the connections; shorter windows are noisier. |
| A-03 | Win rates shrunk to prior **0.11** with weight **20** pseudo-starts | `features._shrunk_rate` | `PRIOR_WIN_RATE`, `PRIOR_RIDES` | Stops tiny samples (1/2 = 50%) dominating. Prior ≈ 1 / typical field size. |
| A-04 | "Similar distance" = within **±1 furlong** | `features._runner_features` (`distance_fit`) | `DISTANCE_TOLERANCE_FURLONGS` | Sprint vs route matters; exact-distance matching would leave too little data. |
| A-05 | Track conditions collapse to **wet / dry** (wet = sloppy, muddy, wet fast, heavy, soft, yielding, good) | `features._cond_group` | `WET_CONDITIONS` | "good" is treated as wet, a judgement call. Turf firm counts as dry. |
| A-06 | Layoff days capped at **365** before `log` | `features._runner_features` (`log_layoff`) | `MAX_LAYOFF_DAYS` | Diminishing difference beyond a year. The effect is learned linearly in log days; real effect is likely U-shaped (too fresh / too stale). |
| A-07 | Missing features are imputed with the **median of the runner's own race field** (training-set mean if the whole field lacks the value), plus a `has_history` flag | `model.ConditionalLogit._prepare`, `features.FEATURES` | `IMPUTE_STRATEGY` | A debut horse is treated as "average for this race". This works on any figure scale, so a model trained on one data source isn't thrown off by another source's numbers. It used to be the training mean, which made debut horses look best in races on lower figure scales. |
| A-08 | Chronological split: the **latest 20 %** of races are the test set | `pipeline.chronological_split`, `evaluate` | `TEST_FRACTION` | Avoids training on the future. A rolling/walk-forward backtest would be more robust. |
| A-09 | L2 penalty **0.01** on standardised weights | `model.ConditionalLogit.fit` | `L2_PENALTY` | Mild regularisation; not tuned. Tune by cross-validation on the training period. |
| A-10 | Full-batch gradient descent, lr **0.5** (halved on overshoot), ≤ 2000 iterations, tol 1e-7 | `model.ConditionalLogit.fit` | `LEARNING_RATE`, `MAX_ITERATIONS`, `TOLERANCE` | The loss is convex, so the optimum is unique; these only affect speed and precision. |
| A-11 | Market odds are **not** a model input by default | `features.FEATURES`, `features._runner_features` | `USE_MORNING_LINE_FEATURE` | Keeps the model an independent opinion to compare with the odds. Benter-style blending (model + market) usually scores better; flip the flag to try. |
| A-12 | Implied probabilities from odds are **normalised to sum to 1** per race | `odds.implied_probabilities`, `evaluate` | `NORMALISE_IMPLIED_PROBS` | Removes the takeout. Proportional normalisation ignores the favourite–longshot bias. |
| A-13 | "Value" = model_prob × decimal M/L odds − 1 > **0.10** | `pipeline.race_cards`, `evaluate.evaluate` | `VALUE_EDGE_THRESHOLD` | Dashboard uses **morning line** (known before the race); backtest uses **final** odds. |
| A-14 | Backtest bets are flat **1 unit** at **final odds**, ignoring the odds moving because of our own bets | `evaluate._roi` | `FLAT_STAKE` | Final odds aren't known when you bet, so backtest ROI is optimistic. |
| A-15 | Calibration uses **10** equal-width probability bins | `evaluate.calibration_table` | `CALIBRATION_BINS` | Display only. |
| A-16 | Dead heats: co-winners **split** the win target equally | `model.win_targets` | – | Standard treatment; dead heats are rare. |
| A-17 | Fit features (distance/surface/condition) are **0** when the horse has figures but none under today's condition; **missing** when it has no figures at all | `features._fit` | – | "No evidence" is treated as neutral rather than negative. |
| A-18 | Unknown post position → **middle** of the field (0.5); unknown weight → **race average** | `features._runner_features`, `features.build_features` | – | Neutral defaults. |
| A-19 | Live predictions use a model **refit on all history** (the backtest model is trained on the training split only) | `pipeline.run` | – | Uses the latest form. The reported backtest therefore describes a slightly different (less-trained) model. |
| A-20 | If a start appears in both a result chart and a PP file, the **result chart wins** | `xml_io.load_history` | – | Result charts are the official record. Matching is by `race_id` + `horse_id`, so IDs must be consistent across files. |
| A-21 | **Scratched** runners are dropped and not counted in `field_size` | `xml_io.parse_card` | – | They didn't run. |
| A-22 | Runners with **no prior starts** in the data are never flagged as value bets, and the dashboard labels them "no form data" | `pipeline.race_cards` | – | Their features are imputed as average [A-07], which can make a debut horse at long odds look like a huge edge. |
| A-23 | TrackMaster IDs: horses = name + foaling date; jockeys and trainers = display name | `trackmaster.horse_id`, `trackmaster.person_id` | – | The file has no horse key, and its PP lines have no jockey key. Two jockeys with the same display name would be merged. |
| A-24 | TrackMaster units: distance unit F = hundredths of a furlong, Y = yards, M = metres; age = race year − foaling year | `trackmaster._furlongs`, `trackmaster._age` | – | Checked against `dist_disp` ("1M" = 800.0 F, "350Y" = 350 Y, "6F*" = 1200 M). Ages follow the North American 1 January birthday rule. |
| A-25 | TrackMaster PP lines with field size 0 or finish 0 are skipped, and speed figure 0 means missing | `trackmaster._pp_row` | – | These lines aren't races: they have no finish, odds or figure. |
| A-26 | Speed-figure model = **ridge regression** (penalty **0.05** on standardised features) predicting a start's speed figure from the horse's earlier starts | `figure_model.SpeedFigureModel.fit` | `FIGURE_RIDGE` | Linear and fast enough to train in the browser. Training examples are single starts, so it needs no complete fields. |
| A-27 | "Sprint" = under **7.5 furlongs**; a sprint↔route switch is a feature | `figure_model._row_features` | `SPRINT_MAX_FURLONGS` | The US convention is roughly sprints < 1 mile. |
| A-28 | Predicted-figure uncertainty σ = residual standard deviation for horses with **0, 1–2, 3–5 or 6+** prior starts | `figure_model._bucket`, `SpeedFigureModel.fit` | `FIGURE_SIGMA_BUCKETS` | Lightly raced horses are less predictable. Buckets with fewer than 10 examples use the overall σ. |
| A-29 | Win probability = share of **20,000** simulated races won, each horse's figure drawn from N(predicted, σ), with fixed seed **0** | `figure_model.SpeedFigureModel.predict_proba` | `SIMULATIONS`, `SIMULATION_SEED` | Treats horses as independent, with no pace interaction. The fixed seed makes browser and pipeline results identical. |
| A-30 | Model per card: TrackMaster cards → speed-figure model trained **only on that source's PP lines** (in the browser: the uploaded file's); other cards → whichever model has the lower backtest log loss | `pipeline.run`, `browser.predict_card` | `WIN_MODEL` | Figure scales never mix between sources. A TrackMaster file with fewer than 20 figured starts falls back to the exported model. |
| A-31 | Speed figures are compared only **within a breed group**: Thoroughbred (TB) and Quarter Horse (QH, which includes mixed-breed "MX" races). One speed-figure model is trained per group, and each race uses its own breed's model | `trackmaster.breed_group`, `pipeline.trackmaster_cards` | – | QH figures use the AQHA scale, which isn't comparable with Thoroughbred figures. MX races are run at QH distances. |

## Known limitations / ideas for whoever improves this next

* Speed figures are used as supplied. Figures from different vendors aren't comparable; pick one source.
* No pace or running-style features (early speed, closers), no pedigree, no workouts, no equipment or medication changes.
* Linear-in-features model. Next steps: interactions, a gradient-boosted ranker, or Benter's two-stage model (fundamental model, then blend with log market probability).
* A single train/test split. Walk-forward validation would give confidence intervals on log loss and ROI.

## Change log

| Date | Change | Test log loss (model / market) | Notes |
|------|--------|-------------------------------|-------|
| 2026-10-23 | Initial model | 1.602 / 1.402 (synthetic data) | Baseline. |
| 2026-10-24 | [A-07] imputation → race-field median | unchanged (1.602) | Fixed debut horses being over-rated on TrackMaster data. |
| 2026-10-24 | Added speed-figure model [A-26]–[A-30]; it is now used for predictions | Speed-figure model 1.564 vs conditional logit 1.602 (market 1.402), synthetic data | On the real Finger Lakes TrackMaster file, figure error ±9.5, the same as "average of last 3" (±9.5) and better than "last figure" (±11.3). That is 1,045 starts, too few to pull ahead. |
