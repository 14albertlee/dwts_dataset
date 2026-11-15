# Modeling Runbook: judge scores and elimination forecast

Commands only. Run from the repository root. Rationale: [MODELING_PLAN.md](./MODELING_PLAN.md);
experiments: [LAYER1_NEXT_EXPERIMENTS.md](./LAYER1_NEXT_EXPERIMENTS.md); elimination model:
[LAYER2_POC_PLAN.md](./LAYER2_POC_PLAN.md). Settings live in
[`config.toml`](../src/dwts_model/config.toml). Outputs go to `outputs/layer1/` and `outputs/layer2/`.

## Weekly workflow (in season)

The loop to run every week. One-time setup (sections 1 and 3) must be done first; the rest of this
document is reference.

| When | Step | Commands |
|---|---|---|
| After an episode airs and its Wikipedia scores are up | 1. Update the data | section 2 (scrape, parse, build the database) |
| Right after step 1 | 2. Retrain the elimination model on all data | `run_rule_check.py`, then `run_layer2.py --fit-final` |
| Before the next episode, once theme and dances are known | 3. Write the schedule CSV, `order` left empty | `predict_upcoming.py --template`, or Claude with [PROMPT_UPCOMING_SCHEDULE.md](./PROMPT_UPCOMING_SCHEDULE.md) |
| Same time | 4. Score pre-show | `predict_upcoming.py --schedule ...` |
| During the episode, once the running order is announced | 5. Fill the `order` column and re-score (better estimate) | same command as step 4 |

```sh
# 1. update the data (after the episode airs)
cd dwts_scrapy
scrapy crawl dwts_scores -O scores.json
cd ..
python scripts/parse_scores_json_to_tables.py
python scripts/build_duckdb.py

# 2. retrain the elimination model on all data
python scripts/run_rule_check.py
python scripts/run_layer2.py --fit-final

# 3. schedule for the next week (e.g. season 35 week 10); fill theme and styles, leave order empty
python scripts/predict_upcoming.py --template prediction_files/inputs/S35W10.csv

# 4. pre-show forecast -> prediction_files/outputs/S35W10_noorder.csv
python scripts/predict_upcoming.py --schedule prediction_files/inputs/S35W10.csv --save-file

# 5. order announced: fill the order column in the same CSV, re-run -> S35W10_order.csv
python scripts/predict_upcoming.py --schedule prediction_files/inputs/S35W10.csv --save-file
```

Checks each week:
- The printed header names the week you expect (the one after the last scored week). If it shows the
  week that just aired, step 1 did not pick it up yet (Wikipedia not updated, or the scrape failed).
- No `Warning: final_params.json covers data through ...` line; if it appears, step 2 was skipped.
- Guest judge or a different panel size: add `--judges N`.
- `p_elim` assumes exactly one couple goes home; read it with care in double-elimination or
  no-elimination weeks, and in weeks where couples dance different numbers of times (the script prints a note).

Not weekly: re-run section 3 (alpha tuning) only after a config or feature change.

## 1. Setup

```sh
pip install -r requirements.txt
```

## 2. Scrape and build the database

Needed only when new episodes have aired (weekly workflow step 1).

```sh
cd dwts_scrapy
scrapy crawl dwts_scores -O scores.json
cd ..
python scripts/parse_scores_json_to_tables.py   # scores.json -> data/raw/*.csv
python scripts/build_duckdb.py                  # data/raw/*.csv -> data/database/dwts_dataset.duckdb
```

## 3. Fit and validate

Re-run after any data or config change. Saves the tuned alphas that steps 4 and 5 use
(`selected_params.json`).

```sh
python scripts/run_layer1.py --mode validate --models ridge
```

Optional:

```sh
python scripts/run_layer1.py --mode validate                    # also LightGBM and LMM (~10 min)
python scripts/score_predictions.py --mode validate             # re-score saved predictions
python scripts/plot_layer1.py --mode validate                   # diagnostics image
python scripts/plot_double_lift.py --mode validate              # double lift chart
```

One-off experiments from `docs/LAYER1_NEXT_EXPERIMENTS.md` live in `scripts/experiments/` (validation seasons only):

```sh
python scripts/experiments/run_order_experiment.py              # running order (3.1)
python scripts/experiments/run_pro_experiment.py                # pro partner history (3.2)
python scripts/experiments/run_celebrity_experiment.py          # celebrity background (3.4)
python scripts/experiments/run_lgbm_experiment.py               # LightGBM with kept features (3.6)
python scripts/experiments/run_lmm_experiment.py                # crossed-effects LMM + score distribution (3.8, ~20 min)
```

## 4. Score the holdout (seasons 33-35)

Only for a final check, never for tuning.

```sh
python scripts/run_layer1.py --mode holdout --confirm-holdout --models ridge --feature-sets main_prev_bottom3 main_order
python scripts/run_rule_check.py
python scripts/report_holdout.py    # -> outputs/reports/holdout/report.md (~5 min)
```

`report_holdout.py` compares both score models and the elimination forecast with baselines on the holdout and
on all out-of-sample seasons 17+: top-1/top-2 tables, double lift and calibration charts, results by season phase
(weeks 2-4, 5-7, 8+), and plain-language findings.

## Effects of the score-model predictors (OLS)

```sh
python scripts/report_ols.py        # -> outputs/reports/ols/report.md
```

OLS on all seasons with the predictors of the pre-show and order-announced Ridge models: effect of each predictor
on the average judge score in points (couple-clustered 95% CIs), dance styles vs the average dance, importance by
predictor group, collinearity, and a comparison with the Ridge coefficients.

## 5. Score an upcoming episode

Reference for weekly workflow steps 2-5. The script scores the week after the last scored week of
the latest season in the database. The printed header shows the season and week it used.

### Retrain on all data (weekly workflow step 2)

```sh
python scripts/run_rule_check.py        # rebuild the Layer 2 couple-week table and usable weeks
python scripts/run_layer2.py --fit-final  # judges-only elimination model on ALL usable weeks
```

`--fit-final` saves `outputs/layer2/final_params.json` (`rho`, weeks used, `data_through`), which
`predict_upcoming.py` reads for `p_elim`. It warns if the database has weeks the file does not cover.
The Ridge score models need no retrain step: they are refit on every scored week before the target
week on each run (alphas from step 3). Re-run step 3 only after a config or feature change.

There are two models:

| Run when | Schedule has `order`? | Model | Output |
|---|---|---|---|
| Theme and dances are known | no | `ridge_main_prev_bottom3` | printed; `--save-file` writes `<schedule>_noorder.csv` |
| Running order is announced | yes | `ridge_main_order` (better), shown next to the pre-show model | printed, both models in the `--backtest` layout; `--save-file` writes `<schedule>_order.csv` |

```sh
# 1. template of couples still competing (optional if Claude writes the CSV from the prompt)
python scripts/predict_upcoming.py --template schedule.csv

# 2. fill in theme and styles (prompt for Claude: PROMPT_UPCOMING_SCHEDULE.md), then score pre-show
python scripts/predict_upcoming.py --schedule schedule.csv

# 3. once the order is announced, fill the order column and re-run the same command
python scripts/predict_upcoming.py --schedule schedule.csv
```

Optional flags: `--theme "Disney Night"` (instead of the column), `--coefs` (print coefficients),
`--judges N` (judges scoring that week; default is last week's panel size, so set it for guest-judge
weeks), `--draws N` (Monte Carlo draws, default 10000), `--save-file [PATH]` (see below).

### Reading the output

Besides the predicted mean score, each couple gets, from the Ridge mean plus a Gaussian spread
(`src/dwts_model/judge_distribution.py`, docs/LAYER1_NEXT_EXPERIMENTS.md 3.8):
- `range80`: 80% range of the couple's mean score (low-high).
- `p_lowest`: probability the couple has the lowest judge total that night. Totals are simulated per
  judge, rounded to integers, and summed, so ties happen at realistic rates; every couple tied for
  lowest counts, so the column can sum to a little over 1.

These are judge-score probabilities only. The spread comes from the same model's out-of-sample residuals in
earlier seasons, so the pre-show model's spread already includes the unknown running order.

- `p_elim`: probability the couple is eliminated, from the judges-only Layer 2 model (the default; fan
  models did not beat it, docs/LAYER2_POC_PLAN.md section 7) applied to the same simulated totals. The
  k-th lowest judge total gets weight `rho^(k-1)` (ties share a position), normalized to sum to 1;
  `rho` (about 0.58) is fit on real eliminations, so it accounts for fans rescuing low scorers on
  average. Unlike `p_lowest`, the column sums to 1. It assumes exactly one couple goes home; fan
  votes are not modeled per couple, and weeks with uneven dance counts get a note (the model was fit
  on even weeks only).

In announced mode and `--backtest` the columns are `p_low_*` and `p_elim_*` per model. For a backtest
or an `--season/--week` run of a week already in `final_params.json`, `rho` is refit on usable weeks
before that week, so the week's own result is never used.

By default predictions are only printed. `--save-file` also writes the CSV: bare, it goes to
`prediction_files/outputs/`; `PATH` can be another directory or a `.csv` file. The file is named
after the input schedule plus the mode, e.g. `prediction_files/inputs/S35W10.csv` gives
`S35W10_noorder.csv` (no `order` column) or `S35W10_order.csv`. Backtests save as
`S<season>W<week>_backtest.csv`. CSVs under `prediction_files/` are gitignored; the `inputs/` and
`outputs/` folders are kept with `.gitkeep`.

```sh
python scripts/predict_upcoming.py --schedule prediction_files/inputs/S35W10.csv --save-file
```

### Backtest an aired week (no CSV)

```sh
python scripts/predict_upcoming.py --backtest --season 34 --week 6
```

Reads the theme, dances and running order from the database, trains only on weeks before it, and
prints both models (pre-show and order-announced) next to the actual scores, with RMSE and pair
accuracy, each model's `p_low_*` (probability of the lowest judge total, using the week's actual
number of judge scores) and `p_elim_*`, and `went_home` for the couple actually eliminated. Add `--save-file` to write `S<season>W<week>_backtest.csv`. Week 1 is not supported.

### Schedule CSV

Pre-show (`order` empty, or the column left out):

```csv
celebrity,professional,theme,styles,order
Ezra Frech,Daniella Karagach,Semifinal,Quickstep; Jive,
Harry Shum Jr.,Jenna Johnson,Semifinal,Quickstep; Paso Doble,
Tatyana Ali,Jan Ravnik,Semifinal,Waltz,
```

Order announced (same file, `order` filled):

```csv
celebrity,professional,theme,styles,order
Ezra Frech,Daniella Karagach,Semifinal,Quickstep; Jive,3; 2
Harry Shum Jr.,Jenna Johnson,Semifinal,Quickstep; Paso Doble,1; 1
Tatyana Ali,Jan Ravnik,Semifinal,Waltz,2
```

Rules:
- `celebrity`: matched by name to the season's couples. Week-1 couples are not supported.
- `theme`: one value for every row.
- `styles`: every dance the couple performs, separated by `;`.
- `order`: the couple's position in each dance's round, in the same `;` layout as `styles`. Round 1 is
  everyone's first dance, round 2 the couples with a second dance. Each round must be numbered
  1..n with no repeats. Fill it for every couple or leave the column out.

## 6. Tests

```sh
python -m pytest tests -q
```
