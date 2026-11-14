# Layer 1 Runbook: judge-score model

Commands only. Run from the repository root. Rationale: [MODELING_PLAN.md](./MODELING_PLAN.md);
experiments: [LAYER1_NEXT_EXPERIMENTS.md](./LAYER1_NEXT_EXPERIMENTS.md). Settings live in
[`config.toml`](../src/dwts_model/config.toml). Outputs go to `outputs/layer1/`.

## 1. Setup

```sh
pip install -r requirements.txt
```

## 2. Scrape and build the database

Needed only when new episodes have aired.

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
```

## 5. Score an upcoming episode

The script scores the week after the last scored week of the latest season in the database, so
re-run step 2 after each episode airs. The printed header shows the season and week it used.

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

These are judge-score probabilities only. They are not elimination probabilities: fan votes come in
with Layer 2 and the combination. The spread comes from the same model's out-of-sample residuals in
earlier seasons, so the pre-show model's spread already includes the unknown running order.

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
accuracy and each model's `p_low_*` (probability of the lowest judge total, using the week's actual
number of judge scores). Add `--save-file` to write `S<season>W<week>_backtest.csv`. Week 1 is not supported.

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
