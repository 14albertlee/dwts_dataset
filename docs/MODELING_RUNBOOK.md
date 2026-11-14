# Layer 1 Runbook: judge-score model

How to prepare the data, train and validate the model, and score a new week. The modeling
rationale is in [MODELING_PLAN.md](./MODELING_PLAN.md). All commands run from the repository root.

## 1. Setup

```sh
pip install -r requirements.txt
```

The model code lives in `src/dwts_model/`. Entry points are in `scripts/`. Outputs go to
`outputs/layer1/` (git-ignored).

## 2. Data prep

There are two stages.

**Build the database (only when the raw data changes).** This is the dataset build described in the
[README](../README.md):

```sh
python scripts/parse_scores_json_to_tables.py   # scores.json -> data/raw/*.csv
python scripts/build_duckdb.py                  # data/raw/*.csv -> data/database/dwts_dataset.duckdb
```

**Model data prep (automatic, runs inside every train/validate/score command).** There is no separate
script. Each run reads the database read-only and builds the modeling table:

| Step | Code |
|---|---|
| One row per (season, week, couple), score = mean per-judge score, performances averaged | `data.py` (`load_weekly_scores`) |
| Theme text and dance style for each performance | `data.py` (`load_performance_context`) |
| Theme category and style groupings (keyword rules, no id tables) | `categories.py` |
| Features: `prev_score`, `running_mean`, `week`, `prev_bottom3`, theme and style indicators; drops each couple's first week | `feature_engineering.py` |

Every feature for a week uses only earlier weeks, plus the schedule-known theme and dances for that week.

## 3. Config

Everything is in [`src/dwts_model/config.toml`](../src/dwts_model/config.toml):

| Section | What it controls |
|---|---|
| `[data]` | `db_path` (default `data/database/dwts_dataset.duckdb`), `schema` |
| `[splits]` | `train_end = 25`, `validation = [26, 32]`, `holdout = [33, 35]`, `short_seasons` |
| `[model]` | `main` (the main model, `ridge_main_prev_bottom3`), `ridge_alphas` grid, score clip range |
| `[model.lgbm]` | LightGBM tuning grid |
| `[model.feature_sets]` | Named feature lists. Each set becomes a Ridge model named `ridge_<set>` |
| `[output]` | `dir` (default `outputs/layer1`) |

To try new features, add a named list under `[model.feature_sets]`. Pass `--config <path>` to any
script to use a different config file.

## 4. Training and validation

There is no saved model file. Ridge fits in seconds, so every run refits from scratch.

```sh
python scripts/run_layer1.py --mode validate --models ridge
```

What it does:
- Rolling origin by season: for each test season S in the validation range, train on all seasons before S and predict S.
- Tunes each model's hyperparameters (Ridge alpha) on the validation folds only, and saves them to `outputs/layer1/selected_params.json`.
- Scores the baselines (last week, last week + drift, running mean, field-adjusted) next to each model.

Options:
- `--models ridge lgbm lmm` picks model types. The default runs all three, and the LMM takes about 10 minutes, so use `--models ridge` for quick runs.
- `--feature-sets phase1 style_onehot ...` restricts which feature sets run.

Outputs in `outputs/layer1/` (suffix `_validate`):
- `predictions_validate.csv`: every test row with each model's prediction.
- `metrics_overall_*.csv`, `metrics_by_season_*.csv`, `metrics_by_week_*.csv`.

Metrics: RMSE, Spearman rank correlation (average ranks for ties), and pairwise accuracy (pairs tied on the actual score skipped; a predicted tie counts half). Pairwise accuracy is reported per week and pooled.

Related scripts (no refitting):

```sh
python scripts/score_predictions.py --mode validate     # re-score saved predictions
python scripts/plot_layer1.py --mode validate           # diagnostics image
python scripts/plot_double_lift.py --mode validate      # double lift chart vs the best baseline
```

### Holdout

The holdout seasons are scored deliberately, using the parameters frozen by the validate run:

```sh
python scripts/run_layer1.py --mode holdout --confirm-holdout --models ridge --feature-sets phase1 style_onehot_theme_emotional main_prev_bottom3
```

The flag is required. The holdout has already been looked at several times, so don't use it for tuning.

## 5. Scoring a new week

```sh
python scripts/predict_upcoming.py --template schedule.csv
# fill in the `theme` and `styles` columns (see below)
python scripts/predict_upcoming.py --schedule schedule.csv
```

### Schedule CSV format

One row per couple still competing. The template provides `celebrity` and `professional`; you fill in `theme` and `styles`.

| Column | Required | Meaning |
|---|---|---|
| `celebrity` | yes | Matched to the season's couples by name (case-insensitive). |
| `professional` | no | Informational only. |
| `theme` | yes, unless `--theme` is given | The week's theme, e.g. `Disney Night`. Use the same value on every row. |
| `styles` | yes | The dances the couple performs that week. Separate several with `;`. |

```csv
celebrity,professional,theme,styles
Ezra Frech,Daniella Karagach,Disney Night,Foxtrot; Team Dance
Another Celeb,Some Pro,Disney Night,Jive
```

- **Multiple styles:** put them in one cell, separated by `;`. Spaces around `;` are trimmed. Quote the cell only if it contains a comma.
- Every couple needs at least one style. The number of styles sets that couple's `n_performances`, so list every dance they will do.
- A style with no dedicated indicator is treated as `other`, and the script prints a note.
- **Theme:** `--theme "Disney Night"` can be passed instead of the column and overrides it. If the column is used, it must hold exactly one distinct value. The theme is mapped to a category by keyword rules, so new theme names work without changes.

- Defaults to the latest season and the week after its last scored week. Use `--season` and `--week` to predict a week "as of" an earlier point (a backtest).
- Uses the main model from the config, trained on all scored weeks before the target week, with the alpha from `selected_params.json`. Run the validate step first.
- Couples are matched by celebrity name. Week-1 couples are not supported (no history).
- Output: a table ranked by predicted score, saved as `outputs/layer1/upcoming_s<season>_w<week>.csv`.

### Do validation and new-week scoring share code?

They share the feature and model code but not the driver or the metrics.

| Piece | Validation (`run_layer1.py`) | New week (`predict_upcoming.py`) |
|---|---|---|
| Feature building | `build_features` on all history | The same `build_features`, called through `build_upcoming_features` in `predict.py`, with placeholder rows for the unaired week |
| Model | `RidgeModel` fit/predict | The same class |
| Hyperparameters | Tuned on validation folds | Read from `selected_params.json` |
| Training rows | Seasons before each test season | All weeks before the target week |
| Metrics (`evaluate.py`) | Yes | No (no actual scores yet) |

A test (`test_upcoming_features_match_training_features`) checks that the placeholder path yields exactly the same features as the training path.

## 6. Tests

```sh
python -m pytest tests -q
```

Covers leakage (features and fitted statistics use only earlier data, splits never mix seasons), the LMM's prediction independence from the target score, the pairwise-accuracy tie handling, and the training/prediction feature match.
