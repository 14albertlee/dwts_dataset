# Layer 1: Next Experiments

## 0. Starting point

- **Current main model:** `ridge_main_prev_bottom3` (validation 26-32: RMSE 0.692, Spearman 0.682, pair accuracy 0.807).
- **It is effectively tied with `ridge_style_onehot_theme_emotional`.** Its pair-accuracy gain comes almost entirely from season 31 (+0.031). In the other six seasons the change is between -0.001 and +0.006, and RMSE is slightly worse in 4 of 7 seasons. In contrast, style + theme over `phase1` lowered RMSE in all 7 seasons. That is what a real gain looks like here.
- Already tested with no gain: `prev_rank`, `prev_dev`, and their combination (`main_prev_rank`, `main_prev_standing`). Don't re-run them as new experiments.
- Already implemented: LightGBM (`--models lgbm`) and the mixed-effects model (`--models lmm`).

## 1. Rules for every experiment

1. **No leakage.** Features for (season, week) use only earlier weeks, earlier seasons, or schedule-known facts about the target week (theme, dances, announced judges). Never the target week's scores.
2. **Same splits as the pipeline:** rolling origin, test seasons 26-32 (`[splits]` in `config.toml`). Each variant gets its own alpha, tuned on these folds. **Don't touch the holdout (33-35)** until a final candidate is chosen. It has already been looked at several times, so treat it as a sanity check, not proof.
3. **One change at a time,** added to the current main model. Decision order matters (forward selection), so re-check earlier "drop" decisions once the base model changes a lot.
4. **Decision rule (validation is only ~530 rows over 7 seasons):** keep a change only if
   - pooled pair accuracy or RMSE improves, the other does not get worse beyond noise, **and**
   - the improvement has the same sign in at least 5 of 7 seasons (report the per-season paired differences), **and**
   - it is not driven by one season (check the result with the best season removed).

   Prefer the simpler model when a result is borderline. Eight or so experiments on the same folds will produce some false positives by chance.
5. **Record results** with the same metric scripts (`metrics_by_week`, `metrics_by_season`) and keep the predictions file for each experiment.

## 2. Step zero: error analysis (do first, cheap)

Use `predictions_validate.csv` for the current main model to locate the error before adding features:
- residuals by week, by `n_performances` (single vs multi-dance nights), by theme, and by style;
- residuals on guest-judge nights vs regular nights;
- residuals by running-order position (done: see 3.1);
- residuals by pro and by celebrity background (rough grouping);
- the largest misses: are they first weeks after a long gap, team dances, or one-off formats?

Reorder section 3 by what this shows. The order below is the default.

## 3. Experiments, in priority order

### 3.1 Within-night running order (top priority: strongest signal found so far)
- **Source: already in the data.** Every Wikipedia week table states "Couples are listed in the order they performed". The spider yields table rows in page order, and `parse_scores_json_to_tables.py` assigns `performance_id` in that row order. So within (season, week), sorting by `performance_id` gives the running order.
- **Make it explicit.** Don't rely on the implicit `performance_id` order in model code. Add `running_order` (1-based position within the table) and `round` (table index within the week, i.e. first dance, second dance) to `performances` in the parse script. Spot-check a few weeks against Wikipedia, including a multi-dance week and a two-night week. Rows the parser drops (team dances, mismatched score counts) leave gaps, but the relative order is unaffected.
- **Evidence from the data (seasons 1-35, rounds with at least 4 couples; score is the mean per-judge score):**
  - **Producer placement is weak.** The Spearman correlation between running position and the couple's prior standing (rank of `running_mean`) is 0.11 (0.06 in seasons 29+). The last slot does go to stronger couples (mean prior-standing percentile 0.70 vs 0.53 in the middle; the first slot is 0.60).
  - **The order effect survives that control.** Regressing the within-week score deviation on prior standing (`running_mean`, `prev_score`) plus position as a fraction of the night: +0.41 per judge from first to last (season-clustered SE 0.04; single-dance weeks +0.45). The first slot scores -0.22 (single-dance weeks -0.32) despite above-average prior standing. So the shape here is mostly a linear rise with a first-slot penalty, not the J shape of Collins, McKenzie, and Vaughan Williams (*Economics Letters*, 2019).
  - **The current main model misses it.** Out of sample, within-week residuals of `ridge_main_prev_bottom3` rise with position: +0.43 from first to last (SE 0.11). The slope is positive in 6 of 7 validation seasons.
  - Unmeasured information producers might use (e.g., rehearsal quality) can't be ruled out. But the residual test shows a usable predictive signal either way.
- **Features:** `order_frac` (position as a fraction of the round, 0 to 1), plus `order_first` and `order_last`. For multi-dance weeks, compute them within each round and average over the couple's performances. Also try `order_frac**2`.
- **Decision: train on the actual order, score in two modes.** The model is always fit on the actual order, so the other coefficients aren't biased by the order effect. One fitted model serves both modes; only the order features at scoring time differ:
  1. **Pre-show (base rates):** set the order features to their expectation under a random order: `order_frac = 0.5`, and `order_first = order_last = 1 / n_couples` in the round.
  2. **Order announced:** the order is announced at the start of the episode, so the model can be re-run with the actual order before the dances are scored.
  - **Why base rates, not a secondary order model.** Tested on seasons 1-35 (first round of each week, at least 4 couples, n = 2311):
    - Order is barely predictable before the show. Prior standing, last week's rank, and last week's position (with first/last flags) explain R² = 0.033 of `order_frac`. The pattern: stronger couples go later, and last week's closer rarely closes again (logit -1.7).
    - A predicted order would shift the score prediction by about 0.024 per judge (standard deviation across couples), against 0.134 once the actual order is known and a model RMSE of about 0.69. That is not worth a second model.
    - Last week's top scorer closes the show 21% of the time, vs 10% for other couples. That is still only about 0.02 points of expected score.
  - **Evaluation: score every test row in both modes.**
    - **Validation:** report base-rate and announced-order metrics side by side. Tuning (alpha) and keep/drop decisions for all other features use the **base-rate** metrics. That is the harder case and is always available, and one fitted model has to serve both modes. **Exception:** the order features themselves are judged on announced-order metrics, where their value shows up, with the condition that base-rate metrics don't get worse.
    - **Holdout:** report both modes as separate results. Neither is used for selection.
  - **What to expect:** base-rate values are the same for every couple in a round, so in pre-show mode the within-week ranking changes only through the re-estimated coefficients of the other features. Expect small gains there. In announced mode, the in-sample effect (+0.41 per judge from first to last) suggests a real gain.
  - **Pipeline (`predict_upcoming.py`):**
    - Add an optional `order` column to the schedule CSV: the couple's 1-based running position, one value per style in the same `;`-separated layout as `styles` (e.g. `3; 11` for two dances). Positions are within each round, and the round size is the number of couples dancing in it.
    - An empty value, or a missing column, means base rate.
    - Allow only all-or-none per week. If some couples have an order and others don't, stop with an error rather than mix the modes.
    - `build_upcoming_features` fills either the base-rate values or the given order. Print which mode was used, and include it in the output file name.
    - The train/predict feature-match test needs an explicit exception for the order features, plus a test that the base-rate path yields the expected constants.
  - **Uncertainty (Monte Carlo, section 4):** in pre-show mode, sample random permutations of the order in each simulated episode and apply the order coefficients. The unknown order is a real ±0.13 source of spread per couple. In announced mode, use the given order.
- **Also check for Layer 2:** order may affect fan votes too (later performances may be more memorable).
- **Result (validation 26-32, `scripts/experiments/run_order_experiment.py`):** announced-order mode lowers RMSE 0.692 to 0.678 (6 of 7 seasons; -0.013 without the best season; bootstrap 95% CI [-0.025, -0.003]); pair accuracy is flat (+0.001); shuffling the order is worse than no order, so the signal is real. Base-rate mode gives no gain and slightly lower pair accuracy (-0.005, 0 of 7 seasons), so one model for both modes fails the base-rate condition. `order_first`, `order_last` and `order_frac_sq` add nothing over `order_frac`.
- **Decision: keep `order_frac` as a separate announced-order model** (`ridge_main_order`, feature set `main_order`). Pre-show predictions keep using `ridge_main_prev_bottom3`. `predict_upcoming.py` picks the model by whether the schedule has an `order` column. Pro merge note: "Valentin" and "Val" Chmerkovskiy are now one `professional_id`.

### 3.2 Pro partner history
- **Why:** a 2026 small-journal study (XGBoost with SHAP) attributed much of the variation in judge scores to the pro. That is suggestive only. About 90% of pros in seasons 26-35 have earlier-season history, so coverage is good.
- **Use earlier seasons only.** A pro normally has one celebrity per season, so "earlier weeks of the current season" would just repeat that couple's `running_mean`.
- **Encodings (cheapest first):**
  1. `pro_prior_dev`: the pro's mean deviation from the field mean (couple score minus week mean) over all earlier-season couple-weeks, shrunk as `mean * n / (n + k)` with k tuned on the validation folds. No nested model needed. Pros with no history get 0.
  2. `pro_mean_residual`: the same, but using residuals from a pro-free baseline model. Those residuals must be out-of-fold predictions inside each training fold, which is much more work. Do this only if (1) helps.
  3. A random intercept per pro in the existing LMM.
  4. One-hot `professional_id` with Ridge (try a stronger alpha). Pros new to a test season get the all-zero baseline.
- **Confounding:** producers assign pros to celebrities non-randomly, so a pro effect partly reflects partner quality. Report whether it survives with `running_mean` in the model, and look at weeks 2-3 separately. The value is likely to be concentrated there.
- **Implementation:** `build_pro_history` in `feature_engineering.py` (`pro_prior_dev`, `pro_prior_n`; pro per couple from `load_couple_info`), experiment in `scripts/experiments/run_pro_experiment.py` (k and alpha tuned jointly on pooled validation MAE; predictions in `outputs/layer1/pro_experiment_*.csv`).
- **Result (validation 26-32, vs `ridge_main_prev_bottom3` at RMSE 0.692, pooled pair accuracy 0.789; 91% of validation rows have earlier-season pro history):**
  - `pro_prior_dev`: validation MAE improves monotonically as k grows (k = 0 to 160), i.e. the tuning itself pushes toward no feature. At k = 160: RMSE 0.693 (+0.001, better in 3 of 7 seasons), pooled pair accuracy -0.008 (0 of 7 seasons better; bootstrap 95% CI [-0.015, -0.002]). Weeks 2-3 are no better (pair accuracy -0.013).
  - One-hot `professional_id` (alpha re-tuned, picked 100): RMSE 0.695, pair accuracy -0.023 (1 of 7 seasons), Spearman 0.615 vs 0.683. Worse in weeks 2-3 (RMSE +0.041).
  - **Confounding:** the in-sample `pro_prior_dev` coefficient is about 1.5 points per point without the couple's own history and falls to about 0.5 with `prev_score` and `running_mean`. Most of the pro signal is the couple's own level, which the main model already has from week 2 on; what remains does not transfer to new seasons.
- **Decision: drop.** Encoding (2) is skipped because (1) did not help, as planned. (3) (pro random intercept in the LMM) is not run: (1) and (4) already bracket it (pooled shrunk mean vs unshrunk per-pro fixed effects) and both fail. Revisit the pro only for the week-1 model, where the couple has no own history.

### 3.3 Judge panel effects (replaces "judge experience")
- **Why the original idea is weak here:** the core panel barely changes (Bruno Tonioli and Carrie Ann Inaba throughout, Len Goodman to S31, Derek Hough from S29). Panel experience is therefore close to a plain time trend. A season-level shift is also already absorbed by `prev_score` and `running_mean` from the same season. What this model cannot see is a **change in the panel between the history weeks and the target week**.
- **Features (schedule-known; guest judges are announced days before the episode, e.g. the season 32 Wikipedia article):**
  1. `guest_judge`: share of the target week's panel who are not core judges.
  2. `panel_leniency_shift`: leniency of the target week's panel minus the average leniency of the panels in the couple's prior weeks. A judge's leniency is their mean deviation from the other judges on the same performances, computed from training seasons only. Guests with no history get 0.
  3. Optional, compared separately: `panel_experience` (mean prior episodes judged, log-transformed), and a `week x era` interaction in case within-season inflation has steepened (O'Connor and Cheema report rising core-judge scores over seasons).
- **Related alternative:** model each judge's score (judge leniency as a fixed effect) and average the predictions over the announced panel. Try this only if 1-2 show signal.

### 3.4 Celebrity background
- **Data work first:** `celebrities.notability` has about 270 distinct free-text values, most used once. Map them to a few levels with keyword rules in `categories.py`, the same pattern as for themes, and add a test for the mapping.
- **Levels should reflect the likely dance ability, not fame type:** e.g., `movement_trained` (gymnast, figure skater, ice dancer, dancer, Broadway, singer-dancer), `athlete_other`, `actor`, `musician`, `reality_social`, `other`. Merge rare levels.
- **Expectation:** helps in weeks 2-3 at most, since the couple's own scores dominate later. Its main use is probably the week-1 model (MODELING_PLAN §4.6), so build the mapping once and reuse it there.
- **Implementation:** `celebrity_group` / `CELEB_RULES` in `categories.py` (6 levels, first match wins; tested in `tests/test_celebrity_groups.py`), `build_celebrity_features` in `feature_engineering.py` (indicators, reference `actor`), notability from `load_couple_info`. Experiment: `scripts/experiments/run_celebrity_experiment.py`, predictions in `outputs/layer1/celebrity_experiment_*.csv`.
- **Levels (422 celebrities, 267 distinct values):** `movement_trained` 67 (gymnasts, skaters, dancers, Broadway, cheer, wrestlers, fitness instructors, choreographed pop groups such as NSYNC or Fifth Harmony), `athlete_other` 86, `musician` 62, `actor` 150 (incl. comedians), `reality_social` 68 (reality TV, social media, models), `other` 55 (hosts, journalists, chefs, politicians, business). Mean week-1 deviation from the field: +0.63, -0.13, +0.16, -0.06, 0.00, -0.55. The gap fades with weeks but persists (weeks 4+: +0.39 ... -0.36).
- **Result (validation 26-32, vs `ridge_main_prev_bottom3`):**
  - `celeb_onehot`: RMSE 0.692 to 0.689 (5 of 7 seasons; -0.002 without the best season; bootstrap CI [-0.010, +0.003]), but pooled pair accuracy -0.006 (4 of 7) and Spearman 0.683 to 0.664. Weeks 2-3 get slightly worse (dRMSE +0.003, dPair -0.011), against the expectation. The fitted `movement_trained` coefficient is a stable +0.15 per judge across folds.
  - `celeb_onehot_decay` (adds indicators / (week - 1)): RMSE 0.687, pair accuracy -0.007, same pattern.
  - Week 1 alone (no own history; group means from earlier seasons): pair accuracy 0.573 vs 0.5 for a constant, Spearman 0.23, RMSE 0.957 vs 0.975.
- **Decision: drop from the main model** (the RMSE gain is within noise and the ranking metrics get worse). Keep the mapping for the week-1 model, where it has modest signal.

### 3.5 Cheap format features (test as one batch, then prune)
- `n_couples_remaining` (known before the episode).
- `multi_dance`: whether the couple dances more than once that night, and whether the score includes a team or group dance (these share one score across couples).
- `prev_season_mean`: last season's mean score at the same week. Expect little value because same-season history already captures the level. Include it only as a check.
- Keep only the features that pass the decision rule individually after the batch.

### 3.6 LightGBM with the kept features
- Use the existing implementation and grid. Don't add XGBoost unless LightGBM is competitive. On this data size the two will behave about the same.
- Run on the main feature set plus anything kept from 3.1-3.5. Try `professional_id` and style as native categoricals.
- Small data favors Ridge. Judge on pair accuracy and Spearman, not just RMSE, and inspect SHAP directions (week, `running_mean`, `prev_score`).
- **Implementation:** `scripts/experiments/run_lgbm_experiment.py` (existing `LGBMModel` and `[model.lgbm]` grid, params tuned per variant on pooled validation MAE; predictions in `outputs/layer1/lgbm_experiment_*.csv` and `lgbm_order_experiment_validate.csv`). Every variant picked `target = "delta"`, 4 leaves, 400 trees. Style as a categorical is the couple's first dance of the week. SHAP is LightGBM's built-in TreeSHAP (`pred_contrib=True`).
- **Result (validation 26-32):**

  | model | RMSE | pair acc | Spearman |
  |---|---|---|---|
  | `ridge_main` | **0.692** | **0.789** | **0.683** |
  | `lgbm_main` | 0.708 | 0.761 | 0.630 |
  | 50/50 blend | 0.695 | 0.778 | 0.662 |
  | `ridge_main_order` (announced) | 0.678 | 0.790 | 0.687 |
  | `lgbm_order` (announced) | 0.687 | 0.782 | 0.673 |
  | 50/50 blend (announced) | **0.676** | 0.786 | 0.678 |

  - `lgbm_main` vs Ridge: pair accuracy -0.028 (1 of 7 seasons better; bootstrap 95% CI [-0.040, -0.016]), RMSE +0.016 (3 of 7). Worse in both weeks 2-3 and weeks 4+. The blend still loses pair accuracy (1 of 7; CI [-0.018, -0.003]). The announced-order blend's RMSE gain (-0.001) is noise, and its pair accuracy is worse.
  - Native categoricals (vs `lgbm_main`): style as a categorical instead of one-hots changes nothing (4 of 7 either way, ~0 without the best season). `professional_id` is noise (4 of 7; weeks 2-3 RMSE +0.024).
  - Ridge rejects re-tested in trees (vs `lgbm_main`): `pro_prior_dev` (k = 40) passes the rule *within LightGBM* (RMSE 6 of 7, pair accuracy 5 of 7, holds without the best season), and `prev_rank` + `prev_dev` improves pair accuracy in 7 of 7 (+0.009; CI [-0.000, +0.020]) but not RMSE. Celebrity group and all theme categories lower RMSE slightly but not pair accuracy. **None of these closes the gap to Ridge**: the best variant (`lgbm_pro_dev`, 0.700 / 0.770) is still 0.019 behind in pair accuracy.
  - SHAP (last fold, delta target, so `prev_score` also enters 1:1 outside the trees): `prev_score` (negative on the delta: mean reversion), `running_mean` (+), and week (+) dominate, then `style_contemporary` (+) and `prev_bottom3` (+). The directions match the Ridge coefficients; the trees don't find a new structure.
- **Decision: drop LightGBM and the blend.** Ridge stays the main model in both modes. Don't add XGBoost (LightGBM isn't competitive). The tree-only gains from `pro_prior_dev` and `prev_rank`/`prev_dev` don't transfer to the Ridge model (3.2 and section 0), so they don't reopen those decisions.

### 3.7 Recency weighting (lowest priority)
- Weight training rows by `exp(-lambda * (S_test - s))` via `fit(X, y, ridge__sample_weight=w)`. Tune lambda on the validation folds.
- Try a sliding window of the last K seasons first. It is simpler and tests the same idea.
- Weighting reduces the effective sample size. It is worth trying only if per-season residuals show drift.

## 4. Needed for the combination layer (not optional)

The Monte Carlo step (MODELING_PLAN §6.2) needs a predictive distribution, not just a point estimate:
- Estimate the residual spread of the main model by week (and by number of couples remaining). Check the coverage of 50% and 90% intervals on the validation seasons.
- For pre-show simulations, sample a random running order per simulated episode. Once the order is announced, use it (section 3.1).
- Add one metric that targets eliminations: how often the couple with the lowest predicted score is in the bottom 2 of actual judge scores, by week.

## 5. Deliverables per experiment

- Per-season and per-week tables comparing the variant to the current main model and the field-adjusted baseline, including the per-season paired differences used by the decision rule.
- A one-line decision (keep, drop, or revisit) with the reason.
- If kept: the feature set added to `[model.feature_sets]`, `main` updated in `config.toml`, and the feature definitions added to this document.
- Final step, once: score the chosen model on the holdout, then track it on new episodes as they air. That is the only truly unseen test left.
