# DWTS Prediction: Modeling Plan

Audience: developers and AI coding assistants (e.g., GitHub Copilot). This document describes WHAT to model and WHY. It does not prescribe file layout.

## 1. Goal

Predict weekly eliminations on *Dancing with the Stars* (US) in two layers, then combine them:

1. **Layer 1:** predict each couple's judge score for an upcoming week from information available before that week.
2. **Layer 2:** estimate fan support (votes are unobserved) for each couple.
3. **Combination:** apply the show's judge+fan elimination rule to produce per-couple elimination probabilities.

Build and validate Layer 1 first. Layers 2 and the combination come later.

## 2. Data

DuckDB database `dwts_db_compat`, schema `main`.

| Table | Key columns |
|---|---|
| `performances` | `performance_id`, `season`, `week`, `week_theme_id`, `couple_id`, `primary_dance_style_id`, `primary_song_id`, `eliminated` (BOOLEAN) |
| `scores` | `score_id`, `performance_id`, `judge_id`, `judge_score` (DOUBLE) |
| `couples` | `couple_id`, `couple`, `celebrity_id`, `professional_id` |
| `celebrities` | `celebrity_id`, `celebrity`, `notability` (VARCHAR) |
| `professionals` | `professional_id`, `professional` |
| `judges` | `judge_id`, `judge` |
| `week_themes` | `week_theme_id`, `week_theme` |
| `primary_dance_styles` | `primary_dance_style_id`, `primary_dance_style` |
| `primary_songs`, `primary_song_artists` | song and artist lookups |

Notes:
- A couple can have several performances in one week (multi-dance or team nights). Aggregate before modeling.
- Panel size varies. Always use per-judge averages, never raw sums.
- Check for `judge_score` placeholders (0 or NULL for absent judges) before averaging.
- Fan votes are NOT in the database. Layer 2 requires external data.

## 3. Global rules (apply to every layer)

- **No leakage.** A feature for (season, week) may use only data from earlier weeks, or from earlier seasons for cross-season features such as pro history. Never use the target week's scores, results, or later seasons.
- **Time-based validation only.** Split by season: train on seasons before S, test on season S (rolling origin). Never shuffle across seasons.
- **Evaluate per week.** Early weeks (2) and late weeks (finale) behave differently. Report metrics by week number, not just overall.
- **Beat baselines before adding complexity.** Every model is compared against the baselines in section 4.3 on the same splits.
- **Save predictions, not only metrics,** so errors can be re-sliced by week, style, or contestant.

## 4. Layer 1: predict judge scores

### 4.1 Target and grain

- Grain: one row per (season, week, couple).
- Target `score`: mean per-judge score for the couple that week (average of per-performance means).
- Exclude each couple's first week from supervised evaluation (no history). Handle week 1 separately (section 4.6).

### 4.2 Features (add in phases; re-evaluate after each phase)

**Phase 1 (proof of concept):**
- `prev_score`: last week's score for the couple.
- `running_mean`: mean of the couple's prior weeks this season.
- `week`: week number within the season (captures grade inflation and general improvement).

**Phase 2:**
- Dance style of the target week (known in advance from the schedule; one-hot or target-encoded; watch for sparse categories).
- Week theme.
- Prior-week RANK of the couple's score among all couples that week (normalized to 0-1 to handle varying field size).
- Prior-week score relative to that week's field mean (deviation, separates skill from inflation).

**Phase 3:**
- Pro partner history: the pro's average residual (actual minus expected) from prior seasons only.
- Celebrity background (`notability` category, grouped to a few levels).
- Judge panel / era indicators, number of couples remaining, double-dance or team-night flags.

### 4.3 Baselines

1. **Last week:** predict `prev_score`.
2. **Last week + drift:** `prev_score` plus the mean week-over-week change for that week number, estimated on training seasons only.
3. **Running mean:** predict `running_mean`.
4. **Field-adjusted:** predict (couple's prior deviation from the field mean) + (expected field mean for the target week, estimated from training seasons). This is the strongest naive baseline and the main bar to clear.

### 4.4 Candidate models (in order)

1. **Ridge / linear regression** on Phase 1 features. Fast, interpretable, good first benchmark.
2. **Linear mixed-effects model** with a random intercept per couple (latent skill) and optionally per pro. Early-week scores are noisy, so shrinking each couple toward the group mean should help most here. Recommended main model.
3. **Gradient boosting** (LightGBM or similar) once Phase 2-3 features exist, to capture interactions (e.g., style x week).
4. **Bounded or ordinal outputs (optional).** Individual judge scores are integers from 1 to 10, so clip predictions to [1, 10] at minimum. A per-judge ordinal model is an option if late-season ceiling effects matter.

### 4.5 Evaluation

- Rolling-origin by season (train on seasons < S, test on S), starting after about 5 training seasons.
- Metrics: MAE and RMSE overall and by week; Spearman rank correlation of predicted vs actual score within each (season, week), averaged by week.
- Late-season caveat: scores cluster at 9-10, so MAE can look excellent while ranking is poor. Always report the rank metric next to MAE.
- Report all baselines and models in the same table.

### 4.6 Week 1

No within-season history exists. Use a separate model based on celebrity background, pro history, dance style, and the historical average first-week score. Expect weak accuracy; evaluate it separately.

## 5. Layer 2: fan support

### 5.1 Problem

Fan votes are a latent (unobserved) quantity. Only eliminations and judge scores are observed. The goal is a per-couple, per-week fan-support signal, ideally ordinal (rank), because the elimination rule may depend only on rank (see section 6).

### 5.2 Inputs needed (not in the current database)

- Social media: follower counts and growth for each contestant, post/engagement volume, mentions during the show week.
- Search interest (e.g., Google Trends) by contestant and week.
- Public discussion volume and sentiment (Reddit, X) around each episode. Volume is likely more informative than sentiment polarity.
- Static fame: pre-show following, celebrity category (athlete, actor, reality star, etc.).
- Optional: the couple's previous-week results (safe vs bottom), as a momentum signal.

### 5.3 Approach

- **Weak supervision from eliminations.** Each week reveals one fact: the eliminated couple had the lowest combined score. When a couple is eliminated despite a high judge rank, that implies low fan rank. Treat these as constraints or labels on latent fan rank.
- **Model type:** a latent-utility ranking model (Bradley-Terry or Plackett-Luce style) where each couple's fan-support utility is a function of the external features above, fit so that implied eliminations match observed ones. Keep it simple: a logistic or ranking model with few features.
- **Evaluation:** by week, using log loss or Brier score on elimination probability, not accuracy. Many weeks have an obvious bottom couple, so accuracy can look good without calibrated probabilities.
- Quick diagnostic before investing heavily: how often does the lowest judge-scored couple get eliminated? That bounds how much Layer 2 can add.

## 6. Combination: elimination probabilities

### 6.1 Rule hypotheses (verify before relying on them)

The rule has changed over time and sources disagree. Treat both as hypotheses to test against historical eliminations.

- **H1 (rank-sum, recent seasons):** judges' scores and fan votes are each converted to rank points (best gets N points among N couples), the two are added, and the lowest total is eliminated; ties are broken in favor of fan votes. Reported by the executive producer in 2025 (secondary source, unverified).
- **H2 (percent-share, earlier seasons):** each couple's share of total judge points is combined with its share of total fan votes.
- A "judges' save" (bottom two, judges choose) existed from season 28 until it was removed in season 32. The data should be segmented by era.

### 6.2 Procedure

1. Encode H1 and H2 as functions: given judge scores and fan support, return the eliminated couple.
2. **Test the rules.** For each historical week, check which fan-rank (or fan-share) orderings are consistent with the actual elimination under each rule, by era. A rule that is frequently inconsistent with eliminations is rejected for that era.
3. **Produce probabilities.** Draw samples from the uncertainty in Layer 1 (predicted judge scores) and Layer 2 (fan support), apply the rule in each sample, and report how often each couple is eliminated (Monte Carlo). Output a per-couple elimination probability for each week.
4. **Evaluate** by week with log loss and Brier score, and check calibration (predicted vs observed elimination frequency in probability bins).

### 6.3 Notes

- Under H1 only ordinal fan information is needed, which is much easier than predicting vote shares.
- Judge-score ties near the top of the season matter under rank-sum; define a consistent tie-handling rule and test the sensitivity of the results to it.
- Weeks with no elimination, double eliminations, and finale formats need explicit handling.

## 7. Milestones

1. Phase 1 features + baselines 1-4 + Ridge, evaluated by week. Confirm leakage tests pass.
2. Add Phase 2 features; compare. Add the mixed-effects model.
3. Add Phase 3 features and gradient boosting; settle on the Layer 1 model.
4. Run the rule-consistency test (section 6.2 step 2) using actual judge scores and eliminations. This needs no fan data, and can start as soon as Layer 1 is working.
5. Collect external fan-signal data and build Layer 2.
6. Combine via Monte Carlo and evaluate calibration.

## 8. Open questions

- Which rule applies in which seasons (H1 vs H2, and when the save applied)?
- How to treat multi-dance and team nights in the Layer 1 target?
- Which external fan-signal sources can be collected historically (not just for the current season)?
