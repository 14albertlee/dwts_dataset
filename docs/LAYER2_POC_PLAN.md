# Layer 2 Proof of Concept: Fan Support, Combination, Evaluation

Goal: estimate each couple's probability of elimination in a week by combining Layer 1 judge-score simulations with a minimal fan-support model, and backtest it.

## 1. Setup and assumptions

- **Combination rule by era** (set in `config.toml`, `[layer2]`):
  - **Rank (H1), every season outside the judges' save:** judge scores and fan votes are each converted to rank points; the points are added; the lowest total is eliminated; ties on the total are broken by fan rank. Seasons 3-27 are sometimes reported as percent-share (H2), but that is unconfirmed, the rule check does not rule H1 out there (1 of 161 weeks), and one scheme keeps the model simple and the data in line with the current show.
  - **Judges' save, seasons 28-31:** the H1 bottom two go to the judges, who send home the lower judge total with probability `save_q` (fitted; half each if the totals are equal).
- **Tie convention is a variant to test:** tied judge scores get best-position, average, or worst-position points. Fan ranks are never tied.
- **Weeks used:** single-elimination weeks consistent with their era's rule. Skipped: no-elimination weeks, double eliminations, withdrawal weeks (a withdrawal is not a vote), and weeks where couples danced different numbers of times (mean and total judge scores rank them differently).
- **Uniform mixing:** every forecast is `(1 - eps) * P + eps / N`, so an elimination the model rules out costs a bounded amount. `eps` is fitted with theta.
- **Layer 1 is fixed** while fitting Layer 2.

## 2. Fan-support model (start here)

Working assumptions: fan support is mostly a stable trait of the couple across a season (popular couples stay popular), and it is positively correlated with dance quality (very good dancers rarely get poor fan support). Fan ranks are therefore not independent of judge ranks.

Latent utility for couple i in week w:

`u_iw = beta_skill * skill_iw + beta_surv * survival_surplus_iw + beta_notab[k(i)] + eps_iw`

- `skill_iw`: the couple's season-to-date average judge score through week w - 1, as a within-week z-score (or percentile) among the couples. Slow-moving by design; do not use only last week's score. Week 1 has no history, so set to 0.
- `survival_surplus_iw`: number of weeks before w in which couple i was in the judges' bottom 3 (ties included) and survived. Zero in week 1. It overlaps with `skill`, so check that the coefficients are stable together.
- `beta_notab[k]`: one coefficient per notability group; one reference group fixed at 0.
- `eps_iw`: fixed-scale Gumbel noise (Plackett-Luce), so there is no separate noise parameter.
- **theta** = {beta_skill, beta_surv, beta_notab[k] for non-reference groups}, about 6 numbers. Add a small ridge penalty on the betas.
- Fan rank in a week = ordering of `u_iw` across couples.

## 3. Fitting theta

- **Data per historical week:** actual judge ranks (including real ties) and who was eliminated.
- **Week likelihood:** P(the couple who actually left is eliminated | judge ranks, theta), summed over fan rankings and mixed with the uniform `eps`. Compute by simulation (fixed seed, a few thousand draws), or by exact enumeration of rankings when 8 or fewer couples remain. Sampled weeks report an effective sample size at the fitted theta; it shrinks as the fitted utilities spread out.
- **Objective:** minimize the negative sum of log week-likelihoods (plus the ridge penalty), using Nelder-Mead on theta with `eps` and `save_q` profiled out.
- Fit with a rolling origin: parameters for season S use only earlier seasons, and `survival_surplus` uses only earlier weeks.

## 4. Combining with Layer 1 (forecasting a new week)

Repeat N times (about 10,000):
1. Draw each couple's integer judge total from the Layer 1 integer simulation (ties arise naturally).
2. Convert to judge rank points using the chosen tie convention.
3. Draw fan utilities from the fitted fan model and convert to fan rank points.
4. Add the points; the lowest total is eliminated (tie broken by fan rank). In the judges' save era the bottom two go to the judges.

Output: the share of draws in which each couple is eliminated, which is its elimination probability.

## 5. Evaluation

**Backtest:** for each test week, use theta fit on earlier data only, forecast, and record where the actual eliminated couple falls in the model's ranking of elimination probability.

**Metrics:**
- Top-1, top-2, top-3 hit rates and mean rank of the eliminated couple.
- Log loss and Brier score on the probability assigned to the actual eliminated couple, plus a calibration table.
- Report everything by number of couples remaining, against the random baseline of 1/N.

**Baselines to beat:**
1. Lower judge score goes home: P proportional to rho^(k - 1) for the couple k-th from the bottom of the judge totals (ties share a position), rho fitted on earlier seasons. rho = 0 is "the lowest goes home, ties broken randomly"; rho > 0 keeps the order among the rest, so the ranking metrics are fair to it.
2. Null fan model: every couple has the same expected fan support, so the fan ranking is a uniformly random ordering (all orderings equally likely). This is theta = 0 in the fan model, i.e., the fitted model with the features switched off. Fan ranks are still a strict ordering with no ties; "equal" refers to equal expected support, not tied ranks.
3. **Skill-only fan model:** only `beta_skill` fitted. Because fan support tracks quality, the random-ranking null is too easy a floor; this is the realistic one, and the extra features must beat it.

**Two backtest versions:**
- **Actual judge scores:** isolates Layer 2.
- **Layer 1 simulated scores:** the real forecast; the gap between the two shows what Layer 1 error costs.

Use season-level bootstrap confidence intervals. The sample is a few hundred eliminations at most, so ignore differences of a few percentage points.

## 6. Order of work

The logic: fix the data and the rule first, build one evaluation harness, then climb a ladder of fan models, keeping a rung only if it improves out-of-sample log loss. Every model is always evaluated both ways (actual judge scores and Layer 1 simulated scores), so there is no separate "combine with Layer 1" stage.

1. **Couple-week table.** Judge total, judge rank (with ties), eliminated flag, notability group, `skill` and `survival_surplus` (both from earlier weeks only), couples remaining. Add a leakage test (features unchanged if the target week and later are deleted).
2. **Rule check (judge scores and eliminations only).** Test whether each actual elimination is consistent with some fan ordering under its era's rule (each tie convention for the rank-based eras). Output: the usable seasons/weeks. The check has little power (most couples are feasible under any rule), so it only removes clear contradictions; it cannot identify the rule or the tie convention.
3. **Harness.** One function that takes judge totals (actual, or Layer 1 draws) plus fan utilities and returns elimination probabilities; a fitter (theta fit on actual judge scores, rolling origin by season); and a backtest that reports the section 5 metrics under both judge-score inputs. Validate it with the two baselines: "lowest judge score goes home" and the random-fan null.
4. **Model ladder.** Fit and backtest each rung, and keep it only if it beats the previous rung:
   - **M1:** skill only. This is the realistic floor; check that `beta_skill` is positive.
   - **M2:** M1 + survival surplus + notability group. Tested and removed (section 7).
   - **M3 (optional):** M2 + persistent per-couple popularity `alpha_i ~ Normal(0, tau^2)`, drawn once per couple per season. Fit by Monte Carlo over alpha within each season; when forecasting, reweight alpha draws by how well they explain the season's earlier weeks. It may make survival surplus redundant. Skipped (section 7).
5. **Final report for the best model.** Calibration table, hit rates by couples remaining against 1/N, season-level bootstrap intervals, and the gap between the actual-score and Layer 1 versions (the cost of Layer 1 error).
6. **Later extensions, one at a time:** current-week judge score as a fan feature (actual when fitting, simulated in combined runs, so fan support and judge rank stay correlated within each draw), surprise-weighted survival surplus, and pre-season follower counts if historical data exists (pre-season only, to avoid leakage).

## 7. Results (status: fan-model ladder closed, judges-only is the default forecast)

Reproduce with `python scripts/run_rule_check.py` then `python scripts/run_layer2.py --models lowest_judge null m1 --compare-to lowest_judge null --validate-null`. Average tie convention; test seasons 6-35, each fit on earlier seasons only. For live forecasts, `python scripts/run_layer2.py --fit-final` refits the default model on all usable weeks (rho 0.584 on 222 weeks through S35W8); `predict_upcoming.py` reports it as `p_elim`.

**Data.** 222 of 228 single-elimination weeks are usable (196 rank-rule, 26 judges' save). Dropped: 5 weeks with uneven dance counts and S17 W5 (a co-top judge scorer went home, impossible under H1). The 3 withdrawals (S9 W3, S16 W2, S31 W5) are excluded as non-votes. Test weeks: 186 with actual judge scores, 101 with Layer 1 forecast scores (Layer 1 needs seasons 17+ and a previous week for every couple).

**Model ladder** (log loss, lower is better; random guess 2.116 on 186 weeks, 2.168 on 101):

| Model | Actual scores (186 weeks) | Layer 1 scores (101 weeks) |
|---|---|---|
| Lower judge score goes home (rho fitted, 0.51-0.59) | **1.618** | 1.757 |
| Random fans under the era rule (null) | 1.678 | 1.773 |
| M1: skill (`beta_skill` grows to about 0.3) | 1.653 | **1.744** |
| M2: M1 + survival surplus + notability | 1.722 | not kept |

- Judges-only beats the random-fan null with actual scores (-0.060, CI -0.112 to -0.007). M1 does not beat judges-only (+0.035, CI -0.018 to +0.082) and M2 is significantly worse than both. With Layer 1 scores every difference is within noise.
- **Fan features that failed** (all tested without notability, against M1): survival surplus as a count, surprise-weighted (sum of the judges-only elimination probability of each week survived), and survived-while-judges'-lowest. Alone each gets a negative coefficient, because surviving a dangerous week mostly marks a weak couple about to leave; next to skill the coefficient turns slightly positive with more data but never improves log loss. Notability coefficients were unstable (reality stars -2.1 early, -0.3 late).
- **Why:** having survived as the judges' lowest before does not change the chance of going home the next time a couple is lowest (relative to field size it stays about 3.2x random). Memorable fan favourites (Harry Jowsey S32, Bristol Palin S11) exist, but a shared coefficient cannot separate them from the many couples rescued once and then eliminated.
- An ensemble (50/50 judges-only and M1) ties the better of the two under both inputs and is not worth its cost. M3 (per-couple popularity) was skipped: its premise, that past rescues predict future rescues, is what the survived-while-lowest check found no evidence for.
- The uniform mixing weight `eps` settles at about 0.19 in the fan models: about one elimination in five is unexplained by judge ranks and skill.

**Default model, lower judge score goes home:**

| | Actual scores (186 weeks) | Layer 1 scores (101 weeks) | Random |
|---|---|---|---|
| Top-1 | 0.390 (CI 0.315-0.463) | 0.347 (0.265-0.429) | 0.13 / 0.12 |
| Top-3 | 0.774 (0.707-0.832) | 0.703 (0.582-0.815) | 0.39 / 0.36 |
| Log loss | 1.618 (1.477-1.773) | 1.757 (1.642-1.888) | 2.116 / 2.168 |

By couples remaining (actual scores), top-1 vs random: <=4 couples 0.55 vs 0.25 (11 weeks), 5-6 0.33 vs 0.18, 7-8 0.43 vs 0.13, 9-10 0.41 vs 0.11, 11+ 0.36 vs 0.08. The model's lift over random is largest in big fields.

**Calibration** (couple-weeks in prediction quintiles, mean predicted vs observed elimination rate):

| Quintile | Actual: predicted / observed | Layer 1: predicted / observed |
|---|---|---|
| 1 | 0.003 / 0.003 | 0.022 / 0.005 |
| 2 | 0.016 / 0.018 | 0.052 / 0.027 |
| 3 | 0.054 / 0.074 | 0.087 / 0.065 |
| 4 | 0.149 / 0.141 | 0.139 / 0.152 |
| 5 | 0.348 / 0.334 | 0.248 / 0.297 |

With actual scores the model is well calibrated. With Layer 1 scores it is somewhat too flat: it gives too much probability to safe couples and too little to the likeliest ones, because Layer 1's score uncertainty spreads the judges' ranking.

**Cost of Layer 1 error.** On the same 101 weeks, Layer 1 scores vs actual scores: log loss -0.016 (CI -0.201 to +0.167), top-1 +0.021. No measurable cost. The lower Layer 1 numbers in the tables come from the weeks it covers (seasons 17+, the harder recent seasons; actual scores on those weeks give 1.773), not from Layer 1 error. The calibration above shows where the error does appear: flatter probabilities.

**Harness checks.** The random-fan forecast agrees with an independent simulation of random-fan worlds (all |z| < 1). The fitter recovers known parameters on simulated data, and its gradient matches finite differences (tests in `tests/test_layer2_*.py`).
