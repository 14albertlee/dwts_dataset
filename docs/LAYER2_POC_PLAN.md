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
3. **Skill-only fan model:** only `beta_skill` fitted (survival and notability off). Because fan support tracks quality, the random-ranking null is too easy a floor; this is the realistic one, and the extra features must beat it.

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
   - **M2:** M1 + survival surplus + notability group.
   - **M3 (optional):** M2 + persistent per-couple popularity `alpha_i ~ Normal(0, tau^2)`, drawn once per couple per season. Fit by Monte Carlo over alpha within each season; when forecasting, reweight alpha draws by how well they explain the season's earlier weeks. It may make survival surplus redundant.
5. **Final report for the best model.** Calibration table, hit rates by couples remaining against 1/N, season-level bootstrap intervals, and the gap between the actual-score and Layer 1 versions (the cost of Layer 1 error).
6. **Later extensions, one at a time:** current-week judge score as a fan feature (actual when fitting, simulated in combined runs, so fan support and judge rank stay correlated within each draw), surprise-weighted survival surplus, and pre-season follower counts if historical data exists (pre-season only, to avoid leakage).
