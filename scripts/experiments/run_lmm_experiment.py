"""Experiment: crossed-effects LMM vs the main Ridge model (validation 26-32 only).

    python scripts/experiments/run_lmm_experiment.py

Variants (one change at a time; the couple's level comes from random effects, not prev_score):
  L0  existing LMMModel (main features + couple-season intercept)
  L1  CrossedLMM: couple + season + night intercepts; fixed: week shape, theme, style one-hots
  L2  L1 + correlated couple week-slope
  L3  L2 + pro intercept (crossed across seasons)
  L4  L3 + celebrity group (fixed)
  H   hybrid: Ridge's main features (incl. prev_score, running_mean) + pro, season, night intercepts
  order: best of L1-L4 + order_frac, vs ridge_main_order (announced order)
Week shape (week vs week + week^2) is tuned on L1 by pooled validation MAE, then used for all.

A/B. Point metrics vs ridge on the same rows, decision rule, and non-inferiority
     (bootstrap 95% CI: dRMSE upper < +0.005 and dPair lower > -0.005).
C.   Distribution vs Ridge + Gaussian (night + within-night variance by week bucket, from
     earlier seasons' out-of-sample residuals): coverage, CRPS, NLL, pairwise Brier, P(lowest),
     and integer judge totals simulated by rounding (tie rates vs actual, integer P(lowest)).
     Also Ridge means with L1's covariance, and night-block bootstrap CIs vs Ridge.
D.   Week-1 predictions, variance components by fold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dwts_model.categories import CELEB_GROUPS, CELEB_REFERENCE  # noqa: E402
from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import (  # noqa: E402
    load_couple_info,
    load_judge_scores,
    load_performance_context,
    load_weekly_scores,
)
from dwts_model.evaluate import (  # noqa: E402
    gaussian_scores,
    lowest_indicator,
    pairwise_probs,
    simulate_integer_totals,
    summarize,
    tie_stats,
)
from dwts_model.feature_engineering import build_context_features, build_features, build_lmm_frame  # noqa: E402
from dwts_model.models import CrossedLMM, LMMModel  # noqa: E402
from dwts_model.splits import test_seasons  # noqa: E402
from run_order_experiment import pooled, week_bootstrap  # noqa: E402
from run_pro_experiment import report  # noqa: E402
from dwts_model.judge_distribution import (  # noqa: E402
    KEYS,
    RESID_FROM,
    judge_noise_var as judge_spread,
    night_covariance,
    residual_spread as ridge_spread,
    ridge_oos,
    week_buckets as bucket,
)

SEED = 0
N_DRAWS = 4000
N_BOOT = 2000
NI_RMSE, NI_PAIR = 0.005, 0.005


# ---------------------------------------------------------------- model runs


def run_crossed(full, cfg, fixed, effects, name):
    """Rolling-origin CrossedLMM. Returns row predictions, per-night (couple_ids, mean, cov), variance comps."""
    rows, nights, vcs = [], {}, []
    for season in test_seasons(cfg, "validate"):
        train, test = full[full.season < season], full[full.season == season]
        m = CrossedLMM(fixed, effects, name).fit(train)
        mean, covs = m.predict_dist(test)
        out = test[KEYS + ["score"]].copy()
        out[name] = mean
        out[f"{name}_sd"] = np.nan
        for key, (idx, cov) in covs.items():
            out.loc[idx, f"{name}_sd"] = np.sqrt(np.diag(cov))
            nights[key] = (test.loc[idx, "couple_id"].to_numpy(), mean.loc[idx].to_numpy(), cov)
        rows.append(out)
        vcs.append({"season": season, "converged": m.converged_, **m.variance_components})
    return pd.concat(rows, ignore_index=True), nights, pd.DataFrame(vcs)


def ridge_nights(oos, cfg):
    nights = {}
    for season in test_seasons(cfg, "validate"):
        spread = ridge_spread(oos, season)
        for (s, w), g in oos[oos.season == season].groupby(["season", "week"]):
            cov, _ = night_covariance(len(g), w, spread)
            nights[(s, w)] = (g["couple_id"].to_numpy(), g["pred"].to_numpy(), cov)
    return nights


# ---------------------------------------------------------------- distribution metrics


def dist_metrics(nights, actual, slots, totals, judges, iid_var, eval_keys):
    """Distribution metrics on complete eval nights. `iid_var(key)` is the model's per-couple independent variance.

    Draws use a per-night seed, so every model sees the same random numbers (paired comparison).
    Returns the summary, per-row marginal scores, per-night tie rates, and per-night sums for bootstrapping.
    """
    marg, pair_p, pair_y, low_rows, ties, per_night = [], [], [], [], [], []
    judge_var = {}
    eps = 1e-4
    for (s, w), (couples, mean, cov) in sorted(nights.items()):
        if (s, w) not in eval_keys:
            continue
        rng = np.random.default_rng([SEED, int(s), int(w)])
        a = actual.loc[[(s, w, c) for c in couples]].to_numpy()
        sd = np.sqrt(np.diag(cov))
        g = gaussian_scores(a, mean, sd)
        marg.append(g.assign(week=w))
        p, y = pairwise_probs(a, mean, cov)
        pair_p.append(p)
        pair_y.append(y)

        tot = totals.loc[[(s, w, c) for c in couples]].to_numpy()
        n_slots = slots.loc[[(s, w, c) for c in couples]].to_numpy()
        if s not in judge_var:
            judge_var[s] = judge_spread(judges, s)
        sj2 = judge_var[s]
        # Latent (shared-across-judges) covariance: remove the judge-noise share of the iid part.
        reduce = np.minimum(sj2 / n_slots, 0.9 * iid_var((s, w)))
        latent_cov = cov - np.diag(reduce)
        draws = rng.multivariate_normal(mean, cov, size=N_DRAWS, method="cholesky")
        latent = rng.multivariate_normal(mean, latent_cov, size=N_DRAWS, method="cholesky")
        ints = simulate_integer_totals(latent, n_slots, np.sqrt(max(sj2 - 1 / 12, 0.01)), rng)
        act_low = lowest_indicator(tot[None, :])[0]
        p_cont = lowest_indicator(draws).mean(axis=0)
        p_int = lowest_indicator(ints).mean(axis=0)
        sim_ties = {k: v.mean() for k, v in tie_stats(ints).items()}
        act_ties = {k: bool(v[0]) for k, v in tie_stats(tot[None, :]).items()}
        ties.append({"season": s, "week": w, "n": len(couples), **{f"sim_{k}": v for k, v in sim_ties.items()},
                     **{f"act_{k}": v for k, v in act_ties.items()}})
        for pc, pi, yy in zip(p_cont, p_int, act_low):
            low_rows.append({"season": s, "week": w, "p_cont": pc, "p_int": pi, "y": yy})
        pi = np.clip(p_int, eps, 1 - eps)
        per_night.append({
            "season": s, "week": w, "n": len(couples), "pairs": len(p),
            "crps": g.crps.sum(), "pair_brier": float(((p - y) ** 2).sum()),
            "low_ll_int": float(-(act_low * np.log(pi) + (1 - act_low) * np.log(1 - pi)).sum()),
        })

    m = pd.concat(marg, ignore_index=True)
    pp, py = np.concatenate(pair_p), np.concatenate(pair_y)
    low = pd.DataFrame(low_rows)

    def bll(p, y):
        p = np.clip(p, eps, 1 - eps)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    summary = {
        "cover50": m.cover50.mean(), "cover90": m.cover90.mean(), "crps": m.crps.mean(), "nll": m.nll.mean(),
        "pair_brier": float(((pp - py) ** 2).mean()), "pair_logloss": bll(pp, py),
        "low_brier_cont": float(((low.p_cont - low.y) ** 2).mean()), "low_ll_cont": bll(low.p_cont, low.y),
        "low_brier_int": float(((low.p_int - low.y) ** 2).mean()), "low_ll_int": bll(low.p_int, low.y),
    }
    return summary, m, pd.DataFrame(ties), pd.DataFrame(per_night)


def night_bootstrap(a: pd.DataFrame, b: pd.DataFrame, rng) -> dict[str, list[float]]:
    """95% CI of (b - a) for per-couple CRPS, per-pair Brier and per-couple integer P(lowest) log loss, resampling nights."""
    a, b = a.sort_values(["season", "week"]), b.sort_values(["season", "week"])
    idx = [rng.integers(0, len(a), len(a)) for _ in range(N_BOOT)]
    out = {}
    for col, denom in (("crps", "n"), ("pair_brier", "pairs"), ("low_ll_int", "n")):
        x, y, d = a[col].to_numpy(), b[col].to_numpy(), a[denom].to_numpy()
        diffs = [(y[i].sum() - x[i].sum()) / d[i].sum() for i in idx]
        out[col] = np.percentile(diffs, [2.5, 97.5]).round(4).tolist()
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    cfg = load_config()
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 160

    weekly = load_weekly_scores(cfg)
    context = build_context_features(load_performance_context(cfg))
    df = build_features(weekly, context)
    full = build_lmm_frame(weekly, context, load_couple_info(cfg))
    judges = load_judge_scores(cfg)
    wk_tot = judges.groupby(KEYS)["judge_score"].agg(["sum", "count"])
    totals, slots = wk_tot["sum"], wk_tot["count"]
    actual = full.set_index(KEYS)["score"]

    saved = json.loads((cfg.output_dir / "selected_params.json").read_text())["ridge_alpha"]
    main_set = cfg.main_model.removeprefix("ridge_")
    order_set = cfg.order_model.removeprefix("ridge_")
    main_feats, order_feats = list(cfg.features[main_set]), list(cfg.features[order_set])
    styles = [f for f in main_feats if f.startswith("style_")]
    celeb = [f"celeb_{g}" for g in CELEB_GROUPS if g != CELEB_REFERENCE]

    # Ridge references (out-of-sample from RESID_FROM on, for the baseline spread).
    oos_main = ridge_oos(df, main_feats, saved[main_set], RESID_FROM, cfg.validation[1])
    oos_order = ridge_oos(df, order_feats, saved[order_set], cfg.validation[0], cfg.validation[1])
    val = df[df.season.between(*cfg.validation)][KEYS + ["score"]].copy()
    val = val.merge(oos_main[KEYS + ["pred"]].rename(columns={"pred": "ridge"}), on=KEYS)
    val = val.merge(oos_order[KEYS + ["pred"]].rename(columns={"pred": "ridge_order"}), on=KEYS)

    # L0: existing LMMModel on the main features.
    l0 = []
    for season in test_seasons(cfg, "validate"):
        train, test = df[df.season < season], df[df.season == season]
        out = test[KEYS].copy()
        out["L0"] = LMMModel(main_feats).fit(train).predict(test)
        l0.append(out)
    val = val.merge(pd.concat(l0), on=KEYS)

    def clip(c):
        return val[c].clip(cfg.score_min, cfg.score_max)

    def mae(preds, c):
        p = val[KEYS + ["score"]].merge(preds[KEYS + [c]], on=KEYS)
        return float((p[c].clip(cfg.score_min, cfg.score_max) - p["score"]).abs().mean())

    base_fixed = ["theme_personal_emotional"] + styles
    shapes = {"lin": ["week"], "quad": ["week", "week_sq"]}
    effects = {
        "L1": ("couple", "season", "night"),
        "L2": ("couple", "couple_slope", "season", "night"),
        "L3": ("couple", "couple_slope", "pro", "season", "night"),
    }
    runs, vcs = {}, {}
    shape_mae = {}
    for sh, cols in shapes.items():
        p, n, v = run_crossed(full, cfg, cols + base_fixed, effects["L1"], f"L1_{sh}")
        shape_mae[sh] = mae(p, f"L1_{sh}")
        runs[f"L1_{sh}"] = (p, n)
        vcs[f"L1_{sh}"] = v
    shape = min(shape_mae, key=shape_mae.get)
    print(f"week shape MAE (L1): {shape_mae} -> {shape}")
    fixed = shapes[shape] + base_fixed
    p, n = runs.pop(f"L1_{shape}")
    p = p.rename(columns={f"L1_{shape}": "L1", f"L1_{shape}_sd": "L1_sd"})
    runs["L1"], vcs["L1"] = (p, n), vcs.pop(f"L1_{shape}")
    for name in ("L2", "L3"):
        p, n, v = run_crossed(full, cfg, fixed, effects[name], name)
        runs[name], vcs[name] = (p, n), v
    p, n, v = run_crossed(full, cfg, fixed + celeb, effects["L3"], "L4")
    runs["L4"], vcs["L4"] = (p, n), v

    for name in ("L1", "L2", "L3", "L4"):
        val = val.merge(runs[name][0][KEYS + [name]], on=KEYS)
    best = min(("L1", "L2", "L3", "L4"), key=lambda c: mae(val, c))
    best_fixed = fixed + (celeb if best == "L4" else [])
    best_eff = effects["L3"] if best == "L4" else effects[best]
    p, n, v = run_crossed(full, cfg, best_fixed + ["order_frac"], best_eff, "Lorder")
    runs["Lorder"], vcs["Lorder"] = (p, n), v
    val = val.merge(p[KEYS + ["Lorder"]], on=KEYS)
    print(f"best LMM by pooled validation MAE: {best}")

    # H: hybrid. Ridge's main features (own history included) plus pro/season/night random effects.
    # A couple intercept is redundant here: its variance fits to ~0 once prev_score and running_mean are in.
    dfp = df.merge(full[KEYS + ["professional_id"]], on=KEYS, how="left")
    p, n, v = run_crossed(dfp, cfg, main_feats, ("pro", "season", "night"), "H")
    runs["H"], vcs["H"] = (p, n), v
    val = val.merge(p[KEYS + ["H"]], on=KEYS)

    models = ["ridge", "L0", "L1", "L2", "L3", "L4", "H", "ridge_order", "Lorder"]
    for c in models:
        val[c] = clip(c)

    # ---- A/B: point metrics and decision rule
    print("\n=== A. Point metrics (validation 26-32, weeks 2+, same rows as Ridge) ===")
    print(pd.DataFrame([dict(model=c, mae=float((val[c] - val.score).abs().mean()),
                             **dict(zip(["rmse", "pair_acc", "spearman"], pooled(val, c)))) for c in models])
          .to_string(index=False))
    brng = np.random.default_rng(SEED)
    ni = []
    for ref, c in [("ridge", x) for x in ("L0", "L1", "L2", "L3", "L4", "H")] + [("ridge_order", "Lorder")]:
        report(val, ref, c, brng)
        ci_r, ci_p = week_bootstrap(val, ref, c, np.random.default_rng(SEED))
        ni.append({"model": c, "ref": ref, "dRMSE_hi": ci_r[1], "dPair_lo": ci_p[0],
                   "non_inferior": bool(ci_r[1] < NI_RMSE and ci_p[0] > -NI_PAIR)})
    print("\n=== B. Non-inferiority (95% week-block bootstrap) ===")
    print(pd.DataFrame(ni).to_string(index=False))

    # ---- C: distribution metrics on complete nights (every couple that night has a weeks-2+ eval row)
    full_n = full[full.season.between(*cfg.validation) & (full.week >= 2)].groupby(["season", "week"]).size()
    val_n = val.groupby(["season", "week"]).size()
    eval_keys = {k for k, n_ in val_n.items() if full_n.get(k) == n_ and n_ >= 3}
    print(f"\n=== C. Distribution ({len(eval_keys)} complete nights with >= 3 couples) ===")

    spreads = {s: ridge_spread(oos_main, s) for s in test_seasons(cfg, "validate")}
    oos_order_full = ridge_oos(df, order_feats, saved[order_set], RESID_FROM, cfg.validation[1])
    spreads_o = {s: ridge_spread(oos_order_full, s) for s in test_seasons(cfg, "validate")}
    dist_models = {
        "ridge": (ridge_nights(oos_main, cfg), lambda k: spreads[k[0]][str(bucket(np.array([k[1]]))[0])][0]),
        "ridge_order": (ridge_nights(oos_order_full, cfg),
                        lambda k: spreads_o[k[0]][str(bucket(np.array([k[1]]))[0])][0]),
    }
    for name in ("L1", "L2", "L3", "L4", "H", "Lorder"):
        sig2 = {r.season: r.resid_sd**2 for r in vcs[name].itertuples()}
        dist_models[name] = (runs[name][1], lambda k, s2=sig2: s2[k[0]])

    # Ridge point predictions with L1's covariance: does the LMM's spread help on its own?
    ridge_n, l1_n = dist_models["ridge"][0], runs["L1"][1]
    hybrid_n = {}
    for k, (couples, _, cov) in l1_n.items():
        if k in ridge_n:
            rmean = dict(zip(ridge_n[k][0], ridge_n[k][1]))
            if all(c in rmean for c in couples):
                hybrid_n[k] = (couples, np.array([rmean[c] for c in couples]), cov)
    dist_models["ridge_mean+L1cov"] = (hybrid_n, dist_models["L1"][1])

    summaries, marg_all, tie_all, nightly = [], [], [], {}
    for name, (nights, iid) in dist_models.items():
        summ, marg, ties, per_night = dist_metrics(nights, actual, slots, totals, judges, iid, eval_keys)
        summaries.append({"model": name, **summ})
        marg_all.append(marg.assign(model=name))
        tie_all.append(ties.assign(model=name))
        nightly[name] = per_night
    summ = pd.DataFrame(summaries)
    print(summ.to_string(index=False))

    print("\nNight-block bootstrap 95% CI of (model - reference); negative = better:")
    brng = np.random.default_rng(SEED)
    for name in dist_models:
        ref = "ridge_order" if name == "Lorder" else "ridge"
        if name not in ("ridge", "ridge_order"):
            print(f"  {name} vs {ref}: {night_bootstrap(nightly[ref], nightly[name], brng)}")

    marg = pd.concat(marg_all)
    marg["bucket"] = bucket(marg["week"])
    print("\nCoverage by week bucket (target 0.50 / 0.90):")
    print(marg.groupby(["model", "bucket"])[["cover50", "cover90", "crps"]].mean().unstack("bucket").to_string())
    print("\nPIT deciles (uniform = 0.10 each):")
    print(marg.groupby("model")["pit"].apply(lambda x: pd.Series(np.histogram(x, bins=10, range=(0, 1))[0] / len(x)))
          .unstack().to_string())

    ties = pd.concat(tie_all)
    ties["era"] = np.select([ties.season <= 27, ties.season <= 31], ["26-27", "28-31"], "32")
    ties["field"] = np.select([ties.n <= 5, ties.n <= 8], ["3-5", "6-8"], "9+")
    cols = ["any_tie", "bottom_tie", "bottom3_tie"]
    print("\nTie rates per night, actual vs simulated integer totals:")
    for by in (None, "era", "field"):
        g = ties.groupby(["model"] + ([by] if by else []))
        t = g[[f"act_{c}" for c in cols] + [f"sim_{c}" for c in cols]].mean()
        t["nights"] = g.size()
        print(t.to_string(), "\n")

    # ---- D: week 1 and variance components
    print("=== D. Week-1 predictions (LMM only; Ridge needs a prior week) ===")
    w1 = full[full.season.between(*cfg.validation) & (full.week == 1)][KEYS + ["score"]]
    for name in ("L1", "L3", "L4"):
        w1 = w1.merge(runs[name][0][KEYS + [name]], on=KEYS)
    w1["const"] = w1.groupby("season")["score"].transform(lambda s: 0) + np.nan
    for s in w1.season.unique():
        prior = full[(full.season < s) & (full.season >= s - 3) & (full.week == 1)]["score"].mean()
        w1.loc[w1.season == s, "const"] = prior
    print(pd.DataFrame([dict(model=c, **dict(zip(["rmse", "pair_acc", "spearman"], pooled(w1, c))))
                        for c in ("const", "L1", "L3", "L4")]).to_string(index=False))

    print("\nVariance components by fold (SD, score scale):")
    for name, v in vcs.items():
        print(f"-- {name}\n{v.to_string(index=False)}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    val.to_csv(cfg.output_dir / "lmm_experiment_validate.csv", index=False)
    summ.to_csv(cfg.output_dir / "lmm_experiment_distribution_validate.csv", index=False)
    ties.to_csv(cfg.output_dir / "lmm_experiment_ties_validate.csv", index=False)
    tables = summarize(val, models, cfg.short_seasons)
    tables["by_season"].to_csv(cfg.output_dir / "lmm_experiment_by_season_validate.csv", index=False)
    tables["by_week"].to_csv(cfg.output_dir / "lmm_experiment_by_week_validate.csv", index=False)


if __name__ == "__main__":
    main()
