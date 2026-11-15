"""Holdout performance report: judge-score model (Layer 1) and elimination forecast (Layer 2) vs baselines.

    python scripts/report_holdout.py            # -> outputs/reports/holdout/report.md (+ CSVs and PNGs)

Needs the holdout predictions (run_layer1.py --mode holdout --confirm-holdout ... main_prev_bottom3 main_order),
the validation predictions, and run_rule_check.py. Holdout = config [splits].holdout. Every Layer 2 forecast
uses rho fit on earlier seasons only; Layer 1 judge totals come from Ridge fits on weeks before the target week.
Because holdout has few elimination weeks, each table is repeated on all out-of-sample seasons with Layer 1
forecasts (17+, which includes the holdout).
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import (  # noqa: E402
    load_judge_scores,
    load_performance_context,
    load_season_couples,
    load_weekly_scores,
)
from dwts_model.evaluate import _metrics  # noqa: E402
from dwts_model.feature_engineering import build_features  # noqa: E402
from dwts_model.judge_distribution import RESID_FROM  # noqa: E402
from dwts_model.layer2.backtest import MODELS, SeasonParams, backtest, fit_season_params  # noqa: E402
from dwts_model.layer2.baselines import fit_rho  # noqa: E402
from dwts_model.layer2.judge_inputs import ActualJudgeInput, Layer1JudgeInput, PrevWeekJudgeInput  # noqa: E402
from dwts_model.layer2.metrics import (  # noqa: E402
    PHASES,
    bootstrap_ci,
    calibration,
    paired_ci,
    summarize,
    week_metrics,
    week_phase,
)
from dwts_model.reporting import md_table  # noqa: E402

SCORE_BASELINES = ["last_week", "last_week_drift", "running_mean", "field_adjusted"]
PROD = "production"
ELIM_LABELS = {
    "random": "random pick (1/n)",
    "bottom3_last_week": "uniform over last week's bottom 3",
    "prev_week_lowest": "lowest score last week (rho refit)",
    "strict_lowest": "Layer 1, lowest predicted goes home (rho=0)",
    PROD: "**Layer 1 pre-show + rho (production)**",
    "production_order": "Layer 1 order-announced + rho",
    "null": "fan model null (saved run)",
    "m1": "fan model m1 (saved run)",
    "oracle_actual": "actual judge scores + rho (ceiling, not a forecast)",
}
ELIM_METRICS = ["top1", "top2", "top3", "rank", "logloss", "brier"]


class CachedInput:
    """Reuse a judge input's simulated totals across models that score the same weeks."""

    def __init__(self, inner):
        self.inner, self.name, self._cache = inner, inner.name, {}

    def available(self, season, week, couple_ids):
        return self.inner.available(season, week, couple_ids)

    def totals(self, season, week, couple_ids):
        if (season, week) not in self._cache:
            self._cache[(season, week)] = self.inner.totals(season, week, couple_ids)
        return self._cache[(season, week)]


# ---------------------------------------------------------------- Layer 1 (judge scores)

def lowest_hits(preds: pd.DataFrame, model: str, n_shuffle: int = 200, seed: int = 0) -> pd.DataFrame:
    """Per week: did the model's lowest (or two lowest) predicted couples include the actual lowest scorer?

    Ties in the actual score: any tied-lowest couple counts. Ties in the prediction are broken at random
    (averaged over `n_shuffle` draws), so tie-heavy baselines like last_week are not favored by row order.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (s, w), g in preds.groupby(["season", "week"]):
        actual, pred = g["score"].to_numpy(), g[model].to_numpy()
        n, low = len(g), actual == actual.min()
        k_low = int(low.sum())
        hit1 = hit2 = 0.0
        for _ in range(n_shuffle):
            order = np.lexsort((rng.random(n), pred))
            hit1 += low[order[0]]
            hit2 += low[order[:2]].any()
        rows.append({
            "season": s, "week": w, "n_couples": n,
            "low1": hit1 / n_shuffle, "low2": hit2 / n_shuffle,
            "random_low1": k_low / n,
            "random_low2": 1 - math.comb(n - k_low, 2) / math.comb(n, 2) if n >= 2 else 1.0,
        })
    return pd.DataFrame(rows)


def rmse_diff_ci(preds: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """RMSE(a) - RMSE(b) with a week-level bootstrap 95% interval."""
    d = preds.assign(ea=(preds[a] - preds["score"]) ** 2, eb=(preds[b] - preds["score"]) ** 2)
    wk = d.groupby(["season", "week"])[["ea", "eb"]].agg(["sum", "count"])
    sa, sb, n = wk[("ea", "sum")].to_numpy(), wk[("eb", "sum")].to_numpy(), wk[("ea", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wk), size=(n_boot, len(wk)))
    draws = np.sqrt(sa[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
    point = np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def score_table(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        hits = lowest_hits(preds, m)
        met = _metrics(preds, m)
        rows.append({
            "model": m, "rows": met["n"], "rmse": met["rmse"], "spearman": met["spearman"],
            "pair_acc": met["pair_acc"], "lowest_top1": hits["low1"].mean(), "lowest_top2": hits["low2"].mean(),
        })
    hits = lowest_hits(preds, models[0])
    rows.append({"model": "random", "lowest_top1": hits["random_low1"].mean(), "lowest_top2": hits["random_low2"].mean()})
    return pd.DataFrame(rows).astype({"rows": "Int64"})


def score_by_phase(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for phase, g in preds.groupby(week_phase(preds["week"]), observed=True):
        for m in models:
            hits = lowest_hits(g, m)
            met = _metrics(g, m)
            rows.append({
                "phase": phase, "model": m, "weeks": len(hits), "rmse": met["rmse"], "pair_acc": met["pair_acc"],
                "lowest_top1": hits["low1"].mean(), "lowest_top2": hits["low2"].mean(),
                "random_top1": hits["random_low1"].mean(), "random_top2": hits["random_low2"].mean(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Layer 2 (eliminations)

def prev_week_params(cw, usable, prev, seasons) -> dict:
    """rho for the previous-week baseline, fit per test season on earlier seasons' usable weeks."""
    groups = cw.groupby(["season", "week"])
    weeks = {}
    for s, w in zip(usable["season"], usable["week"]):
        g = groups.get_group((s, w))
        if prev.available(s, w, g["couple_id"]):
            weeks[(s, w)] = (prev.totals(s, w, g["couple_id"])[0], int(np.flatnonzero(g["eliminated"].to_numpy())[0]))
    return {
        season: SeasonParams(season, rho=fit_rho([v for (s, _), v in weeks.items() if s < season])[0])
        for season in seasons
    }


def bottom3_probs(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Uniform over couples flagged bottom 3 last week (uniform over all if none are flagged)."""
    flag = frame.merge(history[["season", "week", "couple_id", "prev_bottom3"]], on=["season", "week", "couple_id"], how="left")
    f = flag["prev_bottom3"].fillna(0).to_numpy()
    k = flag.groupby(["season", "week"])["prev_bottom3"].transform(lambda s: s.fillna(0).sum()).to_numpy()
    n = flag["n_couples"].to_numpy()
    return flag.drop(columns="prev_bottom3").assign(p_elim=np.where(k > 0, f / np.maximum(k, 1), 1 / n))


def elim_table(wms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    prod = wms[PROD]
    rows.append({"model": ELIM_LABELS["random"], "weeks": len(prod),
                 **{m: prod[f"random_{m}"].mean() for m in ("top1", "top2", "top3", "logloss", "brier")},
                 "rank": ((prod["n_couples"] + 1) / 2).mean()})
    for name, wm in wms.items():
        rows.append({"model": ELIM_LABELS.get(name, name), "weeks": len(wm), **wm[ELIM_METRICS].mean().to_dict()})
    return pd.DataFrame(rows)[["model", "weeks", *ELIM_METRICS]]


def by_week_id(wm: pd.DataFrame) -> pd.DataFrame:
    """Treat every week as its own bootstrap cluster (the holdout has too few seasons to resample)."""
    return wm.assign(season=wm["season"] * 100 + wm["week"])


def phase_table(wms: dict[str, pd.DataFrame], names: list[str], by: str = "phase") -> pd.DataFrame:
    out = None
    for name in names:
        s = summarize(wms[name], by=by).rename(columns={"top1": f"{name}_top1", "top2": f"{name}_top2"})
        cols = ["group", "weeks", f"{name}_top1", f"{name}_top2"]
        if out is None:
            out = s[cols + ["random_top1", "random_top2"]]
        else:
            out = out.merge(s[["group", f"{name}_top1", f"{name}_top2"]], on="group")
    return out.rename(columns={"group": by})


def eliminated_rank_table(probs: dict[str, pd.DataFrame], wms: dict[str, pd.DataFrame], names: pd.DataFrame) -> pd.DataFrame:
    p = probs[PROD].merge(names, on=["season", "couple_id"], how="left")
    rows = []
    for (s, w), g in p.groupby(["season", "week"]):
        e = g[g["eliminated"]].iloc[0]
        top = g.loc[g["p_elim"].idxmax()]
        row = {"week": f"S{s}W{w}", "couples": len(g), "went home": e["celebrity"], "p_elim": e["p_elim"]}
        for name, label in ((PROD, "rank (production)"), ("production_order", "rank (order)"), ("prev_week_lowest", "rank (last week)")):
            wm = wms[name]
            row[label] = wm.loc[(wm["season"] == s) & (wm["week"] == w), "rank"].iloc[0]
        row["model's top pick"] = f"{top['celebrity']} ({top['p_elim']:.0%})"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts

def double_lift(ax, df, actual, a, b, bins, ylabel, title, fmt_pct=False):
    ratio = df[a] / df[b].clip(lower=1e-9)
    binned = pd.qcut(ratio.rank(method="first"), bins, labels=False)
    g = df.groupby(binned)[[actual, a, b]].mean()
    x = np.arange(1, bins + 1)
    ax.plot(x, g[actual], "k-o", lw=2.5, label="actual")
    ax.plot(x, g[a], "--s", label=a)
    ax.plot(x, g[b], "--^", label=b)
    ax.set(xlabel=f"bin of {a} / {b} (1 = {a} lowest relative to {b})", ylabel=ylabel, title=f"{title} (n={len(df)})")
    if fmt_pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    g["closer"] = np.where((g[a] - g[actual]).abs() < (g[b] - g[actual]).abs(), a, b)
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--draws", type=int, default=None, help="Layer 1 Monte Carlo draws per week (default: config)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir.parent / "reports" / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    h0, h1 = cfg.holdout
    holdout = list(range(h0, h1 + 1))
    main_m, order_m = cfg.main_model, cfg.order_model
    tie = cfg.tie_convention

    # ---- Layer 1
    hold = pd.read_csv(cfg.output_dir / "predictions_holdout.csv")
    val = pd.read_csv(cfg.output_dir / "predictions_validate.csv")
    missing = [m for m in (main_m, order_m) if m not in hold.columns]
    if missing:
        sys.exit(f"predictions_holdout.csv lacks {missing}: run run_layer1.py --mode holdout --confirm-holdout "
                 "--models ridge --feature-sets main_prev_bottom3 main_order")
    vmet = pd.DataFrame([{"model": b, **_metrics(val, b)} for b in SCORE_BASELINES])
    best_base = vmet.loc[vmet["rmse"].idxmin(), "model"]  # chosen on validation, not holdout
    cols = ["season", "week", "couple_id", "score", *SCORE_BASELINES, main_m, order_m]
    both = pd.concat([val[cols], hold[cols]], ignore_index=True)
    score_models = [main_m, order_m, *SCORE_BASELINES]
    v0, v1 = int(val["season"].min()), int(hold["season"].max())

    l1_hold = score_table(hold, score_models)
    l1_both = score_table(both, score_models)
    l1_phase_hold = score_by_phase(hold, [main_m, best_base])
    l1_phase_both = score_by_phase(both, [main_m, best_base])
    rmse_ci = {m: rmse_diff_ci(hold, m, best_base, args.n_boot, cfg.seed) for m in (main_m, order_m)}
    rmse_ci_order = rmse_diff_ci(hold, order_m, main_m, args.n_boot, cfg.seed)
    for name, t in (("layer1_holdout", l1_hold), (f"layer1_s{v0}_{v1}", l1_both),
                    ("layer1_phase_holdout", l1_phase_hold), (f"layer1_phase_s{v0}_{v1}", l1_phase_both)):
        t.to_csv(out_dir / f"{name}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    lift_main = double_lift(axes[0], hold, "score", main_m, best_base, 10, "mean judge score",
                            f"Holdout: {main_m} vs best baseline")
    lift_order = double_lift(axes[1], hold, "score", order_m, main_m, 10, "mean judge score",
                             "Holdout: order-announced vs pre-show")
    fig.tight_layout()
    fig.savefig(out_dir / "double_lift_scores.png", dpi=110)
    plt.close(fig)

    # ---- Layer 2
    print("Layer 2: fitting rho and simulating Layer 1 judge totals (a few minutes)...")
    cw = pd.read_csv(cfg.layer2_dir / "couple_week.csv")
    usable = pd.read_csv(cfg.layer2_dir / "usable_weeks.csv")
    weekly, perf, judges = load_weekly_scores(cfg), load_performance_context(cfg), load_judge_scores(cfg)
    l1_main = CachedInput(Layer1JudgeInput(cfg, weekly, perf, judges, args.draws))
    l1_order = Layer1JudgeInput(cfg, weekly, perf, judges, args.draws, model_name=order_m)
    prev = PrevWeekJudgeInput(weekly)
    spec = MODELS["lowest_judge"]
    params = fit_season_params(cw, usable, spec, cfg, tie)
    params = {s: p for s, p in params.items() if s > RESID_FROM}  # Layer 1 spreads need earlier residuals
    rho0 = {s: SeasonParams(s, rho=0.0) for s in params}
    prev_params = prev_week_params(cw, usable, prev, list(params))

    probs = {PROD: backtest(cw, usable, spec, l1_main, cfg, params, tie)}
    keys = probs[PROD][["season", "week"]].drop_duplicates()
    sub = usable.merge(keys, on=["season", "week"])  # every model scores the production weeks
    probs["production_order"] = backtest(cw, sub, spec, l1_order, cfg, params, tie)
    probs["strict_lowest"] = backtest(cw, sub, spec, l1_main, cfg, rho0, tie)
    probs["prev_week_lowest"] = backtest(cw, sub, spec, prev, cfg, prev_params, tie)
    probs["bottom3_last_week"] = bottom3_probs(probs[PROD].drop(columns="p_elim"), build_features(weekly))
    probs["oracle_actual"] = backtest(cw, sub, spec, ActualJudgeInput(cw), cfg, params, tie)
    for fan in ("null", "m1"):
        path = cfg.layer2_dir / f"forecasts_{fan}_layer1_{tie}.csv"
        if path.exists():
            probs[fan] = pd.read_csv(path).merge(keys, on=["season", "week"])
    for name, p in probs.items():
        p.to_csv(out_dir / f"elim_forecasts_{name}.csv", index=False)

    wms_all = {name: week_metrics(p) for name, p in probs.items()}
    wms_hold = {name: wm[wm["season"].isin(holdout)] for name, wm in wms_all.items()}
    probs_hold = {name: p[p["season"].isin(holdout)] for name, p in probs.items()}
    s_lo, s_hi = int(keys["season"].min()), int(keys["season"].max())

    elim_hold, elim_all = elim_table(wms_hold), elim_table(wms_all)
    paired = []
    for name in wms_hold:
        if name == PROD:
            continue
        d = paired_ci(by_week_id(wms_hold[PROD]), by_week_id(wms_hold[name]), ["top1", "top2", "logloss"], args.n_boot, cfg.seed)
        paired.append(d.assign(vs=ELIM_LABELS.get(name, name)))
    paired = pd.concat(paired, ignore_index=True)
    prod_ci = bootstrap_ci(by_week_id(wms_hold[PROD]), ["top1", "top2", "logloss"], args.n_boot, cfg.seed)
    comp = [PROD, "production_order", "prev_week_lowest"]
    phase_hold, phase_all = phase_table(wms_hold, comp), phase_table(wms_all, comp)
    field_all = phase_table(wms_all, comp, by="field")
    names = load_season_couples(cfg)[["season", "couple_id", "celebrity"]]
    per_week = eliminated_rank_table(probs_hold, wms_hold, names)
    for name, t in (("elim_holdout", elim_hold), (f"elim_s{s_lo}_{s_hi}", elim_all), ("elim_paired_holdout", paired),
                    ("elim_phase_holdout", phase_hold), (f"elim_phase_s{s_lo}_{s_hi}", phase_all),
                    (f"elim_field_s{s_lo}_{s_hi}", field_all), ("elim_by_week_holdout", per_week)):
        t.to_csv(out_dir / f"{name}.csv", index=False)

    # Charts: p_elim double lift, calibration, eliminated couple's rank per holdout week.
    lift = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (label, sample, bins) in zip(axes, (("Holdout", probs_hold, 5), (f"S{s_lo}-{s_hi}", probs, 10))):
        d = sample[PROD][["season", "week", "couple_id", "eliminated", "p_elim"]].rename(columns={"p_elim": PROD})
        d = d.merge(sample["prev_week_lowest"][["season", "week", "couple_id", "p_elim"]].rename(
            columns={"p_elim": "prev_week_lowest"}), on=["season", "week", "couple_id"])
        d["went_home"] = d["eliminated"].astype(float)
        lift[label] = double_lift(ax, d, "went_home", PROD, "prev_week_lowest", bins, "elimination rate",
                                  f"{label}: P(eliminated)", fmt_pct=True)
    fig.tight_layout()
    fig.savefig(out_dir / "double_lift_elimination.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, p, bins, style in (("holdout", probs_hold[PROD], 5, "o-"), (f"S{s_lo}-{s_hi}", probs[PROD], 10, "s--")):
        c = calibration(p, bins)
        ax.plot(c["mean_pred"], c["observed"], style, label=f"{label} ({bins} bins)")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    hi = max(probs[PROD]["p_elim"].max(), 0.1) * 1.05
    ax.set(xlim=(0, hi), ylim=(0, 1), xlabel="mean predicted P(eliminated)", ylabel="observed elimination rate",
           title="Calibration of production p_elim")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_elimination.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(per_week))
    width = 0.27
    for i, (col, label) in enumerate((("rank (production)", "production"), ("rank (order)", "order-announced"),
                                      ("rank (last week)", "lowest last week"))):
        ax.bar(x + (i - 1) * width, per_week[col], width, label=label)
    ax.plot(x, (per_week["couples"] + 1) / 2, "k_", ms=20, mew=2, label="random (expected)")
    ax.set_xticks(x, per_week["week"], rotation=45)
    ax.set(ylabel="eliminated couple's rank in the forecast (1 = top pick)", title="Holdout weeks: where the eliminated couple ranked")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eliminated_rank_by_week.png", dpi=110)
    plt.close(fig)

    # ---- Plain-language insights
    def pct(v):
        return f"{v:.0%}"

    insights = []
    ho = wms_hold[PROD]
    insights.append(
        f"Across the {len(ho)} holdout elimination weeks (seasons {h0}-{h1}), the couple the model rated most likely to go home "
        f"went home {pct(ho['top1'].mean())} of the time, and the eliminated couple was the model's first or second pick "
        f"{pct(ho['top2'].mean())} of the time (random picks: {pct(ho['random_top1'].mean())} and {pct(ho['random_top2'].mean())})."
    )
    for phase in PHASES:
        rh, ra = phase_hold[phase_hold["phase"] == phase], phase_all[phase_all["phase"] == phase]
        if ra.empty:
            continue
        ra = ra.iloc[0]
        text = f"In {phase}, "
        if not rh.empty:
            rh = rh.iloc[0]
            text += (f"the couple with the highest or second-highest predicted chance of going home went home "
                     f"{pct(rh[f'{PROD}_top2'])} of the time in the holdout (n={int(rh['weeks'])} weeks; top pick alone "
                     f"{pct(rh[f'{PROD}_top1'])}) and ")
        else:
            text += "there are no holdout weeks; "
        text += (f"{pct(ra[f'{PROD}_top2'])} across seasons {s_lo}-{s_hi} (n={int(ra['weeks'])}; top pick {pct(ra[f'{PROD}_top1'])}). "
                 f"Random picks would manage {pct(ra['random_top2'])} (top 2) / {pct(ra['random_top1'])} (top 1); "
                 f"\"lowest score last week\" got {pct(ra['prev_week_lowest_top2'])} / {pct(ra['prev_week_lowest_top1'])}.")
        insights.append(text)
    pa = probs[PROD]
    for thr in (0.5, 0.3):
        big = pa[pa["p_elim"] >= thr]
        if len(big) >= 5:
            insights.append(
                f"When the model gave a couple at least {pct(thr)} to go home (seasons {s_lo}-{s_hi}, {len(big)} couple-weeks, "
                f"average forecast {pct(big['p_elim'].mean())}), they went home {pct(big['eliminated'].mean())} of the time."
            )
            break
    lh = l1_hold.set_index("model")
    insights.append(
        f"Judge scores: on the holdout the pre-show model misses a couple's average judge score by about "
        f"{lh.loc[main_m, 'rmse']:.2f} points (RMSE; the best simple baseline, {best_base}, misses by {lh.loc[best_base, 'rmse']:.2f}), "
        f"and orders {pct(lh.loc[main_m, 'pair_acc'])} of pairs of couples correctly (who scores higher)."
    )
    insights.append(
        f"The couple predicted to score lowest did score lowest (ties count) {pct(lh.loc[main_m, 'lowest_top1'])} of holdout weeks; "
        f"the actual lowest scorer was among the two lowest predictions {pct(lh.loc[main_m, 'lowest_top2'])} of the time "
        f"(random: {pct(lh.loc['random', 'lowest_top1'])} / {pct(lh.loc['random', 'lowest_top2'])})."
    )
    for phase in PHASES:
        r = l1_phase_hold[(l1_phase_hold["phase"] == phase) & (l1_phase_hold["model"] == main_m)]
        if not r.empty:
            r = r.iloc[0]
            insights.append(
                f"In {phase} of the holdout ({int(r['weeks'])} weeks), the predicted lowest scorer was the actual lowest "
                f"{pct(r['lowest_top1'])} of the time (top 2: {pct(r['lowest_top2'])}); RMSE {r['rmse']:.2f}."
            )

    # ---- Markdown
    pct_cols = {"top1", "top2", "top3", "pair_acc", "lowest_top1", "lowest_top2", "random_top1", "random_top2",
                *(f"{n}_top{k}" for n in comp for k in (1, 2))}
    d_main, d_order = rmse_ci[main_m], rmse_ci[order_m]
    lines = [
        "# Holdout performance report",
        "",
        f"Holdout seasons {h0}-{h1}. Generated by `scripts/report_holdout.py`. Pre-show model `{main_m}`; "
        f"order-announced model `{order_m}`; elimination model: judges-only (k-th lowest judge total weighted rho^(k-1)).",
        "",
        "**How clean is this holdout?** Layer 1 alphas were tuned on validation seasons only, so the judge-score results "
        "are a true holdout. Every elimination forecast uses rho fit on earlier seasons only, but the judges-only model was "
        f"chosen over the fan models after viewing backtests that included seasons {h0}-{h1}. The holdout has only "
        f"{len(ho)} usable elimination weeks, so every elimination table is repeated on all out-of-sample seasons with Layer 1 "
        f"forecasts ({s_lo}-{s_hi}, {len(wms_all[PROD])} weeks; includes the holdout). Intervals are week-level bootstrap 95% CIs.",
        "",
        "## Key findings",
        "",
        *[f"- {s}" for s in insights],
        "",
        "## 1. Judge-score model (Layer 1)",
        "",
        f"Best simple baseline (picked on validation RMSE): `{best_base}`. `lowest_top1`/`lowest_top2`: share of weeks "
        "where the actual lowest scorer was the model's lowest / among its two lowest predictions.",
        "",
        f"### Holdout (seasons {h0}-{h1})",
        "",
        md_table(l1_hold, pct=pct_cols),
        "",
        f"RMSE difference vs `{best_base}`: pre-show {d_main[0]:+.3f} [{d_main[1]:+.3f}, {d_main[2]:+.3f}], "
        f"order-announced {d_order[0]:+.3f} [{d_order[1]:+.3f}, {d_order[2]:+.3f}]; order-announced minus pre-show "
        f"{rmse_ci_order[0]:+.3f} [{rmse_ci_order[1]:+.3f}, {rmse_ci_order[2]:+.3f}] (negative = better).",
        "",
        f"### Validation + holdout (seasons {v0}-{v1})",
        "",
        md_table(l1_both, pct=pct_cols),
        "",
        "### By season phase",
        "",
        "Holdout:",
        "",
        md_table(l1_phase_hold, pct=pct_cols),
        "",
        f"Seasons {v0}-{v1}:",
        "",
        md_table(l1_phase_both, pct=pct_cols),
        "",
        "### Double lift",
        "",
        "Rows sorted by the ratio of the two models' predictions and cut into 10 equal-count bins; the better model "
        "tracks the actual line, especially in the outer bins where the models disagree most.",
        "",
        "![double lift scores](double_lift_scores.png)",
        "",
        f"{main_m} closer to actual than {best_base} in {(lift_main['closer'] == main_m).sum()}/10 bins; "
        f"{order_m} closer than {main_m} in {(lift_order['closer'] == order_m).sum()}/10 bins.",
        "",
        "## 2. Elimination forecast (Layer 2)",
        "",
        "`top1`/`top2`/`top3`: share of weeks where the eliminated couple was among the model's k most likely "
        "(ties split fractionally). `rank`: the eliminated couple's average position (1 = top pick). Log loss and Brier: lower is better. "
        "All models are scored on the same weeks (the fan models on the overlap with their saved run).",
        "",
        f"### Holdout (seasons {h0}-{h1})",
        "",
        md_table(elim_hold, pct=pct_cols),
        "",
        "Production with 95% CI:",
        "",
        md_table(prod_ci, pct=set()),
        "",
        "Production minus each alternative (positive top-k / negative log loss = production better):",
        "",
        md_table(paired[["vs", "metric", "mean", "lo", "hi", "weeks"]]),
        "",
        f"### All out-of-sample seasons {s_lo}-{s_hi}",
        "",
        md_table(elim_all, pct=pct_cols),
        "",
        "### By season phase",
        "",
        "Holdout:",
        "",
        md_table(phase_hold, pct=pct_cols),
        "",
        f"Seasons {s_lo}-{s_hi}:",
        "",
        md_table(phase_all, pct=pct_cols),
        "",
        f"By couples remaining, seasons {s_lo}-{s_hi}:",
        "",
        md_table(field_all, pct=pct_cols),
        "",
        "### Holdout week by week",
        "",
        md_table(per_week, digits=2, pct={"p_elim"}),
        "",
        "![eliminated rank by week](eliminated_rank_by_week.png)",
        "",
        "### Double lift: production vs lowest score last week",
        "",
        "Couple-weeks sorted by the ratio of the two forecasts' P(eliminated), equal-count bins; the black line is "
        "the observed elimination rate.",
        "",
        "![double lift elimination](double_lift_elimination.png)",
        "",
        *[f"- {k}: production closer in {(g['closer'] == PROD).sum()}/{len(g)} bins." for k, g in lift.items()],
        "",
        "### Calibration",
        "",
        "![calibration](calibration_elimination.png)",
        "",
        "## Notes",
        "",
        "- Production here re-fits rho once per test season on earlier seasons; `predict_upcoming.py` re-fits it on all "
        "weeks before the target week, so live numbers can differ slightly.",
        "- Elimination weeks exclude double eliminations, no-elimination weeks, withdrawals and uneven dance counts "
        "(see docs/LAYER2_POC_PLAN.md).",
        "- Fan model rows (null, m1) are read from the last `run_layer2.py` run, not refit here.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(insights))
    print(f"\nSaved {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
