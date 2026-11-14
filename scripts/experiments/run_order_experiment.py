"""Experiment 3.1: within-night running order features on top of the main model.

    python scripts/experiments/run_order_experiment.py

Validation seasons only. Each variant is fit on the actual order and scored in two modes:
base-rate (order features replaced by random-order expectations) and announced (actual order).
Alpha is tuned per variant on base-rate pooled MAE, as in docs/LAYER1_NEXT_EXPERIMENTS.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_performance_context, load_weekly_scores  # noqa: E402
from dwts_model.evaluate import _metrics, summarize  # noqa: E402
from dwts_model.feature_engineering import build_context_features, build_features  # noqa: E402
from dwts_model.models import RidgeModel  # noqa: E402
from dwts_model.splits import rolling_origin  # noqa: E402

VARIANTS = {
    "order_frac": ["order_frac"],
    "order_frac_fl": ["order_frac", "order_first", "order_last"],
    "order_frac_fl_sq": ["order_frac", "order_first", "order_last", "order_frac_sq"],
}
SEED = 0
N_BOOT = 2000


def base_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("order_frac", "order_first", "order_last", "order_frac_sq"):
        out[col] = out[f"base_{col}"]
    return out


def shuffled_view(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Placebo: permute the order features among couples within each (season, week)."""
    out = df.copy()
    cols = ["order_frac", "order_first", "order_last", "order_frac_sq"]
    for _, idx in out.groupby(["season", "week"]).groups.items():
        out.loc[idx, cols] = out.loc[rng.permutation(idx), cols].to_numpy()
    return out


def run_variant(df, cfg, name, feats, alpha, alpha_announced):
    rng = np.random.default_rng(SEED)
    frames = []
    for _, train, test in rolling_origin(df, cfg, "validate"):
        model = RidgeModel(feats, alpha).fit(train)
        out = test[["season", "week", "couple_id", "score"]].copy()
        out[f"{name}|base"] = model.predict(base_view(test)).clip(cfg.score_min, cfg.score_max)
        out[f"{name}|announced"] = model.predict(test).clip(cfg.score_min, cfg.score_max)
        out[f"{name}|placebo"] = model.predict(shuffled_view(test, rng)).clip(cfg.score_min, cfg.score_max)
        own = RidgeModel(feats, alpha_announced).fit(train)
        out[f"{name}|announced_own_alpha"] = own.predict(test).clip(cfg.score_min, cfg.score_max)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def tune_alpha_base(df, cfg, feats, view=base_view):
    maes = {}
    for alpha in cfg.ridge_alphas:
        errs = []
        for _, train, test in rolling_origin(df, cfg, "validate"):
            pred = RidgeModel(feats, alpha).fit(train).predict(view(test)).clip(cfg.score_min, cfg.score_max)
            errs.append((pred - test["score"]).abs())
        maes[alpha] = float(pd.concat(errs).mean())
    return min(maes, key=maes.get)


def season_table(preds, cols):
    rows = []
    for s, g in preds.groupby("season"):
        for c in cols:
            m = _metrics(g, c)
            rows.append({"season": s, "model": c, "rmse": m["rmse"], "pair_acc": m["pair_acc_pooled"], "n": m["n"]})
    return pd.DataFrame(rows)


def paired(preds, a, b):
    """Per-season paired differences (b - a): RMSE (negative is better) and pair accuracy (positive is better)."""
    t = season_table(preds, [a, b])
    ra = t[t.model == a].set_index("season")
    rb = t[t.model == b].set_index("season")
    return pd.DataFrame({"d_rmse": rb.rmse - ra.rmse, "d_pair_acc": rb.pair_acc - ra.pair_acc})


def pooled(preds, c):
    m = _metrics(preds, c)
    return m["rmse"], m["pair_acc_pooled"], m["spearman"]


def week_bootstrap(preds, a, b, rng):
    """Resample (season, week) blocks; CI for the pooled RMSE and pair-accuracy difference (b - a)."""
    blocks = [g for _, g in preds.groupby(["season", "week"])]
    pair = {}
    for c in (a, b):
        pair[c] = [_metrics_block(g, c) for g in blocks]
    sq = {c: np.array([x[0] for x in pair[c]]) for c in (a, b)}
    n = np.array([x[1] for x in pair[a]])
    cor = {c: np.array([x[2] for x in pair[c]]) for c in (a, b)}
    pr = np.array([x[3] for x in pair[a]])
    d_rmse, d_pair = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(blocks), len(blocks))
        d_rmse.append(np.sqrt(sq[b][i].sum() / n[i].sum()) - np.sqrt(sq[a][i].sum() / n[i].sum()))
        d_pair.append(cor[b][i].sum() / pr[i].sum() - cor[a][i].sum() / pr[i].sum())
    q = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
    return q(d_rmse), q(d_pair)


def _metrics_block(g, c):
    from dwts_model.evaluate import _pair_counts

    correct, pairs = _pair_counts(g["score"].to_numpy(), g[c].to_numpy())
    return float(((g[c] - g["score"]) ** 2).sum()), len(g), correct, pairs


def main() -> None:
    cfg = load_config()
    df = build_features(load_weekly_scores(cfg), build_context_features(load_performance_context(cfg)))
    main_feats = list(cfg.features[cfg.main_model.removeprefix("ridge_")])
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 140

    n_order = df["order_frac"].ne(df["base_order_frac"]).mean()
    print(f"share of rows with non-neutral actual order: {n_order:.2f}")

    # Control: main model re-tuned the same way, so every variant differs only by its order features.
    alphas, alphas_ann = {}, {}
    frames = []
    rng = np.random.default_rng(SEED)
    for name, extra in [("main", [])] + list(VARIANTS.items()):
        feats = main_feats + extra
        alphas[name] = tune_alpha_base(df, cfg, feats)
        alphas_ann[name] = tune_alpha_base(df, cfg, feats, view=lambda t: t)
        frames.append(run_variant(df, cfg, name, feats, alphas[name], alphas_ann[name]))
    preds = frames[0]
    for f in frames[1:]:
        preds = preds.merge(f.drop(columns="score"), on=["season", "week", "couple_id"])
    print("alpha tuned on base-rate MAE:", alphas)
    print("alpha tuned on announced MAE:", alphas_ann)

    cols = [c for c in preds.columns if "|" in c]
    print("\nPooled (validation 26-32):")
    print(pd.DataFrame([dict(model=c, **dict(zip(["rmse", "pair_acc", "spearman"], pooled(preds, c)))) for c in cols])
          .to_string(index=False))

    ref = "main|base"
    for v in VARIANTS:
        for mode in ("base", "announced", "announced_own_alpha", "placebo"):
            if mode == "placebo" and v != "order_frac_fl":
                continue
            c = f"{v}|{mode}"
            d = paired(preds, ref, c)
            best = d.d_pair_acc.idxmax()
            pr = preds[preds.season != best]
            r0, p0, _ = pooled(pr, ref)
            r1, p1, _ = pooled(pr, c)
            ci_r, ci_p = week_bootstrap(preds, ref, c, rng)
            print(f"\n== {c} vs {ref}")
            print(d.to_string())
            print(f"seasons better: RMSE {int((d.d_rmse < 0).sum())}/7, pair_acc {int((d.d_pair_acc > 0).sum())}/7")
            print(f"without best season ({best}): dRMSE {r1 - r0:+.4f}, dPair {p1 - p0:+.4f}")
            print(f"week-block bootstrap 95% CI: dRMSE [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}], dPair [{ci_p[0]:+.4f}, {ci_p[1]:+.4f}]")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cfg.output_dir / "order_experiment_validate.csv", index=False)
    tables = summarize(preds.rename(columns={c: c.replace("|", "_") for c in cols}), [c.replace("|", "_") for c in cols], cfg.short_seasons)
    tables["by_season"].to_csv(cfg.output_dir / "order_experiment_by_season_validate.csv", index=False)
    tables["by_week"].to_csv(cfg.output_dir / "order_experiment_by_week_validate.csv", index=False)


if __name__ == "__main__":
    main()
