"""Experiment 3.2: pro partner history on top of the main (pre-show) model.

    python scripts/experiments/run_pro_experiment.py

Validation seasons only. Variants:
  pro_prior_dev  - the pro's shrunk mean deviation from the field over earlier seasons;
                   shrinkage k and Ridge alpha tuned jointly on pooled validation MAE.
  pro_onehot     - one indicator per professional_id (pros new to a test season are all-zero).
Confounding checks: the pro_prior_dev coefficient with and without the couple's own history
(prev_score, running_mean), and metrics on weeks 2-3 vs later weeks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_couple_info, load_performance_context, load_weekly_scores  # noqa: E402
from dwts_model.evaluate import summarize  # noqa: E402
from dwts_model.feature_engineering import (  # noqa: E402
    build_context_features,
    build_features,
    build_pro_history,
)
from dwts_model.models import RidgeModel, tune_ridge_alpha  # noqa: E402
from dwts_model.splits import rolling_origin  # noqa: E402
from run_order_experiment import paired, pooled, week_bootstrap  # noqa: E402

K_GRID = [0, 5, 10, 20, 40, 80, 160]
SEED = 0


def with_pro_dev(df, weekly, couple_pros, k):
    pro = build_pro_history(weekly, couple_pros, k)
    return df.drop(columns=["pro_prior_dev", "pro_prior_n"], errors="ignore").merge(
        pro, on=["season", "couple_id"], how="left"
    ).fillna({"pro_prior_dev": 0.0, "pro_prior_n": 0.0})


def predict(df, cfg, feats, alpha, name):
    frames, coefs = [], []
    for season, train, test in rolling_origin(df, cfg, "validate"):
        model = RidgeModel(feats, alpha).fit(train)
        out = test[["season", "week", "couple_id", "score"]].copy()
        out[name] = model.predict(test).clip(cfg.score_min, cfg.score_max)
        frames.append(out)
        ridge = model.pipe_[-1]
        scale = model.pipe_[1].scale_[: len(feats)]
        coefs.append(pd.Series(ridge.coef_[: len(feats)] / scale, index=feats, name=season))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coefs)


def report(preds, ref, col, rng):
    d = paired(preds, ref, col)
    best = d.d_pair_acc.idxmax()
    pr = preds[preds.season != best]
    r0, p0, _ = pooled(pr, ref)
    r1, p1, _ = pooled(pr, col)
    ci_r, ci_p = week_bootstrap(preds, ref, col, rng)
    print(f"\n== {col} vs {ref}")
    print(d.to_string())
    print(f"seasons better: RMSE {int((d.d_rmse < 0).sum())}/7, pair_acc {int((d.d_pair_acc > 0).sum())}/7")
    print(f"without best season ({best}): dRMSE {r1 - r0:+.4f}, dPair {p1 - p0:+.4f}")
    print(f"week-block bootstrap 95% CI: dRMSE [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}], dPair [{ci_p[0]:+.4f}, {ci_p[1]:+.4f}]")
    for label, mask in (("weeks 2-3", preds.week <= 3), ("weeks 4+", preds.week > 3)):
        r0, p0, _ = pooled(preds[mask], ref)
        r1, p1, _ = pooled(preds[mask], col)
        print(f"  {label} (n={int(mask.sum())}): dRMSE {r1 - r0:+.4f}, dPair {p1 - p0:+.4f}")


def main() -> None:
    cfg = load_config()
    weekly = load_weekly_scores(cfg)
    couple_pros = load_couple_info(cfg)
    base = build_features(weekly, build_context_features(load_performance_context(cfg)))
    main_feats = list(cfg.features[cfg.main_model.removeprefix("ridge_")])
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 140

    df = with_pro_dev(base, weekly, couple_pros, 0)
    val = df[df.season.between(*cfg.validation)]
    print(f"validation rows with earlier-season pro history: {(val.pro_prior_n > 0).mean():.2f}")

    # Joint (k, alpha) search on pooled validation MAE.
    feats_dev = main_feats + ["pro_prior_dev"]
    grid = {}
    for k in K_GRID:
        alpha, maes = tune_ridge_alpha(with_pro_dev(base, weekly, couple_pros, k), cfg, feats_dev)
        grid[k] = (alpha, maes[alpha])
    print("k -> (alpha, MAE):", {k: (a, round(m, 4)) for k, (a, m) in grid.items()})
    best_k = min(grid, key=lambda k: grid[k][1])
    df = with_pro_dev(base, weekly, couple_pros, best_k)

    pro_id = df.merge(couple_pros, on=["season", "couple_id"], how="left")["professional_id"]
    onehot = pd.get_dummies(pro_id, prefix="pro").astype(float)
    df = pd.concat([df, onehot], axis=1)

    variants = {
        "main": main_feats,
        f"pro_prior_dev_k{best_k}": feats_dev,
        "pro_onehot": main_feats + list(onehot.columns),
    }
    frames, coefs = [], {}
    for name, feats in variants.items():
        alpha, _ = tune_ridge_alpha(df, cfg, feats)
        print(f"{name}: alpha={alpha}")
        p, coefs[name] = predict(df, cfg, feats, alpha, name)
        frames.append(p)
    preds = frames[0]
    for f in frames[1:]:
        preds = preds.merge(f.drop(columns="score"), on=["season", "week", "couple_id"])

    print("\nPooled (validation 26-32):")
    print(pd.DataFrame([dict(model=c, **dict(zip(["rmse", "pair_acc", "spearman"], pooled(preds, c))))
                        for c in variants]).to_string(index=False))
    rng = np.random.default_rng(SEED)
    for name in list(variants)[1:]:
        report(preds, "main", name, rng)

    # Confounding: does the pro effect shrink once the couple's own history is in the model?
    dev_name = f"pro_prior_dev_k{best_k}"
    no_hist = [f for f in feats_dev if f not in ("prev_score", "running_mean")]
    _, c_no_hist = predict(df, cfg, no_hist, grid[best_k][0], "tmp")
    print("\npro_prior_dev coefficient (points per point of prior dev), by fold:")
    print(pd.DataFrame({"with_history": coefs[dev_name]["pro_prior_dev"],
                        "without_history": c_no_hist["pro_prior_dev"]}).to_string())

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cfg.output_dir / "pro_experiment_validate.csv", index=False)
    tables = summarize(preds, list(variants), cfg.short_seasons)
    tables["by_season"].to_csv(cfg.output_dir / "pro_experiment_by_season_validate.csv", index=False)
    tables["by_week"].to_csv(cfg.output_dir / "pro_experiment_by_week_validate.csv", index=False)


if __name__ == "__main__":
    main()
