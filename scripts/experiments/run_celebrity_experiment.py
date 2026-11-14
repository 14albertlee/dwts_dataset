"""Experiment 3.4: celebrity background on top of the main (pre-show) model.

    python scripts/experiments/run_celebrity_experiment.py

Validation seasons only. Variants:
  celeb_onehot        - indicators for the celebrity_group levels (reference: actor).
  celeb_onehot_decay  - the same plus indicators / (week - 1), so the effect can be large when the
                        couple has little history of its own and fade as weeks accumulate.
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
    build_celebrity_features,
    build_context_features,
    build_features,
)
from dwts_model.models import tune_ridge_alpha  # noqa: E402
from run_order_experiment import pooled  # noqa: E402
from run_pro_experiment import predict, report  # noqa: E402

SEED = 0


def main() -> None:
    cfg = load_config()
    base = build_features(load_weekly_scores(cfg), build_context_features(load_performance_context(cfg)))
    celeb = build_celebrity_features(load_couple_info(cfg))
    celeb_cols = [c for c in celeb.columns if c.startswith("celeb_")]
    df = base.merge(celeb, on=["season", "couple_id"], how="left")
    df[celeb_cols] = df[celeb_cols].fillna(0.0)
    decay_cols = [f"{c}_decay" for c in celeb_cols]
    df[decay_cols] = df[celeb_cols].div(df["week"] - 1, axis=0).to_numpy()

    main_feats = list(cfg.features[cfg.main_model.removeprefix("ridge_")])
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 140
    val = df[df.season.between(*cfg.validation)]
    print("validation rows by group:", val[celeb_cols].sum().astype(int).to_dict(),
          "actor:", int((val[celeb_cols].sum(axis=1) == 0).sum()))

    variants = {
        "main": main_feats,
        "celeb_onehot": main_feats + celeb_cols,
        "celeb_onehot_decay": main_feats + celeb_cols + decay_cols,
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

    for name in list(variants)[1:]:
        print(f"\n{name} coefficients (points), by fold:")
        print(coefs[name][[c for c in variants[name] if c.startswith("celeb_")]].to_string())

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cfg.output_dir / "celebrity_experiment_validate.csv", index=False)
    tables = summarize(preds, list(variants), cfg.short_seasons)
    tables["by_season"].to_csv(cfg.output_dir / "celebrity_experiment_by_season_validate.csv", index=False)
    tables["by_week"].to_csv(cfg.output_dir / "celebrity_experiment_by_week_validate.csv", index=False)


if __name__ == "__main__":
    main()
