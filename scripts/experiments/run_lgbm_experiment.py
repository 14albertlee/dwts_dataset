"""Experiment 3.6: LightGBM with the kept features, plus a re-test of predictors Ridge rejected.

    python scripts/experiments/run_lgbm_experiment.py

Validation seasons only. Every LightGBM variant gets its own params from the [model.lgbm] grid,
tuned on pooled validation MAE (same folds as Ridge). Variants:
  ridge_main / lgbm_main      - the main (pre-show) feature set.
  blend_main                  - fixed 50/50 average of the two (no tuned weight).
  lgbm_style_cat              - style as one native categorical (the couple's first dance of the
                                week) instead of the one-hot shares.
  lgbm_pro_cat                - + professional_id as a native categorical.
  lgbm_standing / _celeb_cat / _pro_dev / _theme_all
                              - + predictors that did not help Ridge (3.2, 3.4, prev standing,
                                theme categories); trees can use interactions Ridge cannot.
  ridge_order / lgbm_order    - announced-order models (main_order), scored on the actual order;
                                lgbm_order is also scored at base rates as a check.
SHAP directions come from LightGBM's own TreeSHAP (pred_contrib=True) on the last fold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dwts_model.categories import celebrity_group, normalize_style  # noqa: E402
from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_couple_info, load_performance_context, load_weekly_scores  # noqa: E402
from dwts_model.evaluate import summarize  # noqa: E402
from dwts_model.feature_engineering import (  # noqa: E402
    build_context_features,
    build_features,
    build_pro_history,
)
from dwts_model.models import LGBMModel, RidgeModel, tune_lgbm, tune_ridge_alpha  # noqa: E402
from dwts_model.splits import rolling_origin  # noqa: E402
from run_order_experiment import base_view, pooled  # noqa: E402
from run_pro_experiment import report  # noqa: E402

SEED = 0
PRO_K = 40
KEYS = ["season", "week", "couple_id"]


def load_frame(cfg) -> pd.DataFrame:
    perf = load_performance_context(cfg)
    weekly = load_weekly_scores(cfg)
    info = load_couple_info(cfg)
    df = build_features(weekly, build_context_features(perf))

    first = perf.sort_values(["round", "performance_id"]).groupby(KEYS, as_index=False).first()
    first["style_cat"] = first["style"].map(normalize_style)
    df = df.merge(first[KEYS + ["style_cat"]], on=KEYS, how="left")

    info = info.assign(celeb_cat=info["notability"].map(celebrity_group))
    df = df.merge(info[["season", "couple_id", "professional_id", "celeb_cat"]].drop_duplicates(["season", "couple_id"]),
                  on=["season", "couple_id"], how="left")
    df = df.merge(build_pro_history(weekly, info, PRO_K), on=["season", "couple_id"], how="left")
    df[["pro_prior_dev", "pro_prior_n"]] = df[["pro_prior_dev", "pro_prior_n"]].fillna(0.0)

    # Categories fixed on the full frame so train and test folds share one encoding.
    for c in ("style_cat", "professional_id", "celeb_cat"):
        df[c] = df[c].astype(str).astype("category")
    return df


def fold_predict(df, cfg, make, name, view=None):
    frames, models = [], []
    for _, train, test in rolling_origin(df, cfg, "validate"):
        model = make().fit(train)
        out = test[KEYS + ["score"]].copy()
        out[name] = model.predict(test if view is None else view(test)).clip(cfg.score_min, cfg.score_max)
        frames.append(out)
        models.append((model, test))
    return pd.concat(frames, ignore_index=True), models


def shap_summary(model: LGBMModel, test: pd.DataFrame) -> pd.DataFrame:
    x = test[model.features]
    contrib = model.model_.predict(x, pred_contrib=True)[:, :-1]
    rows = []
    for i, f in enumerate(model.features):
        v = x[f]
        num = pd.api.types.is_numeric_dtype(v) and v.nunique() > 1
        rows.append({
            "feature": f,
            "mean_abs_shap": float(np.abs(contrib[:, i]).mean()),
            "spearman_value_vs_shap": float(pd.Series(contrib[:, i]).corr(v.reset_index(drop=True), method="spearman"))
            if num else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def main() -> None:
    cfg = load_config()
    df = load_frame(cfg)
    main_feats = list(cfg.features[cfg.main_model.removeprefix("ridge_")])
    order_feats = list(cfg.features[cfg.order_model.removeprefix("ridge_")])
    style_onehots = [f for f in main_feats if f.startswith("style_")]
    theme_extra = [f for f in cfg.features["theme_all"] if f.startswith("theme_") and f not in main_feats]
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 140

    lgbm_variants = {
        "lgbm_main": main_feats,
        "lgbm_style_cat": [f for f in main_feats if f not in style_onehots] + ["style_cat"],
        "lgbm_pro_cat": main_feats + ["professional_id"],
        "lgbm_standing": main_feats + ["prev_rank", "prev_dev"],
        "lgbm_celeb_cat": main_feats + ["celeb_cat"],
        "lgbm_pro_dev": main_feats + ["pro_prior_dev", "pro_prior_n"],
        "lgbm_theme_all": main_feats + theme_extra,
    }

    frames, fitted, params = [], {}, {}
    alpha, _ = tune_ridge_alpha(df, cfg, main_feats)
    p, _ = fold_predict(df, cfg, lambda: RidgeModel(main_feats, alpha), "ridge_main")
    frames.append(p)
    for name, feats in lgbm_variants.items():
        params[name], _ = tune_lgbm(df, cfg, feats)
        print(f"{name}: {params[name]}", flush=True)
        p, fitted[name] = fold_predict(df, cfg, lambda f=feats, n=name: LGBMModel(f, params[n]), name)
        frames.append(p)
    preds = frames[0]
    for f in frames[1:]:
        preds = preds.merge(f.drop(columns="score"), on=KEYS)
    preds["blend_main"] = (preds["ridge_main"] + preds["lgbm_main"]) / 2

    cols = ["ridge_main", "lgbm_main", "blend_main"] + list(lgbm_variants)[1:]
    print("\nPooled, pre-show (validation 26-32):")
    print(pd.DataFrame([dict(model=c, **dict(zip(["rmse", "pair_acc", "spearman"], pooled(preds, c))))
                        for c in cols]).to_string(index=False))
    rng = np.random.default_rng(SEED)
    report(preds, "ridge_main", "lgbm_main", rng)
    report(preds, "ridge_main", "blend_main", rng)
    for name in list(lgbm_variants)[1:]:
        report(preds, "lgbm_main", name, rng)

    # Announced order: own models on main_order; lgbm also scored at base rates.
    alpha_o, _ = tune_ridge_alpha(df, cfg, order_feats)
    params["lgbm_order"], _ = tune_lgbm(df, cfg, order_feats)
    print(f"\nlgbm_order: {params['lgbm_order']}")
    po, _ = fold_predict(df, cfg, lambda: RidgeModel(order_feats, alpha_o), "ridge_order")
    for name, view in (("lgbm_order", None), ("lgbm_order_base", base_view)):
        p, _ = fold_predict(df, cfg, lambda: LGBMModel(order_feats, params["lgbm_order"]), name, view)
        po = po.merge(p.drop(columns="score"), on=KEYS)
    po["blend_order"] = (po["ridge_order"] + po["lgbm_order"]) / 2
    po = po.merge(preds[KEYS + ["ridge_main", "lgbm_main"]], on=KEYS)
    ocols = ["ridge_order", "lgbm_order", "blend_order", "lgbm_order_base"]
    print("\nPooled, order models (validation 26-32):")
    print(pd.DataFrame([dict(model=c, **dict(zip(["rmse", "pair_acc", "spearman"], pooled(po, c))))
                        for c in ocols]).to_string(index=False))
    report(po, "ridge_order", "lgbm_order", rng)
    report(po, "ridge_order", "blend_order", rng)
    report(po, "lgbm_main", "lgbm_order_base", rng)

    model, test = fitted["lgbm_main"][-1]
    print(f"\nlgbm_main SHAP on season {int(test.season.iloc[0])} (target={model.target_}; "
          "for delta, prev_score also enters 1:1 outside the trees):")
    print(shap_summary(model, test).head(12).to_string(index=False))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cfg.output_dir / "lgbm_experiment_validate.csv", index=False)
    po.to_csv(cfg.output_dir / "lgbm_order_experiment_validate.csv", index=False)
    tables = summarize(preds, cols, cfg.short_seasons)
    tables["by_season"].to_csv(cfg.output_dir / "lgbm_experiment_by_season_validate.csv", index=False)
    tables["by_week"].to_csv(cfg.output_dir / "lgbm_experiment_by_week_validate.csv", index=False)


if __name__ == "__main__":
    main()
