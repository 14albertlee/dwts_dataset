"""Layer 1 entry point: train and evaluate judge-score models.

    python scripts/run_layer1.py --mode validate   # tune on seasons 26-32, save alpha
    python scripts/run_layer1.py --mode holdout    # final run on 33-35 with the saved alpha
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.baselines import all_baselines  # noqa: E402
from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_performance_context, load_weekly_scores  # noqa: E402
from dwts_model.evaluate import summarize  # noqa: E402
from dwts_model.feature_engineering import build_context_features, build_features  # noqa: E402
from dwts_model.models import RidgeModel, tune_ridge_alpha  # noqa: E402
from dwts_model.splits import rolling_origin  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=["validate", "holdout"], required=True)
    parser.add_argument("--confirm-holdout", action="store_true", help="required for --mode holdout")
    parser.add_argument("--feature-sets", nargs="+", default=None, help="subset of config feature sets to run")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.mode == "holdout" and not args.confirm_holdout:
        sys.exit("Holdout can only be scored deliberately: pass --confirm-holdout.")

    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    params_path = cfg.output_dir / "selected_params.json"

    df = build_features(load_weekly_scores(cfg), build_context_features(load_performance_context(cfg)))

    feature_sets = {k: v for k, v in cfg.features.items() if args.feature_sets is None or k in args.feature_sets}
    unknown = set(args.feature_sets or []) - set(cfg.features)
    if unknown:
        sys.exit(f"Unknown feature sets: {sorted(unknown)}")

    if args.mode == "validate":
        alphas, maes = {}, {}
        for name, feats in feature_sets.items():
            alphas[name], maes[name] = tune_ridge_alpha(df, cfg, feats)
            print(f"ridge_{name}: alpha={alphas[name]} validation MAE by alpha {maes[name]}")
        params_path.write_text(json.dumps({"ridge_alpha": alphas, "validation_mae": maes}, indent=2))
    else:
        if not params_path.exists():
            sys.exit("No selected_params.json: run --mode validate first.")
        alphas = json.loads(params_path.read_text())["ridge_alpha"]
        missing = set(feature_sets) - set(alphas)
        if missing:
            sys.exit(f"No validated alpha for {sorted(missing)}: run --mode validate first.")
        print(f"Holdout run with frozen alphas={ {k: alphas[k] for k in feature_sets} }")

    models = all_baselines() + [
        RidgeModel(feats, alphas[name], name=f"ridge_{name}") for name, feats in feature_sets.items()
    ]
    names = [m.name for m in models]

    frames = []
    for season, train, test in rolling_origin(df, cfg, args.mode):
        out = test[["season", "week", "couple_id", "score"]].copy()
        for model in models:
            pred = model.fit(train).predict(test)
            out[model.name] = pred.clip(cfg.score_min, cfg.score_max)
        frames.append(out)
    preds = pd.concat(frames, ignore_index=True)

    tables = summarize(preds, names, cfg.short_seasons)
    preds.to_csv(cfg.output_dir / f"predictions_{args.mode}.csv", index=False)
    for key, table in tables.items():
        table.to_csv(cfg.output_dir / f"metrics_{key}_{args.mode}.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 120
    print("\nOverall:\n", tables["overall"].to_string(index=False))
    print("\nBy season:\n", tables["by_season"].to_string(index=False))
    print("\nBy week:\n", tables["by_week"].to_string(index=False))


if __name__ == "__main__":
    main()
