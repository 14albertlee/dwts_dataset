"""Layer 2 harness: backtest elimination-probability models on the usable weeks (needs run_rule_check.py first).

The default forecast is `lowest_judge` (lower judge total goes home, fitted rho) with Layer 1 judge scores.
The fan models (`null`, `m1`) are the research path; none beat the default (docs/LAYER2_POC_PLAN.md, section 7).

    python scripts/run_layer2.py                      # default model, actual and Layer 1 judge scores
    python scripts/run_layer2.py --models lowest_judge null m1 --compare-to lowest_judge null --validate-null
    python scripts/run_layer2.py --fit-final          # refit the default model on ALL usable weeks (for predict_upcoming)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_judge_scores, load_performance_context, load_weekly_scores  # noqa: E402
from dwts_model.layer2.backtest import (  # noqa: E402
    DEFAULT_MODEL,
    FINAL_PARAMS_FILE,
    MODELS,
    backtest,
    fit_final_params,
    fit_season_params,
    params_table,
)
from dwts_model.layer2.judge_inputs import ActualJudgeInput, Layer1JudgeInput  # noqa: E402
from dwts_model.layer2.metrics import bootstrap_ci, calibration, paired_ci, summarize, week_metrics  # noqa: E402
from dwts_model.layer2.validation import null_self_consistency  # noqa: E402

PAIRED_METRICS = ["logloss", "brier", "top1", "top2", "top3"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=[DEFAULT_MODEL])
    parser.add_argument("--judge-inputs", nargs="+", choices=["actual", "layer1"], default=["actual", "layer1"])
    parser.add_argument("--tie", default=None, help="tie convention (default: config)")
    parser.add_argument("--validate-null", action="store_true", help="simulated-world check of the null model")
    parser.add_argument("--compare-to", nargs="+", choices=sorted(MODELS), default=["null"],
                        help="reference models for paired season-bootstrap differences")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fit-final", action="store_true",
                        help=f"fit the default model on every usable week, save {FINAL_PARAMS_FILE} and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tie = args.tie or cfg.tie_convention
    cw = pd.read_csv(cfg.layer2_dir / "couple_week.csv")
    usable = pd.read_csv(cfg.layer2_dir / "usable_weeks.csv")

    if args.fit_final:
        final = fit_final_params(cw, usable)
        path = cfg.layer2_dir / FINAL_PARAMS_FILE
        path.write_text(json.dumps(final, indent=2))
        s, w = final["last_usable_week"]
        print(f"{final['model']}: rho {final['rho']:.3f} fit on {final['n_weeks']} usable weeks "
              f"(through S{s}W{w}); saved {path}")
        return

    inputs = {"actual": ActualJudgeInput(cw)}
    if "layer1" in args.judge_inputs:
        inputs["layer1"] = Layer1JudgeInput(cfg, load_weekly_scores(cfg), load_performance_context(cfg), load_judge_scores(cfg))

    results = {}
    pd.set_option("display.width", 200)
    for name in args.models:
        spec = MODELS[name]
        params = fit_season_params(cw, usable, spec, cfg, tie)
        table = params_table(params, spec, cw, cfg)
        table.to_csv(cfg.layer2_dir / f"params_{name}_{tie}.csv", index=False)
        print(f"\n{name}: fitted parameters by test season (fit on earlier seasons)")
        print(table.round(3).to_string(index=False))
        for kind in args.judge_inputs:
            probs = backtest(cw, usable, spec, inputs[kind], cfg, params, tie)
            results[(name, kind)] = probs
            probs.to_csv(cfg.layer2_dir / f"forecasts_{name}_{kind}_{tie}.csv", index=False)

    rows, per_week = [], {}
    for kind in args.judge_inputs:
        runs = {k: p for k, p in results.items() if k[1] == kind}
        for (name, _), probs in runs.items():
            wm = week_metrics(probs)
            per_week[(name, kind)] = wm
            rows.append({"model": name, "judge_input": kind, **summarize(wm).iloc[0].drop("group")})
            calibration(probs).to_csv(cfg.layer2_dir / f"calibration_{name}_{kind}_{tie}.csv", index=False)
            summarize(wm, by="field").to_csv(cfg.layer2_dir / f"metrics_by_field_{name}_{kind}_{tie}.csv", index=False)
            summarize(wm, by="era").to_csv(cfg.layer2_dir / f"metrics_by_era_{name}_{kind}_{tie}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(cfg.layer2_dir / f"metrics_{tie}.csv", index=False)
    print(f"\nBacktest metrics, tie={tie} (Layer 1 covers fewer weeks than actual scores; compare within an input)")
    print(table.round(3).to_string(index=False))

    for key, wm in per_week.items():
        by_era = summarize(wm, by="era")[["group", "weeks", "top1", "top3", "logloss", "random_logloss"]]
        print(f"\n{key[0]} / {key[1]} by era")
        print(by_era.round(3).to_string(index=False))

    for (name, kind), wm in per_week.items():
        for ref in args.compare_to:
            if name == ref or (ref, kind) not in per_week:
                continue
            diff = paired_ci(wm, per_week[(ref, kind)], PAIRED_METRICS, args.n_boot, cfg.seed)
            diff.to_csv(cfg.layer2_dir / f"paired_{name}_vs_{ref}_{kind}_{tie}.csv", index=False)
            print(f"\n{name} minus {ref}, {kind} judge scores (season bootstrap 95% CI; log loss and Brier: lower is better)")
            print(diff.round(3).to_string(index=False))

    for key, wm in per_week.items():
        ci = bootstrap_ci(wm, ["top1", "logloss", "brier"], args.n_boot, cfg.seed)
        print(f"\n{key[0]} / {key[1]} (season bootstrap 95% CI)")
        print(ci.round(3).to_string(index=False))

    if args.validate_null:
        if ("null", "actual") not in results:
            sys.exit("--validate-null needs --models null with --judge-inputs actual.")
        check = null_self_consistency(results[("null", "actual")], cw, tie, seed=cfg.seed)
        check.to_csv(cfg.layer2_dir / f"null_validation_{tie}.csv", index=False)
        print("\nNull self-consistency (simulated worlds with random fan ranks vs forecast)")
        print(check.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
