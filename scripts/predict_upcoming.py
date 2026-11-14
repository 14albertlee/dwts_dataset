"""Predict judge scores for a week that has not aired yet.

1. Generate a schedule template for the couples still competing:
       python scripts/predict_upcoming.py --template schedule.csv
2. Fill in the `theme` column (e.g. "Disney Night"; same value on every row) and the `styles`
   column (several dances separated by ';'), then run:
       python scripts/predict_upcoming.py --schedule schedule.csv
   --theme "Disney Night" can be passed instead of (and overrides) the theme column.
3. Optional, once the running order is announced: add an `order` column with each couple's
   1-based position per dance, in the same ';' layout as `styles` (e.g. `3; 11`). Positions are
   within each round (dance 1 of everyone, then dance 2). All couples or none. With an order the
   announced-order model (`order_model` in config.toml) scores the week alongside the pre-show
   model, in the same layout as --backtest (pred, range80 and p_low for each), and the output
   file ends in `_order`; without it only the main model is used.

By default it predicts the week after the last scored week of the latest season, with the main
model from config.toml trained on every scored week before that. Use --season/--week to
predict a different week "as of" that point (history after it is ignored).

Each couple also gets an 80% range and the probability of the lowest judge total (p_lowest, ties
included): the model's mean plus a Gaussian spread from earlier
seasons' out-of-sample residuals, simulated per judge and rounded to integers. Judge scores only;
fan votes are not included. --judges N sets the panel size (default: last scored week's).

Backtest an aired week (no CSV; theme, dances and running order come from the database):
       python scripts/predict_upcoming.py --backtest --season 34 --week 6
It scores the week with both models (pre-show and order-announced), each trained only on weeks
before it, and prints them next to the actual scores.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import (  # noqa: E402
    load_judge_scores,
    load_performance_context,
    load_season_couples,
    load_weekly_scores,
)
from dwts_model.evaluate import _metrics  # noqa: E402
from dwts_model.feature_engineering import build_context_features, build_features  # noqa: E402
from dwts_model.judge_distribution import (  # noqa: E402
    RESID_FROM,
    forecast_night,
    judge_noise_var,
    residual_spread,
    ridge_oos,
)
from dwts_model.models import RidgeModel  # noqa: E402
from dwts_model.predict import build_upcoming_features, check_order, unrecognized_styles  # noqa: E402

BASE_RATE_COLS = ("order_frac", "order_first", "order_last", "order_frac_sq")
N_DRAWS = 10000
SEED = 0


def fit_ridge(cfg, params, model_name, history, season, week):
    set_name = model_name.removeprefix("ridge_")
    if not model_name.startswith("ridge_") or set_name not in params:
        sys.exit(f"No validated Ridge alpha for {model_name}: run --mode validate first.")
    train = history[(history["season"] < season) | ((history["season"] == season) & (history["week"] < week))]
    model = RidgeModel(cfg.features[set_name], params[set_name], name=model_name).fit(train)
    return model, len(train)


def forecast(cfg, params, model_name, history, judges, season, week, mean, n_slots, draws=N_DRAWS):
    """Interval and P(lowest judge total) per couple, with spread from the same model's earlier-season residuals.

    `mean` is the model's (unclipped) prediction per couple. Only seasons before `season` set the spread.
    """
    set_name = model_name.removeprefix("ridge_")
    oos = ridge_oos(history, cfg.features[set_name], params[set_name], RESID_FROM, season - 1)
    spread = residual_spread(oos, season)
    return forecast_night(
        mean, week, n_slots, spread, judge_noise_var(judges, season),
        np.random.default_rng([SEED, season, week]), n_draws=draws, score_range=(cfg.score_min, cfg.score_max),
    )


def panel_size(judges, season, week) -> int:
    """Most common number of judges per performance in the given week (0 if the week has no scores)."""
    w = judges[(judges["season"] == season) & (judges["week"] == week)]
    counts = w.groupby("performance_id")["judge_score"].count()
    return int(counts.mode().iloc[0]) if len(counts) else 0


DEFAULT_SAVE_DIR = Path("prediction_files") / "outputs"


def save_csv(df, save_file, default_name):
    """Write df to save_file (a .csv path, or a directory that gets default_name); return the path."""
    path = Path(save_file)
    if path.suffix.lower() != ".csv":
        path = path / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def backtest(cfg, weekly, perf, couples, judges, season, week, save_file=None) -> None:
    """Score an aired week with both models and compare with the actual scores.

    The week's own rows from build_features hold only earlier-week history plus that week's
    schedule facts, the same inputs a schedule CSV would give. The pre-show run swaps in the
    random-order base rates. P(lowest) uses the week's actual number of judge scores per couple.
    """
    params = json.loads((cfg.output_dir / "selected_params.json").read_text())["ridge_alpha"]
    history = build_features(weekly, build_context_features(perf))
    rows = history[(history["season"] == season) & (history["week"] == week)].copy()
    if rows.empty:
        sys.exit(f"No scored rows for season {season} week {week} (week 1 has no history and is not supported).")

    main, n_train = fit_ridge(cfg, params, cfg.main_model, history, season, week)
    order, _ = fit_ridge(cfg, params, cfg.order_model, history, season, week)
    preshow = rows.copy()
    for col in BASE_RATE_COLS:
        preshow[col] = preshow[f"base_{col}"]
    raw_pre, raw_ord = main.predict(preshow), order.predict(rows)
    rows["pred_preshow"] = raw_pre.clip(cfg.score_min, cfg.score_max)
    rows["pred_order"] = raw_ord.clip(cfg.score_min, cfg.score_max)

    slots = judges[(judges["season"] == season) & (judges["week"] == week)].groupby("couple_id")["judge_score"].count()
    n_slots = rows["couple_id"].map(slots).fillna(slots.median()).to_numpy()
    for tag, model_name, raw in (("preshow", cfg.main_model, raw_pre), ("order", cfg.order_model, raw_ord)):
        fc = forecast(cfg, params, model_name, history, judges, season, week, raw.to_numpy(), n_slots)
        rows[f"p_low_{tag}"] = fc["p_lowest"].to_numpy()
        rows[f"range80_{tag}"] = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(fc["lo"], fc["hi"])]

    wk = perf[(perf["season"] == season) & (perf["week"] == week)].sort_values(["round", "performance_id"])
    by_couple = wk.groupby("couple_id")
    info = pd.DataFrame(
        {
            "styles": by_couple["style"].agg(lambda s: "; ".join(s.fillna("?"))),
            "order": by_couple.apply(
                lambda g: "; ".join(f"{int(o)}/{int(n)}" for o, n in zip(g["running_order"], g["round_size"])),
                include_groups=False,
            ),
        }
    ).reset_index()
    theme = wk["theme"].dropna().iloc[0] if wk["theme"].notna().any() else "?"
    names = couples[couples["season"] == season][["couple_id", "celebrity", "professional"]]
    out = rows.merge(names, on="couple_id").merge(info, on="couple_id")
    out = out.rename(columns={"score": "actual"}).sort_values("actual", ascending=False).reset_index(drop=True)
    out["actual_rank"] = out["actual"].rank(ascending=False, method="min").astype(int)
    out = out[["actual_rank", "celebrity", "professional", "styles", "order", "prev_score", "actual",
               "pred_preshow", "range80_preshow", "p_low_preshow", "pred_order", "range80_order", "p_low_order"]]

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 220
    print(f"Backtest season {season}, week {week} ({theme}); trained on {n_train} rows before it")
    print("order = position/round size per dance; range80 = 80% range of the mean score; "
          "p_low = P(lowest judge total, ties included), judges only")
    print(out.to_string(index=False))
    scored = rows.assign(season=season, week=week)
    for label, col, name in (("pre-show", "pred_preshow", cfg.main_model), ("order", "pred_order", cfg.order_model)):
        m = _metrics(scored, col)
        print(f"{label:9s} {name}: RMSE {m['rmse']:.3f}, pair accuracy {m['pair_acc_pooled']:.3f}, "
              f"Spearman {m['spearman']:.3f}")
    if save_file:
        print(f"Saved {save_csv(out, save_file, f'S{season}W{week}_backtest.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--schedule", help="CSV with columns celebrity, styles, optional theme")
    parser.add_argument("--theme", help="theme of the upcoming week, e.g. 'Disney Night'")
    parser.add_argument("--template", help="write a schedule template to this path and exit")
    parser.add_argument("--backtest", action="store_true",
                        help="score an aired week (--season, --week) from the database, no CSV")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--save-file", nargs="?", const=str(DEFAULT_SAVE_DIR), default=None, metavar="PATH",
                        help="also save the output CSV; PATH is a directory or .csv file "
                             f"(default {DEFAULT_SAVE_DIR}). Without the flag nothing is written")
    parser.add_argument("--coefs", action="store_true", help="also print the fitted Ridge coefficients")
    parser.add_argument("--judges", type=int, default=None,
                        help="judges scoring the upcoming week (default: panel size of the last scored week)")
    parser.add_argument("--draws", type=int, default=N_DRAWS, help="Monte Carlo draws for P(lowest) (default %(default)s)")
    args = parser.parse_args()
    if args.backtest and (args.season is None or args.week is None):
        parser.error("--backtest needs --season and --week")
    if not (args.template or args.schedule or args.backtest):
        parser.error("give --template, --schedule (with a theme column or --theme), or --backtest")

    cfg = load_config(args.config)
    weekly = load_weekly_scores(cfg)
    perf = load_performance_context(cfg)
    couples = load_season_couples(cfg)
    judges = load_judge_scores(cfg)

    if args.backtest:
        backtest(cfg, weekly, perf, couples, judges, args.season, args.week, args.save_file)
        return

    season = args.season or int(weekly["season"].max())
    last_week = int(weekly.loc[weekly["season"] == season, "week"].max())
    week = args.week or last_week + 1
    prior = weekly[(weekly["season"] == season) & (weekly["week"] == week - 1)]

    names = couples[couples["season"] == season]
    if args.template:
        remaining = prior[prior["eliminated"] != True]  # noqa: E712  (NA-safe)
        out = remaining.merge(names, on=["season", "couple_id"])[["celebrity", "professional"]].assign(theme="", styles="", order="")
        out.to_csv(args.template, index=False)
        print(f"Wrote {len(out)} couples still competing in season {season} to {args.template}")
        return

    sched = pd.read_csv(args.schedule)
    theme = args.theme
    if not theme:
        themes = sched["theme"].dropna().astype(str).str.strip().unique().tolist() if "theme" in sched else []
        themes = [t for t in themes if t]
        if len(themes) != 1:
            sys.exit("Give --theme, or fill the theme column with a single theme (e.g. 'Disney Night').")
        theme = themes[0]
    sched["styles"] = sched["styles"].fillna("").map(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])
    if (sched["styles"].map(len) == 0).any():
        sys.exit("Every couple needs at least one dance in the styles column.")
    if "order" in sched:
        def parse_order(v):
            if pd.isna(v) or not str(v).strip():
                return float("nan")
            try:
                return [int(x) for x in str(v).split(";") if x.strip()]
            except ValueError:
                sys.exit(f"Order must be integers separated by ';', got {v!r}.")
        sched["order"] = sched["order"].map(parse_order).astype(object)
    lookup = names.assign(key=names["celebrity"].str.lower().str.strip())
    sched["key"] = sched["celebrity"].str.lower().str.strip()
    sched = sched.drop(columns="professional", errors="ignore")  # use the season's record, not the CSV
    sched = sched.merge(lookup[["key", "couple_id", "professional"]], on="key", how="left")
    if sched["couple_id"].isna().any():
        sys.exit(f"Not in season {season}: {sched.loc[sched['couple_id'].isna(), 'celebrity'].tolist()}")
    sched["couple_id"] = sched["couple_id"].astype(int)

    odd = unrecognized_styles(sched)
    if odd:
        print(f"Note: no dedicated indicator for {sorted(odd)}; treated as 'other'.")

    params = json.loads((cfg.output_dir / "selected_params.json").read_text())["ridge_alpha"]
    try:
        mode = check_order(sched)
    except ValueError as e:
        sys.exit(str(e))
    history = build_features(weekly, build_context_features(perf))
    n_judges = args.judges or panel_size(judges, season, week - 1)
    if n_judges < 1:
        sys.exit("Could not infer the judge panel size from last week; pass --judges N.")
    n_dances = sched.set_index("couple_id")["styles"].map(len)

    def score(model_name, with_order):
        """Fit the model, score the schedule (actual order or base rates), and add the forecast columns."""
        model, n_train = fit_ridge(cfg, params, model_name, history, season, week)
        cols = ["couple_id", "styles"] + (["order"] if with_order else [])
        rows = build_upcoming_features(weekly, perf, sched[cols], season, week, theme)
        missing = set(sched["couple_id"]) - set(rows["couple_id"])
        if missing:
            names_missing = sched.loc[sched["couple_id"].isin(missing), "celebrity"].tolist()
            sys.exit(f"No earlier weeks this season for {names_missing}; week-1 couples are not supported.")
        raw = model.predict(rows)
        n_slots = n_judges * rows["couple_id"].map(n_dances).to_numpy()
        fc = forecast(cfg, params, model_name, history, judges, season, week, raw.to_numpy(), n_slots, args.draws)
        rows["predicted"] = raw.clip(cfg.score_min, cfg.score_max)
        rows["range80"] = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(fc["lo"], fc["hi"])]
        rows["p_lowest"] = fc["p_lowest"].to_numpy()
        return rows, model, n_train

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 220
    info = sched[["couple_id", "celebrity", "professional", "styles"]].copy()
    info["styles"] = info["styles"].map("; ".join)

    if mode == "announced":
        # Both models, laid out like --backtest: pre-show (base-rate order) next to order-announced.
        pre, _, n_train = score(cfg.main_model, False)
        ordered, model, _ = score(cfg.order_model, True)
        model_name = cfg.order_model
        out = ordered.merge(pre[["couple_id", "predicted", "range80", "p_lowest"]], on="couple_id",
                            suffixes=("_order", "_preshow"))
        out = out.rename(columns={"predicted_order": "pred_order", "p_lowest_order": "p_low_order",
                                  "predicted_preshow": "pred_preshow", "p_lowest_preshow": "p_low_preshow"})
        out = out.merge(info, on="couple_id").merge(
            sched[["couple_id", "order"]].assign(order=lambda d: d["order"].map(lambda o: "; ".join(map(str, o)))),
            on="couple_id",
        )
        out = out.sort_values("pred_order", ascending=False).reset_index(drop=True)
        out.insert(0, "pred_rank", range(1, len(out) + 1))
        out = out[["pred_rank", "celebrity", "professional", "styles", "order", "prev_score", "running_mean",
                   "prev_bottom3", "pred_preshow", "range80_preshow", "p_low_preshow",
                   "pred_order", "range80_order", "p_low_order"]]
        header = f"announced mode: pre-show {cfg.main_model} and order {cfg.order_model} (ranked by order model)"
        legend = "range80 = 80% range of the mean score; p_low = P(lowest judge total, ties included)"
    else:
        model_name = cfg.main_model
        rows, model, n_train = score(model_name, False)
        out = rows.merge(info, on="couple_id").sort_values("predicted", ascending=False).reset_index(drop=True)
        out.insert(0, "pred_rank", range(1, len(out) + 1))
        out = out[["pred_rank", "celebrity", "professional", "styles", "prev_score", "running_mean",
                   "prev_bottom3", "predicted", "range80", "p_lowest"]]
        header = f"base-rate mode, model {model_name}"
        legend = "range80 = 80% range of the mean score; p_lowest = P(lowest judge total, ties included)"

    suffix = "_order" if mode == "announced" else "_noorder"
    print(f"Season {season}, week {week} ({theme}); trained on {n_train} rows; {header}")
    print(f"{legend}; {n_judges} judges. Judges only, fan votes not included.")
    print(out.to_string(index=False))
    if args.save_file:
        # Named after the input schedule (e.g. S35W10.csv -> S35W10_order.csv)
        print(f"Saved {save_csv(out, args.save_file, Path(args.schedule).stem + suffix + '.csv')}")
    if args.coefs:
        names_out = model.pipe_[0].get_feature_names_out(model.features)
        coefs = pd.Series(model.pipe_[-1].coef_, index=names_out).sort_values(key=abs, ascending=False)
        print("\nRidge coefficients (per 1 SD of each predictor):")
        print(coefs.to_string())


if __name__ == "__main__":
    main()
