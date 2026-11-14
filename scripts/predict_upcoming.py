"""Predict judge scores for a week that has not aired yet.

1. Generate a schedule template for the couples still competing:
       python scripts/predict_upcoming.py --template schedule.csv
2. Fill in the `theme` column (e.g. "Disney Night"; same value on every row) and the `styles`
   column (several dances separated by ';'), then run:
       python scripts/predict_upcoming.py --schedule schedule.csv
   --theme "Disney Night" can be passed instead of (and overrides) the theme column.

By default it predicts the week after the last scored week of the latest season, with the main
model from config.toml trained on every scored week before that. Use --season/--week to
predict a different week "as of" that point (history after it is ignored).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_performance_context, load_season_couples, load_weekly_scores  # noqa: E402
from dwts_model.feature_engineering import build_context_features, build_features  # noqa: E402
from dwts_model.models import RidgeModel  # noqa: E402
from dwts_model.predict import build_upcoming_features, unrecognized_styles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--schedule", help="CSV with columns celebrity, styles, optional theme")
    parser.add_argument("--theme", help="theme of the upcoming week, e.g. 'Disney Night'")
    parser.add_argument("--template", help="write a schedule template to this path and exit")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if not args.template and not args.schedule:
        parser.error("give --template, or --schedule (with a theme column or --theme)")

    cfg = load_config(args.config)
    weekly = load_weekly_scores(cfg)
    perf = load_performance_context(cfg)
    couples = load_season_couples(cfg)

    season = args.season or int(weekly["season"].max())
    last_week = int(weekly.loc[weekly["season"] == season, "week"].max())
    week = args.week or last_week + 1
    prior = weekly[(weekly["season"] == season) & (weekly["week"] == week - 1)]

    names = couples[couples["season"] == season]
    if args.template:
        remaining = prior[prior["eliminated"] != True]  # noqa: E712  (NA-safe)
        out = remaining.merge(names, on=["season", "couple_id"])[["celebrity", "professional"]].assign(theme="", styles="")
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
    set_name = cfg.main_model.removeprefix("ridge_")
    if not cfg.main_model.startswith("ridge_") or set_name not in params:
        sys.exit(f"No validated Ridge alpha for main model {cfg.main_model}: run --mode validate first.")
    features = cfg.features[set_name]

    history = build_features(weekly, build_context_features(perf))
    train = history[(history["season"] < season) | ((history["season"] == season) & (history["week"] < week))]
    model = RidgeModel(features, params[set_name], name=cfg.main_model).fit(train)

    rows = build_upcoming_features(weekly, perf, sched[["couple_id", "styles"]],     season, week, theme)
    missing = set(sched["couple_id"]) - set(rows["couple_id"])
    if missing:
        names_missing = sched.loc[sched["couple_id"].isin(missing), "celebrity"].tolist()
        sys.exit(f"No earlier weeks this season for {names_missing}; week-1 couples are not supported.")

    rows["predicted"] = model.predict(rows).clip(cfg.score_min, cfg.score_max)
    out = rows.merge(sched[["couple_id", "celebrity", "professional", "styles"]], on="couple_id")
    out["styles"] = out["styles"].map("; ".join)
    out = out.sort_values("predicted", ascending=False).reset_index(drop=True)
    out.insert(0, "pred_rank", range(1, len(out) + 1))
    out = out[["pred_rank", "celebrity", "professional", "styles", "prev_score", "running_mean",
               "prev_bottom3", "predicted"]]

    path = cfg.output_dir / f"upcoming_s{season}_w{week}.csv"
    out.to_csv(path, index=False)
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 160
    print(f"Season {season}, week {week} ({theme}); trained on {len(train)} rows; main model {cfg.main_model}")
    print(out.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
