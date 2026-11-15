"""Layer 2 steps 1-2: build the couple-week table and run the rule check.

    python scripts/run_rule_check.py
"""
import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwts_model.config import load_config  # noqa: E402
from dwts_model.data import load_couple_info, load_weekly_scores, load_withdrawals  # noqa: E402
from dwts_model.layer2.couple_week import build_couple_week  # noqa: E402
from dwts_model.layer2.rule_check import (  # noqa: E402
    check_weeks,
    season_summary,
    summarize_rules,
    usable_weeks,
)
from dwts_model.layer2.rules import era_for_season  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.layer2_dir.mkdir(parents=True, exist_ok=True)

    withdrawals = load_withdrawals(cfg)
    cw = build_couple_week(load_weekly_scores(cfg), load_couple_info(cfg), withdrawals)
    cw.to_csv(cfg.layer2_dir / "couple_week.csv", index=False)
    print(f"couple_week: {len(cw)} rows, {cw.groupby(['season', 'week']).ngroups} weeks")
    print(cw.drop_duplicates(["season", "week"])["week_type"].value_counts().to_string())
    print("withdrawals:", cw.loc[cw["withdrew"], ["season", "week", "couple_id"]].to_dict("records"), "\n")

    era_of = partial(era_for_season, save_seasons=cfg.save_seasons)
    checks = check_weeks(cw, cfg.tie_conventions, era_of, cfg.exact_enum_max, cfg.mc_draws, cfg.seed)
    summary = summarize_rules(checks, cfg.tie_conventions)
    by_season = season_summary(checks)
    usable = usable_weeks(checks)

    checks.to_csv(cfg.layer2_dir / "rule_check.csv", index=False)
    summary.to_csv(cfg.layer2_dir / "rule_check_summary.csv", index=False)
    by_season.to_csv(cfg.layer2_dir / "rule_check_by_season.csv", index=False)
    usable.to_csv(cfg.layer2_dir / "usable_weeks.csv", index=False)

    pd_wide = summary.pivot(index="rule", columns="era", values="share_consistent").round(3)
    print("Share of weeks consistent with each rule, by era:")
    print(pd_wide.to_string(), "\n")
    inconsistent = by_season[by_season["share_consistent"] < 1]
    print(inconsistent.to_string(index=False) if len(inconsistent) else "all seasons consistent with their era's rule")
    print(f"\nUsable weeks: {len(usable)} of {len(checks)}  {usable['era'].value_counts().to_dict()}")
    dropped = checks.merge(usable[["season", "week"]], how="left", on=["season", "week"], indicator=True)
    cols = ["season", "week", "era", "n_couples", "elim_judge_rank", "consistent", "uneven_performances"]
    print(dropped.loc[dropped["_merge"] == "left_only", cols].to_string(index=False))


if __name__ == "__main__":
    main()
