"""Rule-consistency check (plan step 2), using judge scores and eliminations only.

For each single-elimination, non-finale week, asks whether the actual elimination is possible under
some fan ordering for each rule (H1 and the judges' save, under each tie convention) or with no fan
input at all (judges only). A week is usable when its own era's rule is consistent with the elimination
under every tie convention and every couple danced the same number of times (otherwise mean and total
judge scores rank couples differently).
"""
import numpy as np
import pandas as pd

from .rules import (
    feasible_judges_only,
    feasible_rank_rule,
    feasible_save,
    judge_points,
    share_of_orderings_eliminating,
)


def eligible_weeks(cw: pd.DataFrame) -> pd.DataFrame:
    """Rows of single-elimination, non-finale weeks (the weeks the rules are tested on)."""
    return cw[cw["week_type"] == "single"]


def check_weeks(
    cw: pd.DataFrame, tie_conventions, era_of=lambda season: "rank",
    exact_max: int = 8, draws: int = 20000, seed: int = 0,
) -> pd.DataFrame:
    """One row per eligible week. `era_of(season)` gives the week's rule era (see `rules.era_for_season`)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (season, week), g in eligible_weeks(cw).groupby(["season", "week"]):
        scores = g["score"].to_numpy()
        e = int(np.flatnonzero(g["eliminated"].to_numpy())[0])
        era = era_of(season)
        row = {
            "season": season,
            "week": week,
            "era": era,
            "n_couples": len(g),
            "elim_judge_rank": int(g["judge_rank"].iloc[e]),
            "elim_bottom_rank": int(g["bottom_rank"].iloc[e]),
            "uneven_performances": bool(g["uneven_performances"].iloc[0]),
            "judges_only": feasible_judges_only(scores, e),
        }
        for tie in tie_conventions:
            jp = judge_points(scores, tie)
            row[f"rank_{tie}"] = feasible_rank_rule(jp, e)
            row[f"save_{tie}"] = feasible_save(jp, e)
            row[f"p_random_{tie}"] = share_of_orderings_eliminating(jp, e, exact_max, draws, rng)
        row["consistent"] = all(row[f"{era}_{t}"] for t in tie_conventions)
        rows.append(row)
    return pd.DataFrame(rows)


def rule_columns(tie_conventions) -> list[str]:
    return ["judges_only"] + [f"{r}_{t}" for r in ("rank", "save") for t in tie_conventions]


def summarize_rules(checks: pd.DataFrame, tie_conventions) -> pd.DataFrame:
    """Weeks consistent with each rule, by era. A rule nearly every couple could satisfy says little."""
    out = []
    for era, g in checks.groupby("era"):
        for name in rule_columns(tie_conventions):
            consistent = int(g[name].sum())
            out.append({"era": era, "rule": name, "weeks": len(g), "consistent": consistent,
                        "share_consistent": consistent / len(g)})
    return pd.DataFrame(out)


def season_summary(checks: pd.DataFrame) -> pd.DataFrame:
    g = checks.groupby("season")
    out = pd.DataFrame(
        {
            "era": g["era"].first(),
            "weeks": g.size(),
            "consistent": g["consistent"].sum(),
            "judges_only": g["judges_only"].sum(),
            "max_bottom_rank": g["elim_bottom_rank"].max(),
        }
    )
    out["share_consistent"] = out["consistent"] / out["weeks"]
    return out.reset_index()


def usable_weeks(checks: pd.DataFrame) -> pd.DataFrame:
    """Weeks consistent with their era's rule and with even performance counts."""
    keep = checks[checks["consistent"] & ~checks["uneven_performances"]]
    return keep[["season", "week", "era", "n_couples"]].reset_index(drop=True)
