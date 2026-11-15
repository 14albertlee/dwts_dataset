"""Couple-week table for Layer 2 (plan step 1).

One row per (season, week, couple). `skill` and `survival_surplus` use only
earlier weeks of the same season; everything else describes the week itself.
"""
import numpy as np
import pandas as pd

from ..categories import celebrity_group
from ..feature_engineering import BOTTOM3_MIN_FIELD, BOTTOM_K

KEY = ["season", "week", "couple_id"]
SCORE_DECIMALS = 6  # mean-of-means scores are floats; round so true ties compare equal


def build_couple_week(
    weekly: pd.DataFrame, couple_info: pd.DataFrame | None = None, withdrawals: pd.DataFrame | None = None
) -> pd.DataFrame:
    """`weekly` needs season, week, couple_id, score, n_performances, eliminated.

    `withdrawals` lists (season, couple_id) of couples who withdrew; their exit is flagged `withdrew`
    rather than counted as a vote, and the week gets week_type 'withdrawal'.
    """
    df = weekly.sort_values(KEY).reset_index(drop=True).copy()
    df["score"] = df["score"].round(SCORE_DECIMALS)
    df["eliminated"] = df["eliminated"].fillna(False).astype(bool)
    df["withdrew"] = df["eliminated"] & _in_pairs(df, withdrawals)
    wk = df.groupby(["season", "week"])

    df["n_couples"] = wk["couple_id"].transform("size")
    df["n_elim"] = wk["eliminated"].transform("sum").astype(int)
    df["n_withdrew"] = wk["withdrew"].transform("sum").astype(int)
    df["is_final_week"] = df["week"] == df.groupby("season")["week"].transform("max")
    df["uneven_performances"] = wk["n_performances"].transform("nunique") > 1
    df["week_type"] = np.select(
        [df["is_final_week"], df["n_withdrew"] > 0, df["n_elim"] == 0, df["n_elim"] == 1],
        ["finale", "withdrawal", "none", "single"],
        default="multi",
    )

    # Competition ranking (1 = best), ties share the best rank. Lowest-first rank is used for bottom-k.
    df["judge_rank"] = wk["score"].rank(method="min", ascending=False).astype(int)
    df["bottom_rank"] = wk["score"].rank(method="min", ascending=True).astype(int)
    df["in_bottom3"] = (df["bottom_rank"] <= BOTTOM_K) & (df["n_couples"] >= BOTTOM3_MIN_FIELD)

    df = _add_history_features(df)
    df["notability_group"] = _notability_group(df, couple_info)
    return df


def _add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    by_couple = df.groupby(["season", "couple_id"], sort=False)

    # Season-to-date mean through the previous week, as a within-week z-score; week 1 has no history.
    prior_mean = by_couple["score"].transform(lambda s: s.shift(1).expanding().mean())
    week_group = prior_mean.groupby([df["season"], df["week"]])
    spread = week_group.transform("std")
    z = (prior_mean - week_group.transform("mean")) / spread.where(spread > 0)
    df["skill"] = z.fillna(0.0)

    # Earlier weeks in which the couple was in the judges' bottom 3 and survived a vote (not just a withdrawal).
    voted_off = df["n_elim"] - df["n_withdrew"]
    survived_bottom = (df["in_bottom3"] & ~df["eliminated"] & (voted_off > 0)).astype(int)
    df["survival_surplus"] = (
        survived_bottom.groupby([df["season"], df["couple_id"]], sort=False).cumsum()
        - survived_bottom
    )
    return df


def _in_pairs(df: pd.DataFrame, pairs: pd.DataFrame | None) -> pd.Series:
    if pairs is None or pairs.empty:
        return pd.Series(False, index=df.index)
    keys = set(zip(pairs["season"].astype(int), pairs["couple_id"].astype(int)))
    return pd.Series([k in keys for k in zip(df["season"], df["couple_id"])], index=df.index)


def _notability_group(df: pd.DataFrame, couple_info: pd.DataFrame | None) -> pd.Series:
    if couple_info is None:
        return pd.Series("unknown", index=df.index)
    groups = couple_info.assign(notability_group=couple_info["notability"].map(celebrity_group))
    groups = groups[["season", "couple_id", "notability_group"]].drop_duplicates(["season", "couple_id"])
    merged = df[["season", "couple_id"]].merge(groups, how="left", on=["season", "couple_id"])
    return merged["notability_group"].fillna("unknown").set_axis(df.index)
