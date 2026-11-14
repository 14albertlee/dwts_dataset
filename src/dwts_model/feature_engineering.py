import pandas as pd

from .categories import (
    ONEHOT_STYLES,
    THEME_DEFAULT,
    THEME_RULES,
    normalize_style,
    slug,
    style_difficulty,
    style_group,
    theme_category,
)

THEME_CATEGORIES = [c for c, _ in THEME_RULES] + [THEME_DEFAULT]
THEME_REFERENCE = THEME_DEFAULT  # omitted from the indicators to avoid collinearity


def build_context_features(perf: pd.DataFrame) -> pd.DataFrame:
    """Schedule-known features per (season, week, couple): theme category and dance style.

    These describe the week being predicted and are known before it airs.
    Style features are shares over the couple's performances that week.
    """
    p = perf.copy()
    p["theme_cat"] = p["theme"].map(theme_category)
    p["group"] = p["style"].map(style_group)
    p["difficulty"] = p["style"].map(style_difficulty)
    base = p["style"].map(normalize_style)
    keys = ["season", "week", "couple_id"]

    out = p.groupby(keys).agg(
        theme_cat=("theme_cat", "first"),
        style_difficulty=("difficulty", "mean"),
    )
    for cat in THEME_CATEGORIES:
        if cat != THEME_REFERENCE:
            out[f"theme_{slug(cat)}"] = (out["theme_cat"] == cat).astype(float)
    for grp in ("Latin", "Ballroom"):
        out[f"style_share_{slug(grp)}"] = (p["group"] == grp).groupby([p[k] for k in keys]).mean()
    for style in ONEHOT_STYLES:
        out[f"style_{slug(style)}"] = (base == style).groupby([p[k] for k in keys]).mean()
    return out.drop(columns="theme_cat").reset_index()


def build_features(weekly: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add Phase 1/2 features and drop each couple's first week.

    Every feature for a row uses only that couple's earlier weeks (and earlier
    weeks' field stats), plus schedule-known context for the row's own week.
    Never the row's own score.
    """
    df = weekly.sort_values(["season", "couple_id", "week"]).copy()
    if context is not None:
        df = df.merge(context, on=["season", "week", "couple_id"], how="left")
        df["prev_style_difficulty"] = df.groupby(["season", "couple_id"], sort=False)[
            "style_difficulty"
        ].shift(1)
    by_couple = df.groupby(["season", "couple_id"], sort=False)["score"]

    df["prev_score"] = by_couple.shift(1)
    df["running_mean"] = by_couple.transform(lambda s: s.shift(1).expanding().mean())

    # Week-level quantities are only consumed one week later, via the shift below.
    week_group = df.groupby(["season", "week"])["score"]
    df["_dev"] = df["score"] - week_group.transform("mean")
    # Normalized rank in [0, 1]: 1 = best in the field, 0 = worst; ties share the average rank.
    n_field = week_group.transform("count")
    df["_rank"] = ((week_group.rank(method="average") - 1) / (n_field - 1)).where(n_field > 1)
    shifted = df.groupby(["season", "couple_id"], sort=False)[["_dev", "_rank"]].shift(1)
    df["prev_dev"] = shifted["_dev"]
    df["prev_rank"] = shifted["_rank"]
    df = df.drop(columns=["_dev", "_rank"])

    df = df.dropna(subset=["prev_score"]).reset_index(drop=True)
    return df.sort_values(["season", "week", "couple_id"]).reset_index(drop=True)
