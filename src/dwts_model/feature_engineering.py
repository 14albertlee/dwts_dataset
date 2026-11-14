import pandas as pd

from .categories import (
    CELEB_GROUPS,
    CELEB_REFERENCE,
    ONEHOT_STYLES,
    THEME_DEFAULT,
    THEME_RULES,
    celebrity_group,
    normalize_style,
    slug,
    style_difficulty,
    style_group,
    theme_category,
)

THEME_CATEGORIES = [c for c, _ in THEME_RULES] + [THEME_DEFAULT]
THEME_REFERENCE = THEME_DEFAULT  # omitted from the indicators to avoid collinearity
BOTTOM_K = 3
BOTTOM3_MIN_FIELD = 6


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
    if "running_order" in p.columns:
        out = out.join(build_order_features(p))
    return out.drop(columns="theme_cat").reset_index()


ORDER_MIN_ROUND = 4  # smaller rounds (finales, trios) get the neutral base-rate values


def _order_columns(position: pd.Series, size: pd.Series) -> pd.DataFrame:
    """Actual-order features for each performance, plus the base rates expected under a random order."""
    n = size.astype(float)
    pos = position.astype(float)
    ok = (n >= ORDER_MIN_ROUND) & pos.notna()
    base_first = 1 / n
    base_sq = (2 * n - 1) / (6 * (n.clip(lower=2) - 1))
    frac = (pos - 1) / (n - 1)
    return pd.DataFrame(
        {
            "order_frac": frac.where(ok, 0.5),
            "order_first": (pos == 1).astype(float).where(ok, base_first),
            "order_last": (pos == n).astype(float).where(ok, base_first),
            "order_frac_sq": (frac**2).where(ok, base_sq),
            "base_order_frac": 0.5,
            "base_order_first": base_first,
            "base_order_last": base_first,
            "base_order_frac_sq": base_sq,
        }
    )


def build_order_features(p: pd.DataFrame) -> pd.DataFrame:
    """Within-night running order features per (season, week, couple), averaged over the couple's dances.

    The order is announced at the start of the episode. Fit on the actual order (`order_*`); the
    pre-show variant substitutes the `base_order_*` columns, which are the random-order expectations.
    """
    cols = _order_columns(p["running_order"], p["round_size"])
    keys = ["season", "week", "couple_id"]
    return cols.groupby([p[k] for k in keys]).mean()


def build_celebrity_features(couple_info: pd.DataFrame) -> pd.DataFrame:
    """Celebrity background indicators per (season, couple_id), from `notability` (3.4).

    Known before the season starts. The reference group (actor) is omitted.
    """
    group = couple_info["notability"].map(celebrity_group)
    out = couple_info[["season", "couple_id"]].copy()
    for g in CELEB_GROUPS:
        if g != CELEB_REFERENCE:
            out[f"celeb_{g}"] = (group == g).astype(float).to_numpy()
    return out.drop_duplicates(["season", "couple_id"]).reset_index(drop=True)


def build_pro_history(weekly: pd.DataFrame, couple_pros: pd.DataFrame, k: float) -> pd.DataFrame:
    """Pro partner history per (season, couple_id), from earlier seasons only (3.2).

    `pro_prior_dev` is the pro's mean deviation from the week's field mean over all couple-weeks
    in seasons before the row's season, shrunk toward 0 as mean * n / (n + k). `pro_prior_n` is
    that count n. Pros with no earlier season get 0 for both.
    """
    w = weekly[["season", "week", "couple_id", "score"]].copy()
    w["dev"] = w["score"] - w.groupby(["season", "week"])["score"].transform("mean")
    w = w.merge(couple_pros, on=["season", "couple_id"])
    totals = (
        w.groupby(["professional_id", "season"])["dev"].agg(["sum", "count"]).reset_index()
        .sort_values("season")
    )
    totals[["cum_sum", "cum_n"]] = totals.groupby("professional_id")[["sum", "count"]].cumsum()

    target = couple_pros[["season", "couple_id", "professional_id"]].sort_values("season")
    out = pd.merge_asof(
        target, totals[["season", "professional_id", "cum_sum", "cum_n"]],
        on="season", by="professional_id", allow_exact_matches=False,
    )
    n = out["cum_n"].fillna(0)
    mean = (out["cum_sum"] / out["cum_n"]).fillna(0)
    out["pro_prior_dev"] = mean * n / (n + k) if k > 0 else mean
    out["pro_prior_n"] = n
    return out[["season", "couple_id", "pro_prior_dev", "pro_prior_n"]].reset_index(drop=True)



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
    # Bottom-3 flag (ties at the cutoff included). Meaningless once the field is small, so only
    # flagged when at least BOTTOM3_MIN_FIELD couples were dancing.
    df["_bottom3"] = ((week_group.rank(method="min") <= BOTTOM_K) & (n_field >= BOTTOM3_MIN_FIELD)).astype(float)
    shifted = df.groupby(["season", "couple_id"], sort=False)[["_dev", "_rank", "_bottom3"]].shift(1)
    df["prev_dev"] = shifted["_dev"]
    df["prev_rank"] = shifted["_rank"]
    df["prev_bottom3"] = shifted["_bottom3"]
    df = df.drop(columns=["_dev", "_rank", "_bottom3"])

    df = df.dropna(subset=["prev_score"]).reset_index(drop=True)
    return df.sort_values(["season", "week", "couple_id"]).reset_index(drop=True)
