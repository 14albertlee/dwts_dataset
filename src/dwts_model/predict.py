import pandas as pd

from .categories import ONEHOT_STYLES, normalize_style
from .feature_engineering import build_context_features, build_features


def build_upcoming_features(
    weekly: pd.DataFrame,
    perf: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
    theme: str,
) -> pd.DataFrame:
    """Feature rows for a week that has not aired yet.

    `schedule` has one row per couple still competing: `couple_id` and `styles` (list of the
    dances they will perform), and optionally `order` (the couple's 1-based running position in
    each of those dances' rounds). With no `order` the order features take their random-order
    base rates; with it they take the announced values. History features come from earlier weeks of that season, via the
    same build_features used for training, so training and prediction cannot drift apart.
    Anything at or after (season, week) in `weekly`/`perf` is dropped first, so this can also be
    run "as of" a past week for backtests.
    """
    def before(df):
        return df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]

    new_weekly = pd.DataFrame(
        {
            "season": season,
            "week": week,
            "couple_id": schedule["couple_id"].to_numpy(),
            "score": float("nan"),
            "n_performances": schedule["styles"].map(len).to_numpy(),
            "eliminated": pd.NA,
        }
    )
    # Round k holds every couple with more than k dances; its size is the field for that round.
    round_size = [sum(len(s) > k for s in schedule["styles"]) for k in range(schedule["styles"].map(len).max())]
    orders = schedule["order"] if "order" in schedule else [None] * len(schedule)
    has_order = "running_order" in perf.columns
    new_perf = pd.DataFrame(
        [
            {"season": season, "week": week, "couple_id": row.couple_id,
             "performance_id": -1, "theme": theme, "style": style,
             **({"round": k + 1, "round_size": round_size[k],
                 "running_order": order[k] if order is not None else float("nan")} if has_order else {})}
            for row, order in zip(schedule.itertuples(), orders)
            for k, style in enumerate(row.styles)
        ]
    )
    weekly_all = pd.concat([before(weekly), new_weekly], ignore_index=True)
    perf_all = pd.concat([before(perf), new_perf], ignore_index=True)
    feats = build_features(weekly_all, build_context_features(perf_all))
    return feats[(feats["season"] == season) & (feats["week"] == week)].reset_index(drop=True)


def check_order(schedule: pd.DataFrame) -> str:
    """Validate the optional `order` column and return the mode: 'announced' or 'base-rate'.

    Orders are all-or-none per week. Each round's positions must be 1..n with no repeats,
    where n is the number of couples dancing in that round.
    """
    if "order" not in schedule or schedule["order"].isna().all():
        return "base-rate"
    if schedule["order"].isna().any():
        raise ValueError("Order given for some couples but not others; fill all or none.")
    for orders, styles in zip(schedule["order"], schedule["styles"]):
        if len(orders) != len(styles):
            raise ValueError(f"Order {orders} needs one position per dance in {styles}.")
    for k in range(schedule["styles"].map(len).max()):
        positions = sorted(o[k] for o, s in zip(schedule["order"], schedule["styles"]) if len(s) > k)
        if positions != list(range(1, len(positions) + 1)):
            raise ValueError(f"Round {k + 1} positions must be 1..{len(positions)} without repeats, got {positions}.")
    return "announced"


def unrecognized_styles(schedule: pd.DataFrame) -> set[str]:
    """Scheduled dances with no style indicator of their own (they fall into 'other')."""
    return {
        normalize_style(s) for styles in schedule["styles"] for s in styles
    } - set(ONEHOT_STYLES)
