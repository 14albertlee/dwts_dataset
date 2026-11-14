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
    dances they will perform). History features come from earlier weeks of that season, via the
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
    new_perf = pd.DataFrame(
        [
            {"season": season, "week": week, "couple_id": row.couple_id,
             "performance_id": -1, "theme": theme, "style": style}
            for row in schedule.itertuples()
            for style in row.styles
        ]
    )
    weekly_all = pd.concat([before(weekly), new_weekly], ignore_index=True)
    perf_all = pd.concat([before(perf), new_perf], ignore_index=True)
    feats = build_features(weekly_all, build_context_features(perf_all))
    return feats[(feats["season"] == season) & (feats["week"] == week)].reset_index(drop=True)


def unrecognized_styles(schedule: pd.DataFrame) -> set[str]:
    """Scheduled dances with no style indicator of their own (they fall into 'other')."""
    return {
        normalize_style(s) for styles in schedule["styles"] for s in styles
    } - set(ONEHOT_STYLES)
