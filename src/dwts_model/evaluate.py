import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _week_spearman(group: pd.DataFrame, pred_col: str) -> float:
    if len(group) < 3 or group[pred_col].nunique() < 2 or group["score"].nunique() < 2:
        return np.nan
    return spearmanr(group[pred_col], group["score"])[0]


def _metrics(df: pd.DataFrame, model: str) -> dict:
    err = df[model] - df["score"]
    per_week = df.groupby(["season", "week"]).apply(
        lambda g: _week_spearman(g, model), include_groups=False
    )
    return {
        "n": len(df),
        "mae": err.abs().mean(),
        "rmse": float(np.sqrt((err**2).mean())),
        "spearman": per_week.mean(),
    }


def summarize(preds: pd.DataFrame, models: list[str], short_seasons=()) -> dict[str, pd.DataFrame]:
    """Metric tables for models in `models` (columns of `preds`).

    overall: pooled over all test rows, and again excluding short seasons.
    by_week: per week number. by_season: per test season.
    """
    scopes = {"all": preds}
    if short_seasons:
        scopes["excl_short"] = preds[~preds["season"].isin(short_seasons)]

    overall = pd.DataFrame(
        [{"scope": s, "model": m, **_metrics(d, m)} for s, d in scopes.items() for m in models]
    )
    by_week = pd.DataFrame(
        [
            {"week": w, "model": m, **_metrics(g, m)}
            for w, g in preds.groupby("week")
            for m in models
        ]
    )
    by_season = pd.DataFrame(
        [
            {"season": s, "model": m, **_metrics(g, m)}
            for s, g in preds.groupby("season")
            for m in models
        ]
    )
    return {"overall": overall, "by_week": by_week, "by_season": by_season}
