import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _pair_counts(actual: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """(correctly ordered pairs, comparable pairs) within one week.

    Pairs tied on the actual score have no right answer and are skipped.
    A predicted tie on a comparable pair counts as half correct.
    """
    da = np.sign(actual[:, None] - actual[None, :])
    dp = np.sign(pred[:, None] - pred[None, :])
    upper = np.triu(np.ones_like(da, dtype=bool), k=1) & (da != 0)
    correct = (da[upper] == dp[upper]).sum() + 0.5 * (dp[upper] == 0).sum()
    return float(correct), int(upper.sum())


def _group_stats(group: pd.DataFrame, pred_col: str) -> pd.Series:
    actual, pred = group["score"].to_numpy(), group[pred_col].to_numpy()
    correct, pairs = _pair_counts(actual, pred)
    varied = len(group) >= 3 and group[pred_col].nunique() >= 2 and group["score"].nunique() >= 2
    return pd.Series(
        {
            "spearman": spearmanr(pred, actual)[0] if varied else np.nan,
            "kendall_tau": kendalltau(pred, actual, variant="b")[0] if varied else np.nan,
            "pair_acc": correct / pairs if pairs else np.nan,
            "correct_pairs": correct,
            "pairs": pairs,
        }
    )


def _metrics(df: pd.DataFrame, model: str) -> dict:
    err = df[model] - df["score"]
    per_week = df.groupby(["season", "week"]).apply(lambda g: _group_stats(g, model), include_groups=False)
    return {
        "n": len(df),
        "mae": err.abs().mean(),
        "rmse": float(np.sqrt((err**2).mean())),
        "spearman": per_week["spearman"].mean(),
        "kendall_tau": per_week["kendall_tau"].mean(),
        # Mean of per-week accuracies (each week equal) and pooled over all pairs (big weeks count more).
        "pair_acc": per_week["pair_acc"].mean(),
        "pair_acc_pooled": per_week["correct_pairs"].sum() / per_week["pairs"].sum(),
        "pairs": int(per_week["pairs"].sum()),
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
