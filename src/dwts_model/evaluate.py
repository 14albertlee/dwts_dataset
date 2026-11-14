import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


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
    # Spearman uses average ranks for ties.
    varied = len(group) >= 3 and group[pred_col].nunique() >= 2 and group["score"].nunique() >= 2
    return pd.Series(
        {
            "spearman": spearmanr(pred, actual)[0] if varied else np.nan,
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
        "rmse": float(np.sqrt((err**2).mean())),
        "spearman": per_week["spearman"].mean(),
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


# ---- Predictive-distribution metrics (Gaussian marginals, joint draws per night) ----


def gaussian_scores(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> pd.DataFrame:
    """Per-row central-interval coverage (50%, 90%), CRPS and negative log density under N(mu, sd^2)."""
    z = (y - mu) / sd
    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return pd.DataFrame(
        {
            "cover50": np.abs(z) <= norm.ppf(0.75),
            "cover90": np.abs(z) <= norm.ppf(0.95),
            "crps": crps,
            "nll": -norm.logpdf(y, mu, sd),
            "pit": norm.cdf(z),
        }
    )


def pairwise_probs(actual: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(i beats j) and the outcome for every pair in one night with differing actual scores."""
    i, j = np.triu_indices(len(actual), k=1)
    keep = actual[i] != actual[j]
    i, j = i[keep], j[keep]
    var = np.diag(cov)[i] + np.diag(cov)[j] - 2 * cov[i, j]
    p = norm.cdf((mean[i] - mean[j]) / np.sqrt(np.maximum(var, 1e-12)))
    return p, (actual[i] > actual[j]).astype(float)


def lowest_indicator(totals: np.ndarray) -> np.ndarray:
    """Rows of draws x couples: 1 where the couple has the (possibly shared) lowest value in that draw."""
    return (totals == totals.min(axis=-1, keepdims=True)).astype(float)


def tie_stats(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Per draw (rows): any tie, tie for lowest, and any tie among the three lowest values."""
    s = np.sort(totals, axis=-1)
    d = np.diff(s, axis=-1) == 0
    return {"any_tie": d.any(axis=-1), "bottom_tie": d[..., 0], "bottom3_tie": d[..., :2].any(axis=-1)}


def simulate_integer_totals(
    latent: np.ndarray, n_slots: np.ndarray, judge_sd: float, rng: np.random.Generator,
    score_min: float = 1.0, score_max: float = 10.0,
) -> np.ndarray:
    """Integer judge totals from latent per-judge means (draws x couples).

    Each couple gets `n_slots` judge-dance scores: latent + independent N(0, judge_sd^2) noise, rounded
    and clipped to [score_min, score_max], then summed. `judge_sd` is the pre-rounding judge spread.
    """
    out = np.zeros_like(latent)
    for c, k in enumerate(n_slots.astype(int)):
        raw = latent[:, c : c + 1] + rng.normal(0.0, judge_sd, size=(latent.shape[0], k))
        out[:, c] = np.clip(np.round(raw), score_min, score_max).sum(axis=1)
    return out
