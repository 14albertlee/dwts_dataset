"""Predictive distribution of a night's judge scores: Ridge mean + Gaussian spread + integer rounding.

Layer 1 for the Monte Carlo (docs/LAYER1_NEXT_EXPERIMENTS.md, 3.8 and section 4):
  * mean: the Ridge prediction (mean per-judge score) for each couple;
  * spread: a shared night term plus a per-couple term, both by week bucket, from earlier seasons'
    out-of-sample residuals of the same model (no residual from the target season or later);
  * integer step: the per-couple term is split into a performance shock shared by all judges and
    per-judge noise; each judge-dance score is rounded and clipped to 1-10 and summed, so ties
    in the judges' totals occur at realistic rates.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .evaluate import lowest_indicator, simulate_integer_totals
from .models import RidgeModel

RESID_FROM = 16  # earliest season whose out-of-sample residuals set the spread
JUDGE_WINDOW = 5  # training seasons used for the within-performance judge spread
KEYS = ["season", "week", "couple_id"]
BUCKETS = ("w2-3", "w4-6", "w7+")


def week_buckets(weeks: pd.Series) -> np.ndarray:
    return np.select([weeks <= 3, weeks <= 6], list(BUCKETS[:2]), BUCKETS[2])


def ridge_oos(df: pd.DataFrame, features, alpha: float, first: int, last: int) -> pd.DataFrame:
    """Out-of-sample Ridge predictions for seasons first..last, each trained on the seasons before it."""
    frames = []
    for s in range(first, last + 1):
        train, test = df[df["season"] < s], df[df["season"] == s]
        if train.empty or test.empty:
            continue
        out = test[KEYS + ["score"]].copy()
        out["pred"] = RidgeModel(features, alpha).fit(train).predict(test)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def residual_spread(oos: pd.DataFrame, season: int) -> dict[str, tuple[float, float]]:
    """(within-night variance, night-level variance) of the mean score, by week bucket.

    Uses only seasons RESID_FROM..season-1 of `oos`. The within-night part is the variance of
    residuals around their night mean; the night part is the variance of night-mean residuals
    after removing the sampling share of the within-night part.
    """
    r = oos[(oos["season"] >= RESID_FROM) & (oos["season"] < season)].copy()
    if r.empty:
        raise ValueError(f"No out-of-sample residuals before season {season}.")
    r["e"] = r["score"] - r["pred"]
    r["b"] = week_buckets(r["week"])
    out = {}
    for b, g in r.groupby("b"):
        wk = g.groupby(["season", "week"])["e"]
        n = wk.transform("count")
        dev = g["e"] - wk.transform("mean")
        dof = (n - 1).clip(lower=0).groupby([g["season"], g["week"]]).first().sum()
        var_w = float((dev**2).sum() / dof)
        stats = wk.agg(["mean", "count"])
        var_n = max(float((stats["mean"] ** 2).mean() - (var_w / stats["count"]).mean()), 0.0)
        out[b] = (var_w, var_n)
    return out


def judge_noise_var(judges: pd.DataFrame, season: int) -> float:
    """Pooled within-performance variance across judges (integer scores) in the last JUDGE_WINDOW seasons."""
    j = judges[(judges["season"] < season) & (judges["season"] >= season - JUDGE_WINDOW)]
    g = j.groupby("performance_id")["judge_score"]
    n = g.transform("count")
    dev = (j["judge_score"] - g.transform("mean"))[n >= 2]
    counts = g.count()
    return float((dev**2).sum() / (counts[counts >= 2] - 1).sum())


def night_covariance(n_couples: int, week: int, spread: dict[str, tuple[float, float]]) -> tuple[np.ndarray, float]:
    """Covariance of the couples' mean scores on one night, and the per-couple (diagonal) variance."""
    var_w, var_n = spread[str(week_buckets(pd.Series([week]))[0])]
    return var_w * np.eye(n_couples) + var_n * np.ones((n_couples, n_couples)), var_w


def forecast_night(
    mean: np.ndarray,
    week: int,
    n_slots: np.ndarray,
    spread: dict[str, tuple[float, float]],
    judge_var: float,
    rng: np.random.Generator,
    n_draws: int = 10000,
    interval: float = 0.8,
    score_range: tuple[float, float] = (1.0, 10.0),
) -> pd.DataFrame:
    """Per-couple score interval and elimination-relevant probabilities for one night.

    `n_slots` is the number of judge-dance scores in each couple's total (judges x dances).
    Returns `lo` / `hi` (central `interval` range of the mean score, clipped to the score range),
    `p_lowest` (share of draws where the couple has the lowest integer total, ties counted for
    every tied couple) and `p_tied_lowest` (lowest total shared with at least one other couple).
    """
    mean = np.asarray(mean, dtype=float)
    n_slots = np.asarray(n_slots, dtype=float)
    cov, var_w = night_covariance(len(mean), week, spread)
    # Remove the judge-noise share of the per-couple variance to get the shared performance shock.
    latent_cov = cov - np.diag(np.minimum(judge_var / n_slots, 0.9 * var_w))
    latent = rng.multivariate_normal(mean, latent_cov, size=n_draws, method="cholesky")
    totals = simulate_integer_totals(latent, n_slots, np.sqrt(max(judge_var - 1 / 12, 0.01)), rng)
    lowest = lowest_indicator(totals)
    z = norm.ppf(0.5 + interval / 2)
    sd = np.sqrt(np.diag(cov))
    return pd.DataFrame(
        {
            "lo": np.clip(mean - z * sd, *score_range),
            "hi": np.clip(mean + z * sd, *score_range),
            "p_lowest": lowest.mean(axis=0),
            "p_tied_lowest": (lowest * (lowest.sum(axis=1, keepdims=True) > 1)).mean(axis=0),
        }
    )
