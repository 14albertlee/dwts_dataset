"""Evaluation metrics (plan section 5) on per-week elimination probabilities.

`probs` is a long frame with season, week, couple_id, n_couples, p_elim, eliminated.
"""
import numpy as np
import pandas as pd

P_MIN = 1e-3  # probability floor for log loss, so a zero from a baseline is a large but finite penalty

METRICS = [
    "top1", "top2", "top3", "rank", "logloss", "brier",
    "random_top1", "random_top2", "random_top3", "random_logloss", "random_brier",
]


def week_metrics(probs: pd.DataFrame) -> pd.DataFrame:
    """One row per week: where the actual couple ranks, its probability, log loss, Brier score."""
    rows = []
    for (season, week), g in probs.groupby(["season", "week"]):
        p = g["p_elim"].to_numpy()
        e = int(np.flatnonzero(g["eliminated"].to_numpy())[0])
        n = len(g)
        above = int((p > p[e]).sum())
        tied = int((p == p[e]).sum())  # includes the actual couple
        row = {
            "season": season,
            "week": week,
            "n_couples": n,
            "p_actual": p[e],
            "rank": above + (tied + 1) / 2,  # ties in the model's ranking share the average position
            "logloss": -np.log(max(p[e], P_MIN)),
            "brier": float(((p - (np.arange(n) == e)) ** 2).sum()),
            "random_logloss": np.log(n),
            "random_brier": 1 - 1 / n,
        }
        if "era" in g:
            row["era"] = g["era"].iloc[0]
        for k in (1, 2, 3):
            # A tied block straddling the cutoff counts the share of the block that fits inside the top k.
            row[f"top{k}"] = float(np.clip((k - above) / tied, 0, 1))
            row[f"random_top{k}"] = min(k, n) / n
        rows.append(row)
    return pd.DataFrame(rows)


PHASES = ["weeks 2-4", "weeks 5-7", "weeks 8+"]


def week_phase(week: pd.Series) -> pd.Series:
    """Season phase of a week number: early (2-4), middle (5-7) or late (8+)."""
    return pd.cut(week, [0, 4, 7, np.inf], labels=PHASES)


def summarize(wm: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean of each metric overall (by=None), by number of couples remaining (by='field'), by rule era
    (by='era'), or by season phase (by='phase', see `week_phase`)."""
    if by == "phase":
        g = wm.groupby(week_phase(wm["week"]), observed=True)
    elif by == "field":
        bucket = pd.cut(wm["n_couples"], [0, 4, 6, 8, 10, 20], labels=["<=4", "5-6", "7-8", "9-10", "11+"])
        g = wm.groupby(bucket, observed=True)
    elif by == "era":
        g = wm.groupby("era")
    else:
        g = wm.groupby(lambda _: "all")
    out = g[METRICS].mean()
    out.insert(0, "weeks", g.size())
    return out.reset_index(names="group")


def calibration(probs: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed elimination frequency over all couple-weeks, in quantile bins of the prediction."""
    d = probs[["p_elim", "eliminated"]].astype(float)
    d["bin"] = pd.qcut(d["p_elim"].rank(method="first"), bins, labels=False)
    g = d.groupby("bin")
    return pd.DataFrame(
        {"n": g.size(), "mean_pred": g["p_elim"].mean(), "observed": g["eliminated"].mean()}
    ).reset_index()


def bootstrap_ci(wm: pd.DataFrame, metrics, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Season-level bootstrap: resample whole seasons with replacement, recompute the pooled weekly mean."""
    metrics = list(metrics)
    rng = np.random.default_rng(seed)
    grouped = {s: g[metrics] for s, g in wm.groupby("season")}
    seasons = np.array(list(grouped))
    sums = {s: g.sum().to_numpy() for s, g in grouped.items()}
    counts = {s: len(g) for s, g in grouped.items()}
    draws = np.empty((n_boot, len(metrics)))
    for b in range(n_boot):
        pick = rng.choice(seasons, size=len(seasons))
        draws[b] = sum(sums[s] for s in pick) / sum(counts[s] for s in pick)
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return pd.DataFrame({"metric": metrics, "mean": wm[metrics].mean().to_numpy(), "lo": lo, "hi": hi})


def paired_ci(wm_a: pd.DataFrame, wm_b: pd.DataFrame, metrics, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Season-bootstrap interval of the mean weekly difference a - b, on the weeks both cover."""
    metrics = list(metrics)
    both = wm_a.merge(wm_b, on=["season", "week"], suffixes=("_a", "_b"))
    diff = pd.DataFrame({m: both[f"{m}_a"] - both[f"{m}_b"] for m in metrics}).assign(season=both["season"])
    return bootstrap_ci(diff, metrics, n_boot, seed).assign(weeks=len(both))
