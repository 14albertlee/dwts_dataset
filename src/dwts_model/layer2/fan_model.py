"""Fan-support model: latent utility per couple and Plackett-Luce fan orderings.

u_iw = X_iw . theta, and the fan ranking is the ordering of u + Gumbel noise (fixed scale), which is a
Plackett-Luce ordering. theta = 0 gives uniformly random orderings (the null model).
"""
import numpy as np
import pandas as pd

FEATURE_GROUPS = ("skill", "survival_surplus", "notability")


def design_matrix(cw: pd.DataFrame, features, reference_group: str, groups=None) -> pd.DataFrame:
    """Columns for the requested feature groups; notability gets one dummy per non-reference group.

    `groups` fixes the notability dummies (default: every group in `cw`), so the columns stay the
    same when a single week is passed in.
    """
    unknown = set(features) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    if groups is None:
        groups = notability_groups(cw, reference_group)
    cols = {}
    for name in features:
        if name == "notability":
            for group in groups:
                cols[f"notab_{group}"] = (cw["notability_group"] == group).astype(float)
        else:
            cols[name] = cw[name].astype(float)
    return pd.DataFrame(cols, index=cw.index)


def notability_groups(cw: pd.DataFrame, reference_group: str) -> list[str]:
    return sorted(set(cw["notability_group"]) - {reference_group})


def utilities(X, theta) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X @ np.asarray(theta, dtype=float) if X.shape[1] else np.zeros(X.shape[0])


def sample_fan_points(u: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Fan points (draws x couples, N = best fan rank, 1 = worst) from u + standard Gumbel noise."""
    u = np.asarray(u, dtype=float)
    noisy = u + rng.gumbel(size=(n_draws, len(u)))
    return np.argsort(np.argsort(noisy, axis=1), axis=1) + 1.0


def log_plackett_luce(u: np.ndarray, best_first: np.ndarray) -> np.ndarray:
    """Log probability of each ordering in `best_first` (rows of couple indices, best fan rank first)."""
    us = np.asarray(u, dtype=float)[best_first]
    tail = np.logaddexp.accumulate(us[:, ::-1], axis=1)[:, ::-1]  # log sum of utilities still in the pool
    return (us - tail).sum(axis=1)
