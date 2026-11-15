"""Fan-support model: latent utility per couple and Plackett-Luce fan orderings.

u_iw = X_iw . theta, and the fan ranking is the ordering of u + Gumbel noise (fixed scale), which is a
Plackett-Luce ordering. theta = 0 gives uniformly random orderings (the null model).
"""
import numpy as np
import pandas as pd

# Survival surplus (plain, surprise-weighted, and survived-as-judges'-lowest) and notability groups were
# tried as fan features and did not beat skill alone out of sample (docs/LAYER2_POC_PLAN.md, section 7).
FEATURES = ("skill",)


def design_matrix(cw: pd.DataFrame, features) -> pd.DataFrame:
    """One numeric column per requested feature."""
    unknown = set(features) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    return pd.DataFrame({name: cw[name].astype(float) for name in features}, index=cw.index)


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
