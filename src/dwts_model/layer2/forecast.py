"""Elimination probabilities for one week (plan sections 3-4).

`elimination_probs` takes judge totals (actual, or Layer 1 draws) and fan utilities and returns each
couple's probability of being eliminated under the week's rule (see `rules.Rule`).
"""
import itertools
from functools import lru_cache

import numpy as np

from .fan_model import log_plackett_luce, sample_fan_points
from .rules import Rule


@lru_cache(maxsize=None)
def all_fan_points(n: int) -> np.ndarray:
    """Every fan ordering of n couples as fan points (n! x n)."""
    return np.array(list(itertools.permutations(range(1, n + 1))), dtype=float)


def mix_uniform(p: np.ndarray, eps: float) -> np.ndarray:
    """With probability eps the elimination ignores the model and falls on a uniformly random couple."""
    return (1 - eps) * p + eps / len(p)


def elimination_probs(
    judge_totals,
    u,
    rule: Rule,
    rng: np.random.Generator,
    n_draws: int = 10000,
    exact_max: int = 8,
    eps: float = 0.0,
) -> np.ndarray:
    """P(eliminated) per couple.

    `judge_totals` is (couples,), one row of actual totals, or (draws, couples) from Layer 1. With a
    single row and at most `exact_max` couples, fan orderings are enumerated and weighted by their
    Plackett-Luce probability (no Monte Carlo error). Otherwise each judge draw is paired with an
    independent fan draw (a single row is repeated `n_draws` times).
    """
    totals = np.atleast_2d(np.asarray(judge_totals, dtype=float))
    n = totals.shape[1]
    u = np.asarray(u, dtype=float)
    if totals.shape[0] == 1 and n <= exact_max:
        fans = all_fan_points(n)
        weight = np.exp(log_plackett_luce(u, np.argsort(-fans, axis=1)))
        p = weight @ rule.elimination_weights(totals, fans)
    else:
        if totals.shape[0] == 1:
            totals = np.repeat(totals, n_draws, axis=0)
        fans = sample_fan_points(u, len(totals), rng)
        p = rule.elimination_weights(totals, fans).mean(axis=0)
    return mix_uniform(p, eps)
