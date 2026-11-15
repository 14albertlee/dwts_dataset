"""Fitting theta by maximum penalized likelihood of the observed eliminations (plan section 3).

The likelihood of a week is P(the couple who left is eliminated | actual judge totals, theta), summed over
fan orderings. Fan orderings are enumerated for small fields and sampled uniformly for large ones; either
way each ordering is weighted by its Plackett-Luce probability, so the likelihood is smooth in theta.

Every week is mixed with a uniform elimination, (1 - eps) * P + eps / N, so a surprise the model rules
out costs a bounded amount. eps, and save_q for judges'-save weeks (P is linear in it), are profiled out:
for each theta they are set to their best values.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from .fan_model import log_plackett_luce
from .forecast import all_fan_points
from .rules import Rule

P_FLOOR = 1e-12
EPS_MAX = 0.5


@dataclass
class WeekLikelihood:
    X: np.ndarray  # couples x features
    best_first: np.ndarray  # orderings that can eliminate the actual couple, couple indices best fan rank first
    const: np.ndarray  # P(actual couple eliminated | ordering) = const + coef * save_q, per kept ordering
    coef: np.ndarray
    n_orderings: int  # orderings examined (N! when enumerated)
    log_n_factorial: float
    sampled: bool  # orderings were sampled uniformly (importance sampling) rather than enumerated

    @property
    def n_couples(self) -> int:
        return self.X.shape[0]

    def _log_weights(self, theta) -> np.ndarray:
        u = self.X @ theta if self.X.shape[1] else np.zeros(self.n_couples)
        # Importance weight N! * PL(ordering) against the uniform proposal; theta = 0 gives weight 1.
        return self.log_n_factorial + log_plackett_luce(u, self.best_first)

    def terms(self, theta) -> tuple[float, float]:
        """(a, b) with P(actual couple eliminated | theta) = a + b * save_q."""
        if len(self.best_first) == 0:
            return 0.0, 0.0
        lw = self._log_weights(theta)
        top = lw.max()
        w = np.exp(lw - top)
        scale = np.exp(top - np.log(self.n_orderings))
        return float(scale * (w @ self.const)), float(scale * (w @ self.coef))

    def ess(self, theta) -> float:
        """Effective number of kept orderings at theta (equals the kept count when theta = 0)."""
        if len(self.best_first) == 0:
            return 0.0
        lw = self._log_weights(theta)
        w = np.exp(lw - lw.max())
        return float(w.sum() ** 2 / (w**2).sum())


def prepare_week(
    judge_totals,
    eliminated: int,
    X,
    rule: Rule,
    rng: np.random.Generator,
    exact_max: int = 8,
    draws: int = 20000,
) -> WeekLikelihood:
    totals = np.atleast_2d(np.asarray(judge_totals, dtype=float))
    n = totals.shape[1]
    sampled = n > exact_max
    if sampled:
        fans = np.argsort(np.argsort(rng.random((draws, n)), axis=1), axis=1) + 1.0
    else:
        fans = all_fan_points(n)
    const, coef = rule.elimination_terms(totals, fans)
    const, coef = const[:, eliminated], coef[:, eliminated]
    keep = (const != 0) | (coef != 0)
    return WeekLikelihood(
        X=np.asarray(X, dtype=float),
        best_first=np.argsort(-fans[keep], axis=1),
        const=const[keep],
        coef=coef[keep],
        n_orderings=len(fans),
        log_n_factorial=float(gammaln(n + 1)),
        sampled=sampled,
    )


def mixture_nll(p: np.ndarray, n: np.ndarray, eps: float) -> float:
    return -float(np.log(np.maximum((1 - eps) * p + eps / n, P_FLOOR)).sum())


def _min_scalar(f, lo: float, hi: float) -> tuple[float, float]:
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    return min([(lo, f(lo)), (hi, f(hi)), (float(res.x), float(res.fun))], key=lambda c: c[1])


def best_mix(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> tuple[float, float, float]:
    """(eps, save_q, nll) minimizing the mixture negative log likelihood of P = a + b * save_q.

    The likelihood is convex in eps for fixed save_q; save_q only matters when some week has b != 0.
    """
    def nll_given_q(q):
        p = a + b * q
        return _min_scalar(lambda e: mixture_nll(p, n, e), 0.0, EPS_MAX)

    if not np.any(b):
        eps, nll = nll_given_q(1.0)
        return eps, 1.0, nll
    q, nll = _min_scalar(lambda q: nll_given_q(q)[1], 0.0, 1.0)
    return nll_given_q(q)[0], q, nll


@dataclass
class Fit:
    theta: np.ndarray
    eps: float
    save_q: float
    nll: float  # penalized negative log likelihood at the fitted values


def _week_terms(weeks, theta) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([w.terms(theta) for w in weeks]).reshape(-1, 2)
    return t[:, 0], t[:, 1]


def fit_theta(weeks: list[WeekLikelihood], n_features: int, ridge: float = 0.1, theta0=None) -> Fit:
    """Nelder-Mead on theta with eps and save_q profiled out; with no features only those two are fitted."""
    n = np.array([w.n_couples for w in weeks], dtype=float)

    def objective(theta):
        theta = np.asarray(theta, dtype=float)
        return best_mix(*_week_terms(weeks, theta), n)[2] + ridge * float(theta @ theta)

    theta = np.zeros(n_features)
    if n_features and weeks:
        start = np.zeros(n_features) if theta0 is None else np.asarray(theta0, dtype=float)
        theta = minimize(
            objective, start, method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400 * n_features,
                     "initial_simplex": start + np.vstack([np.zeros(n_features), 0.1 * np.eye(n_features)])},
        ).x
    if not weeks:
        return Fit(theta=theta, eps=0.0, save_q=1.0, nll=0.0)
    eps, q, nll = best_mix(*_week_terms(weeks, theta), n)
    return Fit(theta=theta, eps=eps, save_q=q, nll=nll + ridge * float(theta @ theta))
