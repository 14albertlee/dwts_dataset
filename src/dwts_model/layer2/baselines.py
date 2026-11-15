"""Layer 2 baselines. The fan-model baselines (null, skill-only) are fan models with fewer features."""
import numpy as np
from scipy.optimize import minimize_scalar

P_FLOOR = 1e-12


def lowest_judge_probs(judge_totals, rho: float = 0.0) -> np.ndarray:
    """Baseline 1: the lower the judge total, the likelier the couple goes home.

    P(couple) is proportional to rho^(k - 1), where k is its position from the bottom of the judge totals
    (tied couples share the lowest position). rho = 0 is "the lowest total goes home, ties broken at
    random"; rho > 0 keeps the order among the other couples, so ranking metrics compare fairly.
    Works on one row of actual totals or on (draws, couples) Layer 1 draws.
    """
    totals = np.atleast_2d(np.asarray(judge_totals, dtype=float))
    from_bottom = (totals[:, None, :] < totals[:, :, None]).sum(axis=2)  # couples strictly below
    weight = np.where(from_bottom == 0, 1.0, rho**from_bottom)
    return (weight / weight.sum(axis=1, keepdims=True)).mean(axis=0)


def fit_rho(weeks) -> tuple[float, float]:
    """(rho, nll) maximizing the likelihood of the actual eliminations; `weeks` is [(totals, eliminated), ...]."""
    def nll(rho):
        return -sum(np.log(max(lowest_judge_probs(t, rho)[e], P_FLOOR)) for t, e in weeks)

    res = minimize_scalar(nll, bounds=(1e-6, 1.0), method="bounded", options={"xatol": 1e-4})
    return float(res.x), float(res.fun)
