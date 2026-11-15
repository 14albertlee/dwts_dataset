"""Elimination rules: rank points (H1), and rank points with the judges' save.

Points are "higher is better": the best couple gets N points, the worst gets 1. Fan points work the same
way (N = most fan votes). H1 is assumed in every season outside the judges' save era, including the
seasons sometimes reported as percent-share (3-27): H1 is not ruled out there and keeps one scheme.
"""
import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

TIE_CONVENTIONS = ("best", "average", "worst")
ERAS = ("rank", "save")


def era_for_season(season: int, save_seasons) -> str:
    """'save' (H1 bottom two, judges choose) inside the inclusive `save_seasons` range, else 'rank' (H1)."""
    return "save" if save_seasons[0] <= season <= save_seasons[1] else "rank"


def order_key(judge_pts, fan_pts, n: int):
    """Sort key putting couples in H1 order: lowest combined points first, then fewer fan points.

    Combined points differ by at least 0.5 (average ties give half points) and fan points by at most
    n - 1, so scaling the combined points by 2(n + 1) keeps the fan points a pure tie-break.
    """
    return (judge_pts + fan_pts) * (2 * (n + 1)) + fan_pts


@dataclass(frozen=True)
class Rule:
    era: str = "rank"
    tie: str = "average"
    save_q: float = 1.0  # judges' save: P(the bottom-two couple with the lower judge total goes home)

    def __post_init__(self):
        if self.era not in ERAS:
            raise ValueError(f"era must be one of {ERAS}")
        if self.tie not in TIE_CONVENTIONS:
            raise ValueError(f"tie must be one of {TIE_CONVENTIONS}")

    def elimination_terms(self, totals, fan_pts) -> tuple[np.ndarray, np.ndarray]:
        """(const, coef), each (rows, couples): P(couple eliminated | fan ordering) = const + coef * save_q.

        `totals` is (rows or 1, couples), `fan_pts` is (rows, couples). H1 is deterministic given the fan
        ordering (coef = 0). Under the judges' save the bottom two under H1 go to the judges, who send home
        the lower judge total with probability save_q (half each if equal).
        """
        fan_pts = np.asarray(fan_pts, dtype=float)
        totals = np.broadcast_to(np.atleast_2d(np.asarray(totals, dtype=float)), fan_pts.shape)
        rows, n = np.arange(fan_pts.shape[0]), fan_pts.shape[1]
        const, coef = np.zeros(fan_pts.shape), np.zeros(fan_pts.shape)
        key = order_key(judge_points_matrix(totals, self.tie), fan_pts, n)
        if self.era == "rank":
            const[rows, np.argmin(key, axis=1)] = 1.0
            return const, coef
        pair = np.argsort(key, axis=1)[:, :2]
        a, b = pair[:, 0], pair[:, 1]
        sign = np.sign(totals[rows, b] - totals[rows, a])  # +1 when a has the lower judge total
        const[rows, a], coef[rows, a] = np.where(sign > 0, 0.0, np.where(sign < 0, 1.0, 0.5)), sign
        const[rows, b], coef[rows, b] = 1.0 - const[rows, a], -sign
        return const, coef

    def elimination_weights(self, totals, fan_pts) -> np.ndarray:
        const, coef = self.elimination_terms(totals, fan_pts)
        return const + coef * self.save_q


def judge_points(scores, tie: str = "average") -> np.ndarray:
    """Rank points from judge scores. Tied scores share best, average, or worst points of their block."""
    if tie not in TIE_CONVENTIONS:
        raise ValueError(f"tie must be one of {TIE_CONVENTIONS}")
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    points = np.empty(n)
    for value in np.unique(scores):
        idx = scores == value
        k = int(idx.sum())
        higher = int((scores > value).sum())
        top, bottom = n - higher, n - higher - k + 1  # point values spanned by the tied block
        points[idx] = {"best": top, "average": (top + bottom) / 2, "worst": bottom}[tie]
    return points


def judge_points_matrix(totals, tie: str = "average") -> np.ndarray:
    """`judge_points` for many draws at once: `totals` is (draws, couples)."""
    if tie not in TIE_CONVENTIONS:
        raise ValueError(f"tie must be one of {TIE_CONVENTIONS}")
    t = np.atleast_2d(np.asarray(totals, dtype=float))
    n = t.shape[1]
    higher = (t[:, None, :] > t[:, :, None]).sum(axis=2)
    equal = (t[:, None, :] == t[:, :, None]).sum(axis=2)
    top = n - higher
    bottom = top - equal + 1
    return {"best": top, "average": (top + bottom) / 2, "worst": bottom}[tie].astype(float)


def eliminated_index(judge_pts, fan_pts) -> int:
    """Lowest total goes home; a tie on the total is broken by the worse fan rank (fewer fan points)."""
    total = np.asarray(judge_pts, dtype=float) + np.asarray(fan_pts, dtype=float)
    lowest = np.flatnonzero(total == total.min())
    return int(lowest[np.argmin(np.asarray(fan_pts)[lowest])])


def eliminated_index_save(totals, judge_pts, fan_pts, save_q: float, rng: np.random.Generator) -> int:
    """Judges' save, one week at a time: the H1 bottom two go to the judges, who send home the lower judge
    total with probability save_q (a coin flip if the totals are equal)."""
    totals = np.asarray(totals, dtype=float)
    jp, fp = np.asarray(judge_pts, dtype=float), np.asarray(fan_pts, dtype=float)
    first = eliminated_index(jp, fp)
    rest = np.delete(np.arange(len(jp)), first)
    second = int(rest[eliminated_index(jp[rest], fp[rest])])
    if totals[first] == totals[second]:
        return first if rng.random() < 0.5 else second
    lower, higher = (first, second) if totals[first] < totals[second] else (second, first)
    return lower if rng.random() < save_q else higher


def feasible_rank_rule(judge_pts, eliminated: int) -> bool:
    """Is there any fan ordering under which `eliminated` has the lowest combined total?

    Giving the eliminated couple the worst fan rank is always optimal (it lowers its own total and
    wins the tie-break), so only the others' fan points 2..N need a valid assignment.
    """
    jp = np.asarray(judge_pts, dtype=float)
    n = len(jp)
    needed = np.delete(jp[eliminated] + 1.0 - jp, eliminated)  # fan points each other couple needs
    return bool(np.all(np.sort(needed) <= np.arange(2, n + 1)))


def feasible_save(judge_pts, eliminated: int) -> bool:
    """Is there any fan ordering that puts `eliminated` in the H1 bottom two (so the judges could send it home)?

    Giving the pair fan points 1 and 2 is optimal (it lowers both of their totals and frees the high fan
    points for the others), so for each partner and each split of {1, 2} the others only need a valid
    assignment of fan points 3..N that keeps them above the pair.
    """
    jp = np.asarray(judge_pts, dtype=float)
    n, e = len(jp), eliminated
    scale = 2 * (n + 1)  # order_key(jp, f) = jp * scale + f * (scale + 1)
    for j in range(n):
        if j == e:
            continue
        others = np.array([k for k in range(n) if k not in (e, j)], dtype=int)
        for fe, fj in ((1, 2), (2, 1)):
            threshold = max(order_key(jp[e], fe, n), order_key(jp[j], fj, n))
            # Smallest fan points f with order_key(jp_k, f) > threshold.
            needed = np.floor(np.round((threshold - jp[others] * scale) / (scale + 1), 9)) + 1
            if np.all(np.sort(np.maximum(needed, 3)) <= np.arange(3, n + 1)):
                return True
    return False


def feasible_judges_only(scores, eliminated: int) -> bool:
    s = np.asarray(scores, dtype=float)
    return bool(s[eliminated] <= s.min())


@lru_cache(maxsize=None)
def _all_orderings(n: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(1, n + 1))), dtype=np.int16)


def share_of_orderings_eliminating(
    judge_pts, eliminated: int, exact_max: int = 8, draws: int = 20000, rng=None
) -> float:
    """Share of equally likely fan orderings that eliminate `eliminated`.

    This is the chance a null (random) fan vote reproduces the observed elimination, so it measures
    how informative the rule check is for that week. Exact for small fields, Monte Carlo otherwise.
    """
    jp = np.asarray(judge_pts, dtype=float)
    n = len(jp)
    if n <= exact_max:
        fans = _all_orderings(n)
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        fans = np.argsort(rng.random((draws, n)), axis=1).astype(np.int16) + 1
    winner = np.argmin(order_key(jp, fans, n), axis=1)
    return float(np.mean(winner == eliminated))
