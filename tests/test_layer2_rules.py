import itertools

import numpy as np
import pandas as pd
import pytest

from dwts_model.layer2.rule_check import check_weeks, summarize_rules, usable_weeks
from dwts_model.layer2.rules import (
    Rule,
    eliminated_index,
    eliminated_index_save,
    era_for_season,
    feasible_rank_rule,
    feasible_save,
    judge_points,
    share_of_orderings_eliminating,
)


def test_judge_points_tie_conventions():
    scores = [9, 8, 8, 8, 5]
    assert judge_points(scores, "best").tolist() == [5, 4, 4, 4, 1]
    assert judge_points(scores, "average").tolist() == [5, 3, 3, 3, 1]
    assert judge_points(scores, "worst").tolist() == [5, 2, 2, 2, 1]


def test_lowest_total_goes_home_and_fan_breaks_ties():
    jp = [3, 2, 1]
    assert eliminated_index(jp, [1, 2, 3]) == 0  # totals 4, 4, 4: the worst fan rank (fan points 1) leaves
    assert eliminated_index([3, 2, 1], [3, 2, 1]) == 2
    assert eliminated_index([1, 2, 3], [3, 1, 2]) == 1  # totals 4, 3, 5


def _brute_feasible(jp, e):
    n = len(jp)
    return any(eliminated_index(jp, fan) == e for fan in itertools.permutations(range(1, n + 1)))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("tie", ["best", "average", "worst"])
def test_closed_form_matches_brute_force(seed, tie):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7))
    scores = rng.integers(3, 8, n)  # small range, so ties are common
    jp = judge_points(scores, tie)
    for e in range(n):
        assert feasible_rank_rule(jp, e) == _brute_feasible(jp, e)


def test_share_of_orderings_matches_brute_force_and_monte_carlo():
    jp = judge_points([9, 8, 8, 6, 5], "average")
    for e in range(5):
        fans = list(itertools.permutations(range(1, 6)))
        expected = np.mean([eliminated_index(jp, f) == e for f in fans])
        assert share_of_orderings_eliminating(jp, e) == pytest.approx(expected)
        mc = share_of_orderings_eliminating(jp, e, exact_max=0, draws=40000, rng=np.random.default_rng(0))
        assert mc == pytest.approx(expected, abs=0.01)


def test_rank_rule_has_power_only_through_ties():
    # Without ties, any couple can be eliminated by some fan ordering, even the leader.
    assert feasible_rank_rule(judge_points([10, 6, 5, 4], "average"), 0)
    # A leader facing a tied block that is scored at its worst positions cannot be.
    assert not feasible_rank_rule(judge_points([10, 5, 5, 5], "worst"), 0)
    assert feasible_rank_rule(judge_points([10, 5, 5, 5], "best"), 0)


def test_eras_by_season():
    era = [era_for_season(s, (28, 31)) for s in (1, 3, 27, 28, 31, 32, 35)]
    assert era == ["rank", "rank", "rank", "save", "save", "rank", "rank"]


def _random_week(rng):
    n = int(rng.integers(3, 7))
    return n, rng.integers(3, 8, n).astype(float)  # small range, so ties are common


class _FixedUniform:
    def __init__(self, r):
        self.r = r

    def random(self):
        return self.r


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("era", ["rank", "save"])
def test_vectorized_rule_matches_scalar_rules(seed, era):
    rng = np.random.default_rng(seed)
    n, totals = _random_week(rng)
    rule = Rule(era, "average", save_q=0.8)
    fans = np.array(list(itertools.permutations(range(1, n + 1))), dtype=float)
    weights = rule.elimination_weights(totals, fans)
    assert np.allclose(weights.sum(axis=1), 1.0)
    jp = judge_points(totals, "average")
    for fan, w in zip(fans, weights):
        if era == "rank":
            expected = np.eye(n)[eliminated_index(jp, fan)]
        else:  # the scalar save rule draws a uniform; average it over an evenly spaced grid of draws
            draws = [eliminated_index_save(totals, jp, fan, 0.8, _FixedUniform(r)) for r in np.arange(0.05, 1, 0.1)]
            expected = np.bincount(draws, minlength=n) / len(draws)
        assert np.allclose(w, expected)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("tie", ["best", "average", "worst"])
def test_save_feasibility_matches_brute_force(seed, tie):
    rng = np.random.default_rng(seed)
    n, totals = _random_week(rng)
    jp = judge_points(totals, tie)
    weights = Rule("save", tie, save_q=0.5).elimination_weights(
        totals, np.array(list(itertools.permutations(range(1, n + 1))), dtype=float)
    )
    for e in range(n):
        assert feasible_save(jp, e) == bool(np.any(weights[:, e] > 0))


def _cw(scores, elim, season=1, week=2):
    return pd.DataFrame(
        {
            "season": season,
            "week": week,
            "couple_id": range(len(scores)),
            "score": scores,
            "n_performances": 1,
            "eliminated": [i == elim for i in range(len(scores))],
            "week_type": "single",
            "judge_rank": pd.Series(scores).rank(method="min", ascending=False).astype(int),
            "bottom_rank": pd.Series(scores).rank(method="min").astype(int),
            "uneven_performances": False,
        }
    )


def test_check_weeks_flags_inconsistent_week_and_usable_set():
    consistent = _cw([9, 8, 7, 6, 5], elim=4, week=2)
    impossible = _cw([10, 5, 5, 5, 5, 5, 5, 5], elim=0, week=3)  # leader out with a clear margin
    uneven = _cw([9, 8, 7, 6, 5], elim=4, week=4).assign(uneven_performances=True)
    cw = pd.concat([consistent, impossible, uneven])
    checks = check_weeks(cw, ["best", "average", "worst"])
    row = checks.set_index("week")
    assert row.loc[2, "rank_average"] and row.loc[2, "judges_only"]
    assert not row.loc[3, "rank_average"] and not row.loc[3, "judges_only"]
    usable = usable_weeks(checks)
    assert usable["week"].tolist() == [2]
    summary = summarize_rules(checks, ["best", "average", "worst"]).set_index("rule")
    assert summary.loc["rank_average", "consistent"] == 2


def test_week_is_judged_by_its_era_rule():
    # The leader cannot have the lowest H1 total here, but it can land in the bottom two for the judges.
    cw = _cw([9, 8, 7, 5, 5], elim=0, season=29, week=3)
    checks = check_weeks(cw, ["average"], era_of=lambda s: era_for_season(s, (28, 31)))
    assert checks.loc[0, "era"] == "save" and checks.loc[0, "consistent"]
    assert not checks.loc[0, "rank_average"]


def test_non_single_weeks_are_not_checked():
    cw = _cw([9, 8, 7], elim=2)
    cw["week_type"] = "finale"
    assert check_weeks(cw, ["average"]).empty
