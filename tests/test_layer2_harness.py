import numpy as np
import pandas as pd
import pytest

from dwts_model.layer2.baselines import fit_rho, lowest_judge_probs
from dwts_model.layer2.fan_model import log_plackett_luce, sample_fan_points
from dwts_model.layer2.fit import fit_theta, prepare_week, profiled_nll_and_grad
from dwts_model.layer2.forecast import all_fan_points, elimination_probs
from dwts_model.layer2.metrics import bootstrap_ci, calibration, paired_ci, summarize, week_metrics
from dwts_model.layer2.rules import (
    Rule,
    eliminated_index,
    judge_points,
    judge_points_matrix,
    share_of_orderings_eliminating,
)
from dwts_model.layer2.validation import null_self_consistency

RANK = Rule("rank", "average")


@pytest.mark.parametrize("tie", ["best", "average", "worst"])
def test_judge_points_matrix_matches_scalar(tie):
    rng = np.random.default_rng(0)
    totals = rng.integers(20, 26, size=(50, 7))
    expected = np.array([judge_points(t, tie) for t in totals])
    assert np.array_equal(judge_points_matrix(totals, tie), expected)


def test_plackett_luce_orderings_sum_to_one_and_theta_zero_is_uniform():
    u = np.array([0.5, -0.2, 1.0, 0.0])
    best_first = np.argsort(-all_fan_points(4), axis=1)
    assert np.exp(log_plackett_luce(u, best_first)).sum() == pytest.approx(1.0)
    assert np.allclose(np.exp(log_plackett_luce(np.zeros(4), best_first)), 1 / 24)


def test_gumbel_sampling_follows_plackett_luce():
    u = np.array([1.0, 0.0, -1.0])
    draws = sample_fan_points(u, 60000, np.random.default_rng(0))
    best = np.argmax(draws, axis=1)
    expected = np.exp(u) / np.exp(u).sum()  # P(first place) under Plackett-Luce
    assert np.bincount(best, minlength=3) / len(best) == pytest.approx(expected, abs=0.01)


def test_null_probs_match_ordering_share_and_sum_to_one():
    totals = np.array([27, 25, 25, 22, 20, 20])
    jp = judge_points(totals, "average")
    p = elimination_probs(totals, np.zeros(6), RANK, np.random.default_rng(0))
    assert p.sum() == pytest.approx(1.0)
    for e in range(6):
        assert p[e] == pytest.approx(share_of_orderings_eliminating(jp, e))


@pytest.mark.parametrize("rule", [RANK, Rule("save", save_q=0.7)])
def test_exact_and_monte_carlo_agree_with_fan_utilities(rule):
    totals = np.array([27, 25, 25, 22, 20, 20, 18])
    u = np.array([0.8, -0.3, 0.2, 0.0, -0.5, 0.4, -0.9])
    exact = elimination_probs(totals, u, rule, np.random.default_rng(0))
    mc = elimination_probs(totals, u, rule, np.random.default_rng(1), n_draws=80000, exact_max=0)
    assert exact.sum() == pytest.approx(1.0)
    assert mc == pytest.approx(exact, abs=0.01)


def test_higher_fan_utility_lowers_elimination_probability():
    totals = np.array([24, 24, 24, 24])
    p = elimination_probs(totals, np.array([2.0, 0.0, 0.0, -2.0]), RANK, np.random.default_rng(0))
    assert p[0] < p[1] < p[3]
    assert p[1] == pytest.approx(p[2])


def test_uniform_mixing():
    p = elimination_probs([27, 20, 24], np.zeros(3), RANK, np.random.default_rng(0), eps=0.3)
    base = elimination_probs([27, 20, 24], np.zeros(3), RANK, np.random.default_rng(0))
    assert p == pytest.approx(0.7 * base + 0.1)


def test_layer1_style_draws_are_averaged_over_draws():
    draws = np.array([[20, 25, 25], [25, 20, 25]])  # couple 0 or 1 is the clear lowest, equally often
    p = elimination_probs(draws, np.zeros(3), RANK, np.random.default_rng(0), n_draws=2)
    assert p.sum() == pytest.approx(1.0)
    assert p[2] < p[0] and p[0] == pytest.approx(p[1], abs=0.3)
    assert lowest_judge_probs(draws).tolist() == [0.5, 0.5, 0.0]


def test_lowest_judge_splits_ties():
    assert lowest_judge_probs([20, 20, 25]).tolist() == [0.5, 0.5, 0.0]


def test_lowest_judge_rho_keeps_judge_order_and_is_fitted():
    p = lowest_judge_probs([20, 26, 23, 23], rho=0.5)
    assert p[0] > p[2] == p[3] > p[1]
    assert p == pytest.approx(np.array([1, 0.125, 0.5, 0.5]) / 2.125)
    rng = np.random.default_rng(0)
    weeks = []
    for _ in range(400):
        totals = rng.permutation(6).astype(float)
        weeks.append((totals, int(rng.choice(6, p=lowest_judge_probs(totals, 0.4)))))
    assert fit_rho(weeks)[0] == pytest.approx(0.4, abs=0.08)


def _simulate_weeks(theta, n_weeks, n_couples, seed, eps=0.0, rule=RANK):
    """Weeks whose eliminations come from the model: one feature, actual judge totals, the given rule."""
    rng = np.random.default_rng(seed)
    weeks = []
    for _ in range(n_weeks):
        x = rng.normal(size=(n_couples, 1))
        totals = rng.integers(20, 27, n_couples).astype(float)
        fans = sample_fan_points(x[:, 0] * theta, 1, rng)
        w = rule.elimination_weights(totals, fans)[0]
        e = int(rng.integers(n_couples)) if rng.random() < eps else int(rng.choice(n_couples, p=w))
        weeks.append(prepare_week(totals, e, x, rule, rng))
    return weeks


@pytest.mark.parametrize("theta", [0.0, 1.0])
def test_fitter_recovers_theta(theta):
    weeks = _simulate_weeks(theta, n_weeks=400, n_couples=6, seed=3)
    est = fit_theta(weeks, 1, ridge=0.0)
    assert est.theta[0] == pytest.approx(theta, abs=0.3)
    assert est.eps < 0.1


def test_fitter_recovers_uniform_mixing_weight():
    weeks = _simulate_weeks(1.0, n_weeks=600, n_couples=6, seed=4, eps=0.3)
    est = fit_theta(weeks, 1, ridge=0.0)
    assert est.eps == pytest.approx(0.3, abs=0.12)
    assert est.theta[0] == pytest.approx(1.0, abs=0.4)


def test_fitter_recovers_judges_save_probability():
    weeks = _simulate_weeks(0.0, n_weeks=500, n_couples=6, seed=6, rule=Rule("save", save_q=0.8))
    est = fit_theta(weeks, 1)
    assert est.save_q == pytest.approx(0.8, abs=0.1)


def test_profiled_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    weeks = []
    for i in range(40):
        n = 6 if i % 2 else 10  # enumerated and sampled weeks
        x = rng.normal(size=(n, 2))
        totals = rng.integers(20, 27, n).astype(float)
        rule = Rule("save", save_q=0.8) if i % 3 == 0 else RANK
        e = int(rng.integers(n)) if i % 4 == 0 else int(np.argmin(totals))  # some surprises, so eps > 0
        weeks.append(prepare_week(totals, e, x, rule, rng, draws=4000))
    n = np.array([w.n_couples for w in weeks], dtype=float)
    theta = np.array([0.4, -0.3])
    _, grad = profiled_nll_and_grad(theta, weeks, n, ridge=0.1)
    h = 1e-5
    numeric = [
        (profiled_nll_and_grad(theta + h * d, weeks, n, 0.1)[0] - profiled_nll_and_grad(theta - h * d, weeks, n, 0.1)[0]) / (2 * h)
        for d in np.eye(2)
    ]
    assert grad == pytest.approx(numeric, rel=1e-3, abs=1e-4)


def test_fitter_with_sampled_orderings_for_large_fields():
    rng = np.random.default_rng(5)
    weeks = []
    for _ in range(150):
        x = rng.normal(size=(10, 1))
        totals = rng.integers(20, 27, 10).astype(float)
        fans = sample_fan_points(x[:, 0] * 1.0, 1, rng)[0]
        e = eliminated_index(judge_points(totals, "average"), fans)
        weeks.append(prepare_week(totals, e, x, RANK, rng, exact_max=8, draws=20000))
    est = fit_theta(weeks, 1, ridge=0.0)
    assert est.theta[0] == pytest.approx(1.0, abs=0.45)
    # The diagnostic equals the kept count at theta = 0 and shrinks as utilities spread out.
    w = weeks[0]
    assert w.sampled and w.ess(np.zeros(1)) == pytest.approx(len(w.best_first))
    assert w.ess(est.theta) < w.ess(np.zeros(1))


def _probs(p, e, season=1, week=2):
    return pd.DataFrame(
        {"season": season, "week": week, "couple_id": range(len(p)), "n_couples": len(p),
         "p_elim": p, "eliminated": [i == e for i in range(len(p))]}
    )


def test_week_metrics_ranks_ties_and_losses():
    wm = week_metrics(_probs([0.4, 0.4, 0.2], e=1))  # tied for the top, actual couple is one of the tied pair
    row = wm.iloc[0]
    assert row["top1"] == 0.5 and row["top2"] == 1.0 and row["rank"] == 1.5
    assert row["logloss"] == pytest.approx(-np.log(0.4))
    assert row["brier"] == pytest.approx(0.4**2 + 0.6**2 + 0.2**2)
    assert row["random_top1"] == pytest.approx(1 / 3) and row["random_logloss"] == pytest.approx(np.log(3))


def test_uniform_forecast_matches_random_baseline():
    wm = week_metrics(pd.concat([_probs([0.25] * 4, e=2, week=2), _probs([0.2] * 5, e=0, week=3)]))
    for k in ("top1", "top2", "top3", "logloss", "brier"):
        assert np.allclose(wm[k], wm["random_" + k])


def test_summaries_and_bootstrap():
    probs = pd.concat([_probs([0.5, 0.3, 0.2], e=0, season=s, week=w) for s in (1, 2, 3) for w in (2, 3)])
    wm = week_metrics(probs)
    assert summarize(wm).loc[0, "weeks"] == 6
    ci = bootstrap_ci(wm, ["top1"], n_boot=50)
    assert ci.loc[0, "lo"] <= ci.loc[0, "mean"] <= ci.loc[0, "hi"]
    cal = calibration(probs, bins=3)
    assert cal["n"].sum() == len(probs)
    worse = week_metrics(pd.concat([_probs([0.2, 0.3, 0.5], e=0, season=s, week=w) for s in (1, 2, 3) for w in (2, 3)]))
    diff = paired_ci(wm, worse, ["logloss", "top1"], n_boot=50).set_index("metric")
    assert diff.loc["logloss", "mean"] == pytest.approx(np.log(0.2) - np.log(0.5))
    assert diff.loc["top1", "mean"] == 1.0 and diff.loc["top1", "weeks"] == 6


def test_null_self_consistency_on_synthetic_season():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, 4):
        for w in range(2, 7):
            n = 12 - w
            scores = rng.integers(20, 28, n) / 3
            rows += [(s, w, c, scores[c], c == 0) for c in range(n)]
    cw = pd.DataFrame(rows, columns=["season", "week", "couple_id", "score", "eliminated"])
    eras = {1: Rule("rank"), 2: Rule("save", save_q=0.9), 3: Rule("save", save_q=0.7)}
    probs = pd.concat(
        g.assign(n_couples=len(g), era=eras[s].era, eps=0.2, save_q=eras[s].save_q,
                 p_elim=elimination_probs(g["score"].to_numpy(), np.zeros(len(g)), eras[s], rng, eps=0.2))
        for (s, _), g in cw.groupby(["season", "week"])
    ).reset_index(drop=True)
    check = null_self_consistency(probs, cw, "average", n_rep=200, seed=1).set_index("metric")
    assert (check["z"].abs() < 4).all()
