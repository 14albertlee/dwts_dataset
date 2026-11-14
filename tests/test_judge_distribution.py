import numpy as np
import pandas as pd

from dwts_model.judge_distribution import RESID_FROM, forecast_night, judge_noise_var, residual_spread


def _oos(seed=0, seasons=range(RESID_FROM, RESID_FROM + 6), couples=6):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in range(2, 9):
            night = rng.normal(0, 0.2)
            for c in range(couples):
                rows.append((s, w, c, 8.0, 8.0 + night + rng.normal(0, 0.6)))
    return pd.DataFrame(rows, columns=["season", "week", "couple_id", "pred", "score"])


def test_residual_spread_uses_only_earlier_seasons():
    oos = _oos()
    target = RESID_FROM + 4
    base = residual_spread(oos, target)
    poisoned = oos.copy()
    poisoned.loc[poisoned["season"] >= target, "score"] += 5.0
    assert residual_spread(poisoned, target) == base
    assert set(base) == {"w2-3", "w4-6", "w7+"}
    assert all(var_w > 0 and var_n >= 0 for var_w, var_n in base.values())


def test_judge_noise_uses_only_recent_earlier_seasons():
    rng = np.random.default_rng(0)
    rows = [(s, s * 100 + p, j, float(np.clip(round(rng.normal(8, 1)), 1, 10)))
            for s in range(1, 10) for p in range(6) for j in range(3)]
    judges = pd.DataFrame(rows, columns=["season", "performance_id", "judge_id", "judge_score"])
    base = judge_noise_var(judges, 8)
    poisoned = judges.copy()
    poisoned.loc[poisoned["season"] >= 8, "judge_score"] = 1.0  # target season and later
    poisoned.loc[poisoned["season"] <= 2, "judge_score"] = 10.0  # outside the 5-season window
    assert judge_noise_var(poisoned, 8) == base
    assert base > 0


def test_forecast_night_probabilities():
    spread = {b: (0.25, 0.04) for b in ("w2-3", "w4-6", "w7+")}
    mean = np.array([9.5, 9.0, 8.5, 5.0])
    fc = forecast_night(mean, 5, np.full(4, 3), spread, 0.4, np.random.default_rng(0), n_draws=4000)
    assert fc["p_lowest"].iloc[3] > 0.99
    assert (fc["p_lowest"] >= fc["p_tied_lowest"]).all()
    assert fc["p_lowest"].sum() >= 1 - 1e-9  # ties can only add to the total
    assert (fc["lo"] <= mean.clip(1, 10)).all() and (fc["hi"] >= mean.clip(1, 10)).all()

    # Identical couples share the lowest total often, and tie probability is then positive.
    same = forecast_night(np.full(3, 8.0), 5, np.full(3, 3), spread, 0.4, np.random.default_rng(0), n_draws=4000)
    assert (same["p_tied_lowest"] > 0).all()
    assert same["p_lowest"].sum() > 1.05
