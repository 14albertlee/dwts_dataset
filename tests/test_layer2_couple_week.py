import numpy as np
import pandas as pd

from dwts_model.layer2.couple_week import build_couple_week

KEY = ["season", "week", "couple_id"]


def _weekly(seed=0, seasons=(1, 2, 3), weeks=7, couples=8):
    """Each week the couple with the lowest score leaves; later weeks have fewer couples."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        alive = list(range(s * 100, s * 100 + couples))
        for w in range(1, weeks + 1):
            scores = {c: rng.uniform(4, 10) for c in alive}
            out = min(scores, key=scores.get) if w > 1 else None
            for c in alive:
                rows.append((s, w, c, scores[c], 1, (c == out) if w > 1 else None))
            if out is not None:
                alive.remove(out)
    return pd.DataFrame(rows, columns=KEY + ["score", "n_performances", "eliminated"]).astype({"eliminated": "boolean"})


def test_features_ignore_target_week_and_later():
    weekly = _weekly()
    base = build_couple_week(weekly)
    season, week = 2, 4
    changed = weekly.copy()
    later = (changed["season"] == season) & (changed["week"] >= week)
    rng = np.random.default_rng(1)
    changed.loc[later, "score"] = rng.uniform(1, 10, later.sum())
    changed.loc[later, "eliminated"] = False
    new = build_couple_week(changed)

    merged = base.merge(new, on=KEY, suffixes=("_a", "_b"))
    scope = (merged["season"] != season) | (merged["week"] <= week)
    for col in ["skill", "survival_surplus"]:
        assert np.allclose(merged.loc[scope, col + "_a"], merged.loc[scope, col + "_b"]), col


def test_features_ignore_other_seasons():
    weekly = _weekly()
    base = build_couple_week(weekly[weekly["season"] <= 2])
    full = build_couple_week(weekly).query("season <= 2").reset_index(drop=True)
    pd.testing.assert_frame_equal(base, full)


def test_week_one_has_no_history():
    cw = build_couple_week(_weekly())
    first = cw[cw["week"] == 1]
    assert (first["skill"] == 0).all() and (first["survival_surplus"] == 0).all()


def test_survival_surplus_counts_survived_bottom3_weeks():
    weekly = pd.DataFrame(
        {
            "season": 1,
            "week": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            "couple_id": list(range(7)) * 3,
            "score": [9, 8, 7, 6, 5, 4, 3] * 2 + [5, 4, 9, 8, 7, 6, 3],
            "n_performances": 1,
            "eliminated": [False] * 14 + [False] * 6 + [True],
        }
    )
    weekly.loc[weekly["week"] == 2, "eliminated"] = [False] * 6 + [True]
    cw = build_couple_week(weekly)
    week3 = cw[cw["week"] == 3].set_index("couple_id")
    # Couples 4 and 5 were bottom-3 (with the eliminated 6) in weeks 1 and 2 and survived; week 1 had no elimination.
    assert week3.loc[5, "survival_surplus"] == 1
    assert week3.loc[0, "survival_surplus"] == 0


def test_week_types_and_ranks():
    cw = build_couple_week(_weekly())
    one_week = cw[(cw["season"] == 1) & (cw["week"] == 3)]
    assert one_week["week_type"].eq("single").all()
    assert cw[(cw["season"] == 1) & (cw["week"] == 7)]["week_type"].eq("finale").all()
    ranks = one_week.sort_values("score", ascending=False)["judge_rank"].tolist()
    assert ranks == list(range(1, len(one_week) + 1))


def test_withdrawal_week_is_not_a_vote():
    weekly = _weekly()
    gone = weekly[(weekly["season"] == 1) & (weekly["week"] == 3) & weekly["eliminated"].fillna(False)]
    withdrawals = gone[["season", "couple_id"]]
    cw = build_couple_week(weekly, withdrawals=withdrawals)
    week = cw[(cw["season"] == 1) & (cw["week"] == 3)]
    assert week["week_type"].eq("withdrawal").all() and week["withdrew"].sum() == 1
    # Surviving a withdrawal week is not surviving a vote.
    base = build_couple_week(weekly).set_index(["season", "week", "couple_id"])["survival_surplus"]
    new = cw.set_index(["season", "week", "couple_id"])["survival_surplus"]
    later = new.index.get_level_values("week") > 3
    assert (new[later & (new.index.get_level_values("season") == 1)] <= base[later & (base.index.get_level_values("season") == 1)]).all()
