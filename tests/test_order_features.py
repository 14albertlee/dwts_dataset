import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from parse_scores_json_to_tables import add_running_order  # noqa: E402

from dwts_model.feature_engineering import build_order_features  # noqa: E402


def test_running_order_rounds_and_couple_major_layout():
    couples = ["A", "B", "C", "D",  # round-major: two tables
               "B", "A", "C", "D",
               "X", "X", "Y", "Y"]  # couple-major: one table, one row per dance
    titles = ["Week 1"] * 8 + ["Week 2"] * 4
    out = add_running_order(pd.DataFrame({"season": 1, "week_title": titles, "couple": couples}))
    assert out["round"].tolist() == [1] * 4 + [2] * 4 + [1] * 4
    assert out["running_order"].tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 2]
    assert out["round_size"].tolist() == [4] * 8 + [2] * 4


def test_order_features_actual_and_base_rate():
    n = 5
    perf = pd.DataFrame(
        {"season": 1, "week": 1, "couple_id": range(n), "running_order": range(1, n + 1), "round_size": n}
    )
    f = build_order_features(perf)
    assert f["order_frac"].tolist() == [0, 0.25, 0.5, 0.75, 1]
    assert f["order_first"].tolist() == [1, 0, 0, 0, 0]
    assert f["order_last"].tolist() == [0, 0, 0, 0, 1]
    # Random-order expectations equal the average of the actual features across the round.
    for col in ("frac", "first", "last", "frac_sq"):
        assert np.allclose(f[f"base_order_{col}"], f[f"order_{col}"].mean())


def test_order_features_small_round_is_neutral():
    perf = pd.DataFrame(
        {"season": 1, "week": 1, "couple_id": [0, 1, 2], "running_order": [1, 2, 3], "round_size": 3}
    )
    f = build_order_features(perf)
    assert np.allclose(f["order_frac"], 0.5)
    assert np.allclose(f["order_first"], 1 / 3)


def _upcoming(order):
    from dwts_model.data import load_performance_context, load_weekly_scores
    from dwts_model.config import load_config
    from dwts_model.predict import build_upcoming_features

    cfg = load_config()
    weekly, perf = load_weekly_scores(cfg), load_performance_context(cfg)
    ids = sorted(weekly[(weekly["season"] == 32) & (weekly["week"] == 5)]["couple_id"])[:4]
    sched = pd.DataFrame({"couple_id": ids, "styles": [["Foxtrot"]] * 4})
    if order is not None:
        sched["order"] = [[o] for o in order]
    return build_upcoming_features(weekly, perf, sched, season=32, week=6, theme="Disney Night")


def test_upcoming_order_modes_use_base_rate_or_announced_values():
    base = _upcoming(None)
    assert np.allclose(base["order_frac"], 0.5)
    assert np.allclose(base["order_first"], 0.25) and np.allclose(base["order_last"], 0.25)
    ann = _upcoming([3, 1, 4, 2])
    assert np.allclose(sorted(ann["order_frac"]), [0, 1 / 3, 2 / 3, 1])
    assert ann["order_first"].sum() == 1 and ann["order_last"].sum() == 1


def test_check_order_all_or_none_and_valid_positions():
    import pytest

    from dwts_model.predict import check_order

    styles = [["A"], ["B"], ["C"]]
    assert check_order(pd.DataFrame({"styles": styles})) == "base-rate"
    assert check_order(pd.DataFrame({"styles": styles, "order": [[2], [1], [3]]})) == "announced"
    with pytest.raises(ValueError):
        check_order(pd.DataFrame({"styles": styles, "order": [[2], float("nan"), [3]]}))
    with pytest.raises(ValueError):
        check_order(pd.DataFrame({"styles": styles, "order": [[1], [1], [3]]}))
