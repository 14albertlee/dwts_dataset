import numpy as np
import pandas as pd

from dwts_model.baselines import FieldAdjusted, LastWeekDrift
from dwts_model.config import load_config
from dwts_model.feature_engineering import build_features
from dwts_model.splits import rolling_origin


def _weekly(seed=0, seasons=range(1, 8), weeks=6, couples=5):
    rng = np.random.default_rng(seed)
    rows = [
        (s, w, c, rng.uniform(4, 10))
        for s in seasons for w in range(1, weeks + 1) for c in range(s * 100, s * 100 + couples)
    ]
    return pd.DataFrame(rows, columns=["season", "week", "couple_id", "score"])


def test_features_ignore_own_week_score():
    weekly = _weekly()
    base = build_features(weekly)
    changed = weekly.copy()
    mask = (changed["season"] == 3) & (changed["week"] == 4)
    changed.loc[mask, "score"] = 1.0
    new = build_features(changed)
    key = ["season", "week", "couple_id"]
    cols = ["prev_score", "running_mean", "prev_dev"]
    merged = base.merge(new, on=key, suffixes=("_a", "_b"))
    same_week = (merged["season"] == 3) & (merged["week"] == 4)
    for c in cols:
        assert np.allclose(merged.loc[same_week, c + "_a"], merged.loc[same_week, c + "_b"])


def test_features_ignore_later_seasons():
    weekly = _weekly()
    base = build_features(weekly[weekly["season"] <= 4])
    full = build_features(weekly).query("season <= 4").reset_index(drop=True)
    pd.testing.assert_frame_equal(base, full)


def test_first_week_dropped_and_running_mean_is_prior_only():
    df = build_features(_weekly())
    assert (df["week"] > 1).all()
    row = df[(df["season"] == 2) & (df["week"] == 3)].iloc[0]
    weekly = _weekly()
    prior = weekly[(weekly["season"] == 2) & (weekly["couple_id"] == row["couple_id"]) & (weekly["week"] < 3)]
    assert np.isclose(row["running_mean"], prior["score"].mean())


def test_rolling_origin_never_trains_on_test_or_later_seasons():
    cfg = load_config()
    df = build_features(_weekly(seasons=range(1, 40)))
    for mode in ("validate", "holdout"):
        for season, train, test in rolling_origin(df, cfg, mode):
            assert train["season"].max() < season
            assert (test["season"] == season).all()


def test_validate_mode_never_touches_holdout_seasons():
    cfg = load_config()
    df = build_features(_weekly(seasons=range(1, 40)))
    seasons = {s for s, _, _ in rolling_origin(df, cfg, "validate")}
    assert seasons.isdisjoint(range(cfg.holdout[0], cfg.holdout[1] + 1))


def test_prev_rank_and_dev_use_only_prior_week_field():
    weekly = _weekly()
    df = build_features(weekly)
    row = df[(df["season"] == 2) & (df["week"] == 4)].iloc[0]
    prior = weekly[(weekly["season"] == 2) & (weekly["week"] == 3)].set_index("couple_id")["score"]
    me = prior[row["couple_id"]]
    assert np.isclose(row["prev_dev"], me - prior.mean())
    assert np.isclose(row["prev_rank"], (prior.rank().loc[row["couple_id"]] - 1) / (len(prior) - 1))
    assert df["prev_rank"].between(0, 1).all()


def test_lmm_prediction_ignores_own_and_later_scores():
    from dwts_model.models import LMMModel

    weekly = _weekly()
    df = build_features(weekly)
    train, test = df[df["season"] < 6], df[df["season"] == 6].copy()
    model = LMMModel(["prev_score", "running_mean", "week"]).fit(train)
    base = model.predict(test)
    altered = test.copy()
    mask = (altered["week"] >= 4)
    altered.loc[mask, "score"] = 1.0
    pred = model.predict(altered)
    # Scores from week 4 on cannot affect predictions for weeks <= 4.
    early = altered["week"] <= 4
    assert np.allclose(base[early], pred[early])


def test_pair_counts_handles_ties_and_ordering():
    from dwts_model.evaluate import _pair_counts

    actual = np.array([9.0, 8.0, 8.0, 6.0])
    # Perfect order: the actual-tied pair (8, 8) is skipped, leaving 5 comparable pairs.
    assert _pair_counts(actual, np.array([4.0, 3.0, 2.5, 1.0])) == (5.0, 5)
    # Fully reversed order gets nothing right.
    assert _pair_counts(actual, np.array([1.0, 2.0, 2.5, 4.0])) == (0.0, 5)
    # A predicted tie on a comparable pair counts as half.
    assert _pair_counts(np.array([9.0, 8.0]), np.array([5.0, 5.0])) == (0.5, 1)


def test_baseline_statistics_use_only_training_data():
    df = build_features(_weekly())
    train = df[df["season"] < 6]
    poisoned = df.copy()
    poisoned.loc[poisoned["season"] >= 6, "score"] = 10.0
    poisoned_train = poisoned[poisoned["season"] < 6]
    for cls in (LastWeekDrift, FieldAdjusted):
        a = cls().fit(train)
        b = cls().fit(poisoned_train)
        test = df[df["season"] == 6]
        assert np.allclose(a.predict(test), b.predict(test))
