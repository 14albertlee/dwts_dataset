import numpy as np
import pandas as pd

from dwts_model.baselines import FieldAdjusted, LastWeekDrift
from dwts_model.config import load_config
from dwts_model.feature_engineering import build_context_features, build_features
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


def test_prev_bottom3_flags_only_prior_week_in_large_fields():
    weekly = _weekly(couples=8)
    df = build_features(weekly)
    row = df[(df["season"] == 2) & (df["week"] == 4)].iloc[0]
    prior = weekly[(weekly["season"] == 2) & (weekly["week"] == 3)].set_index("couple_id")["score"]
    assert row["prev_bottom3"] == float(prior.rank(method="min").loc[row["couple_id"]] <= 3)
    small = build_features(_weekly(couples=5))
    assert (small["prev_bottom3"] == 0).all()


def test_upcoming_features_match_training_features():
    from dwts_model.predict import build_upcoming_features

    weekly = _weekly(couples=8).assign(n_performances=1, eliminated=False)
    styles = ["Foxtrot", "Samba", "Jazz", "Rumba"]
    perf = pd.DataFrame(
        [
            (r.season, r.week, r.couple_id, i, "Disney Night" if r.week == 4 else "No Theme", styles[(r.week + r.couple_id) % 4])
            for i, r in enumerate(weekly.itertuples())
        ],
        columns=["season", "week", "couple_id", "performance_id", "theme", "style"],
    )
    full = build_features(weekly, build_context_features(perf))
    target = full[(full["season"] == 3) & (full["week"] == 4)]

    # Pretend week 4 of season 3 hasn't aired: only its schedule is known.
    schedule = pd.DataFrame(
        {
            "couple_id": target["couple_id"].to_numpy(),
            "styles": [[styles[(4 + c) % 4]] for c in target["couple_id"]],
        }
    )
    up = build_upcoming_features(weekly, perf, schedule, season=3, week=4, theme="Disney Night")

    cols = [c for c in full.columns if c not in ("score", "n_performances", "eliminated")]
    pd.testing.assert_frame_equal(
        up.sort_values("couple_id")[cols].reset_index(drop=True),
        target.sort_values("couple_id")[cols].reset_index(drop=True),
    )
    assert up["score"].isna().all()


def test_pro_history_uses_only_earlier_seasons_and_shrinks():
    from dwts_model.feature_engineering import build_pro_history

    weekly = _weekly(seasons=range(1, 4))
    # Pro 7 partners the first couple each season; every other couple has its own new pro.
    couple_pros = weekly[["season", "couple_id"]].drop_duplicates()
    couple_pros["professional_id"] = np.where(couple_pros["couple_id"] % 100 == 0, 7, couple_pros["couple_id"])
    k = 5.0
    out = build_pro_history(weekly, couple_pros, k).set_index(["season", "couple_id"])

    dev = weekly["score"] - weekly.groupby(["season", "week"])["score"].transform("mean")
    prior = dev[(weekly["season"] < 3) & (weekly["couple_id"] % 100 == 0)]
    row = out.loc[(3, 300)]
    assert row["pro_prior_n"] == len(prior)
    assert np.isclose(row["pro_prior_dev"], prior.mean() * len(prior) / (len(prior) + k))
    assert out.loc[(1, 100), "pro_prior_dev"] == 0 and out.loc[(3, 301), "pro_prior_n"] == 0

    changed = weekly.copy()
    changed.loc[changed["season"] == 3, "score"] = 1.0
    again = build_pro_history(changed, couple_pros, k).set_index(["season", "couple_id"])
    pd.testing.assert_frame_equal(out, again)


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
