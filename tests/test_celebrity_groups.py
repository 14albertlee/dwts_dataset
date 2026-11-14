import pandas as pd

from dwts_model.categories import celebrity_group
from dwts_model.feature_engineering import build_celebrity_features


def test_celebrity_group_rule_order_and_defaults():
    cases = {
        "Olympic figure skater": "movement_trained",  # before the generic Olympic athlete rule
        "Glee actress & dancer": "movement_trained",  # before actor
        "NSYNC singer & actor": "movement_trained",  # before musician
        "WWE wrestler": "movement_trained",
        "NFL wide receiver": "athlete_other",
        "Paralympic swimmer & ESPN host": "athlete_other",
        "Actor & singer": "musician",
        "Model & actress": "actor",  # before reality_social's model rule
        "Film & television actress": "actor",
        "The Bachelorette star": "reality_social",
        "Supermodel": "reality_social",
        "Television host": "other",
        "Celebrity chef & author": "other",
        None: "other",
    }
    assert {k: celebrity_group(k) for k in cases} == cases


def test_celebrity_features_one_hot_with_actor_reference():
    info = pd.DataFrame(
        {"season": 1, "couple_id": [1, 2], "notability": ["Film actor", "Olympic artistic gymnast"]}
    )
    f = build_celebrity_features(info).set_index("couple_id")
    celeb = [c for c in f.columns if c.startswith("celeb_")]
    assert "celeb_actor" not in celeb
    assert f.loc[1, celeb].sum() == 0
    assert f.loc[2, "celeb_movement_trained"] == 1 and f.loc[2, celeb].sum() == 1
