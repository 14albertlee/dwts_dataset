"""Judge-score inputs to the Layer 2 harness: actual totals, or Layer 1 simulated integer totals.

Both return an array (rows, couples) in the order of the `couple_ids` passed in: one row for actual
scores, `n_draws` rows for Layer 1.
"""
import json

import numpy as np
import pandas as pd

from ..feature_engineering import build_context_features, build_features
from ..judge_distribution import (
    RESID_FROM,
    judge_noise_var,
    residual_spread,
    ridge_oos,
    simulate_night_totals,
)
from ..models import RidgeModel


class ActualJudgeInput:
    name = "actual"

    def __init__(self, cw: pd.DataFrame):
        self._scores = cw.set_index(["season", "week", "couple_id"])["score"]

    def available(self, season: int, week: int, couple_ids) -> bool:
        return True

    def totals(self, season: int, week: int, couple_ids) -> np.ndarray:
        return self._scores.loc[[(season, week, c) for c in couple_ids]].to_numpy()[None, :]


class Layer1JudgeInput:
    """Layer 1 forecast of a night's integer judge totals, using only earlier data (as in predict_upcoming)."""

    name = "layer1"

    def __init__(self, cfg, weekly, perf, judges, n_draws: int | None = None):
        self.cfg = cfg
        self.n_draws = n_draws or cfg.mc_draws
        self.history = build_features(weekly, build_context_features(perf))
        self._featured = self.history.groupby(["season", "week"])["couple_id"].agg(set).to_dict()
        self.judges = judges
        self.slots = judges.groupby(["season", "week", "couple_id"])["judge_score"].count()
        params = json.loads((cfg.output_dir / "selected_params.json").read_text())["ridge_alpha"]
        self.set_name = cfg.main_model.removeprefix("ridge_")
        self.alpha = params[self.set_name]
        self._oos = None

    def available(self, season: int, week: int, couple_ids) -> bool:
        # The spread needs earlier out-of-sample residuals, and every couple needs a previous week for features
        # (so week 1 and a couple's first appearance cannot be forecast).
        return season > RESID_FROM and set(couple_ids) <= self._featured.get((season, week), set())

    def _residuals(self) -> pd.DataFrame:
        if self._oos is None:
            last = int(self.history["season"].max()) - 1
            self._oos = ridge_oos(self.history, self.cfg.features[self.set_name], self.alpha, RESID_FROM, last)
        return self._oos

    def totals(self, season: int, week: int, couple_ids) -> np.ndarray:
        h = self.history
        feats = self.cfg.features[self.set_name]
        train = h[(h["season"] < season) | ((h["season"] == season) & (h["week"] < week))]
        rows = h[(h["season"] == season) & (h["week"] == week)].set_index("couple_id").loc[list(couple_ids)]
        mean = RidgeModel(feats, self.alpha).fit(train).predict(rows.reset_index())
        n_slots = np.array([self.slots[(season, week, c)] for c in couple_ids], dtype=float)
        spread = residual_spread(self._residuals(), season)
        rng = np.random.default_rng([self.cfg.seed, season, week])
        totals, _ = simulate_night_totals(
            mean, week, n_slots, spread, judge_noise_var(self.judges, season), rng, self.n_draws
        )
        return totals
