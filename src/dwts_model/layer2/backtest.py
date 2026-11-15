"""Rolling-origin backtest (plan section 5): forecast each usable week with parameters fit on earlier seasons.

`fit_season_params` fits once per test season on the actual judge totals of earlier seasons' usable weeks;
`backtest` then forecasts the season with any judge-score input (actual, or Layer 1 draws).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import fit_rho, lowest_judge_probs
from .fan_model import design_matrix, utilities
from .fit import fit_theta, prepare_week
from .forecast import elimination_probs
from .rules import Rule, era_for_season


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "fan" (era rule with a fan model) or "lowest_judge"
    features: tuple[str, ...] = ()


# lowest_judge is the default forecast; the fan models are kept as the research path (none beat it).
MODELS = {
    "lowest_judge": ModelSpec("lowest_judge", "lowest_judge"),
    "null": ModelSpec("null", "fan"),  # theta = 0: every fan ordering equally likely
    "m1": ModelSpec("m1", "fan", ("skill",)),
}
DEFAULT_MODEL = "lowest_judge"


@dataclass
class SeasonParams:
    season: int
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eps: float = 0.0
    save_q: float = 1.0
    rho: float = 0.0
    nll: float = np.nan
    ess_min: float = np.nan  # effective sample size of the sampled-ordering weeks at the fitted theta
    ess_median: float = np.nan


class _Weeks:
    """Per-week data shared by fitting and forecasting."""

    def __init__(self, cw: pd.DataFrame, spec: ModelSpec, cfg, tie: str):
        self.cw, self.cfg, self.tie = cw, cfg, tie
        self.index = cw.groupby(["season", "week"]).indices
        self.columns, self.X = [], np.zeros((len(cw), 0))
        if spec.kind == "fan":
            design = design_matrix(cw, spec.features)
            self.columns, self.X = list(design.columns), design.to_numpy()
        self._prepared = {}

    def era(self, season: int) -> str:
        return era_for_season(season, self.cfg.save_seasons)

    def rows(self, s, w) -> pd.DataFrame:
        return self.cw.iloc[self.index[(s, w)]]

    def eliminated(self, s, w) -> int:
        return int(np.flatnonzero(self.rows(s, w)["eliminated"].to_numpy())[0])

    def likelihood(self, s, w):
        if (s, w) not in self._prepared:
            self._prepared[(s, w)] = prepare_week(
                self.rows(s, w)["score"].to_numpy(), self.eliminated(s, w), self.X[self.index[(s, w)]],
                Rule(self.era(s), self.tie), np.random.default_rng([self.cfg.seed, s, w]),
                self.cfg.exact_enum_max, self.cfg.mc_draws,
            )
        return self._prepared[(s, w)]


def fit_season_params(cw: pd.DataFrame, usable: pd.DataFrame, spec: ModelSpec, cfg, tie: str | None = None):
    """{test season: SeasonParams}, each fit on the usable weeks of earlier seasons (actual judge totals)."""
    tie = tie or cfg.tie_convention
    data = _Weeks(cw, spec, cfg, tie)
    seasons = sorted(usable.loc[usable["season"] >= cfg.first_test_season, "season"].unique())
    out, warm = {}, None
    for season in seasons:
        train = usable[usable["season"] < season]
        keys = list(zip(train["season"], train["week"]))
        if spec.kind == "lowest_judge":
            rho, nll = fit_rho([(data.rows(s, w)["score"].to_numpy(), data.eliminated(s, w)) for s, w in keys])
            out[season] = SeasonParams(season, rho=rho, nll=nll)
            continue
        weeks = [data.likelihood(s, w) for s, w in keys]
        fit = fit_theta(weeks, len(data.columns), cfg.ridge, theta0=warm)
        warm = fit.theta
        ess = [w.ess(fit.theta) for w in weeks if w.sampled]
        out[season] = SeasonParams(
            season, theta=fit.theta, eps=fit.eps, save_q=fit.save_q, nll=fit.nll,
            ess_min=min(ess, default=np.nan), ess_median=float(np.median(ess)) if ess else np.nan,
        )
    return out


def params_table(params: dict, spec: ModelSpec, cw: pd.DataFrame, cfg) -> pd.DataFrame:
    columns = _Weeks(cw, spec, cfg, cfg.tie_convention).columns
    rows = []
    for p in params.values():
        row = {"season": p.season, **dict(zip(columns, p.theta))}
        row.update({k: getattr(p, k) for k in ("eps", "save_q", "rho", "nll", "ess_min", "ess_median")})
        rows.append(row)
    return pd.DataFrame(rows)


def backtest(cw: pd.DataFrame, usable: pd.DataFrame, spec: ModelSpec, judge_input, cfg, params: dict,
             tie: str | None = None) -> pd.DataFrame:
    """One row per couple-week of the usable weeks in the test seasons that `judge_input` can forecast."""
    tie = tie or cfg.tie_convention
    data = _Weeks(cw, spec, cfg, tie)
    weeks = usable[usable["season"].isin(list(params))]
    weeks = weeks[[judge_input.available(s, w, data.rows(s, w)["couple_id"]) for s, w in zip(weeks["season"], weeks["week"])]]
    out = []
    for s, w in zip(weeks["season"], weeks["week"]):
        p, g = params[s], data.rows(s, w)
        totals = judge_input.totals(s, w, g["couple_id"])
        era = data.era(s)
        if spec.kind == "lowest_judge":
            probs = lowest_judge_probs(totals, p.rho)
        else:
            rule = Rule(era, tie, save_q=p.save_q)
            u = utilities(data.X[data.index[(s, w)]], p.theta)
            rng = np.random.default_rng([cfg.seed, s, w, 1])
            probs = elimination_probs(totals, u, rule, rng, cfg.mc_draws, cfg.exact_enum_max, p.eps)
        out.append(
            g[["season", "week", "couple_id", "n_couples", "eliminated"]].assign(
                p_elim=probs, model=spec.name, judge_input=judge_input.name, era=era, eps=p.eps, save_q=p.save_q,
            )
        )
    return pd.concat(out, ignore_index=True)
