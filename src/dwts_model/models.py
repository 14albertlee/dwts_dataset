import itertools

import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .splits import rolling_origin


class RidgeModel:
    def __init__(self, features, alpha: float = 1.0, name: str = "ridge"):
        self.features = list(features)
        self.alpha = alpha
        self.name = name

    def fit(self, train: pd.DataFrame) -> "RidgeModel":
        # Imputer handles style_difficulty for dances outside the ranked list.
        self.pipe_ = make_pipeline(
            SimpleImputer(strategy="mean", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=self.alpha),
        )
        self.pipe_.fit(train[self.features], train["score"])
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        return pd.Series(self.pipe_.predict(test[self.features]), index=test.index)


def tune_ridge_alpha(
    df: pd.DataFrame, cfg: Config, features
) -> tuple[float, dict[float, float]]:
    """Pick alpha by pooled MAE over the validation rolling-origin folds only."""
    maes = {}
    for alpha in cfg.ridge_alphas:
        errors = []
        for _, train, test in rolling_origin(df, cfg, "validate"):
            pred = RidgeModel(features, alpha).fit(train).predict(test)
            pred = pred.clip(cfg.score_min, cfg.score_max)
            errors.append((pred - test["score"]).abs())
        maes[alpha] = float(pd.concat(errors).mean())
    return min(maes, key=maes.get), maes


class LGBMModel:
    """Gradient boosting. target="delta" models the change from prev_score and adds it back,
    which spares the trees from re-learning the strong identity link to last week."""

    def __init__(self, features, params: dict, name: str = "lgbm"):
        self.features = list(features)
        self.params = dict(params)
        self.name = name

    def fit(self, train: pd.DataFrame) -> "LGBMModel":
        params = dict(self.params)
        self.target_ = params.pop("target", "score")
        y = train["score"] - train["prev_score"] if self.target_ == "delta" else train["score"]
        self.model_ = LGBMRegressor(
            objective="regression_l1" if params.pop("l1", False) else "regression",
            subsample_freq=1, random_state=0, n_jobs=1, verbose=-1, **params,
        )
        self.model_.fit(train[self.features], y)
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        pred = pd.Series(self.model_.predict(test[self.features]), index=test.index)
        return pred + test["prev_score"] if self.target_ == "delta" else pred


class LMMModel:
    """Linear mixed model: fixed effects on the features plus a random intercept per couple-season.

    The random intercept is a latent skill. At prediction time it is the BLUP from the couple's
    own earlier weeks in the target season (fixed-effect residuals, shrunk toward 0), so early
    weeks with little history lean on the fixed effects and later weeks lean on the couple.
    """

    def __init__(self, features, name: str = "lmm"):
        self.features = list(features)
        self.name = name

    @staticmethod
    def _group(df: pd.DataFrame) -> pd.Series:
        return df["season"].astype(str) + "_" + df["couple_id"].astype(str)

    def _exog(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        x = df[self.features]
        if fit:
            self.prep_ = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
            self.prep_.fit(x)
        out = pd.DataFrame(self.prep_.transform(x), index=df.index, columns=self.features)
        out.insert(0, "const", 1.0)
        return out

    def fit(self, train: pd.DataFrame) -> "LMMModel":
        exog = self._exog(train, fit=True)
        result = sm.MixedLM(train["score"], exog, groups=self._group(train)).fit(reml=True)
        self.fe_ = result.fe_params
        self.var_u_ = float(result.cov_re.iloc[0, 0])
        self.var_e_ = float(result.scale)
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        fixed = self._exog(test, fit=False) @ self.fe_
        t = pd.DataFrame(
            {"g": self._group(test), "week": test["week"], "r": test["score"] - fixed}, index=test.index
        ).sort_values(["g", "week"])
        # Sum and count of strictly earlier weeks' residuals for the same couple.
        prior_sum = t.groupby("g")["r"].cumsum() - t["r"]
        n_prior = t.groupby("g").cumcount()
        u = self.var_u_ * prior_sum / (n_prior * self.var_u_ + self.var_e_)
        return fixed + u.reindex(test.index)


def lgbm_param_grid(cfg: Config) -> list[dict]:
    keys = list(cfg.lgbm_grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*cfg.lgbm_grid.values())]


def tune_lgbm(df: pd.DataFrame, cfg: Config, features) -> tuple[dict, list[tuple[dict, float]]]:
    """Pick LightGBM params by pooled MAE over the validation rolling-origin folds only."""
    results = []
    for params in lgbm_param_grid(cfg):
        errors = []
        for _, train, test in rolling_origin(df, cfg, "validate"):
            pred = LGBMModel(features, params).fit(train).predict(test)
            errors.append((pred.clip(cfg.score_min, cfg.score_max) - test["score"]).abs())
        results.append((params, float(pd.concat(errors).mean())))
    best = min(results, key=lambda r: r[1])[0]
    return best, results
