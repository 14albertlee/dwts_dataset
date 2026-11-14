import pandas as pd
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
