import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
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


LMM_EFFECTS = ("couple", "couple_slope", "pro", "season", "night")


class CrossedLMM:
    """Linear mixed model with crossed random effects, fit by REML.

    score = X beta + couple intercept (+ couple week-slope) + pro + season + night + noise.
    Effects: "couple" (couple-season intercept), "couple_slope" (correlated week slope, needs "couple"),
    "pro" (professional_id, persists across seasons), "season", and "night" (season x week).
    The couple's level comes from its random effects, not from own-history features.

    Uses the lme4 parametrization u = Lambda b, b ~ N(0, sigma^2 I), with dense Henderson equations;
    a few thousand rows and ~1-2k random-effect columns keep that cheap.

    Prediction for (season S, week w) conditions on all training rows plus test rows from earlier
    seasons or season S weeks before w, with beta and the variance parameters fixed from the fit.
    `predict_dist` also returns the joint predictive covariance of the couples on each night.
    """

    def __init__(self, fixed, effects=("couple", "season", "night"), name: str = "clmm", slope_scale: float = 5.0):
        unknown = set(effects) - set(LMM_EFFECTS)
        if unknown:
            raise ValueError(f"Unknown effects: {sorted(unknown)}")
        if "couple_slope" in effects and "couple" not in effects:
            raise ValueError("couple_slope needs couple")
        self.fixed = list(fixed)
        self.effects = tuple(e for e in LMM_EFFECTS if e in effects)
        self.name = name
        self.slope_scale = slope_scale

    _COLS = ["season", "week", "couple_id", "professional_id", "score"]

    def _keep_cols(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in self._COLS if c in df.columns]
        return cols + [f for f in self.fixed if f not in cols]

    def _x(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if not self.fixed:
            return np.ones((len(df), 1))
        if fit:
            self.prep_ = make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler())
            self.prep_.fit(df[self.fixed])
        return np.column_stack([np.ones(len(df)), self.prep_.transform(df[self.fixed])])

    def _design(self, df: pd.DataFrame) -> list[sp.csc_matrix]:
        """Indicator blocks per random term, in theta order (couple slope gets a t-weighted block)."""
        n = len(df)
        rows = np.arange(n)

        def block(key: pd.Series, weight=None) -> sp.csc_matrix:
            codes, uniq = pd.factorize(key)
            ok = codes >= 0
            vals = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
            return sp.csc_matrix((vals[ok], (rows[ok], codes[ok])), shape=(n, len(uniq)))

        season = df["season"].astype(str)
        blocks = []
        if "couple" in self.effects:
            couple = season + "_" + df["couple_id"].astype(str)
            blocks.append(block(couple))
            if "couple_slope" in self.effects:
                blocks.append(block(couple, (df["week"].to_numpy() - 1) / self.slope_scale))
        if "pro" in self.effects:
            blocks.append(block(df["professional_id"].astype("Int64")))
        if "season" in self.effects:
            blocks.append(block(season))
        if "night" in self.effects:
            blocks.append(block(season + "_" + df["week"].astype(str)))
        return blocks

    def _n_theta(self) -> int:
        return len(self.effects) + ("couple_slope" in self.effects)

    def _lambda_z(self, theta: np.ndarray, blocks: list[sp.csc_matrix]) -> sp.csc_matrix:
        """A = Z Lambda for relative-scale parameters theta."""
        parts, i, j = [], 0, 0
        if "couple" in self.effects and "couple_slope" in self.effects:
            l11, l21, l22 = theta[:3]
            z_int, z_t = blocks[0], blocks[1]
            parts += [l11 * z_int + l21 * z_t, l22 * z_t]
            i, j = 3, 2
        for k in range(j, len(blocks)):
            parts.append(theta[i] * blocks[k])
            i += 1
        return sp.hstack(parts, format="csc")

    def _bounds(self):
        b = [(0.0, None)] * self._n_theta()
        if "couple_slope" in self.effects:
            b[1] = (None, None)
        return b

    def _deviance(self, theta, X, y, blocks, XtX):
        A = self._lambda_z(theta, blocks)
        q, p, n = A.shape[1], X.shape[1], len(y)
        AtX = (A.T @ X)
        C = np.empty((p + q, p + q))
        C[:p, :p] = XtX
        C[:p, p:] = AtX.T
        C[p:, :p] = AtX
        C[p:, p:] = (A.T @ A).toarray() + np.eye(q)
        rhs = np.concatenate([X.T @ y, A.T @ y])
        cf = cho_factor(C, lower=True)
        sol = cho_solve(cf, rhs)
        beta, b = sol[:p], sol[p:]
        r = y - X @ beta - A @ b
        prss = r @ r + b @ b
        dev = 2 * np.log(np.diag(cf[0])).sum() + (n - p) * np.log(prss / (n - p))
        return dev, beta, prss / (n - p)

    def fit(self, train: pd.DataFrame) -> "CrossedLMM":
        train = train[train["score"].notna()]
        X = self._x(train, fit=True)
        y = train["score"].to_numpy(dtype=float)
        blocks = self._design(train)
        # Tiny jitter keeps X'X invertible when one-hot shares sum to the constant.
        XtX = X.T @ X + 1e-8 * np.eye(X.shape[1])
        x0 = np.full(self._n_theta(), 0.5)
        if "couple_slope" in self.effects:
            x0[1] = 0.0
        res = minimize(lambda t: self._deviance(t, X, y, blocks, XtX)[0], x0, method="L-BFGS-B", bounds=self._bounds())
        self.theta_ = res.x
        self.converged_ = bool(res.success)
        _, self.beta_, self.sigma2_ = self._deviance(res.x, X, y, blocks, XtX)
        self.train_ = train[self._keep_cols(train)].copy()
        return self

    @property
    def variance_components(self) -> dict[str, float]:
        """Standard deviations on the score scale (slope per `slope_scale` weeks), plus intercept-slope correlation."""
        s = np.sqrt(self.sigma2_)
        t = self.theta_
        out, i = {}, 0
        if "couple" in self.effects:
            if "couple_slope" in self.effects:
                l11, l21, l22 = t[:3]
                sd_slope = np.hypot(l21, l22)
                out["couple_sd"] = s * l11
                out["couple_slope_sd"] = s * sd_slope
                out["couple_corr"] = float(l21 / sd_slope) if sd_slope > 0 else 0.0
                i = 3
            else:
                out["couple_sd"] = s * t[0]
                i = 1
        for e in ("pro", "season", "night"):
            if e in self.effects:
                out[f"{e}_sd"] = s * t[i]
                i += 1
        out["resid_sd"] = s
        return {k: float(v) for k, v in out.items()}

    def predict_dist(self, test: pd.DataFrame) -> tuple[pd.Series, dict]:
        """Predictive mean per row and, per (season, week), (row index, covariance) of that night's rows.

        Test-row scores are only used to condition later weeks; NaN scores (unaired weeks) are skipped.
        """
        both = pd.concat([self.train_, test[self._keep_cols(test)]], ignore_index=True)
        n_tr = len(self.train_)
        A = self._lambda_z(self.theta_, self._design(both)).tocsr()
        X = self._x(both, fit=False)
        resid = both["score"].to_numpy(dtype=float) - X @ self.beta_
        q = A.shape[1]
        A_tr, A_te = A[:n_tr], A[n_tr:]
        M0 = (A_tr.T @ A_tr).toarray() + np.eye(q)
        rhs0 = A_tr.T @ resid[:n_tr]
        X_te, r_te = X[n_tr:], resid[n_tr:]
        seasons, weeks = test["season"].to_numpy(), test["week"].to_numpy()
        has_score = ~np.isnan(r_te)

        mean = np.empty(len(test))
        covs = {}
        for s, w in sorted(set(zip(seasons, weeks))):
            obs = has_score & ((seasons < s) | ((seasons == s) & (weeks < w)))
            A_o = A_te[obs]
            cf = cho_factor(M0 + (A_o.T @ A_o).toarray(), lower=True)
            b = cho_solve(cf, rhs0 + A_o.T @ r_te[obs])
            tgt = np.flatnonzero((seasons == s) & (weeks == w))
            A_t = A_te[tgt].toarray()
            mean[tgt] = X_te[tgt] @ self.beta_ + A_t @ b
            covs[(s, w)] = (test.index[tgt], self.sigma2_ * (A_t @ cho_solve(cf, A_t.T) + np.eye(len(tgt))))
        return pd.Series(mean, index=test.index), covs

    def predict(self, test: pd.DataFrame) -> pd.Series:
        return self.predict_dist(test)[0]


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
