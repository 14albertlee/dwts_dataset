import pandas as pd


class Baseline:
    name = "baseline"

    def fit(self, train: pd.DataFrame) -> "Baseline":
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class LastWeek(Baseline):
    name = "last_week"

    def predict(self, test):
        return test["prev_score"]


class LastWeekDrift(Baseline):
    """prev_score plus the mean week-over-week change for that week number."""

    name = "last_week_drift"

    def fit(self, train):
        change = train["score"] - train["prev_score"]
        self.drift_by_week_ = change.groupby(train["week"]).mean()
        self.overall_drift_ = change.mean()
        return self

    def predict(self, test):
        drift = test["week"].map(self.drift_by_week_).fillna(self.overall_drift_)
        return test["prev_score"] + drift


class RunningMean(Baseline):
    name = "running_mean"

    def predict(self, test):
        return test["running_mean"]


class FieldAdjusted(Baseline):
    """Couple's prior deviation from the field mean plus the expected field mean."""

    name = "field_adjusted"

    def fit(self, train):
        self.field_mean_by_week_ = train.groupby("week")["score"].mean()
        self.overall_mean_ = train["score"].mean()
        return self

    def predict(self, test):
        expected = test["week"].map(self.field_mean_by_week_).fillna(self.overall_mean_)
        return test["prev_dev"] + expected


def all_baselines() -> list[Baseline]:
    return [LastWeek(), LastWeekDrift(), RunningMean(), FieldAdjusted()]
