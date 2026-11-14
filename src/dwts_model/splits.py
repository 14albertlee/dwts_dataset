from collections.abc import Iterator

import pandas as pd

from .config import Config


def test_seasons(cfg: Config, mode: str) -> list[int]:
    if mode == "validate":
        lo, hi = cfg.validation
    elif mode == "holdout":
        lo, hi = cfg.holdout
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return list(range(lo, hi + 1))


def rolling_origin(
    df: pd.DataFrame, cfg: Config, mode: str
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (season, train, test): train on seasons < S, test on season S."""
    for season in test_seasons(cfg, mode):
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if not test.empty:
            yield season, train, test
