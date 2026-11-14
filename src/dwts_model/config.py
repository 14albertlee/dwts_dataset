import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class Config:
    db_path: Path
    schema: str
    train_end: int
    validation: tuple[int, int]
    holdout: tuple[int, int]
    short_seasons: tuple[int, ...]
    features: dict[str, tuple[str, ...]]
    lgbm_grid: dict[str, list]
    main_model: str
    ridge_alphas: tuple[float, ...]
    score_min: float
    score_max: float
    output_dir: Path


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path | str | None = None) -> Config:
    with open(path or DEFAULT_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)
    splits, model = raw["splits"], raw["model"]
    cfg = Config(
        db_path=_resolve(raw["data"]["db_path"]),
        schema=raw["data"]["schema"],
        train_end=splits["train_end"],
        validation=tuple(splits["validation"]),
        holdout=tuple(splits["holdout"]),
        short_seasons=tuple(splits["short_seasons"]),
        features={k: tuple(v) for k, v in model["feature_sets"].items()},
        lgbm_grid=dict(model["lgbm"]),
        main_model=model["main"],
        ridge_alphas=tuple(model["ridge_alphas"]),
        score_min=model["score_min"],
        score_max=model["score_max"],
        output_dir=_resolve(raw["output"]["dir"]),
    )
    if not (cfg.train_end < cfg.validation[0] <= cfg.validation[1] < cfg.holdout[0] <= cfg.holdout[1]):
        raise ValueError("Splits must be ordered: train < validation < holdout with no overlap")
    return cfg
