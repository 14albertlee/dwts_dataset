import duckdb
import pandas as pd

from .config import Config

# Per-judge mean per performance first, then average performances, so panel
# size and multi-dance nights do not distort the target.
_QUERY = """
WITH perf AS (
    SELECT p.performance_id, p.season, p.week, p.couple_id, p.eliminated,
           AVG(s.judge_score) AS perf_score
    FROM {schema}.performances p
    JOIN {schema}.scores s USING (performance_id)
    WHERE s.judge_score IS NOT NULL AND s.judge_score > 0
      AND p.week IS NOT NULL  -- season 1 finale has no week number
    GROUP BY ALL
)
SELECT season, week, couple_id,
       AVG(perf_score) AS score,
       COUNT(*) AS n_performances,
       BOOL_OR(eliminated) AS eliminated
FROM perf
GROUP BY season, week, couple_id
ORDER BY season, week, couple_id
"""


def load_weekly_scores(cfg: Config) -> pd.DataFrame:
    """One row per (season, week, couple) with the mean per-judge score."""
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        return con.execute(_QUERY.format(schema=cfg.schema)).df()
    finally:
        con.close()


_PERFORMANCE_QUERY = """
SELECT p.season, p.week, p.couple_id, p.performance_id,
       p.round, p.running_order, p.round_size,
       t.week_theme AS theme, d.primary_dance_style AS style
FROM {schema}.performances p
LEFT JOIN {schema}.week_themes t USING (week_theme_id)
LEFT JOIN {schema}.primary_dance_styles d USING (primary_dance_style_id)
WHERE p.week IS NOT NULL
"""

_COUPLES_QUERY = """
SELECT DISTINCT p.season, c.couple_id, ce.celebrity, pr.professional
FROM {schema}.performances p
JOIN {schema}.couples c USING (couple_id)
JOIN {schema}.celebrities ce USING (celebrity_id)
JOIN {schema}.professionals pr USING (professional_id)
"""


def load_season_couples(cfg: Config) -> pd.DataFrame:
    """Couple ids with celebrity and professional names, one row per (season, couple)."""
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        return con.execute(_COUPLES_QUERY.format(schema=cfg.schema)).df()
    finally:
        con.close()


_COUPLE_INFO_QUERY = """
SELECT DISTINCT p.season, c.couple_id, c.professional_id, ce.notability
FROM {schema}.performances p
JOIN {schema}.couples c USING (couple_id)
LEFT JOIN {schema}.celebrities ce USING (celebrity_id)
WHERE p.week IS NOT NULL AND c.professional_id IS NOT NULL
"""


def load_couple_info(cfg: Config) -> pd.DataFrame:
    """One row per (season, couple_id) with the couple's professional_id and celebrity notability."""
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        return con.execute(_COUPLE_INFO_QUERY.format(schema=cfg.schema)).df()
    finally:
        con.close()


_JUDGE_SCORES_QUERY = """
SELECT p.season, p.week, p.couple_id, p.performance_id, s.judge_id, s.judge_score
FROM {schema}.performances p
JOIN {schema}.scores s USING (performance_id)
WHERE s.judge_score IS NOT NULL AND s.judge_score > 0 AND p.week IS NOT NULL
"""


def load_judge_scores(cfg: Config) -> pd.DataFrame:
    """One row per judge score, with the same filters as `load_weekly_scores`."""
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        return con.execute(_JUDGE_SCORES_QUERY.format(schema=cfg.schema)).df()
    finally:
        con.close()


def load_performance_context(cfg: Config) -> pd.DataFrame:
    """One row per performance with its week theme text and dance style name."""
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        return con.execute(_PERFORMANCE_QUERY.format(schema=cfg.schema)).df()
    finally:
        con.close()
