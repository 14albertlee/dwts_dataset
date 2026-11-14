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
       t.week_theme AS theme, d.primary_dance_style AS style
FROM {schema}.performances p
LEFT JOIN {schema}.week_themes t USING (week_theme_id)
LEFT JOIN {schema}.primary_dance_styles d USING (primary_dance_style_id)
WHERE p.week IS NOT NULL
"""


def load_performance_context(cfg: Config) -> pd.DataFrame:
    """One row per performance with its week theme text and dance style name."""
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        return con.execute(_PERFORMANCE_QUERY.format(schema=cfg.schema)).df()
    finally:
        con.close()
