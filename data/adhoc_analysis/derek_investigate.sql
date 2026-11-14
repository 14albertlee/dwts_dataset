CREATE OR REPLACE TEMP TABLE derek_range_diagnostics AS
WITH judge_episode_range AS (
    SELECT
        season
        , week
        , judge
        , COUNT(*) AS performances_scored
        , MAX(judge_score) - MIN(judge_score) AS point_range

    FROM judge_scores_relative
    WHERE week IN (4, 5, 6)
    GROUP BY season, week, judge
),

episode_comparison AS (
    SELECT
        season
        , week
        , judge
        , performances_scored
        , point_range

        , AVG(point_range) FILTER (
            WHERE judge <> 'Derek Hough'
        ) OVER (
            PARTITION BY season, week
        ) AS colleagues_avg_range

    FROM judge_episode_range
)

SELECT
    season
    , week
    , performances_scored
    , point_range AS derek_point_range
    , colleagues_avg_range
    , point_range - colleagues_avg_range AS relative_point_range

FROM episode_comparison
WHERE judge = 'Derek Hough'
;

SELECT
    week
    , COUNT(*) AS seasons

    , AVG(derek_point_range) AS avg_derek_point_range
    , AVG(colleagues_avg_range) AS avg_colleagues_point_range

    , AVG(relative_point_range) AS avg_relative_point_range

    , COUNT(*) FILTER (
        WHERE relative_point_range > 0
    ) AS seasons_above_colleagues

    , COUNT(*) FILTER (
        WHERE relative_point_range < 0
    ) AS seasons_below_colleagues

    , COUNT(*) FILTER (
        WHERE relative_point_range = 0
    ) AS seasons_equal

FROM derek_range_diagnostics
GROUP BY week
ORDER BY week;

SELECT *
FROM derek_range_diagnostics
ORDER BY week, season;

SELECT DISTINCT
    p.season
    , p.week
    , wt.week_theme

FROM "dwts_db_compat"."main"."performances" p
LEFT JOIN "dwts_db_compat"."main"."week_themes" wt
    ON p.week_theme_id = wt.week_theme_id

WHERE p.season >= 29
    AND p.week IN (4, 5)

ORDER BY
    p.season
    , p.week
;

CREATE OR REPLACE TEMP TABLE judge_theme_spread AS

WITH episode_judge_stats AS (
    SELECT
        p.season
        , p.week
        , wt.week_theme
        , j.judge

        , COUNT(*) AS performances_scored
        , AVG(s.judge_score) AS avg_score
        , MAX(s.judge_score) - MIN(s.judge_score) AS point_range
        , STDDEV_POP(s.judge_score) AS score_stddev

    FROM "dwts_db_compat"."main"."performances" p

    LEFT JOIN "dwts_db_compat"."main"."scores" s
        ON p.performance_id = s.performance_id

    LEFT JOIN "dwts_db_compat"."main"."judges" j
        ON s.judge_id = j.judge_id

    LEFT JOIN "dwts_db_compat"."main"."week_themes" wt
        ON p.week_theme_id = wt.week_theme_id

    WHERE j.judge IN (
        'Derek Hough'
        , 'Carrie Ann Inaba'
        , 'Bruno Tonioli'
    )
    AND p.season >= 29

    GROUP BY
        p.season
        , p.week
        , wt.week_theme
        , j.judge
),

classified AS (
    SELECT
        *
        , CASE
            WHEN LOWER(week_theme) LIKE '%disney%'
                THEN 'Disney'

            WHEN REGEXP_MATCHES(
                LOWER(week_theme),
                'story|stories|memory|memorable|dedication|life'
            )
                THEN 'Dedication / Story / Memory / Life'
        END AS theme_category

    FROM episode_judge_stats
)

SELECT *
FROM classified
WHERE theme_category IS NOT NULL
;

SELECT
    judge
    , theme_category

    , COUNT(*) AS episodes
    , SUM(performances_scored) AS total_performances

    -- Average episode-level scoring
    , AVG(avg_score) AS avg_judge_score

    -- Average episode-level point spread
    , AVG(point_range) AS avg_point_range

    -- Average episode-level standard deviation
    , AVG(score_stddev) AS avg_score_stddev

FROM judge_theme_spread

GROUP BY
    judge
    , theme_category

ORDER BY
    theme_category
    , judge
;

SELECT
    season
    , week
    , derek_point_range
    , colleagues_avg_range
    , relative_point_range
FROM derek_range_diagnostics
ORDER BY season, week;