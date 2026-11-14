-- 1. Base table: judge scores joined to performances and judges
CREATE OR REPLACE TEMP TABLE judge_scores AS
SELECT
    performances.performance_id
    , performances.season
    , performances.week
    , judges.judge
    , scores.judge_score
FROM "dwts_db_compat"."main"."performances" performances
LEFT JOIN "dwts_db_compat"."main"."scores" scores
    ON performances.performance_id = scores.performance_id
LEFT JOIN "dwts_db_compat"."main"."judges" judges
    ON scores.judge_id = judges.judge_id
;

-- Average Judge Scores by Week
SELECT week, judge, AVG(judge_score) AS avg_judge_score, count(*) AS num_performances
FROM judge_scores
GROUP BY week, judge
ORDER BY week, judge
;

-- 2. Relative score to the average of the other judges in the same performance
CREATE OR REPLACE TEMP TABLE judge_scores_relative_stage1 AS
SELECT
    performances.performance_id
    , performances.season
    , performances.week
    , judges.judge
    , scores.judge_score

    , AVG(scores.judge_score) OVER (
        PARTITION BY performances.performance_id
        ORDER BY judges.judge_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        EXCLUDE CURRENT ROW
    ) AS other_judges_avg

    , scores.judge_score
        - AVG(scores.judge_score) OVER (
            PARTITION BY performances.performance_id
            ORDER BY judges.judge_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            EXCLUDE CURRENT ROW
        ) AS relative_score

FROM "dwts_db_compat"."main"."performances" performances
LEFT JOIN "dwts_db_compat"."main"."scores" scores
    ON performances.performance_id = scores.performance_id
LEFT JOIN "dwts_db_compat"."main"."judges" judges
    ON scores.judge_id = judges.judge_id
;

-- 3. Add the above/below/equal category
CREATE OR REPLACE TEMP TABLE judge_scores_relative AS
SELECT
    *
    , CASE
        WHEN relative_score < 0 THEN 'below'
        WHEN relative_score > 0 THEN 'above'
        ELSE 'equal'
    END AS relative_score_category
FROM judge_scores_relative_stage1
;

SELECT * FROM judge_scores_relative;

-- 4. Summaries by season-week-judge
CREATE OR REPLACE TEMP TABLE judge_scores_summary_by_season AS
SELECT
    season
    , week
    , judge

    , COUNT(*) AS total_performances

    , COUNT(*) FILTER (WHERE relative_score_category = 'above')
        / COUNT(*)::DOUBLE AS pct_above

    , COUNT(*) FILTER (WHERE relative_score_category = 'below')
        / COUNT(*)::DOUBLE AS pct_below

    , COUNT(*) FILTER (WHERE relative_score_category = 'equal')
        / COUNT(*)::DOUBLE AS pct_equal

    , AVG(relative_score) AS avg_relative_score

FROM judge_scores_relative
GROUP BY
    season
    , week
    , judge
ORDER BY
    season
    , week
    , judge
;

SELECT *
FROM judge_scores_summary_by_season;

-- 5. Summaries by week-judge (across seasons)
CREATE OR REPLACE TEMP TABLE judge_scores_summary_by_week AS
SELECT
    week
    , judge

    , COUNT(*) AS total_performances

    , COUNT(*) FILTER (WHERE relative_score_category = 'above')
        / COUNT(*)::DOUBLE AS pct_above

    , COUNT(*) FILTER (WHERE relative_score_category = 'below')
        / COUNT(*)::DOUBLE AS pct_below

    , COUNT(*) FILTER (WHERE relative_score_category = 'equal')
        / COUNT(*)::DOUBLE AS pct_equal

    , AVG(relative_score) AS avg_relative_score

FROM judge_scores_relative
GROUP BY
    week
    , judge
ORDER BY
    week
    , judge
;

SELECT *
FROM judge_scores_summary_by_week;

-- 6. Score spread by week and judge, relative to the other judges
-- i.e. is the judge willing to differentiate contestants in a given week, or giving everyone the same score?
CREATE OR REPLACE TEMP TABLE judge_score_spread AS
WITH judge_episode_spread AS (
    -- Spread within each season-week-judge
    SELECT
        season
        , week
        , judge

        , COUNT(*) AS performances_scored
        , AVG(judge_score) AS avg_score
        , STDDEV_POP(judge_score) AS score_stddev
        , MAX(judge_score) - MIN(judge_score) AS score_range

    FROM judge_scores_relative
    WHERE season >= 29
    GROUP BY
        season
        , week
        , judge
),

judge_week_summary AS (
    -- Aggregate episode-level spread across seasons
    SELECT
        week
        , judge

        , COUNT(*) AS episodes_scored
        , SUM(performances_scored) AS total_performances

        , AVG(avg_score) AS avg_score
        , AVG(score_stddev) AS avg_score_stddev
        , AVG(score_range) AS avg_score_range

    FROM judge_episode_spread
    GROUP BY
        week
        , judge
)

SELECT
    week
    , judge
    , episodes_scored
    , total_performances
    , avg_score
    , avg_score_stddev
    , avg_score_range

    -- Average spread of the other judges in the same week
    , AVG(avg_score_stddev) OVER (
        PARTITION BY week
        ORDER BY judge
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        EXCLUDE CURRENT ROW
    ) AS other_judges_avg_stddev

    -- This judge's spread relative to the other judges
    , avg_score_stddev
        - AVG(avg_score_stddev) OVER (
            PARTITION BY week
            ORDER BY judge
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            EXCLUDE CURRENT ROW
        ) AS relative_score_spread

FROM judge_week_summary
;

SELECT *
FROM judge_score_spread
ORDER BY week, judge;

-- Still to do:
-- Dance style scoring: normalizing by week, which styles earn the best and worst scores?
-- Celeb type scoring: do certain types of celebrities tend to get better or worse scores?
-- Season to season shifts: has a judge changed scoring behavior over the course of the show?
-- Season to season shifts: have the judges collectively become more generous or more critical?
-- Guest judge stats: how do guest judges compare to the regular judges?