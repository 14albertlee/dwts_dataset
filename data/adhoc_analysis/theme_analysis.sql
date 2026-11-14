WITH theme_mapping AS (
    SELECT *
    FROM (VALUES
        (1, 'Music'),
        (2, 'Competition Stage'),
        (3, 'Competition Stage'),
        (4, 'Dance / Competition'),
        (5, 'Music'),
        (6, 'Music'),
        (7, 'Dance / Competition'),
        (8, 'Dance / Competition'),
        (9, 'Competition Stage'),
        (10, 'Competition Stage'),
        (11, 'General / Other'),
        (12, 'Competition Stage'),
        (13, 'Competition Stage'),
        (14, 'Music'),
        (15, 'Music'),
        (16, 'Music'),
        (17, 'Music'),
        (18, 'Personal / Emotional'),
        (19, 'Movies / TV'),
        (20, 'Disney'),
        (21, 'Music'),
        (22, 'Music'),
        (23, 'Music'),
        (24, 'Movies / TV'),
        (25, 'Movies / TV'),
        (26, 'Music'),
        (27, 'Milestone / Special'),
        (28, 'Movies / TV'),
        (29, 'Milestone / Special'),
        (30, 'Movies / TV'),
        (31, 'Milestone / Special'),
        (32, 'Music'),
        (33, 'Music'),
        (34, 'Disney'),
        (35, 'Personal / Emotional'),
        (36, 'Movies / TV'),
        (37, 'Music'),
        (38, 'Music'),
        (39, 'Music'),
        (40, 'Competition Stage'),
        (41, 'Music'),
        (42, 'Movies / TV'),
        (43, 'Disney'),
        (44, 'Personal / Emotional'),
        (45, 'Music'),
        (46, 'Competition Stage'),
        (47, 'Music'),
        (48, 'Disney'),
        (49, 'Movies / TV'),
        (50, 'Movies / TV'),
        (51, 'Music'),
        (52, 'Competition Stage'),
        (53, 'Music'),
        (54, 'Competition Stage'),
        (55, 'Movies / TV'),
        (56, 'Competition Stage'),
        (57, 'Movies / TV'),
        (58, 'Competition Stage'),
        (59, 'Dance / Competition'),
        (60, 'Music'),
        (61, 'Competition Stage'),
        (62, 'General / Other'),
        (63, 'Personal / Emotional'),
        (64, 'Dance / Competition'),
        (65, 'Music'),
        (66, 'Dance / Competition'),
        (67, 'Competition Stage'),
        (68, 'Dance / Competition'),
        (69, 'Music'),
        (70, 'Movies / TV'),
        (71, 'Competition Stage'),
        (72, 'Competition Stage'),
        (73, 'Music'),
        (74, 'Music'),
        (75, 'Movies / TV'),
        (76, 'Dance / Competition'),
        (77, 'Movies / TV'),
        (78, 'Music'),
        (79, 'Dance / Competition'),
        (80, 'Dance / Competition'),
        (81, 'Dance / Competition'),
        (82, 'Music'),
        (83, 'Dance / Competition'),
        (84, 'Music'),
        (85, 'Music'),
        (86, 'Music'),
        (87, 'Music'),
        (88, 'Music'),
        (89, 'Music'),
        (90, 'Movies / TV'),
        (91, 'Dance / Competition'),
        (92, 'Music'),
        (93, 'Personal / Emotional'),
        (94, 'Movies / TV'),
        (95, 'Music'),
        (96, 'Movies / TV'),
        (97, 'Movies / TV'),
        (98, 'Dance / Competition'),
        (99, 'Personal / Emotional'),
        (100, 'Music'),
        (101, 'Music'),
        (102, 'Music'),
        (103, 'Dance / Competition'),
        (104, 'Personal / Emotional'),
        (105, 'Movies / TV'),
        (106, 'Music'),
        (107, 'Milestone / Special'),
        (108, 'Music'),
        (109, 'Personal / Emotional'),
        (110, 'Dance / Competition'),
        (111, 'Dance / Competition'),
        (112, 'Competition Stage'),
        (113, 'Music'),
        (114, 'Personal / Emotional'),
        (115, 'Dance / Competition'),
        (116, 'Music'),
        (117, 'Personal / Emotional'),
        (118, 'Music'),
        (119, 'Music'),
        (120, 'Personal / Emotional'),
        (121, 'Music'),
        (122, 'Dance / Competition'),
        (123, 'Music'),
        (124, 'Music'),
        (125, 'Music'),
        (126, 'Dance / Competition')
    ) AS t(week_theme_id, theme_group)
)

, week_averages AS (
    SELECT
        week
        , judge
        , AVG(judge_score) AS avg_week_score

    FROM judge_scores

    GROUP BY
        week
        , judge
),

theme_averages AS (
    SELECT
        js.season
        , js.week
        , js.judge
        , tm.theme_group

        , COUNT(DISTINCT js.performance_id) AS performances
        , AVG(js.judge_score) AS avg_theme_score

    FROM judge_scores js

    JOIN "dwts_db_compat"."main"."performances" p
        ON js.performance_id = p.performance_id

    JOIN theme_mapping tm
        ON p.week_theme_id = tm.week_theme_id

    GROUP BY
        js.season
        , js.week
        , js.judge
        , tm.theme_group
),

theme_relative_scores AS (
    SELECT
        ta.season
        , ta.week
        , ta.judge
        , ta.theme_group
        , ta.performances
        , ta.avg_theme_score
        , wa.avg_week_score

        , ta.avg_theme_score - wa.avg_week_score
            AS relative_score

    FROM theme_averages ta

    JOIN week_averages wa
        ON ta.week = wa.week
        AND ta.judge = wa.judge
)

SELECT
    judge
    , theme_group

    , COUNT(*) AS episode_observations
    , COUNT(DISTINCT season) AS seasons
    , SUM(performances) AS total_performances

    , ROUND(AVG(avg_theme_score), 3) AS avg_theme_score
    , ROUND(AVG(avg_week_score), 3) AS avg_week_score

    , ROUND(AVG(relative_score), 3) AS avg_relative_score

FROM theme_relative_scores

GROUP BY
    judge
    , theme_group

ORDER BY
    judge
    , avg_relative_score DESC;