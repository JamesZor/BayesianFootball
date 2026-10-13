-- Additional pre-match features: lineup churn, rest, and end-of-season form.
-- All are computed from the bounded local extract, never the target match lineup.
CREATE TEMP TABLE mm AS SELECT *,left(start_timestamp::VARCHAR,19)::TIMESTAMP AS dt,
 year(left(start_timestamp::VARCHAR,10)::DATE-INTERVAL '6 months') AS sy
FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/matches.csv') WHERE status_type='finished';
CREATE TEMP TABLE pm AS
SELECT m.match_id,m.dt,m.sy,CASE WHEN l.is_home_team THEN m.home_team ELSE m.away_team END AS club,
 l.player_id FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/lineups.csv') l JOIN mm m USING(match_id)
 WHERE NOT substitute;
CREATE TEMP TABLE xi AS SELECT match_id,club,dt,sy,list(DISTINCT player_id) AS xi
FROM pm GROUP BY 1,2,3,4 HAVING count(DISTINCT player_id) BETWEEN 10 AND 11;
CREATE TEMP TABLE churn AS SELECT *,
 list_count(list_intersect(xi,lag(xi) OVER w))*1.0/list_count(xi) AS consecutive_retention
 FROM xi WINDOW w AS(PARTITION BY club ORDER BY dt,match_id);
CREATE TEMP TABLE end_history AS SELECT match_id,dt,sy,home_team AS club,home_score-away_score AS gd FROM mm
 UNION ALL SELECT match_id,dt,sy,away_team,away_score-home_score FROM mm;
COPY (SELECT p.match_id,p.club,
 own.consecutive_retention AS own_xi_continuity,opp.consecutive_retention AS opp_xi_continuity,
 lastseason.gd AS end_season_gd6,
 datediff('day',p.previous_date::TIMESTAMP,p.start_timestamp::TIMESTAMP) AS rest_days,
 datediff('day',p.opp_previous_date::TIMESTAMP,p.start_timestamp::TIMESTAMP) AS opp_rest_days
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/panel.csv') p
 LEFT JOIN LATERAL (SELECT consecutive_retention FROM churn c WHERE c.club=p.club
 AND c.dt<p.start_timestamp::TIMESTAMP AND c.dt>p.start_timestamp::TIMESTAMP-INTERVAL '120 days'
 ORDER BY c.dt DESC LIMIT 1) own ON true
 LEFT JOIN LATERAL (SELECT consecutive_retention FROM churn c WHERE c.club=p.opponent
 AND c.dt<p.start_timestamp::TIMESTAMP AND c.dt>p.start_timestamp::TIMESTAMP-INTERVAL '120 days'
 ORDER BY c.dt DESC LIMIT 1) opp ON true
 LEFT JOIN LATERAL (SELECT avg(gd) AS gd FROM (SELECT gd FROM end_history h WHERE h.club=p.club
 AND h.sy=p.sy-1 ORDER BY dt DESC LIMIT 6)) lastseason ON true)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/extra_features.csv' (HEADER);
