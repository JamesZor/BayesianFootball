-- All are LOCAL queries. Fixed information delays distinguish freshness from
-- signals already observable a week or more before target kick-off.
CREATE TEMP TABLE hist AS
SELECT match_id,CAST(left(start_timestamp::VARCHAR,19) AS TIMESTAMP) AS dt,
 home_team AS club,away_team AS opponent,home_score-away_score AS gd
FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/matches.csv') WHERE status_type='finished'
UNION ALL
SELECT match_id,CAST(left(start_timestamp::VARCHAR,19) AS TIMESTAMP),away_team,home_team,away_score-home_score
FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/matches.csv') WHERE status_type='finished';
CREATE TEMP TABLE shots AS SELECT match_id,
 max(home_value-away_value) FILTER(WHERE stat_type='shotsTotal') AS shotdiff,
 max(home_value-away_value) FILTER(WHERE stat_type='shotsOnTarget') AS sotdiff
FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/bbc_shots.csv')
WHERE stat_cat='basic' AND NOT filled GROUP BY 1;
CREATE TEMP TABLE h AS SELECT hist.*,
 CASE WHEN hist.club=m.home_team THEN s.shotdiff ELSE -s.shotdiff END AS shotdiff,
 CASE WHEN hist.club=m.home_team THEN s.sotdiff ELSE -s.sotdiff END AS sotdiff
FROM hist JOIN read_csv('experiments/scotland/11_qs_transition/research/mae_deep/matches.csv') m USING(match_id)
LEFT JOIN shots s USING(match_id);
CREATE TEMP TABLE delayed AS
SELECT a.match_id,a.club,delay.days,
 own.gd AS own_gd,opp.gd AS opp_gd,own.shotdiff AS own_shots,opp.shotdiff AS opp_shots,
 own.sotdiff AS own_sot,opp.sotdiff AS opp_sot,own.latest AS own_latest,opp.latest AS opp_latest
FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/panel.csv') a
CROSS JOIN (VALUES(7),(14),(28)) delay(days)
LEFT JOIN LATERAL (SELECT avg(gd) AS gd,avg(shotdiff) AS shotdiff,avg(sotdiff) AS sotdiff,max(dt) AS latest
 FROM (SELECT * FROM h WHERE h.club=a.club
 AND h.dt<CAST(left(a.start_timestamp::VARCHAR,19) AS TIMESTAMP)-delay.days*INTERVAL '1 day'
 ORDER BY h.dt DESC LIMIT 6)) own ON true
LEFT JOIN LATERAL (SELECT avg(gd) AS gd,avg(shotdiff) AS shotdiff,avg(sotdiff) AS sotdiff,max(dt) AS latest
 FROM (SELECT * FROM h WHERE h.club=a.opponent
 AND h.dt<CAST(left(a.start_timestamp::VARCHAR,19) AS TIMESTAMP)-delay.days*INTERVAL '1 day'
 ORDER BY h.dt DESC LIMIT 6)) opp ON true;
COPY delayed TO 'experiments/scotland/11_qs_transition/research/mae_deep/delayed_features.csv' (HEADER);

-- Point-in-time provenance diagnostics. These are not predictive features.
COPY (WITH p AS (SELECT player_id,count(DISTINCT proposed_market_value) AS distinct_positive_values,
 count(*) AS valued_appearances FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/lineups.csv')
 WHERE proposed_market_value>0 GROUP BY 1)
 SELECT distinct_positive_values,count(*) AS players,sum(valued_appearances) AS appearances FROM p GROUP BY 1)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/value_vintage_audit.csv' (HEADER);
COPY (SELECT a.club,count(DISTINCT a.manager) AS bbc_distinct_managers,
 count(DISTINCT s.sofa_manager) AS sofa_distinct_managers,
 string_agg(DISTINCT a.manager,'|') AS bbc_managers,string_agg(DISTINCT s.sofa_manager,'|') AS sofa_managers,
 count(*) FILTER(WHERE a.manager_change3=1 AND a.game<=20) AS first20_changed_rows
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/panel.csv') a
 LEFT JOIN (SELECT match_id,home_team AS club,manager_home AS sofa_manager
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/matches.csv')
 UNION ALL SELECT match_id,away_team,manager_away FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/matches.csv')) s
 USING(match_id,club) GROUP BY 1 ORDER BY 1)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/manager_vintage_audit.csv' (HEADER);
-- Historical resource shocks, externally verified announcement dates.
COPY (SELECT club,sy,CASE WHEN start_timestamp::DATE<'2024-10-24' THEN 'before' ELSE 'after' END AS event_period,
 count(*) AS n,avg(residual) AS bias,avg(abs(residual)) AS mae
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/panel.csv')
 WHERE club='inverness-caledonian-thistle' AND sy=2024 AND game<=20 GROUP BY ALL)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/inverness_resource_event.csv' (HEADER);
-- Dates and event-time BBC managers, to establish that archived manager strings
-- distinguish the actual change rather than silently following current metadata.
COPY (SELECT p.match_id,p.start_timestamp,p.club,p.manager,p.previous_manager,p.manager_change3,p.game,p.residual
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/panel.csv') p
 WHERE p.club='inverness-caledonian-thistle' AND p.sy=2024 AND p.game<=20 ORDER BY start_timestamp)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/inverness_manager_timeline.csv' (HEADER);
