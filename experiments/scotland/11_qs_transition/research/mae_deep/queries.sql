-- Read-only investigation; run from repository root through run_sql.sh.
-- 26/27 excluded from all extracts; no target-match price used as a feature.
COPY (SELECT table_schema, table_name, column_name, data_type
 FROM db.information_schema.columns
 WHERE (table_schema='sofascore' AND table_name IN ('matches','match_player_lineups','match_statistics','events','teams','players','venues'))
 OR table_schema='bbc'
 ORDER BY table_schema,table_name,ordinal_position)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/schema.csv' (HEADER);

-- Q1: bounded Scottish league history, including two pre-panel seasons.
CREATE TEMP TABLE matches AS
SELECT match_id,tournament_id,season_id,name,start_timestamp,home_team,away_team,
 status_type,home_score,away_score,manager_home,manager_away,venue_name_home,venue_name_away,
 TRY_CAST(json_extract_string(raw_data,'$.venue.capacity') AS INTEGER) AS venue_capacity,
 TRY_CAST(json_extract_string(raw_data,'$.homeTeam.id') AS INTEGER) AS home_id,
 TRY_CAST(json_extract_string(raw_data,'$.awayTeam.id') AS INTEGER) AS away_id
FROM db.sofascore.matches
WHERE tournament_id IN (54,55,56,57)
 AND left(start_timestamp::VARCHAR,10)>='2021-07-01'
 AND left(start_timestamp::VARCHAR,10)<'2026-07-01';
COPY matches TO 'experiments/scotland/11_qs_transition/research/mae_deep/matches.csv' (HEADER);

-- Q2: player-match facts; omit post-match performance except starter flag.
COPY (SELECT l.match_id,l.player_id,l.team_id,l.is_home_team,l.substitute,
 l.proposed_market_value,l.proposed_market_value_currency,
 TRY_CAST(json_extract_string(l.raw_data,'$.player.dateOfBirthTimestamp') AS BIGINT) AS birth_timestamp
 FROM db.sofascore.match_player_lineups l JOIN matches m USING(match_id))
TO 'experiments/scotland/11_qs_transition/research/mae_deep/lineups.csv' (HEADER);

-- Q3: statistics, total match only; names inventoried locally before use.
COPY (SELECT s.match_id,s.period,s.stat_key,s.stat_name,s.home_value,s.away_value
 FROM db.sofascore.match_statistics s JOIN matches m USING(match_id)
 WHERE s.period='ALL')
TO 'experiments/scotland/11_qs_transition/research/mae_deep/statistics.csv' (HEADER);

-- Q4: BBC shots / on-target; no fitted proxy-xG calibration is imported.
COPY (SELECT s.match_id,s.stat_cat,s.stat_type,s.home_value,s.away_value,s.filled
 FROM db.bbc.match_stats s JOIN matches m USING(match_id)
 WHERE s.stat_type IN ('shotsTotal','shotsOnTarget'))
TO 'experiments/scotland/11_qs_transition/research/mae_deep/bbc_shots.csv' (HEADER);

-- Q5-Q6: additional source/provenance and bounded summer-cup coverage queries.
.read experiments/scotland/11_qs_transition/research/mae_deep/audit.sql
.read experiments/scotland/11_qs_transition/research/mae_deep/cups.sql
-- Q7: construct all lagged club-fixture covariates from LOCAL extracts only.
.read experiments/scotland/11_qs_transition/research/mae_deep/features.sql
-- Q8: fixed information-delay and provenance/event checks (LOCAL only).
.read experiments/scotland/11_qs_transition/research/mae_deep/sensitivity.sql
-- Q9: lagged lineup churn, league rest, previous-season final-six form.
.read experiments/scotland/11_qs_transition/research/mae_deep/extra_features.sql
-- Q10 reporting queries are in final_tables.sql, run AFTER analyze.py,
-- robustness.py and outcome_sanity.py produce the diagnostic predictions.
-- reproduce.sh runs Q7-Q10 offline from the saved bounded extract archive.
