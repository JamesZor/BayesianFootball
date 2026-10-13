-- Additional bounded read-only provenance audit. Run with run_sql.sh.
COPY (SELECT table_schema,table_name,column_name,data_type FROM db.information_schema.columns
 WHERE table_schema='sofascore' AND table_name IN ('tournaments','seasons') ORDER BY 2,ordinal_position)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/lookup_schema.csv' (HEADER);
-- BBC contemporaneous match-page manager candidates, no current-season data.
COPY (SELECT b.match_id,b.home_manager,b.away_manager,b.last_updated,b.has_lineup,b.attendance,b.venue
 FROM db.bbc.match_meta b JOIN db.sofascore.matches m USING(match_id)
 WHERE m.tournament_id IN (54,55,56,57)
 AND left(m.start_timestamp::VARCHAR,10)>='2021-07-01'
 AND left(m.start_timestamp::VARCHAR,10)<'2026-07-01')
TO 'experiments/scotland/11_qs_transition/research/mae_deep/bbc_meta.csv' (HEADER);
COPY (SELECT tournament_id,name,country FROM db.sofascore.tournaments WHERE country='Scotland')
TO 'experiments/scotland/11_qs_transition/research/mae_deep/scottish_tournaments.csv' (HEADER);
