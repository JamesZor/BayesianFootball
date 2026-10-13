-- Bounded summer-cup identity extraction: outcomes deliberately not selected.
CREATE TEMP TABLE cup_matches AS
SELECT match_id,tournament_id,TIMESTAMP '1970-01-01' + start_timestamp*INTERVAL '1 second' AS dt,home_team,away_team
FROM db.sofascore.events
WHERE tournament_id IN (73,982,1520) AND status_type='finished'
AND start_timestamp>=epoch(DATE '2023-07-01')
AND start_timestamp<epoch(DATE '2025-08-01')
AND month(TIMESTAMP '1970-01-01' + start_timestamp*INTERVAL '1 second')=7
;
-- July cup schedule is already a targeted bounded extract; event team labels
-- are audited before joining to slug-based league fixtures.
COPY cup_matches TO 'experiments/scotland/11_qs_transition/research/mae_deep/cup_matches.csv' (HEADER);
COPY (SELECT l.match_id,l.player_id,l.is_home_team,l.substitute
 FROM db.sofascore.match_player_lineups l JOIN cup_matches m USING(match_id))
TO 'experiments/scotland/11_qs_transition/research/mae_deep/cup_lineups.csv' (HEADER);
