-- Local-only SQL. Called after extraction. Historical match date eligibility is
-- independently checked by analyze.py. Season: July--June, no 26/27 data.
CREATE TEMP TABLE m AS SELECT * EXCLUDE(start_timestamp),
 CAST(left(start_timestamp::VARCHAR,19) AS TIMESTAMP) AS start_timestamp,
 year(CAST(left(start_timestamp::VARCHAR,10) AS DATE)-INTERVAL '6 months') AS sy,
 tournament_id-53 AS tier
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/matches.csv')
 WHERE status_type='finished' AND home_score IS NOT NULL AND away_score IS NOT NULL;
CREATE TEMP TABLE bb AS
 SELECT match_id,
 max(home_value) FILTER(WHERE stat_type='shotsTotal' AND stat_cat='basic' AND NOT filled) AS shots_h,
 max(away_value) FILTER(WHERE stat_type='shotsTotal' AND stat_cat='basic' AND NOT filled) AS shots_a,
 max(home_value) FILTER(WHERE stat_type='shotsOnTarget' AND stat_cat='basic' AND NOT filled) AS sot_h,
 max(away_value) FILTER(WHERE stat_type='shotsOnTarget' AND stat_cat='basic' AND NOT filled) AS sot_a
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/bbc_shots.csv') GROUP BY 1;
CREATE TEMP TABLE ss AS SELECT match_id,
 max(home_value) FILTER(WHERE stat_key='expectedGoals') AS xg_h,
 max(away_value) FILTER(WHERE stat_key='expectedGoals') AS xg_a
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/statistics.csv') GROUP BY 1;
-- BBC archival match-page managers vary through time; SofaScore manager fields
-- are club-level backfills (see provenance_audit.csv), so never used as features.
CREATE TEMP TABLE bm AS SELECT * FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/bbc_meta.csv');
CREATE TEMP TABLE tm AS
 SELECT m.match_id,m.sy,m.tier,m.start_timestamp,m.home_team AS club,m.away_team AS opponent,
 1 AS is_home,m.home_score AS gf,m.away_score AS ga,nullif(trim(bm.home_manager),'') AS manager,
 m.venue_name_home AS venue,bb.shots_h AS shots,bb.shots_a AS shots_against,
 bb.sot_h AS sot,bb.sot_a AS sot_against,ss.xg_h AS xg,ss.xg_a AS xg_against
 FROM m LEFT JOIN bb USING(match_id) LEFT JOIN ss USING(match_id) LEFT JOIN bm USING(match_id)
 UNION ALL
 SELECT m.match_id,m.sy,m.tier,m.start_timestamp,m.away_team,m.home_team,
 0,m.away_score,m.home_score,nullif(trim(bm.away_manager),''),m.venue_name_away,
 bb.shots_a,bb.shots_h,bb.sot_a,bb.sot_h,ss.xg_a,ss.xg_h
 FROM m LEFT JOIN bb USING(match_id) LEFT JOIN ss USING(match_id) LEFT JOIN bm USING(match_id);
CREATE TEMP TABLE cs AS SELECT club,sy,min(tier) AS tier,count(*) AS n_games,
 avg(gf-ga) AS gd,avg(CASE WHEN gf>ga THEN 3 WHEN gf=ga THEN 1 ELSE 0 END) AS ppg,
 avg(shots-shots_against) AS shotdiff,avg(sot-sot_against) AS sotdiff,
 count(shots) AS n_shots,avg(xg-xg_against) AS xgdiff
 FROM tm GROUP BY 1,2;
CREATE TEMP TABLE players AS
 SELECT l.*,m.sy,m.tier,m.start_timestamp,
 CASE WHEN l.is_home_team THEN m.home_team ELSE m.away_team END AS club,
 (epoch(m.start_timestamp)-l.birth_timestamp)/(365.25*86400) AS age
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/lineups.csv') l JOIN m USING(match_id);
CREATE TEMP TABLE player_year AS
 SELECT player_id,club,sy,min(tier) AS tier,count(*) AS appearances
 FROM players WHERE NOT substitute GROUP BY 1,2,3;
CREATE TEMP TABLE player_source AS SELECT player_id,sy,min(tier) AS tier FROM player_year GROUP BY 1,2;
CREATE TEMP TABLE lf AS
 SELECT p.match_id,p.club,count(*) AS starters,
 count(p.proposed_market_value) FILTER(WHERE p.proposed_market_value>0 AND p.proposed_market_value_currency='EUR') AS n_valued,
 sum(p.proposed_market_value) FILTER(WHERE p.proposed_market_value>0 AND p.proposed_market_value_currency='EUR') AS value_sum,
 avg(p.age) FILTER(WHERE p.age BETWEEN 15 AND 45) AS age_mean,
 count(p.age) FILTER(WHERE p.age BETWEEN 15 AND 45) AS n_age,
 avg(CASE WHEN py.player_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS retained_share,
 avg(CASE WHEN py.player_id IS NULL AND ps.tier<p.tier THEN 1.0 ELSE 0.0 END) AS higher_arrival_share,
 avg(CASE WHEN ps.player_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS source_coverage
 FROM players p LEFT JOIN player_year py ON py.player_id=p.player_id AND py.club=p.club AND py.sy=p.sy-1
 LEFT JOIN player_source ps ON ps.player_id=p.player_id AND ps.sy=p.sy-1
 WHERE NOT p.substitute GROUP BY 1,2;
CREATE TEMP TABLE obs AS
 SELECT o.match_id,CASE WHEN o.selection='home' THEN m.home_team ELSE m.away_team END AS club,
 o.p_model,o.p_market,100*(o.p_model-o.p_market) AS residual
 FROM read_csv('experiments/scotland/11_qs_transition/grid/observations_qs_control.csv') o JOIN m USING(match_id)
 WHERE o.selection IN ('home','away') AND o.family='1X2';
CREATE TEMP TABLE rates AS
 SELECT match_id,0.5*ln(lambda_h/lambda_a) AS d,0.5*ln(lambda_h*lambda_a) AS s
 FROM read_csv('current_development/market_model/results/QS/rates_sco_*.csv',union_by_name=true)
 WHERE accepted AND season IN ('21/22','22/23','23/24','24/25','25/26');
CREATE TEMP TABLE enriched AS SELECT t.*,o.p_model,o.p_market,o.residual,
 CASE WHEN t.is_home=1 THEN r.d ELSE -r.d END AS market_d,r.s AS market_s,
 row_number() OVER(PARTITION BY t.club,t.sy ORDER BY t.start_timestamp,t.match_id) AS game,
 lag(t.manager IGNORE NULLS) OVER w AS previous_manager,
 lag(t.start_timestamp) OVER w AS previous_date,
 avg(t.gf-t.ga) OVER w6 AS form_gd6,
 avg(CASE WHEN t.gf>t.ga THEN 3 WHEN t.gf=t.ga THEN 1 ELSE 0 END) OVER w6 AS form_ppg6,
 avg(t.shots-t.shots_against) OVER w6 AS form_shots6,
 avg(t.sot-t.sot_against) OVER w6 AS form_sot6,
 avg(t.xg-t.xg_against) OVER w6 AS form_xg6
 FROM tm t LEFT JOIN obs o USING(match_id,club) LEFT JOIN rates r USING(match_id)
 WINDOW w AS (PARTITION BY t.club ORDER BY t.start_timestamp,t.match_id),
 w6 AS (PARTITION BY t.club ORDER BY t.start_timestamp,t.match_id ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING);
CREATE TEMP TABLE history AS
 SELECT e.*,
 lag(manager IS DISTINCT FROM previous_manager AND manager IS NOT NULL AND previous_manager IS NOT NULL) OVER w AS manager_changed_previous,
 last_value(manager IS DISTINCT FROM previous_manager AND manager IS NOT NULL AND previous_manager IS NOT NULL IGNORE NULLS) OVER w3 AS unused_flag,
 max(CASE WHEN manager IS DISTINCT FROM previous_manager AND manager IS NOT NULL AND previous_manager IS NOT NULL THEN 1 ELSE 0 END) OVER w3 AS manager_change3,
 avg(residual) OVER w3 AS residual_lag3,
 avg(market_d) OVER w5 AS market_d_lag5,
 avg(market_s) OVER w5 AS market_s_lag5,
 lag(residual IGNORE NULLS) OVER w AS residual_last,
 lag(market_d IGNORE NULLS) OVER w AS market_d_last
 FROM enriched e
 WINDOW w AS (PARTITION BY club ORDER BY start_timestamp,match_id),
 w3 AS (PARTITION BY club ORDER BY start_timestamp,match_id ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
 w5 AS (PARTITION BY club ORDER BY start_timestamp,match_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING);
CREATE TEMP TABLE own AS
 SELECT h.*,c.tier AS previous_tier,c.gd AS previous_gd,c.ppg AS previous_ppg,
 c.shotdiff AS previous_shotdiff,c.sotdiff AS previous_sotdiff,c.xgdiff AS previous_xgdiff,c.n_games AS previous_games,
 CASE WHEN c.tier=h.tier THEN 'nonmover'
 WHEN c.tier=2 AND h.tier=3 THEN 'relegated_into_L1'
 WHEN c.tier=4 AND h.tier=3 THEN 'promoted_into_L1'
 WHEN c.tier=3 AND h.tier=4 THEN 'relegated_into_L2'
 WHEN c.tier IS NULL THEN 'entered_spfl' ELSE 'other_mover' END AS direction,
 (SELECT avg((tier<=2)::INTEGER) FROM tm past WHERE past.club=h.club AND past.start_timestamp<h.start_timestamp) AS prior_upper_share,
 line.start_timestamp AS lineup_date,line.sy AS lineup_sy,line.starters,line.n_valued,
 CASE WHEN line.n_valued>=8 THEN ln(line.value_sum) ELSE NULL END AS log_value,
 CASE WHEN line.n_age>=8 THEN line.age_mean ELSE NULL END AS age_mean,
 CASE WHEN line.sy=h.sy THEN line.retained_share ELSE NULL END AS retained_share,
 CASE WHEN line.sy=h.sy THEN line.higher_arrival_share ELSE NULL END AS higher_arrival_share,
 CASE WHEN line.sy=h.sy THEN line.source_coverage ELSE NULL END AS source_coverage
 FROM history h LEFT JOIN cs c ON c.club=h.club AND c.sy=h.sy-1
 LEFT JOIN LATERAL (
 SELECT ll.*,tt.start_timestamp,tt.sy FROM lf ll JOIN tm tt USING(match_id,club)
 WHERE ll.club=h.club AND tt.start_timestamp<h.start_timestamp-INTERVAL '3 hours'
 AND tt.start_timestamp>=h.start_timestamp-INTERVAL '120 days' AND ll.starters BETWEEN 10 AND 11
 ORDER BY tt.start_timestamp DESC LIMIT 1
 ) line ON true;
CREATE TEMP TABLE panel AS SELECT a.*,
 b.previous_gd AS opp_previous_gd,b.form_gd6 AS opp_form_gd6,
 b.form_shots6 AS opp_form_shots6,b.form_sot6 AS opp_form_sot6,
 b.log_value AS opp_log_value,b.age_mean AS opp_age_mean,b.retained_share AS opp_retained_share,
 b.higher_arrival_share AS opp_higher_arrival_share,b.manager_change3 AS opp_manager_change3,
 b.market_d_lag5 AS opp_market_d_lag5,b.market_s_lag5 AS opp_market_s_lag5,
 b.residual_lag3 AS opp_residual_lag3,b.prior_upper_share AS opp_prior_upper_share,
 b.direction AS opp_direction,b.lineup_date AS opp_lineup_date,b.previous_date AS opp_previous_date,
 b.market_d_last AS opp_market_d_last
 FROM own a JOIN own b ON a.match_id=b.match_id AND a.opponent=b.club
 WHERE a.sy BETWEEN 2023 AND 2025 AND a.tier IN (3,4) AND a.residual IS NOT NULL;
COPY panel TO 'experiments/scotland/11_qs_transition/research/mae_deep/panel.csv' (HEADER);
COPY (SELECT sy,tier,count(*) AS club_matches,count(shots) AS shots_available,count(xg) AS native_xg_available,
 count(manager) AS managers_available FROM tm GROUP BY 1,2 ORDER BY 1,2)
 TO 'experiments/scotland/11_qs_transition/research/mae_deep/history_coverage.csv' (HEADER);
COPY (SELECT m.sy,m.tier,count(*) AS club_matches,count(l.match_id) AS lineup_matches,
 count(*) FILTER(WHERE l.n_valued>=8) AS value_eligible,count(*) FILTER(WHERE l.n_age>=8) AS age_eligible,
 avg(l.n_valued) AS mean_valued_starters,avg(l.starters) AS mean_starters
 FROM tm m LEFT JOIN lf l USING(match_id,club) GROUP BY 1,2 ORDER BY 1,2)
 TO 'experiments/scotland/11_qs_transition/research/mae_deep/lineup_coverage.csv' (HEADER);
