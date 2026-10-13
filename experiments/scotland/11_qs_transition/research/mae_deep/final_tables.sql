-- Reporting-only queries on completed diagnostic outputs.
COPY (SELECT block,"group",rows,club_seasons,mae_club,delta_vs_zero,delta_lo,delta_hi,delta_vs_context,r2_vs_zero
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/chronological_diagnostics.csv')
 WHERE test_season='both' AND "group" IN ('L1_movers','nonmover') ORDER BY block,"group")
TO 'experiments/scotland/11_qs_transition/research/mae_deep/candidate_comparison.csv' (HEADER);
COPY (WITH r AS (SELECT *,CASE WHEN game<=5 THEN '01-05' WHEN game<=10 THEN '06-10' ELSE '11-20' END AS game_bin
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/residual_predictions.csv')
 WHERE game<=20 AND season>=2024),
 c AS (SELECT direction,club,season,game_bin,count(*) AS n,avg(abs(residual)) AS baseline,
 avg(abs(residual-form_pair)) AS corrected,avg(residual) AS bias FROM r GROUP BY ALL)
 SELECT direction,game_bin,count(*) AS club_seasons,sum(n) AS rows,avg(baseline) AS baseline_mae,
 avg(corrected) AS corrected_mae,avg(corrected-baseline) AS delta,avg(bias) AS bias
 FROM c GROUP BY 1,2 ORDER BY 1,2)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/form_gain_by_game_bin.csv' (HEADER);
COPY (SELECT p.club,p.sy,count(*) AS rows,avg(p.age_mean) AS age,avg(p.retained_share) AS retained,
 avg(p.higher_arrival_share) AS higher_arrivals,avg(p.source_coverage) AS player_source_coverage,
 avg(p.manager_change3) AS manager_change3_share,avg(p.residual) AS bias,avg(abs(p.residual)) AS mae
 FROM read_csv('experiments/scotland/11_qs_transition/research/mae_deep/panel.csv') p
 WHERE game<=20 AND direction IN ('promoted_into_L1','relegated_into_L1') GROUP BY 1,2 ORDER BY 2,1)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/mover_covariates.csv' (HEADER);
COPY (WITH f AS(SELECT match_id,count(*) AS n,sum(p_model) AS model_mass,sum(p_market) AS market_mass
 FROM read_csv('experiments/scotland/11_qs_transition/grid/observations_qs_control.csv') WHERE family='1X2' GROUP BY 1)
 SELECT n,count(*) AS fixtures,min(model_mass) AS min_model_mass,max(model_mass) AS max_model_mass,
 min(market_mass) AS min_market_mass,max(market_mass) AS max_market_mass,
 count(*) FILTER(WHERE abs(model_mass-1)>0.00001) AS model_mass_error_gt1e5 FROM f GROUP BY 1)
TO 'experiments/scotland/11_qs_transition/research/mae_deep/probability_mass_audit.csv' (HEADER);
