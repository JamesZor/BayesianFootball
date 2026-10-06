COPY (
 SELECT 'C3 Gate1 ' || v.measure || ' ' || s.book_type || ' n=' || s.n || ' sd=' || s.spread || ' offset=' || s."offset" AS gate,
 v.value, v.tol, v.value <= v.tol AS pass
 FROM 'current_development/market_model/results/C/laplace_gate.csv' s,
 LATERAL (VALUES ('median marginal',s.median_abs_error,0.01),
 ('p95 marginal',s.p95_abs_error,0.05),('max mean/SD',s.max_mean_error,0.05),
 ('max SD relative',s.max_sd_error,0.05)) v(measure,value,tol)
 ORDER BY s.book_type,s.n,s.spread,s."offset",v.measure
) TO 'current_development/market_model/results/C/engine_gates_c.csv' (HEADER);
SELECT count(*) AS gates, count(*) FILTER(WHERE NOT pass) AS failures FROM 'current_development/market_model/results/C/engine_gates_c.csv';
SELECT count(*) AS fixture_settings FROM 'current_development/market_model/results/C/laplace_gate_fixture.csv';
