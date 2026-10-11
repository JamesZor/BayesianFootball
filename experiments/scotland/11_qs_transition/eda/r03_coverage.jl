# Correct calendar-year cup labels and export regulation-only cup goals.
if !isdefined(Main,:s1audit)
    include(joinpath(@__DIR__,"r00_inventory.jl"))
end
BayesianFootball.DotEnv.load!(ENV,joinpath(pkgdir(BayesianFootball),".env"))
s1audit("coverage") do conn
    m=filter(r->Date(r.match_date)<Date(2026,7,1),s1ds.matches)
    CSV.write(joinpath(S1OUT,"fixtures.csv"),m)
    ids=Set(Int.(m.match_id))
    CSV.write(joinpath(S1OUT,"bookmaker_odds.csv"),filter(r->Int(r.match_id) in ids,s1ds.odds))
    ENV["PGOPTIONS"]="-c default_transaction_read_only=on"
    c=LibPQ.Connection(ENV["BF_DB_URL"])
    try
        rows=QSBacktestPhase0.read_query(c,"SELECT tournament_id,count(*)::bigint AS n_fixtures,min(start_timestamp)::text AS first_date,max(start_timestamp)::text AS last_date FROM sofascore.matches WHERE tournament_id IN (54,55,56,57,73,982,1520) AND start_timestamp >= '2026-07-01' AND start_timestamp < '2027-07-01' GROUP BY tournament_id ORDER BY tournament_id")
        CSV.write(joinpath(S1OUT,"q6_2627_coverage_only.csv"),rows)
        goals=QSBacktestPhase0.read_query(c,"SELECT match_id,tournament_id,start_timestamp::text AS match_date,raw_data::jsonb #>> '{event,homeScore,normaltime}' AS home_regulation,raw_data::jsonb #>> '{event,awayScore,normaltime}' AS away_regulation,raw_data::jsonb #>> '{event,homeScore,period1}' AS home_period1,raw_data::jsonb #>> '{event,homeScore,period2}' AS home_period2,raw_data::jsonb #>> '{event,awayScore,period1}' AS away_period1,raw_data::jsonb #>> '{event,awayScore,period2}' AS away_period2 FROM sofascore.matches WHERE tournament_id IN (73,982,1520) AND start_timestamp >= '2021-07-01' AND start_timestamp < '2026-07-01'")
        CSV.write(joinpath(S1OUT,"q3_cup_regulation.csv"),goals)
        println("CUP_REGULATION_ROWS=",nrow(goals)," CURRENT_COVERAGE_ROWS=",nrow(rows))
    finally
        close(c)
    end
end
