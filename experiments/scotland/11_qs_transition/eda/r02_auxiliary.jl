# Cup market/goals coverage and QS boundary posterior summaries. No fits.
if !isdefined(Main,:s1audit)
    include(joinpath(@__DIR__,"r00_inventory.jl"))
end
using Statistics, UUIDs
# Explicit runtime load: package top-level dotenv execution may be cached by precompile.
BayesianFootball.DotEnv.load!(ENV, joinpath(pkgdir(BayesianFootball), ".env"))
s1audit("auxiliary") do conn
    # Processed SofaScore source; never confuse this with Betfair close.
    ids=Set(Int.(filter(r->String(r.season)<="25/26",s1ds.matches).match_id))
    CSV.write(joinpath(S1OUT,"bookmaker_odds.csv"),filter(r->Int(r.match_id) in ids,s1ds.odds))
    ENV["PGOPTIONS"]="-c default_transaction_read_only=on"
    betconn=LibPQ.Connection(ENV["BF_DB_URL"])
    try
        meta=QSBacktestPhase0.read_query(betconn,"SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema='sofascore' AND table_name IN ('match_odds','matches') ORDER BY table_name,ordinal_position")
        CSV.write(joinpath(S1OUT,"sofascore_schema.csv"),meta)
        raw=QSBacktestPhase0.read_query(betconn,"SELECT o.match_id,o.market_id,o.market_name,o.choice_name,o.choice_group,o.fraction_num,o.fraction_den,o.initial_fraction_num,o.initial_fraction_den FROM sofascore.match_odds o JOIN sofascore.matches m ON o.match_id=m.match_id WHERE m.tournament_id IN (54,55,56,57,73,982,1520) AND m.start_timestamp < '2026-07-01' AND o.market_name='Full time'")
        CSV.write(joinpath(S1OUT,"q3_bookmaker_raw.csv"),raw)
        println("BOOKMAKER rows=",nrow(raw)," matches=",length(unique(raw.match_id)))
    finally
        close(betconn)
    end
    fit=Training.load_fit(Training.PostgresStorage("scottish_lower_qs_wave2_2426"),UUID("81fdf817-f82d-4f66-9bfd-5f456c4fe29f"))
    rows=NamedTuple[]
    for fold in fit.folds
        chain=fold.chain
        names=string.(chain.name_map.parameters)
        for name in filter(n->occursin(r"dyn\.(τ|r)[₀ₛₖ]$",n),names)
            values=vec(Array(chain[:,[Symbol(name)],:]))
            push!(rows,(;fold=fold.fold,target_season=fold.meta.target_season,name,mean=mean(values),sd=std(values),p025=quantile(values,0.025),p975=quantile(values,0.975)))
        end
    end
    CSV.write(joinpath(S1OUT,"q4_qs_scales.csv"),DataFrame(rows))
    println("QS_SCALE_ROWS=",length(rows))
end
