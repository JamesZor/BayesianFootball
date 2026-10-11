module MarketAPMPhaseA
import BayesianFootball, CSV, DataFrames, SHA
const LEAGUES = [("sco_league_one",56,"ScottishLower"),("sco_league_two",57,"ScottishLower"),
    ("sco_premiership",54,"ScottishUpper"),("sco_championship",55,"ScottishUpper"),
    ("eng_premier",1,"MarketModelEnglish"),("eng_championship",2,"MarketModelEnglish"),
    ("eng_league_one",3,"MarketModelEnglish"),("eng_league_two",84,"MarketModelEnglish")]
const HASHES = Dict("MarketModelEnglish"=>"2efe200ca6ed2899a75246570e025d5782af5da945ec26100bfa38331c5168bf",
    "ScottishUpper"=>"56207e6354f0e9fbf5a6d1e29a277ae49b50769d2975de7a7e7ccca85c93630a",
    "ScottishLower"=>"e894ac6d59f912aceb01f639c80be37555a73f4eab2fa24271cddd801f13006a")

"Verify before loading; the English serialized segment requires its original module identity."
function pinned_datastores(root)
    segments = Dict("ScottishLower"=>BayesianFootball.Data.ScottishLower(),
        "ScottishUpper"=>BayesianFootball.Data.ScottishUpper(),
        "MarketModelEnglish"=>Main.QualityStyleEDA.MarketModelEnglish())
    for (name, hash) in HASHES
        actual = open(SHA.sha256, joinpath(root, ".cache", "datastore_$(name).jls"))
        bytes2hex(actual) == hash || error("STOP: SHA mismatch for $name")
    end
    Dict(name=>BayesianFootball.Data.load_datastore_cached(segment; max_age_hours=10^6)
        for (name,segment) in segments)
end

"Coverage on the inversion's included seasons, with disjoint accepted-close failure reasons."
function coverage(root, out)
    stores = pinned_datastores(root)
    rows = NamedTuple[]
    failures = NamedTuple[]
    gates = NamedTuple[]
    for (league, tournament, segment) in LEAGUES
        ds = stores[segment]
        rates = CSV.read(joinpath(root,"current_development/market_model/results/QS/rates_$league.csv"), DataFrames.DataFrame)
        length(unique(rates.match_id)) == DataFrames.nrow(rates) || error("Duplicate rate match IDs: $league")
        fixtures = ds.matches[ds.matches.tournament_id .== tournament,:]
        length(unique(fixtures.match_id)) == DataFrames.nrow(fixtures) || error("Duplicate fixtures: $league")
        fixture_ids = Set(fixtures.match_id)
        starters = ds.lineups[(ds.lineups.tournament_id .== tournament) .& .!ds.lineups.is_substitute,:]
        sides = Dict((g.match_id[1], String(g.team_side[1])) => collect(g.player_id)
            for g in DataFrames.groupby(starters, [:match_id,:team_side]))
        accepted = rates[rates.accepted .== true,:]
        total_joined = 0
        for season in sort(unique(rates.season))
            r = accepted[accepted.season .== season,:]
            complete = 0
            joined = 0
            for m in DataFrames.eachrow(r)
                h = get(sides,(m.match_id,"home"),Int[])
                a = get(sides,(m.match_id,"away"),Int[])
                valid = length(h)==11 && length(a)==11 && length(unique(h))==11 && length(unique(a))==11 && isempty(intersect(h,a))
                complete += valid
                reason = !(m.match_id in fixture_ids) ? "missing_fixture" :
                    isempty(h) || isempty(a) ? "missing_starting_side" :
                    length(h)!=11 || length(a)!=11 ? "starter_count_not_11" :
                    !valid ? "duplicate_or_shared_player_id" :
                    !isfinite(m.lambda_h) || !isfinite(m.lambda_a) || min(m.lambda_h,m.lambda_a)<=0 ? "invalid_rate" : "ok"
                if reason == "ok"
                    joined += 1
                else
                    push!(failures,(league=league,season=season,match_id=m.match_id,reason=reason,home_starters=length(h),away_starters=length(a)))
                end
            end
            n = DataFrames.nrow(r)
            push!(rows,(league=league,tournament=tournament,season=season,
                fixtures=count(==(season),fixtures.season),accepted_closes=n,
                closes_complete_xi=complete,joined_n=joined,loss_fraction=n==0 ? 0.0 : 1-joined/n))
            total_joined += joined
        end
        n = DataFrames.nrow(accepted)
        loss = 1-total_joined/n
        push!(gates,(league=league,accepted=n,joined_n=total_joined,loss_fraction=loss,blocked=loss>0.15))
        println(league, ": accepted=",n," joined=",total_joined," loss=",round(100loss;digits=2),"%",loss>0.15 ? " STOP" : "")
    end
    CSV.write(joinpath(out,"coverage.csv"),DataFrames.DataFrame(rows))
    CSV.write(joinpath(out,"join_failures.csv"),DataFrames.DataFrame(failures))
    CSV.write(joinpath(out,"coverage_gate.csv"),DataFrames.DataFrame(gates))
    return any(r.blocked for r in gates)
end
end
