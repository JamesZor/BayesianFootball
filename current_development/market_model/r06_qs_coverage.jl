# Cross-league quality/style EDA: Phase A coverage and inversion only.
# No model sampling, SQL writes, package changes or score-grid changes.
# All data come from fresh SHA-checked local snapshots (force=true done once separately).
# Run by include in an owned persistent laptop Julia REPL, never one-shot Julia.

# %%
# ===================================================================
# 1. Packages and scientific implementation
# ===================================================================
using BayesianFootball, CSV, DataFrames, SHA
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__, "l06_qs_eda.jl"))
end
const Q06A = QualityStyleEDA

# %%
# ===================================================================
# 2. Configuration and pinned snapshot gate
# ===================================================================
const Q06A_OUT = joinpath(@__DIR__, "results", "QS")
for segment in unique(map(last,Q06A.LEAGUES))
    name = string(nameof(typeof(segment)))
    Q06A.snapshot_hash(segment) == Q06A.SNAPSHOT_HASHES[name] || error("snapshot differs for $name")
end
mkpath(Q06A_OUT)

# %%
# ===================================================================
# 3. Season inventory, exclusions and complete close with price TWA(-20,0]
# ===================================================================
q06a_coverage = DataFrame[]
q06a_inventory = NamedTuple[]
q06a_configurations = NamedTuple[]
for segment in unique(map(last,Q06A.LEAGUES))
    ds = BayesianFootball.Data.load_datastore_cached(segment; max_age_hours=10^6)
    leagues = filter(x -> typeof(x[3]) == typeof(segment), Q06A.LEAGUES)
    provisional = Q06A.MM.MarketModelConfig(; name="inventory", segment,
        tournaments=[x[2] for x in leagues], seasons=String[],
        honest_train=String[], honest_test=String[])
    raw = Q06A.MM.closing_book(ds,provisional)
    book,refusals = Q06A.MM.gate_book(raw)
    for (name,tournament,_) in leagues
        inv = Q06A.season_inventory(ds,tournament)
        scoped = DataFrames.filter(r -> !ismissing(r.tournament_id) && r.tournament_id == tournament &&
            !ismissing(r.season), ds.matches)
        quoted = Set(DataFrames.innerjoin(select(scoped,:match_id,:season),
            unique(select(book,:match_id)); on=:match_id).season)
        config = Q06A.league_config(name,tournament,segment,inv; quoted_seasons=quoted)
        for r in eachrow(inv)
            reason = get(get(Q06A.EXCLUSIONS,name,Dict{String,String}()),r.season,
                r.season in ("26/27","2026") ? "season in progress" :
                !(r.season in quoted) ? "no complete Betfair market" : "")
            push!(q06a_inventory,(; league=name,tournament,season=r.season,
                matches=r.n_matches,first_date=r.first_date,last_date=r.last_date,
                has_complete_market=r.season in quoted,
                included=r.season in config.seasons,exclusion_reason=reason))
        end
        push!(q06a_configurations,(; league=name,tournament,
            seasons=join(config.seasons,";"),honest_train=join(config.honest_train,";"),
            honest_test=only(config.honest_test),price_start=-20.0,price_end=0.0,
            min_selections_ladder=config.min_selections_ladder))
        println("QS coverage ",name," seasons ",config.seasons)
        flush(stdout)
        coverage,rates = Q06A.coverage_and_rates(ds,config,raw,book,refusals)
        CSV.write(joinpath(Q06A_OUT,"rates_$(name).csv"),rates)
        push!(q06a_coverage,coverage)
    end
end
CSV.write(joinpath(Q06A_OUT,"coverage_by_league.csv"),vcat(q06a_coverage...))
CSV.write(joinpath(Q06A_OUT,"season_inventory.csv"),DataFrame(q06a_inventory))
CSV.write(joinpath(Q06A_OUT,"configs.csv"),DataFrame(q06a_configurations))
println("R06_COVERAGE_DONE")
