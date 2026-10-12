# %% 1. Packages and fixed runtime
using BayesianFootball, CSV, DataFrames, Serialization, SHA, LinearAlgebra, ThreadPinning, Test
strip(read(`hostname`,String))=="mcmc-beast" && Threads.nthreads()==4 || error("pool/beast4 only")
pinthreads(:cores); BLAS.set_num_threads(1)
const PoolCandidates=Module(:PoolCandidates)
Base.include(PoolCandidates,joinpath(@__DIR__,"..","candidates.jl"))
const POOL_CACHE=joinpath(pwd(),".cache","datastore_ScottishPyramid.jls")
@test bytes2hex(open(SHA.sha256,POOL_CACHE))=="a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12"
pool_raw=Serialization.deserialize(POOL_CACHE)
pool_candidate=only(filter(c->c.name=="qs_pooled_a2",PoolCandidates.CANDIDATES))

# %% 2. Exact target fixtures and fold signatures against control
pool_ds=Data.apply_scope(pool_raw,pool_candidate.scope)
pool_split=Data.ScopedWalkForwardCV(pool_candidate.scope)
pool_bounds=Data.create_id_boundaries(pool_ds,pool_split)
control_ds=Data.apply_scope(pool_raw,PoolCandidates.SCOPE)
control_split=Data.ScopedWalkForwardCV(PoolCandidates.SCOPE)
control_bounds=Data.create_id_boundaries(control_ds,control_split)
@test length(pool_bounds)==length(control_bounds)==60
pool_panel=NamedTuple[]
for (fold,(pb,cb)) in enumerate(zip(pool_bounds,control_bounds))
    held=Data.get_next_matches(pool_ds,pb,pool_split)
    p=sort(Int.(held[in.(held.tournament_id,Ref([56,57])),:match_id]))
    c=sort(Int.(Data.get_next_matches(control_ds,cb,control_split).match_id))
    @test p==c
    append!(pool_panel,[(;fold,match_id=id) for id in p])
end
@test length(pool_panel)==length(unique(r.match_id for r in pool_panel))==1070
CSV.write(joinpath(@__DIR__,"panel.csv"),DataFrame(pool_panel))

# %% 3. Market mask and Championship goals, including 25/26
market_ids=Set(PoolCandidates.MARKET_FEATURE.match_ids)
champ_ids=Set(pool_raw.matches[pool_raw.matches.tournament_id.==55,:match_id])
@test isempty(intersect(market_ids,champ_ids))
pool_masks=NamedTuple[]
for (fold,pair) in zip((1,2,60),Features.create_features(pool_bounds[[1,2,60]],pool_ds,pool_candidate.model,pool_split))
    fs=first(pair)
    ids=fs.data[:ordered_match_ids]
    championship=in.(ids,Ref(champ_ids))
    @test count(championship)>0
    @test all(iszero,fs.data[:flat_market_available][championship])
    @test all(isfinite,fs.data[:flat_home_goals][championship])
    @test all(isfinite,fs.data[:flat_away_goals][championship])
    lookup=Dict(r.match_id=>r.season for r in eachrow(pool_raw.matches))
    current=championship .& [lookup[id]=="25/26" for id in ids]
    fold==60 && (@test count(current)>0)
    push!(pool_masks,(;fold,championship_goals_rows=count(championship),championship_market_rows=sum(fs.data[:flat_market_available][championship]),championship_2526_rows=count(current)))
end
CSV.write(joinpath(@__DIR__,"market_masks.csv"),DataFrame(pool_masks))
println("POOL_PANEL_PASS 60 folds / 1070 identical targets; Championship goals only")
