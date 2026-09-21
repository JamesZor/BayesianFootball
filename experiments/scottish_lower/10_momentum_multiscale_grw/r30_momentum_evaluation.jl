# Stage 3: evaluate only source-matched, converged production UUIDs. No sampling.
# One identical 710-fixture panel and Betfair TWA(-20,0] close for every arm.
# Framework proper scores, plug-in marginal CRPS, supremacy vs accepted rate
# inversions, favourite tails on ALL available 1X2 closes, and matched portfolios.
# Refused market inversions are reported, not silently folded into the tail panel.

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Statistics
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l10_momentum_grw_loader.jl"))
include(joinpath(@__DIR__,"l12_evaluation.jl"))
const M = MomentumGRW
const E = MomentumEvaluation

# ===================================================================
# 2. Immutable run addresses and fixed scoring/staking contracts
# ===================================================================
const C = M.MomentumGRWConfig()
const SOURCE = M.source_fingerprint()
const MANIFEST = joinpath(C.save_root,"production_manifest.jls")
isfile(MANIFEST) || error("No accepted production manifest; run Stage 2 first")
manifest = M.Serialization.deserialize(MANIFEST)
manifest.source==SOURCE || error("Production manifest is source-stale")
const OUT = joinpath(C.save_root,"evaluation",SOURCE)
mkpath(OUT)
db = M.PostgresStorage(C.experiment)
ds = M.gph_load_data()
odds = M.gph_betfair_closing_odds(ds)
families = M.gph_family_selections(odds)
fits = Dict(name=>M.load_fit(db,M.UUID(manifest.runs[name])) for (name,_) in M.models())
panel = sort(copy(first(values(fits)).latents.match_ids))
length(panel)==C.expected_oos || error("Expected 710 held-out fixtures")
for (name,fit) in fits
    M.gph_assert_coverage(name,fit;folds=40,oos=710)
    M.convergence_pass(fit,40) || error("$name is not converged")
    sort(fit.latents.match_ids)==panel || error("$name has a different OOS panel")
end
rates = E.market_reference(odds,panel)
CSV.write(joinpath(OUT,"market_inversions.csv"),rates)

# ===================================================================
# 3. Proper scoring, decompression, identification and portfolio ledger
# ===================================================================
scores = NamedTuple[]
headlines = NamedTuple[]
posterior = NamedTuple[]
observations = Dict{String,DataFrame}()
for (name,_) in M.models()
    fit = fits[name]
    ctx = M.gph_context(fit,odds,ds)
    append!(scores,M.gph_scores(name,ctx,families))
    observations[name] = M.gph_observation_frame(name,ctx,odds)
    CSV.write(joinpath(OUT,name*"_observations.csv"),observations[name])
    crps = M.GPH_EVAL.compute_metric(M.GPH_EVAL.CRPS(),ctx)
    decompression,supremacy,favourites = E.decompression(fit,odds,ds,rates)
    CSV.write(joinpath(OUT,name*"_supremacy.csv"),supremacy)
    CSV.write(joinpath(OUT,name*"_favourites.csv"),favourites)
    append!(posterior,M.momentum_posterior(fit,collect(1:40)))

    # BookSpec(1X2,OU2.5,BakerMcHale); FlatTrust(.25), Drawdown(20), Cap(.25).
    # The same persisted-fit/portfolio round-trip gate as the smoke run.
    run_id = M.UUID(manifest.runs[name])
    portfolio_id,result = M.portfolio_roundtrip(db,run_id,fit,ds,odds)
    s = result.summary
    allocation = E.capital_allocation(result)
    push!(headlines,(;model=name,run_id=string(run_id),portfolio_id=string(portfolio_id),
        decompression...,crps_home=crps.home.mean,crps_away=crps.away.mean,
        crps_all=crps.all.mean,total_return_pct=s.total_return_pct,
        sharpe_ann=s.sharpe_ann,max_drawdown_pct=s.mdd,flat_roi_pct=s.roi,
        n_bets=s.n_bets,allocation...))
    CSV.write(joinpath(OUT,name*"_bets.csv"),result.trajectory.bets)
    CSV.write(joinpath(OUT,"headlines.csv"),DataFrame(headlines))
end
CSV.write(joinpath(OUT,"proper_scores.csv"),DataFrame(scores))
CSV.write(joinpath(OUT,"momentum_posterior.csv"),DataFrame(posterior))

# ===================================================================
# 4. Paired fixture-clustered LogLoss comparisons, not independent-row CIs
# ===================================================================
comparisons = NamedTuple[]
for (name,_) in M.models(), family in (nothing,"1X2","OU2.5","BTTS")
    for comparator in (:market,"m01_poisson_time_decay","m02_poisson_grw_1st_order")
        comparator==name && continue
        other = comparator===:market ? :market : observations[comparator]
        summary = M.gph_paired_bootstrap(observations[name],other;B=4000,seed=22,family)
        push!(comparisons,(;model=name,comparator=string(comparator),scope=something(family,"all"),summary...))
    end
end
CSV.write(joinpath(OUT,"paired_logloss.csv"),DataFrame(comparisons))
println("EVALUATION COMPLETE: ",OUT)
println(M.gph_markdown_table(DataFrame(headlines)))
