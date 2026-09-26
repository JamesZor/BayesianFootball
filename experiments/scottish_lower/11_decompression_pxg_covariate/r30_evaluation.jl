# Stage 3: evaluate source-matched, converged production UUIDs only. No sampling.
# Every arm uses one 710-fixture latent panel, the same de-vigged Betfair
# TWA(-20,0] close, and one reported common tradeable panel.
# USAGE: julia --project -t 16 experiments/scottish_lower/11_decompression_pxg_covariate/r30_evaluation.jl

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Statistics
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l11_decompression_loader.jl"))
include(joinpath(@__DIR__, "l13_evaluation.jl"))
const D = DecompressionPXG
const E = DecompressionEvaluation

# ===================================================================
# 2. Immutable run addresses and fixed scoring/staking contracts
# ===================================================================
const CONFIG = D.DecompressionConfig()
const SOURCE = D.source_fingerprint()
const MANIFEST = joinpath(CONFIG.save_root, "production_manifest.jls")
isfile(MANIFEST) || error("No accepted production manifest; run Stage 2 first")
manifest = D.Serialization.deserialize(MANIFEST)
manifest.source == SOURCE || error("Production manifest is source-stale")
const OUTPUT = joinpath(CONFIG.save_root, "evaluation", SOURCE)
mkpath(OUTPUT)
db = D.PostgresStorage(CONFIG.experiment)
ds = D.gph_load_data()
odds = D.gph_betfair_closing_odds(ds)
families = D.gph_family_selections(odds)
fits = Dict(name => D.load_fit(db, D.UUID(manifest.runs[name])) for (name, _) in D.models())
panel = sort(copy(first(values(fits)).latents.match_ids))
length(panel) == CONFIG.expected_oos || error("Expected 710 held-out fixtures")
for (name, fit) in fits
    D.gph_assert_coverage(name, fit; folds = 40, oos = 710)
    D.convergence_pass(fit, 40) || error("$name is not converged")
    sort(fit.latents.match_ids) == panel || error("$name has a different OOS panel")
end
rates = E.market_reference(odds, panel)
CSV.write(joinpath(OUTPUT, "market_inversions.csv"), rates)
tradeable, refusals = D.tradeable_panel(fits, odds, ds)
CSV.write(joinpath(OUTPUT, "portfolio_panel.csv"), DataFrame(match_id = tradeable))
CSV.write(joinpath(OUTPUT, "portfolio_refusals.csv"), refusals)

# ===================================================================
# 3. Proper scores, decompression, posterior, and portfolio ledger
# ===================================================================
scores = NamedTuple[]
headlines = NamedTuple[]
observations = Dict{String,DataFrame}()
for (name, _) in D.models()
    fit = fits[name]
    context = D.gph_context(fit, odds, ds)
    append!(scores, D.gph_scores(name, context, families))
    observations[name] = D.gph_observation_frame(name, context, odds)
    CSV.write(joinpath(OUTPUT, name * "_observations.csv"), observations[name])
    crps = D.GPH_EVAL.compute_metric(D.GPH_EVAL.CRPS(), context)
    decompression, supremacy, favourites = E.decompression(fit, odds, ds, rates)
    CSV.write(joinpath(OUTPUT, name * "_supremacy.csv"), supremacy)
    CSV.write(joinpath(OUTPUT, name * "_favourites.csv"), favourites)

    run_id = D.UUID(manifest.runs[name])
    portfolio_id, result = D.portfolio_roundtrip(
        db, run_id, fit, ds, odds; panel = tradeable)
    summary = result.summary
    allocation = E.capital_allocation(result)
    push!(headlines, (;
        model = name,
        run_id = string(run_id),
        portfolio_id = string(portfolio_id),
        decompression...,
        crps_home = crps.home.mean,
        crps_away = crps.away.mean,
        crps_all = crps.all.mean,
        total_return_pct = summary.total_return_pct,
        sharpe_ann = summary.sharpe_ann,
        max_drawdown_pct = summary.mdd,
        flat_roi_pct = summary.roi,
        n_portfolio_fixtures = length(tradeable),
        n_bets = summary.n_bets,
        allocation...,
    ))
    CSV.write(joinpath(OUTPUT, name * "_bets.csv"), result.trajectory.bets)
    CSV.write(joinpath(OUTPUT, "headlines.csv"), DataFrame(headlines))
end
CSV.write(joinpath(OUTPUT, "proper_scores.csv"), DataFrame(scores))

candidate_posterior = D.pxg_posterior(
    fits["m03_negbin_pxg_covariate"], collect(1:CONFIG.expected_folds))
CSV.write(joinpath(OUTPUT, "pxg_posterior.csv"), DataFrame(candidate_posterior))

# ===================================================================
# 4. Paired fixture-clustered LogLoss comparisons
# ===================================================================
comparisons = NamedTuple[]
for (name, _) in D.models(), family in (nothing, "1X2", "OU2.5", "BTTS")
    for comparator in (:market, "m01_poisson_time_decay", "m02_joint_gamma_poisson")
        comparator == name && continue
        other = comparator === :market ? :market : observations[comparator]
        result = D.gph_paired_bootstrap(
            observations[name], other; B = 4000, seed = 24, family)
        push!(comparisons, (;
            model = name,
            comparator = string(comparator),
            scope = something(family, "all"),
            result...,
        ))
    end
end
CSV.write(joinpath(OUTPUT, "paired_logloss.csv"), DataFrame(comparisons))
println("EVALUATION COMPLETE: ", OUTPUT)
println(D.gph_markdown_table(DataFrame(headlines)))
