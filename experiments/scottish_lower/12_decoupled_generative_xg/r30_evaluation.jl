# Stage 3: evaluate source-matched, converged production UUIDs only. No sampling.
# Every arm uses one 710-fixture latent panel, one Betfair TWA(-20,0] close,
# and one reported common tradeable panel.
# USAGE: julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r30_evaluation.jl

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Statistics
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l12_loader.jl"))
include(joinpath(@__DIR__, "l14_evaluation.jl"))
const D = DecoupledGenerativeXG
const E = DecoupledGenerativeXGEvaluation

# ===================================================================
# 2. Immutable run addresses and fixed scoring/staking contracts
# ===================================================================
const CONFIG = D.FunnelConfig()
const SOURCE = D.source_fingerprint()
const MANIFEST = joinpath(CONFIG.save_root, "production_manifest.jls")
isfile(MANIFEST) || error("No accepted production manifest; run Stage 2 first")
manifest = D.Serialization.deserialize(MANIFEST)
manifest.source == SOURCE || error("Production manifest is source-stale")
# An arm may have been ACCEPTED under this source but SAMPLED under an earlier one
# (see r21_resume_m04.jl). That is legitimate only when the difference is confined to
# gate/reporting code off the sampling path, so it is printed here rather than hidden.
if get(manifest, :sampled_source, SOURCE) != SOURCE
    println("PROVENANCE: arms ", get(manifest, :resumed, String[]),
            " sampled under ", get(manifest, :sampled_source, "?"),
            "\n            accepted under ", SOURCE)
end
const OUTPUT = joinpath(CONFIG.save_root, "evaluation", SOURCE)
mkpath(OUTPUT)
db = D.PostgresStorage(CONFIG.experiment)
ds = D.gph_load_data()
splitter_for_folds = D.gph_splitter(CONFIG.target_seasons)
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
# Fold ownership of every priced fixture, for the coverage-stratified reporting below.
fold_of = E.fold_of_match(ds, splitter_for_folds, last(first(D.models())))
CSV.write(joinpath(OUTPUT, "fold_of_match.csv"),
          DataFrame(match_id = collect(keys(fold_of)), fold = collect(values(fold_of))))

rates = E.market_reference(odds, panel)
CSV.write(joinpath(OUTPUT, "market_inversions.csv"), rates)
tradeable, refusals = D.tradeable_panel(fits, odds, ds)
CSV.write(joinpath(OUTPUT, "portfolio_panel.csv"), DataFrame(match_id = tradeable))
CSV.write(joinpath(OUTPUT, "portfolio_refusals.csv"), refusals)

# ===================================================================
# 3. Proper scores, decompression, finishing posterior, and portfolio
# ===================================================================
scores = NamedTuple[]
headlines = NamedTuple[]
observations = Dict{String,DataFrame}()
posterior = NamedTuple[]
for (name, _) in D.models()
    fit = fits[name]
    context = D.gph_context(fit, odds, ds)
    append!(scores, D.gph_scores(name, context, families))
    observations[name] = E.with_fold(D.gph_observation_frame(name, context, odds), fold_of)
    CSV.write(joinpath(OUTPUT, name * "_observations.csv"), observations[name])
    crps = D.GPH_EVAL.compute_metric(D.GPH_EVAL.CRPS(), context)
    decompression, supremacy, favourites = E.decompression(fit, odds, ds, rates)
    CSV.write(joinpath(OUTPUT, name * "_supremacy.csv"), supremacy)
    CSV.write(joinpath(OUTPUT, name * "_favourites.csv"), favourites)
    append!(posterior, [(; model = name, row...) for row in
                        D.kappa_posterior(fit, collect(1:CONFIG.expected_folds))])

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
CSV.write(joinpath(OUTPUT, "kappa_posterior.csv"), DataFrame(posterior))

# ===================================================================
# 4. Paired fixture-clustered LogLoss comparisons
# ===================================================================
comparisons = NamedTuple[]
for (name, _) in D.models(), family in (nothing, "1X2", "OU2.5", "BTTS")
    for comparator in (:market, "m01_poisson_time_decay", "m02_joint_gamma_poisson",
                       "m03_funnel_shared_kappa")
        comparator == name && continue
        other = comparator === :market ? :market : observations[comparator]
        result = D.gph_paired_bootstrap(
            observations[name], other; B = 4000, seed = 25, family)
        push!(comparisons, (;
            model = name,
            comparator = string(comparator),
            scope = something(family, "all"),
            result...,
        ))
    end
end
CSV.write(joinpath(OUTPUT, "paired_logloss.csv"), DataFrame(comparisons))

# ===================================================================
# 5. THE PRE-REGISTERED HEADLINE: folds 21-40 (full proxy coverage)
# ===================================================================
# m02 vs m03 is no longer a Monte Carlo null. The two densities USED to be identical
# -- that was the defect -- and are now a joint posterior versus a cut one, so this
# contrast is the experiment's actual treatment effect: what it costs, in price, to
# forbid goals from updating the team ratings.
#
# It is reported on three fold blocks because the chance layer's proxy coverage is not
# constant across the walk-forward (see `E.fold_of_match`). Folds 21-40 are the
# pre-registered clean comparison; folds 1-20 are shown because hiding a
# disadvantageous block would be the whole point of pre-registering, not a footnote.
blocks = (
    ("folds_21_40_full_proxy", f -> f > 20),     # headline
    ("folds_01_20_thin_proxy", f -> f <= 20),    # coverage-confounded
    ("all_folds", f -> true),
)
cut_rows = NamedTuple[]
for (block, keep) in blocks, family in (nothing, "1X2", "OU2.5", "BTTS")
    sub(name) = observations[name][keep.(observations[name].fold), :]
    for (label, a, b) in (
            ("m03_vs_m02", "m03_funnel_shared_kappa", "m02_joint_gamma_poisson"),
            ("m04_vs_m02", "m04_funnel_hierarchical_kappa", "m02_joint_gamma_poisson"),
            ("m04_vs_m03", "m04_funnel_hierarchical_kappa", "m03_funnel_shared_kappa"),
            ("m03_vs_m01", "m03_funnel_shared_kappa", "m01_poisson_time_decay"))
        result = D.gph_paired_bootstrap(sub(a), sub(b); B = 4000, seed = 25, family)
        push!(cut_rows, (; block, contrast = label,
                           scope = something(family, "all"), result...))
    end
    # And every arm against the closing line, per block.
    for name in first.(D.models())
        result = D.gph_paired_bootstrap(sub(name), :market; B = 4000, seed = 25, family)
        push!(cut_rows, (; block, contrast = "$(name)_vs_market",
                           scope = something(family, "all"), result...))
    end
end
CSV.write(joinpath(OUTPUT, "cut_effect_by_fold_block.csv"), DataFrame(cut_rows))

# Per-arm LogLoss by block, the table the README quotes.
block_scores = NamedTuple[]
for (block, keep) in blocks, name in first.(D.models())
    df = observations[name]
    df = df[keep.(df.fold), :]
    push!(block_scores, (; block, model = name,
                           n_obs = nrow(df),
                           n_fixtures = length(unique(df.match_id)),
                           logloss_model = mean(df.ll_model),
                           logloss_market = mean(df.ll_market)))
end
CSV.write(joinpath(OUTPUT, "logloss_by_fold_block.csv"), DataFrame(block_scores))
println("\n=== LogLoss by fold block (proxy coverage stratified) ===")
println(D.gph_markdown_table(DataFrame(block_scores)))
println("\n=== Cut effect, headline block folds 21-40, all markets ===")
println(D.gph_markdown_table(DataFrame(
    [r for r in cut_rows if r.block == "folds_21_40_full_proxy" && r.scope == "all"])))

println("EVALUATION COMPLETE: ", OUTPUT)
println(D.gph_markdown_table(DataFrame(headlines)))
