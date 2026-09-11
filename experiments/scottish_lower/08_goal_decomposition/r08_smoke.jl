# ==============================================================================
# r08 — Experiment 08 strict real-data smoke ladder
# ==============================================================================
#
# This is a real one-fold, four-candidate smoke.  It is intentionally disabled
# until `L08_RUN_SMOKE=true`: no short, local, or synthetic MCMC substitutes for
# the production geometry.  Every candidate must pass every gate independently.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Test
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 8 || error(
    "Experiment 08 local smoke requires 8 physical-core Julia threads on archpc; got $(Threads.nthreads())")

include(joinpath(@__DIR__, "l08_workflow.jl"))
include(joinpath(@__DIR__, "l08_incident_data.jl"))
include(joinpath(@__DIR__, "l08_decomposed_models.jl"))
include(joinpath(@__DIR__, "l08_model_checks.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R08_RUN_SMOKE = lowercase(get(ENV, "L08_RUN_SMOKE", "false")) in ("1", "true", "yes")
const R08_SMOKE_FOLD = parse(Int, get(ENV, "L08_SMOKE_FOLD", "1"))
const R08_SOURCE_FILES = [
    joinpath(@__DIR__, "l08_workflow.jl"),
    joinpath(@__DIR__, "l08_decomposed_models.jl"),
    joinpath(@__DIR__, "l08_incident_data.jl"),
    joinpath(@__DIR__, "l08_model_checks.jl"),
    @__FILE__,
]

# %%
# ==============================================================================
# 3. Data, immutable registry, and one canonical fold
# ==============================================================================
l08_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(L08_EXPERIMENT)
ensure_schema!(db)
registry = l08_registry(ds, db; source_files = R08_SOURCE_FILES)
splitter = l08_splitter()
incident_registry, incident_snapshot_hash = GoalDecompositionIncidentData.load_registry(joinpath(@__DIR__, "results"))
model_entries = l08_models(incident_registry, incident_snapshot_hash;
                            registry_hash = incident_snapshot_hash)
length(model_entries) == length(L08_CANDIDATE_NAMES) || error("smoke must run all four candidates")
models = [(String(entry.name), entry.model) for entry in model_entries]
configs = Dict(name => l08_fit_config(name * "_smoke_fold$(R08_SMOKE_FOLD)", model, splitter)
               for (name, model) in models)
book = l08_book_spec()
policy = l08_policy_spec()
l08_register!(registry, models, splitter, L08_SAMPLER, configs, book, policy)

boundaries = Data.create_id_boundaries(ds, splitter)
1 <= R08_SMOKE_FOLD <= length(boundaries) || error("smoke fold $R08_SMOKE_FOLD is out of range")
smoke_boundary = boundaries[R08_SMOKE_FOLD:R08_SMOKE_FOLD]

println("\n", "="^108)
println(" EXPERIMENT 08 · STRICT REAL-DATA THREE-CANDIDATE SMOKE")
println("="^108)
println("  mode      : ", R08_RUN_SMOKE ? "SAMPLING AUTHORISED" : "PREFLIGHT ONLY")
println("  fold      : ", R08_SMOKE_FOLD)
println("  sampler   : 4 chains × 1,000 warmup × 1,000 retained · target acceptance 0.95 (fixed pre-smoke)")
println("  gates     : gradients | parameter contract | sampling | six-part audit | latents | grid | DB round trips")
println("  snapshot  : ", registry.snapshot_hash)

# %%
# ==============================================================================
# 4. Feature and AD preflight gates
# ==============================================================================
# `l08_gradient_checks` and the local registry-backed feature materialiser are
# model-owned, so orchestration does not duplicate their density or filtration logic.
prepared = Dict{String,Any}()
for (name, model) in models
    original_features = Features.create_features(smoke_boundary, ds, model, splitter)
    oos = [Data.get_next_matches(ds, feature, splitter) for feature in original_features]
    all(frame -> nrow(frame) > 0, oos) || error("$name smoke fold has no OOS fixture")
    features = l08_declare_prediction_teams(original_features, oos)
    fs = first(first(features))
    String(fs.data[:goal_decomposition_data_hash]) == model.data_hash || error(
        "$name smoke FeatureSet does not carry the frozen incident snapshot hash")
    gradient = l08_gradient_checks(model, fs)
    benchmark = l08_gradient_benchmark(gradient)
    benchmark.allocations == 0 || error(
        "$name compiled gradient allocated $(benchmark.allocations) bytes; require zero after warmup")
    expected = l08_expected_params(Symbol(name), Int(fs.data[:n_teams]), Int(fs.data[:n_referees]))
    length(gradient.θ) == expected || error(
        "$name has $(length(gradient.θ)) parameters; expected structural contract $expected")
    prepared[name] = (; model, features, oos, gradient, benchmark)
    @printf("  %-38s G1 compiled gradient passed | tape %d | params %d | %.4f ms | 0 alloc\n",
        name, gradient.instructions, length(gradient.θ), benchmark.milliseconds)
end

l08_write_manifest!(registry;
    stage = R08_RUN_SMOKE ? "smoke_authorised" : "smoke_preflight",
    extra = Dict(
        "smoke_fold" => R08_SMOKE_FOLD,
        "candidate_names" => first.(models),
        "gradient_checks" => Dict(name => Dict(
            "instructions" => value.gradient.instructions,
            "compiled_fresh" => value.gradient.compiled_fresh,
            "compiled_forward" => value.gradient.compiled_forward,
            "perturbed" => value.gradient.perturbed,
            "gradient_allocations" => value.benchmark.allocations,
            "gradient_milliseconds" => value.benchmark.milliseconds,
        ) for (name, value) in prepared),
    ))

# %%
# ==============================================================================
# 5. Sampling, strict convergence, extraction, and persistence
# ==============================================================================
if !R08_RUN_SMOKE
    println("\nPREFLIGHT ONLY passed.  Set L08_RUN_SMOKE=true only after the parent confirms an idle beast.")
else
    smoke_passed = String[]
    @testset "Experiment 08 real-data strict smoke" begin
        for (name, _) in models
            @testset "$name" begin
                item = prepared[name]
                fit = fit_model(configs[name];
                    feature_sets = item.features,
                    oos_fixtures = item.oos,
                    thresholds = L08_THRESHOLDS,
                    checkpoint_dir = joinpath(configs[name].save_dir, "checkpoints"),
                    cleanup_checkpoints = false,
                    quiet = false)

                # G2/G3/G4: all chains returned, native gate passed, and the stricter
                # requested R-hat/ESS/divergence readings remain explicit.
                @test length(fit.folds) == 1
                l08_assert_promotion(name, fit.diagnostics)

                # G5: extraction is concrete, finite, positive, and OOS aligned.
                @test fit.latents isa CountLatents
                @test n_matches(fit.latents) == sum(nrow, item.oos)
                @test all(isfinite, fit.latents.λ_home)
                @test all(isfinite, fit.latents.λ_away)
                @test all(>(0.0), fit.latents.λ_home)
                @test all(>(0.0), fit.latents.λ_away)

                # G6: the ordinary 12×12 score grid must be buildable from the total
                # intensity that the decomposed model exposes through CountLatents.
                grid = compute_score_grid(fit.latents, 1)
                @test size(grid, 1) == 12
                @test size(grid, 2) == 12
                @test all(isfinite, grid)

                # G7: exact posterior artefact and relational latent panel round-trip.
                run_id = save_fit(fit, db)
                loaded = load_fit(db, run_id)
                @test length(loaded.folds) == length(fit.folds)
                @test Array(loaded.folds[1].chain) == Array(fit.folds[1].chain)
                @test loaded.latents.match_ids == fit.latents.match_ids
                @test latent_matrices(loaded.latents) == latent_matrices(fit.latents)

                # G8: common Betfair book and backtest ledger are also lossless.
                odds = l08_betfair_closing_odds(ds)
                result, _, _ = run_portfolio_simulation(
                    book, policy, fit, odds, ds;
                    bootstrap = false, require_converged = true, quiet = true)
                portfolio_id = save_portfolio_db(result, run_id, db;
                    book_spec = book, policy_spec = policy,
                    metadata = (; smoke = true, candidate = name,
                                  odds_source = "betfair_twa_minus20_to_close"))
                reloaded = load_portfolio_db(portfolio_id, db)
                @test reloaded.summary.total_return_pct == result.summary.total_return_pct
                @test isequal(reloaded.trajectory.bets, result.trajectory.bets)
                push!(smoke_passed, name)
                println("  PASS $name · run $run_id · portfolio $portfolio_id")
            end
        end
    end
    Set(smoke_passed) == Set(first.(models)) || error(
        "production is blocked: strict smoke did not pass all four candidates")
    println("ALL FOUR strict smokes passed; only now may r08_production_grid.jl be authorised.")
end

# %%
# ==============================================================================
# 6. Final report
# ==============================================================================
println("Finished: ", Dates.now())
