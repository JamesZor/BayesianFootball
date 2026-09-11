# ==============================================================================
# Task 007 — MultiScaleGRW staged training and evaluation runner
# ==============================================================================
#
# WHAT THIS IS
#   A state-space dynamics experiment. It asks whether macro season steps plus
#   micro target-season match-biweek steps improve proper scores over the otherwise
#   matched 180-day TimeDecayDynamics control.
#
# WHAT THIS IS NOT
#   It is not a portfolio or staking study. No betting result is used for promotion.
#
# FILTRATION / COMPARABILITY CONTRACT
#   Pooled Scottish League One/Two (tournaments 56/57), two history seasons,
#   walk-forward target seasons 24/25 and 25/26. Every comparison requires the
#   exact same 710 held-out match IDs as its TimeDecayDynamics control.
#
# PERSISTENCE
#   Production Fits are persisted to PostgreSQL namespace
#   `scottish_lower_multiscale_grw_2426`. The loader must be included before a
#   prototype Fit is deserialized. Per-fold checkpoints live under
#   `current_development/multiscale_grw/results/` and are removed only after a
#   complete fit lands. Stage summaries are replaceable `.jls` files; run UUIDs
#   and PostgreSQL fit artifacts are immutable.
#
# USAGE ON MCMC-BEAST
#   L01_STAGE=preflight /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
#   L01_STAGE=phase1   /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
#   L01_STAGE=phase2   /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
#   L01_STAGE=auto     /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
#
# `auto` runs preflight, Phase 1, and only promotes to Phase 2 when both Phase 1
# fits clear the strict convergence gate and score comparisons complete.
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
using Serialization
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l01_loader.jl"))
using .MultiScaleGRWPrototype

Threads.nthreads() == 16 || error(
    "Task 007 remote work must run with `julia --project -t 16`; " *
    "got $(Threads.nthreads()) threads")

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R01_STAGE = lowercase(get(ENV, "L01_STAGE", "auto"))
const R01_VALID_STAGES = Set(["preflight", "phase1", "phase2", "auto", "report"])
R01_STAGE in R01_VALID_STAGES || error(
    "L01_STAGE must be one of $(sort!(collect(R01_VALID_STAGES))); got '$R01_STAGE'")

const R01_CONFIG = L01Config(
    max_concurrent_tasks = Threads.nthreads(),
)
const R01_RETRY_SAMPLES = parse(Int, get(ENV, "L01_RETRY_SAMPLES", "800"))
const R01_RETRY_WARMUP = parse(Int, get(ENV, "L01_RETRY_WARMUP", "800"))

# The two-fold preflight tests the geometry without pretending a short chain has
# production Monte Carlo precision. Zero divergences, BFMI >= 0.30, and <5%
# depth saturation are hard gates; R̂ <= 1.10 and ESS >= 50 are advisory because
# the run retains only 400 draws per chain. Gradient replay
# allocations are measured and compared with the TimeDecay control; this installed
# Turing/ReverseDiff stack allocates in the control too, so zero bytes is recorded
# as an unmet performance target rather than falsely reported as a passing gate.
const R01_PREFLIGHT_THRESHOLDS = ConvergenceThresholds(
    max_rhat = 1.10,
    min_ess = 50.0,
    max_divergence_rate = eps(Float64),
    min_bfmi = 0.30,
    max_treedepth_rate = 0.05,
)

const R01_BASELINES = Dict(
    "m00_baseline_grw" => ("scottish_lower_poisson_2426", "m00_baseline"),
    "m05_production_wealth_grw" =>
        ("scottish_lower_poisson_2426", "m05_production_wealth"),
    "m05_joint_production_wealth_grw" =>
        ("scottish_lower_joint_2426", "m05_joint_production_wealth"),
)

# %%
# ==============================================================================
# 3. Runtime, output directory, and shared experiment state
# ==============================================================================
mkpath(R01_CONFIG.save_root)

println("\n" * "="^108)
println(" TASK 007 · MULTISCALE GAUSSIAN RANDOM WALK · ", uppercase(R01_STAGE))
println("="^108)
println("  started    : ", Dates.now())
println("  repository : ", pwd())
println("  threads    : ", Threads.nthreads(), " physical-core-pinned Julia threads")
println("  BLAS       : ", LinearAlgebra.BLAS.get_num_threads(), " thread")
println("  output     : ", R01_CONFIG.save_root)
println("  database   : ", R01_CONFIG.experiment)

models = l01_models()
splitter = l01_splitter(R01_CONFIG)
production_sampler = l01_production_sampler(R01_CONFIG)
production_configs = l01_fit_configs(
    R01_CONFIG,
    models,
    splitter,
    production_sampler,
)

ds = l01_load_data()
db = l01_database(R01_CONFIG)
registry = l01_register!(
    db,
    models,
    splitter,
    production_sampler,
    production_configs,
)

println("  matches    : ", nrow(ds.matches))
println("  odds       : ", nrow(ds.odds))
println("  registry   : models=", registry.model_ids,
        " splitter=", registry.splitter_id,
        " sampler=", registry.sampler_id)

# %%
# ==============================================================================
# 4. Data snapshot and temporal split gate
# ==============================================================================
function r01_split_inventory(model)
    inputs = l01_scored_inputs(ds, splitter, model)
    held_out = sum(nrow(frame) for frame in inputs.oos; init = 0)
    length(inputs.boundaries) == R01_CONFIG.expected_folds || error(
        "G-A split inventory: got $(length(inputs.boundaries)) boundaries; " *
        "expected $(R01_CONFIG.expected_folds)")
    length(inputs.feature_sets) == R01_CONFIG.expected_folds || error(
        "G-A split inventory: built $(length(inputs.feature_sets)) feature folds; " *
        "expected $(R01_CONFIG.expected_folds)")
    held_out == R01_CONFIG.expected_oos_matches || error(
        "G-A split inventory: got $held_out held-out matches; " *
        "expected $(R01_CONFIG.expected_oos_matches)")
    return inputs
end

phase1_inventory = r01_split_inventory(models["m00_baseline_grw"])
println("\nG-A PASS · ", length(phase1_inventory.feature_sets),
        " scored boundaries · ", R01_CONFIG.expected_oos_matches,
        " held-out fixtures · dynamics_col=:match_biweek")

# %%
# ==============================================================================
# 5. Engine / model construction
# ==============================================================================
println("\nModel recipes")
for name in MultiScaleGRWPrototype.L01_MODEL_NAMES
    println("  ", rpad(name, 40), " : ", models[name])
end
println("  dynamics priors: α σ₀~Gamma(2,0.06), σₛ~Gamma(2,0.03), σₖ~Gamma(2,0.015);")
println("                   β σ₀~Gamma(2,0.10), σₛ~Gamma(2,0.055), σₖ~Gamma(2,0.012)")

# %%
# ==============================================================================
# 6. Feature construction and two-fold preflight gates
# ==============================================================================
function r01_run_preflight()
    rows = NamedTuple[]
    sampler = l01_preflight_sampler(R01_CONFIG)

    println("\n" * "-"^108)
    println(" TWO-FOLD PREFLIGHT")
    println("-"^108)
    println("  sampler: 4 chains × $(R01_CONFIG.preflight_warmup) warmup × " *
            "$(R01_CONFIG.preflight_samples) retained · target 0.90")

    for name in MultiScaleGRWPrototype.L01_MODEL_NAMES
        model = models[name]
        inputs = l01_scored_inputs(
            ds,
            splitter,
            model;
            limit = R01_CONFIG.preflight_folds,
        )
        length(inputs.feature_sets) == R01_CONFIG.preflight_folds || error(
            "$name preflight built $(length(inputs.feature_sets)) folds; " *
            "expected $(R01_CONFIG.preflight_folds)")

        for fold in eachindex(inputs.feature_sets)
            audit = l01_gradient_audit(
                model,
                inputs.feature_sets[fold],
                R01_CONFIG;
                seed = 20260910 + fold,
            )
            @printf("  %-39s fold %d · %d params · %d tape · %.3f ms · %d B\n",
                    name, fold, audit.n_parameters, audit.tape_instructions,
                    audit.gradient_ms, audit.allocated_bytes)
        end

        preflight_name = name * "_preflight_2fold"
        preflight_config = FitConfig(
            name = preflight_name,
            model = model,
            splitter = splitter,
            sampler = sampler,
            execution = QueuedExecution(
                max_concurrent_tasks = R01_CONFIG.max_concurrent_tasks),
            tags = [MultiScaleGRWPrototype.L01_TAGS; "preflight"],
            description = "Two-scored-fold Task 007 preflight for $name.",
            save_dir = joinpath(R01_CONFIG.save_root, "preflight", name),
        )

        fit = fit_model(
            preflight_config;
            feature_sets = inputs.feature_sets,
            oos_fixtures = inputs.oos,
            thresholds = R01_PREFLIGHT_THRESHOLDS,
            checkpoint_dir = joinpath(preflight_config.save_dir, "checkpoints"),
            cleanup_checkpoints = true,
            quiet = false,
        )
        fit.latents isa CountLatents || error(
            "$name two-fold preflight failed held-out latent extraction: $(fit.config.tags)")
        n_matches(fit.latents) == sum(nrow(frame) for frame in inputs.oos; init = 0) || error(
            "$name two-fold preflight latent coverage does not match held-out fixtures")
        fit.diagnostics.n_divergent == 0 || error(
            "$name two-fold preflight recorded $(fit.diagnostics.n_divergent) divergences")
        fit.diagnostics.min_bfmi >= 0.30 || error(
            "$name two-fold preflight BFMI $(fit.diagnostics.min_bfmi) is below 0.30")
        fit.diagnostics.treedepth_rate < 0.05 || error(
            "$name two-fold preflight depth saturation $(fit.diagnostics.treedepth_rate) is not below 5%")
        if !fit.diagnostics.passed
            @warn "$name short-chain preflight did not clear diagnostic advisory thresholds" failures=fit.diagnostics.failures
        end

        for fold in eachindex(inputs.feature_sets)
            audit = l01_gradient_audit(
                model,
                inputs.feature_sets[fold],
                R01_CONFIG;
                seed = 20260910 + fold,
            )
            diagnostic = fit.diagnostics.folds[fold]
            push!(rows, (;
                name,
                fold,
                audit.n_parameters,
                audit.tape_instructions,
                audit.gradient_ms,
                audit.allocated_bytes,
                rhat = diagnostic.max_rhat,
                ess = min(diagnostic.min_ess_bulk, diagnostic.min_ess_tail),
                divergences = diagnostic.n_divergent,
            ))
        end
    end

    path = l01_save_stage_result(R01_CONFIG, "preflight", rows)
    println("\nG-B/G-C/G-D PASS · gradient parity and zero divergences")
    println("  allocation finding: replay bytes are reported; literal zero is not achieved " *
            "by the installed TimeDecay control either")
    println("  summary: ", path)
    return rows
end

# %%
# ==============================================================================
# 7. Stage-summary loading and report refresh
# ==============================================================================
function r01_load_stage(name::String, default)
    path = joinpath(R01_CONFIG.save_root, name * ".jls")
    return isfile(path) ? Serialization.deserialize(path) : default
end

function r01_refresh_report(; phase2_status = nothing)
    preflight_rows = r01_load_stage("preflight", NamedTuple[])
    run_rows = vcat(
        r01_load_stage("phase1_runs", NamedTuple[]),
        r01_load_stage("phase2_runs", NamedTuple[]),
    )
    comparisons = vcat(
        r01_load_stage("phase1_comparisons", NamedTuple[]),
        r01_load_stage("phase2_comparisons", NamedTuple[]),
    )
    status = phase2_status === nothing ?
        r01_load_stage("phase2_status", "not started") : phase2_status
    path = l01_write_report!(
        R01_CONFIG;
        preflight_rows,
        run_rows,
        comparisons,
        phase2_status = status,
    )
    println("  report: ", path)
    return path
end

# %%
# ==============================================================================
# 8. Production training, convergence, persistence, and proper scoring
# ==============================================================================
function r01_run_phase(names; stage::String)
    run_rows = NamedTuple[]
    comparisons = NamedTuple[]

    println("\n" * "-"^108)
    println(" ", uppercase(stage), " PRODUCTION GRID")
    println("-"^108)
    println("  sampler   : QueuedNUTSConfig(800 warmup, 800 retained, 4 chains, target 0.90)")
    println("  execution : 40 folds × 4 single-threaded chains through the native queue")

    for name in names
        fit_config = production_configs[name]
        println("\n", "-"^96)
        println(" FIT: ", name, " · ", Dates.now())
        println("-"^96)

        result = l01_fit_or_load(db, fit_config, ds, R01_CONFIG)
        fit = result.fit
        l01_assert_coverage(name, fit, R01_CONFIG)
        if !fit.diagnostics.passed
            println("  strict gate missed; appending ", R01_RETRY_SAMPLES,
                    " retained draws per chain before the final verdict")
            fit = l01_extend_sampling(
                fit,
                fit_config,
                ds,
                R01_CONFIG;
                n_samples = R01_RETRY_SAMPLES,
                n_warmup = R01_RETRY_WARMUP,
            )
            l01_assert_coverage(name, fit, R01_CONFIG)
        end
        l01_assert_promotion(name, fit.diagnostics)

        run_id = save_fit(fit, db)
        reloaded = load_fit(db, run_id)
        reloaded.latents.match_ids == fit.latents.match_ids || error(
            "$name PostgreSQL round-trip changed latent match IDs")
        latent_matrices(reloaded.latents) == latent_matrices(fit.latents) || error(
            "$name PostgreSQL round-trip changed latent draw matrices")

        scores = l01_evaluate(reloaded, ds)
        l01_persist_scores!(db, run_id, scores)

        baseline_experiment, baseline_name = R01_BASELINES[name]
        baseline_fit = l01_load_baseline(baseline_experiment, baseline_name)
        comparison = l01_compare(name, reloaded, baseline_name, baseline_fit, ds)
        push!(comparisons, comparison)

        diagnostic = reloaded.diagnostics
        push!(run_rows, (;
            name,
            folds = length(reloaded),
            oos = n_matches(reloaded.latents),
            rhat = diagnostic.max_rhat,
            ess = min(diagnostic.min_ess_bulk, diagnostic.min_ess_tail),
            divergences = diagnostic.n_divergent,
            seconds = reloaded.metadata.elapsed_seconds,
            run_id,
            reused = result.reused,
        ))

        @printf("  persisted: %s\n", string(run_id))
        @printf("  diagnostics: R̂ %.4f · bulk ESS %.0f · tail ESS %.0f · div %d/%d\n",
                diagnostic.max_rhat, diagnostic.min_ess_bulk,
                diagnostic.min_ess_tail, diagnostic.n_divergent,
                diagnostic.n_transitions)
        @printf("  scores: LogLoss %.5f (%+.5f) · Brier %.5f (%+.5f) · RPS %.5f (%+.5f) · CRPS %.5f (%+.5f)\n",
                comparison.grw.logloss, comparison.delta_logloss,
                comparison.grw.brier, comparison.delta_brier,
                comparison.grw.rps, comparison.delta_rps,
                comparison.grw.crps, comparison.delta_crps)
    end

    l01_save_stage_result(R01_CONFIG, stage * "_runs", run_rows)
    l01_save_stage_result(R01_CONFIG, stage * "_comparisons", comparisons)
    return (; run_rows, comparisons)
end

function r01_run_phase1()
    result = r01_run_phase(MultiScaleGRWPrototype.L01_PHASE1_NAMES; stage = "phase1")
    l01_phase1_passed(db) || error(
        "Phase 1 fits were persisted but did not both pass strict convergence; refusing Phase 2")
    println("\nPHASE 1 PROMOTION PASS · both Poisson candidates satisfy R̂/ESS/divergence gates")
    return result
end

function r01_run_phase2()
    l01_phase1_passed(db) || error(
        "Phase 2 is gated on completed, strictly converged Phase 1 PostgreSQL fits")
    l01_save_stage_result(
        R01_CONFIG,
        "phase2_status",
        "running since $(Dates.now()) on mcmc-beast",
    )
    r01_refresh_report(phase2_status = "running since $(Dates.now()) on mcmc-beast")

    result = r01_run_phase(MultiScaleGRWPrototype.L01_PHASE2_NAMES; stage = "phase2")
    status = "completed and persisted at $(Dates.now()); strict convergence and score comparison passed"
    l01_save_stage_result(R01_CONFIG, "phase2_status", status)
    println("\nPHASE 2 PASS · joint Gamma-Poisson candidate persisted and compared")
    return result
end

# %%
# ==============================================================================
# 9. Stage dispatch
# ==============================================================================
if R01_STAGE == "preflight"
    r01_run_preflight()
    r01_refresh_report()
elseif R01_STAGE == "phase1"
    r01_run_phase1()
    r01_refresh_report(
        phase2_status = "not started; Phase 1 completed and is eligible for promotion")
elseif R01_STAGE == "phase2"
    r01_run_phase2()
    r01_refresh_report()
elseif R01_STAGE == "auto"
    r01_run_preflight()
    r01_run_phase1()
    r01_refresh_report(
        phase2_status = "queued by auto stage after Phase 1 strict convergence pass")
    r01_run_phase2()
    r01_refresh_report()
elseif R01_STAGE == "report"
    r01_refresh_report()
end

# %%
# ==============================================================================
# 10. Final report
# ==============================================================================
println("\n" * "="^108)
println(" TASK 007 STAGE COMPLETE · ", uppercase(R01_STAGE))
println("="^108)
println("  finished : ", Dates.now())
println("  report   : ", R01_CONFIG.report_path)
println("  database : ", db)
