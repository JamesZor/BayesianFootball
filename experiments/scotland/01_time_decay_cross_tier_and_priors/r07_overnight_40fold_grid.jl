# ==============================================================================
# Scottish Cross-Tier Time-Decay — 40-Fold Overnight Benchmark Grid
# ==============================================================================
# Runs all 5 candidate models across all 40 walk-forward folds on mcmc-beast
# with QueuedExecution(max_concurrent_tasks = 16), 4 chains × 1,000 retained draws,
# 500 adaptation, acceptance 0.65.
#
# Models:
#   1. m00_control: ScottishLower baseline (56, 57; 180d time decay, flat priors)
#   2. m01_all_spfl_league_offsets: All 4 tiers (54, 55, 56, 57) + zero-sum league offsets
#   3. m02_all_spfl_hierarchical_tiers: All 4 tiers + hierarchical tier steps
#   4. m03_prior_structural_offset: ScottishLower + structural relegation prior
#   5. m04_prior_market_derived: ScottishLower + market supremacy prior
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using JSON3
using LinearAlgebra
using Printf
using ThreadPinning

include("l01_cross_tier_loader.jl")
include("l04_production_helpers.jl")
include("r02_production_grid.jl")

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n", "="^96)
println(" TODO028 — 40-FOLD SCOTTISH CROSS-TIER BENCHMARK GRID (OVERNIGHT)")
println("="^96)
println("  Host:           mcmc-beast (16 pinned cores)")
println("  Queue:          40 folds × 4 chains = 160 tasks per candidate")
println("  Sampling:       4 chains × 1,000 draws; warmup = 500; accept_rate = 0.65")
println("  Candidates:     5 models (m00, m01, m02, m03, m04)")
println("  Target calendar: 710 unique OOS fixtures (seasons 24/25 & 25/26)")
println("="^96, "\n")

ds, data_manifest = ct02_load_snapshot()
candidates = ct02_candidates()
db = PostgresStorage("scotland_cross_tier_time_decay")

outcomes = Dict{String, Any}()
outcomes_path = joinpath(@__DIR__, "results", "overnight_40fold_outcomes.json")
mkpath(dirname(outcomes_path))

for candidate in candidates
    println("\n>>> [$(Dates.format(now(), "HH:MM:SS"))] Starting 40-fold grid for $(candidate.name)...")
    t0 = time()
    
    prepared = ct02_prepare_candidate(candidate, ds)
    println("    Prepared 40 folds with $(length(prepared.oos_ids)) unique OOS match IDs.")
    
    config = Training.FitConfig(
        name = "scotland_cross_tier_time_decay_" * candidate.name,
        model = candidate.model,
        splitter = candidate.splitter,
        sampler = ct02_sampler(),
        execution = QueuedExecution(max_concurrent_tasks = 16),
        tags = ["todo028", "stage1", "poisson", "overnight_40fold", "diagnostic_benchmark"],
        description = "40-fold walk-forward benchmark grid for $(candidate.name).",
        save_dir = joinpath(@__DIR__, "results", "overnight_40fold", candidate.name),
    )
    
    fit = fit_model(
        config;
        feature_sets = prepared.feature_sets,
        oos_fixtures = prepared.oos,
        thresholds = ConvergenceThresholds(
            max_rhat = 1.05,
            min_ess = 100.0,
            max_divergence_rate = 0.01,
            min_bfmi = 0.25,
            max_treedepth_rate = 0.05,
        ),
        cleanup_checkpoints = false,
        quiet = false,
    )
    
    elapsed_sec = time() - t0
    println("<<< [$(Dates.format(now(), "HH:MM:SS"))] Completed $(candidate.name) in $(round(elapsed_sec / 60, digits=1)) min.")
    
    run_id = save_fit(fit, db)
    println("    Persisted Fit to PostgreSQL with run_id: $run_id")
    
    n_conv_passed = count(f -> f.diagnostics.passed, fit.folds)
    println("    Convergence summary: $n_conv_passed / $(length(fit.folds)) folds passed strict thresholds.")
    
    outcomes[candidate.name] = (;
        run_id = string(run_id),
        status = "completed",
        elapsed_sec = elapsed_sec,
        folds = length(fit.folds),
        folds_passed = n_conv_passed,
        completed_at = string(now()),
    )
    
    open(outcomes_path, "w") do io
        JSON3.pretty(io, outcomes)
    end
end

println("\n", "="^96)
println(" ALL 5 CANDIDATES COMPLETED SUCCESSFULLY!")
println(" Results saved to PostgreSQL mcmc_experiments (namespace: scotland_cross_tier_time_decay)")
println(" Outcomes written to: $outcomes_path")
println("="^96)
