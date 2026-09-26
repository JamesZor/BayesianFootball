# ============================================================================
# TODO028 — separate one-fold longer-chain convergence pilot
# ============================================================================
# User approved 4×1,000 retained draws, 500 adaptation, acceptance 0.65.
# This does NOT replace the original 2×100 smoke or launch the 40-fold grid.
# Hypothesis: the short smoke's R-hat/ESS failures diminish with more sampling.
# Filtration, model equations, source data and convergence thresholds stay fixed.
# Allocation and grid-tail failures remain independent promotion blockers.
# Reuse exact persisted recipes; never retry seeds to obtain a green audit.
#
# Run on mcmc-beast: julia --startup-file=no --project -t 16 <this file>

# ============================================================================
# 1. Packages and definitions
# ============================================================================
using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Random
using SHA
using Test
using ThreadPinning
include("l01_cross_tier_loader.jl")
include("l02_smoke_helpers.jl")
include("l04_production_helpers.jl")
include("l05_pilot_helpers.jl")

# ============================================================================
# 2. Explicit approved configuration
# ============================================================================
const CT06_SAMPLES = 1_000
const CT06_CHAINS = 4
const CT06_WARMUP = 500
const CT06_ACCEPT = 0.65
const CT06_SEED = 2806
const CT06_THRESHOLDS = ConvergenceThresholds(
    max_rhat = 1.05, min_ess = 100.0, max_divergence_rate = eps(),
    min_bfmi = 0.30, max_treedepth_rate = 0.05,
)
const CT06_SAMPLER = NUTSConfig(
    n_samples = CT06_SAMPLES, n_chains = CT06_CHAINS,
    n_warmup = CT06_WARMUP, accept_rate = CT06_ACCEPT,
    max_depth = 10, show_progress = false,
)
const CT06_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CT06_SOURCE_FILES = [joinpath(@__DIR__, name) for name in
    ("l01_cross_tier_loader.jl", "l02_smoke_helpers.jl", "l04_production_helpers.jl",
     "l05_pilot_helpers.jl", "r06_longer_chain_pilot.jl")]
push!(CT06_SOURCE_FILES, joinpath(CT06_ROOT, "src", "Data", "fetchers", "segments.jl"))
const CT06_SOURCE_DIGEST = bytes2hex(SHA.sha256(join(read.(CT06_SOURCE_FILES, String), "\u001e")))

# ============================================================================
# 3. Runtime and isolated output
# ============================================================================
Threads.nthreads() >= 4 || error("pilot requires at least four Julia threads")
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
const CT06_STAMP = Dates.format(now(), "yyyymmdd_HHMMSS")
const CT06_OUTPUT = joinpath(@__DIR__, "results", "longer_pilot", CT06_STAMP)
ispath(CT06_OUTPUT) && error("pilot output directory already exists: $CT06_OUTPUT")
mkpath(CT06_OUTPUT)
println("One-fold pilot: 4×1000 retained, 500 adaptation, acceptance=0.65")
println("Output: ", CT06_OUTPUT)
println("Source digest: ", CT06_SOURCE_DIGEST)

# ============================================================================
# 4. Same snapshot, five models and same lower OOS boundary
# ============================================================================
function ct06_run_pilot()
    ds = Data.load_datastore_cached(Data.ScottishAll(); max_age_hours = 100_000)
    data_manifest = ct02_data_manifest(ds)
    candidates = cross_tier_smoke_candidates(sampler = CT06_SAMPLER,
                                             execution = SequentialExecution())
    prepared = [cross_tier_smoke_prepare_one_fold(candidate, ds) for candidate in candidates]
    lower_ids = Set(Int.(ds.matches.match_id[in.(ds.matches.tournament_id, Ref([56, 57]))]))
    for item in prepared
        @test Set(item.heldout_ids) == Set(prepared[1].heldout_ids)
        @test intersect(Set(item.fitted_ids), lower_ids) == Set(prepared[1].fitted_ids)
        @test isempty(intersect(item.fitted_ids, item.heldout_ids))
        fitted = DataFrames.subset(ds.matches, :match_id => ByRow(in(Set(item.fitted_ids))))
        @test all(Date.(fitted.match_date) .< item.cutoff)
    end
    db = PostgresStorage("scotland_cross_tier_time_decay")
    records = []

    for (index, candidate) in enumerate(candidates)
        item = prepared[index]
        core_digest = ct02_core_recipe_digest(candidate, data_manifest)
        config = FitConfig(
            name = candidate.name * "_pilot_4x1000_accept065",
            model = candidate.model, splitter = candidate.splitter,
            sampler = CT06_SAMPLER, execution = SequentialExecution(),
            tags = ["todo028", "stage1", "longer_pilot", "seed:$(CT06_SEED + index)",
                    "source_sha256:" * CT06_SOURCE_DIGEST, "core_sha256:" * core_digest],
            description = "User-approved one-fold 4x1000/500 pilot at acceptance 0.65; not production promotion.",
            save_dir = CT06_OUTPUT,
        )

        # ====================================================================
        # 5. Canonical recipe and AD correctness preflight (no weakening)
        # ====================================================================
        preflight = ct_pilot_register_preflight(db, config)
        Random.seed!(CT06_SEED + index)
        tape = ct_smoke_tape_gate(candidate.model, item.feature_set)
        println("\n", candidate.name, " tape: ", tape.elapsed * 1000, " ms, ",
                tape.allocation, " bytes; gradient correctness verified")

        # ====================================================================
        # 6. One-fold sampling or exact saved-run reuse
        # ====================================================================
        reused = preflight.run_id !== nothing
        fit = if reused
            load_fit(db, preflight.run_id)
        else
            fit_model(config;
                feature_sets = [(item.feature_set, item.meta)], oos_fixtures = Any[item.oos],
                thresholds = CT06_THRESHOLDS, cleanup_checkpoints = false,
                checkpoint_dir = joinpath(CT06_OUTPUT, candidate.name, "checkpoints"),
                quiet = false)
        end
        @test length(fit.folds) == 1
        @test size(fit.folds[1].chain, 1) == CT06_SAMPLES
        @test size(fit.folds[1].chain, 3) == CT06_CHAINS
        run_id = reused ? preflight.run_id : string(save_fit(fit, db))
        actual_hash = Training.config_hash(fit, db)
        @test actual_hash in (preflight.base_hash, preflight.failed_hash)

        # ====================================================================
        # 7. Unchanged convergence audit, extraction and exact DB parity
        # ====================================================================
        diagnostics = ct_pilot_diagnostics(fit, CT06_THRESHOLDS)
        @test fit.latents isa Models.CountLatents
        @test Models.n_draws(fit.latents) == CT06_SAMPLES * CT06_CHAINS
        @test Set(fit.latents.match_ids) == Set(item.heldout_ids)
        ct_smoke_fit_parity(fit, load_fit(db, run_id))
        grid = ct_smoke_gate("same 12x12 grid-tail diagnostic") do
            ct_smoke_score_grid_gate(fit)
        end
        record = (; candidate = candidate.name, run_id, config_hash = actual_hash, reused,
            samples = CT06_SAMPLES, chains = CT06_CHAINS, warmup = CT06_WARMUP,
            accept_rate = CT06_ACCEPT, seed = CT06_SEED + index,
            diagnostics, source_digest = CT06_SOURCE_DIGEST, core_digest, data_manifest,
            cutoff = string(item.cutoff), fitted_count = length(item.fitted_ids),
            heldout_ids = item.heldout_ids, prior_metrics = item.prior_metrics,
            tape_ms = tape.elapsed * 1000, tape_bytes = tape.allocation,
            tape_gate_passed = tape.performance_ok, db_roundtrip = true,
            grid_status = grid.status, grid_detail = grid.detail,
            production_promoted = false)
        push!(records, record)
        ct_pilot_write_progress(joinpath(CT06_OUTPUT, "pilot_results.json"), records)
        println(candidate.name, ": convergence=", diagnostics.passed,
            " Rhat=", diagnostics.max_rhat, " bulkESS=", diagnostics.min_ess_bulk,
            " tailESS=", diagnostics.min_ess_tail, " divergences=", diagnostics.divergences,
            " run_id=", run_id)
    end

    # ========================================================================
    # 8. Report only — never launch or authorise a production grid
    # ========================================================================
    println("\nPilot finished: ", count(r -> r.diagnostics.passed, records), "/", length(records),
            " pass the unchanged convergence audit. Production was NOT launched.")
    return records
end

@testset "TODO028 longer pilot filtration, sampler shape and persistence" begin
    ct06_run_pilot()
end
