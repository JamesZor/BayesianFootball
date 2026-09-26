# ==============================================================================
# Scottish cross-tier time-decay — Stage-1 production grid
# ==============================================================================
#
# This runner promotes only smoke-approved recipes. It deliberately does not soften
# a gate, reconstruct a missing smoke result, trim fixtures, or sample in prepare-only
# mode. Run the production branch only on mcmc-beast after the smoke manifests exist.
# ============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using SHA
using Test
using ThreadPinning

if !isdefined(@__MODULE__, :CrossTierPoissonModel)
    include("l01_cross_tier_loader.jl")
end
if !isdefined(@__MODULE__, :CrossTierProductionManifest)
    include("l04_production_helpers.jl")
end

# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const CT02_EXPERIMENT = "scotland_cross_tier_time_decay"
const CT02_TARGET_SEASONS = ["24/25", "25/26"]
const CT02_HISTORY_SEASONS = 2
const CT02_SAMPLES = 1_000
const CT02_CHAINS = 4
const CT02_MCMC_WARMUP = 500
const CT02_BIWEEK_WARMUP = 0
# User correction on 2026-09-24 supersedes WORK_PACKAGE_PROMPT Step 4.
const CT02_ACCEPT_RATE = 0.65
const CT02_MAX_DEPTH = 10
const CT02_QUEUE_TASKS = 16
const CT02_EXPECTED_FOLDS = 40
const CT02_EXPECTED_OOS_FIXTURES = 710
const CT02_PREPARE_ONLY = lowercase(get(ENV, "CT02_PREPARE_ONLY", "true")) in ("1", "true", "yes")
const CT02_OUTPUT_ROOT = joinpath(@__DIR__, "results", "production")
const CT02_SMOKE_MANIFEST_DIR = get(ENV, "CT02_SMOKE_MANIFEST_DIR", "")
const CT02_LOADER_PATH = joinpath(@__DIR__, "l01_cross_tier_loader.jl")
const CT02_HELPER_PATH = joinpath(@__DIR__, "l02_smoke_helpers.jl")
const CT02_TAGS = ["todo028", "stage1", "poisson", "production", "two_season_history"]
const CT02_CANDIDATES = ["m00_control", "m01_all_spfl_league_offsets",
                         "m02_all_spfl_hierarchical_tiers", "m03_prior_structural_offset",
                         "m04_prior_market_derived"]

ct02_sampler() = QueuedNUTSConfig(
    n_samples = CT02_SAMPLES,
    n_chains = CT02_CHAINS,
    n_warmup = CT02_MCMC_WARMUP,
    accept_rate = CT02_ACCEPT_RATE,
    max_depth = CT02_MAX_DEPTH,
    show_progress = false,
)

# %%
# ===================================================================
# 3. Runtime and immutable output directory
# ===================================================================

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const CT02_RUN_STAMP = Dates.format(now(), "yyyymmdd_HHMMSS")
const CT02_OUTPUT_DIR = joinpath(CT02_OUTPUT_ROOT, CT02_RUN_STAMP)
const CT02_LOADER_DIGEST = ct02_file_digest(CT02_LOADER_PATH)
const CT02_HELPER_DIGEST = ct02_file_digest(CT02_HELPER_PATH)

println("\n", "="^96)
println(" TODO028 STAGE-1 — CROSS-TIER POISSON PRODUCTION GRID")
println("="^96)
println("  Queue:          $(CT02_EXPECTED_FOLDS) folds × $CT02_CHAINS chains = $(CT02_EXPECTED_FOLDS * CT02_CHAINS) tasks")
println("  Concurrency:    $CT02_QUEUE_TASKS native queued tasks; BLAS threads = 1")
println("  Retained draws: $CT02_CHAINS chains × $CT02_SAMPLES samples; MCMC adaptation = $CT02_MCMC_WARMUP; biweek warmup = $CT02_BIWEEK_WARMUP")
println("  Mode:           ", CT02_PREPARE_ONLY ? "prepare-only (no sampling)" : "production")

# %%
# ===================================================================
# 4. ScottishAll snapshot and temporal splits
# ===================================================================

function ct02_load_snapshot()
    ds = Data.load_datastore_cached(Data.ScottishAll(); max_age_hours = 100_000)
    data_manifest = ct02_data_manifest(ds)
    println("  ScottishAll snapshot: $(data_manifest.match_count) matches, $(data_manifest.odds_count) odds rows")
    return ds, data_manifest
end

function ct02_candidates()
    candidates = cross_tier_smoke_candidates(
        target_seasons = CT02_TARGET_SEASONS,
        history_seasons = CT02_HISTORY_SEASONS,
        sampler = ct02_sampler(),
        execution = QueuedExecution(max_concurrent_tasks = CT02_QUEUE_TASKS),
    )
    names = String[candidate.name for candidate in candidates]
    names == CT02_CANDIDATES || error(
        "production requires all five frozen candidates in canonical order; got $(join(names, ", "))")
    return candidates
end

function ct02_prepare_candidate(candidate, ds)
    splits = Data.create_id_boundaries(ds, candidate.splitter)
    length(splits) == CT02_EXPECTED_FOLDS || error(
        "$(candidate.name): splitter produced $(length(splits)) folds; expected $CT02_EXPECTED_FOLDS")
    oos = Any[Data.get_next_matches(ds, splits[i][2], candidate.splitter)
              for i in eachindex(splits)]
    all(frame -> nrow(frame) > 0, oos) || error("$(candidate.name): an OOS fold is empty")
    all(frame -> all(Int.(frame.tournament_id) .∈ Ref([56, 57])), oos) || error(
        "$(candidate.name): OOS panel includes a non-lower-tier fixture")
    lower_training_ids = Vector{Vector{Int}}(undef, length(splits))
    for i in eachindex(splits)
        boundary, _ = splits[i]
        fitted_ids = Int.(vcat(boundary.history_match_ids, boundary.target_match_ids))
        heldout_ids = Int.(oos[i].match_id)
        isempty(intersect(fitted_ids, heldout_ids)) || error(
            "$(candidate.name) fold $i: fitted and held-out IDs overlap")
        cutoff = minimum(Date.(oos[i].match_date))
        fitted = DataFrames.subset(ds.matches, :match_id => ByRow(in(Set(fitted_ids))))
        nrow(fitted) == length(fitted_ids) || error(
            "$(candidate.name) fold $i: cannot resolve every fitted match ID")
        any(ismissing, fitted.match_date) && error("$(candidate.name) fold $i: fitted kickoff is missing")
        all(Date.(fitted.match_date) .< cutoff) || error(
            "$(candidate.name) fold $i: fitted kickoff is not strictly before OOS cutoff $cutoff")
        lower_training_ids[i] = sort!(Int.(fitted.match_id[Int.(fitted.tournament_id) .∈ Ref([56, 57])]))
    end
    feature_sets = Features.create_features(splits, ds, candidate.model, candidate.splitter)
    ids = reduce(vcat, Int.(frame.match_id) for frame in oos)
    length(unique(ids)) == length(ids) || error("$(candidate.name): OOS calendar repeats match IDs")
    length(ids) == CT02_EXPECTED_OOS_FIXTURES || error(
        "$(candidate.name): OOS panel has $(length(ids)) fixtures; expected $CT02_EXPECTED_OOS_FIXTURES — refusing to trim")
    return (; splits, feature_sets, oos, oos_ids = sort!(ids), lower_training_ids)
end

# %%
# ===================================================================
# 5. Smoke promotion and all-candidate comparability gates
# ===================================================================

function ct02_manifest_path(candidate_name::AbstractString)
    isempty(CT02_SMOKE_MANIFEST_DIR) && error(
        "CT02_SMOKE_MANIFEST_DIR must name the one immutable, smoke-approved manifest directory")
    isdir(CT02_SMOKE_MANIFEST_DIR) || error(
        "CT02_SMOKE_MANIFEST_DIR does not exist: $CT02_SMOKE_MANIFEST_DIR")
    path = joinpath(CT02_SMOKE_MANIFEST_DIR,
                    String(candidate_name) * "_production_manifest.json")
    isfile(path) || error("$(candidate_name): required smoke promotion manifest is absent: $path")
    return path
end

function ct02_validate_promotion(candidate, data_manifest)
    manifest = ct02_load_smoke_manifest(ct02_manifest_path(candidate.name))
    ct02_validate_smoke_manifest(manifest, candidate, CT02_LOADER_DIGEST,
                                 CT02_HELPER_DIGEST, data_manifest)
    return manifest
end

function ct02_comparability_gate!(prepared_by_name, manifests)
    reference = prepared_by_name["m00_control"].oos_ids
    for name in CT02_CANDIDATES
        manifest = manifests[name]
        prepared = prepared_by_name[name]
        prepared.oos_ids == reference || error(
            "$name: production OOS IDs differ from the m00 lower calendar")
        production_first_fold = sort!(Int.(prepared.oos[1].match_id))
        manifest.heldout_ids == production_first_fold || error(
            "$name: smoke first-fold OOS manifest does not match production fold 1")
        prepared.lower_training_ids == prepared_by_name["m00_control"].lower_training_ids || error(
            "$name: lower-tier fitted likelihood IDs are not paired to m00 on every fold")
    end
    return nothing
end

# %%
# ===================================================================
# 6. Canonical recipes, deduplication and checkpoint preparation
# ===================================================================

function ct02_fit_config(candidate, source_digest, core_recipe_digest)
    config = Training.FitConfig(
        name = "scotland_cross_tier_time_decay_" * candidate.name,
        model = candidate.model,
        splitter = candidate.splitter,
        sampler = ct02_sampler(),
        execution = QueuedExecution(max_concurrent_tasks = CT02_QUEUE_TASKS),
        tags = [CT02_TAGS; "source_sha256:" * source_digest;
                "core_recipe_sha256:" * core_recipe_digest],
        description = "TODO028 Stage-1 40-fold production pure-Poisson recipe for $(candidate.name).",
        save_dir = CT02_OUTPUT_DIR,
    )
    ct02_validate_production_execution(config)
    return config
end

function ct02_checkpoint_dir(db, config)
    return joinpath(CT02_OUTPUT_ROOT, "checkpoints", ct02_checkpoint_key(db, config))
end

# %%
# ===================================================================
# 7. Training — native QueuedExecution only
# ===================================================================

function ct02_train(db, config, prepared)
    return fit_model(
        config;
        feature_sets = prepared.feature_sets,
        oos_fixtures = prepared.oos,
        thresholds = ConvergenceThresholds(),
        checkpoint_dir = ct02_checkpoint_dir(db, config),
        cleanup_checkpoints = false,
        quiet = false,
    )
end

# %%
# ===================================================================
# 8. Convergence and immutable persistence
# ===================================================================

function ct02_promote_or_reject!(db, candidate, config, prepared)
    preflight = ct02_register_and_preflight!(
        db, candidate, config,
        "TODO028 Stage-1 production recipe; smoke-approved exact source/model/data manifest.",
        CT02_TAGS,
    )
    if preflight.existing_run_id !== nothing
        preflight.prior_convergence_failed && error(
            "$(candidate.name): prior exact production recipe failed convergence in run " *
            "$(preflight.existing_run_id); refusing to resample or promote it")
        fit = load_fit(db, preflight.existing_run_id)
        fit.diagnostics.passed || error("$(candidate.name): reused run is not converged")
        return (; fit, run_id = preflight.existing_run_id, reused = true)
    end
    CT02_PREPARE_ONLY && return (; fit = nothing, run_id = nothing, reused = false)
    fit = ct02_train(db, config, prepared)
    if !fit.diagnostics.passed
        run_id = ct02_save_blocked_fit!(fit, db)
        error("$(candidate.name): convergence failed; persisted blocked diagnostic Fit $run_id")
    end
    fit.latents isa Models.CountLatents || error("$(candidate.name): CountLatents extraction failed")
    return (; fit, run_id = save_fit(fit, db), reused = false)
end

# %%
# ===================================================================
# 9. Final report
# ===================================================================

function ct02_run_production()
    !CT02_PREPARE_ONLY && Threads.nthreads() >= CT02_QUEUE_TASKS || CT02_PREPARE_ONLY || error(
        "production requires at least $CT02_QUEUE_TASKS Julia threads; got $(Threads.nthreads())")
    ds, data_manifest = ct02_load_snapshot()
    candidates = ct02_candidates()
    manifests = Dict(candidate.name => ct02_validate_promotion(candidate, data_manifest)
                     for candidate in candidates)
    prepared = Dict(candidate.name => ct02_prepare_candidate(candidate, ds) for candidate in candidates)
    ct02_comparability_gate!(prepared, manifests)
    source_digest = ct02_sha256(join((CT02_LOADER_DIGEST, CT02_HELPER_DIGEST), "\u001e"))
    db = PostgresStorage(CT02_EXPERIMENT)
    outcomes = Dict{String, Any}()
    outcome_path = joinpath(CT02_OUTPUT_DIR, "production_outcomes.json")
    for candidate in candidates
        core_digest = ct02_core_recipe_digest(candidate, data_manifest)
        config = ct02_fit_config(candidate, source_digest, core_digest)
        try
            result = ct02_promote_or_reject!(db, candidate, config, prepared[candidate.name])
            outcomes[candidate.name] = (; status = result.reused ? "reused" :
                                        (CT02_PREPARE_ONLY ? "prepared" : "completed"),
                                        run_id = result.run_id,
                                        config_hash = ct02_checkpoint_key(db, config),
                                        core_recipe_digest = core_digest)
        catch err
            outcomes[candidate.name] = (; status = "blocked", run_id = nothing,
                                        config_hash = ct02_checkpoint_key(db, config),
                                        core_recipe_digest = core_digest,
                                        detail = sprint(showerror, err))
            ct02_write_production_outcome(outcome_path, outcomes)
            rethrow()
        end
        ct02_write_production_outcome(outcome_path, outcomes)
    end
    CT02_PREPARE_ONLY && println("Production preparation passed; no MCMC was launched.")
    return outcomes
end

@testset "TODO028 production sampler API contract" begin
    config = Training.FitConfig(
        name = "ct02_api_contract",
        model = CTM00Control(),
        splitter = CrossTierPriorCV(),
        sampler = ct02_sampler(),
        execution = QueuedExecution(max_concurrent_tasks = CT02_QUEUE_TASKS),
    )
    @test ct02_validate_production_execution(config) === nothing
    @test config.sampler.n_samples == 1_000
    @test config.sampler.n_chains == 4
    @test config.sampler.n_warmup == 500
    @test config.splitter.warmup_period == 0
    @test config.sampler.accept_rate == 0.65
    @test config.execution.max_concurrent_tasks == 16
end

if get(ENV, "CT02_RUN_PRODUCTION", "false") in ("1", "true", "yes")
    ct02_run_production()
else
    println("Set CT02_RUN_PRODUCTION=true after smoke promotion to run production preparation/sampling.")
end
