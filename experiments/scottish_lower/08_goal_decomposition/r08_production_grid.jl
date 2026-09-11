# ==============================================================================
# r08 — Experiment 08 production goal-decomposition walk-forward grid
# ==============================================================================
#
# WHAT THIS IS. The four-candidate, canonical Scottish Lower 24/25–25/26
# walk-forward grid.  Candidate differences are confined to the decomposed
# intensity model.  The splitter, data snapshot, Betfair comparator, sampler,
# native queued execution, convergence gates, and portfolio recipe are shared.
#
# WHAT THIS IS NOT. It does not launch sampling by default and it does not claim
# the stale 40-fold/710-fixture headline is reproduced.  Prepare-only freezes the
# actual all-model intersection after it verifies fold construction and filtration.
# A parent process must explicitly authorise a host-idle launch with
# `L08_RUN_GRID=true`.
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
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 16 || error(
    "Experiment 08 production queue requires 16 physical-core Julia threads on mcmc-beast; got $(Threads.nthreads())")

include(joinpath(@__DIR__, "l08_workflow.jl"))
include(joinpath(@__DIR__, "l08_incident_data.jl"))
include(joinpath(@__DIR__, "l08_decomposed_models.jl"))
include(joinpath(@__DIR__, "l08_model_checks.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R08_RUN_GRID = lowercase(get(ENV, "L08_RUN_GRID", "false")) in ("1", "true", "yes")
const R08_OUTPUT_DIR = joinpath(@__DIR__, "results")
const R08_SOURCE_FILES = [
    joinpath(@__DIR__, "l08_workflow.jl"),
    joinpath(@__DIR__, "l08_decomposed_models.jl"),
    joinpath(@__DIR__, "l08_incident_data.jl"),
    joinpath(@__DIR__, "l08_model_checks.jl"),
    @__FILE__,
]

# %%
# ==============================================================================
# 3. Data snapshot, registry, and immutable recipes
# ==============================================================================
# The cache is an input.  Its match-ID/date/season hash enters the manifest before
# model construction, and the same explicit registry is passed to every candidate.
l08_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(L08_EXPERIMENT)
ensure_schema!(db)
registry = l08_registry(ds, db; output_dir = R08_OUTPUT_DIR, source_files = R08_SOURCE_FILES)
splitter = l08_splitter()
incident_registry, incident_snapshot_hash = GoalDecompositionIncidentData.load_registry(R08_OUTPUT_DIR)
models = l08_models(incident_registry, incident_snapshot_hash;
                    registry_hash = incident_snapshot_hash)
length(models) == length(L08_CANDIDATE_NAMES) || error(
    "models API returned $(length(models)) candidates; require $(length(L08_CANDIDATE_NAMES))")
Set(String(entry.name) for entry in models) == Set(L08_CANDIDATE_NAMES) || error(
    "models API candidate names do not match the registered Experiment 08 family")
models = [(String(entry.name), entry.model) for entry in models]
configs = Dict(name => l08_fit_config(name, model, splitter) for (name, model) in models)
book = l08_book_spec()
policy = l08_policy_spec()
registered = l08_register!(registry, models, splitter, L08_SAMPLER, configs, book, policy)

println("\n", "="^110)
println(" EXPERIMENT 08 · GOAL DECOMPOSITION · CANONICAL WALK-FORWARD GRID")
println("="^110)
println("  mode      : ", R08_RUN_GRID ? "SAMPLING AUTHORISED" : "PREPARE ONLY")
println("  threads   : ", Threads.nthreads(), " physical-core queue slots")
println("  sampler   : 4 chains × 1,000 warmup × 1,000 retained · acceptance 0.95 (fixed pre-smoke)")
println("  execution : native QueuedExecution() only; no second scheduler")
println("  snapshot  : ", registry.snapshot_hash)
println("  database  : ", db)
println("  started   : ", Dates.now())

# %%
# ==============================================================================
# 4. All-fold feature, count, and filtration preflight
# ==============================================================================
# `l08_decomposed_models.jl` owns the local registry-backed feature materialiser.
# It must reject a component count whose incident timestamp/identity is not visible
# under each fold cutoff; the model only receives concrete Float64 vectors.
boundaries = Data.create_id_boundaries(ds, splitter)
length(boundaries) == L08_EXPECTED_FOLDS || error(
    "splitter made $(length(boundaries)) boundaries rather than canonical $(L08_EXPECTED_FOLDS)")

feature_sets = Dict{String,Any}()
oos_by_name = Dict{String,Any}()
preflight_rows = NamedTuple[]
for (name, model) in models
    # The data loader supplies the native FeatureConfig.  It must retain total-score
    # rows and route quarantined component rows through the model's binary mask.
    original_features = Features.create_features(boundaries, ds, model, splitter)
    oos = [Data.get_next_matches(ds, original_features[i], splitter) for i in eachindex(original_features)]
    features = l08_declare_prediction_teams(original_features, oos)
    actual_oos = l08_assert_prepare!(name, model, features, oos, splitter)
    all(String(first(feature).data[:goal_decomposition_data_hash]) == model.data_hash
        for feature in features) || error("$name FeatureSets do not carry the frozen incident snapshot hash")
    feature_sets[name] = features
    oos_by_name[name] = oos
    push!(preflight_rows, (
        model = name,
        folds = length(features),
        oos_fixtures = actual_oos,
        incident_rows = nrow(incident_registry.incidents),
        filtration_refusals = nrow(incident_registry.quarantines),
        queue_tasks = length(features) * L08_CHAINS,
    ))
end

ids_by_name = Dict(name => reduce(vcat, Int.(frame.match_id) for frame in oos)
                   for (name, oos) in oos_by_name)

# The canonical empirical baseline fixture set is the genuine Gen-3 m05 40-fold
# run.  Candidate coverage must equal it exactly: an intersection would silently
# discard every candidate failure and turn missing OOS latents into a score gain.
baseline_db = PostgresStorage("scottish_lower_joint_2426")
baseline_run = UUID("5eff755c-3591-48d1-a2cc-5fc2744ddf88")
baseline_fit = load_fit(baseline_db, baseline_run)
length(baseline_fit.folds) == L08_EXPECTED_FOLDS || error("Gen-3 m05 baseline is not the genuine 40-fold fit")
n_matches(baseline_fit.latents) == L08_EXPECTED_OOS || error("Gen-3 m05 baseline is not the genuine 710-fixture fit")
canonical_ids = sort!(Int.(baseline_fit.latents.match_ids))
length(unique(canonical_ids)) == L08_EXPECTED_OOS || error("Gen-3 m05 baseline has duplicate latent match IDs")
for (name, ids) in ids_by_name
    Set(ids) == Set(canonical_ids) || error(
        "$name OOS fixture coverage differs from the canonical genuine Gen-3 m05 set; " *
        "refuse to drop mismatches through an intersection")
end

manifest_path = l08_write_manifest!(registry;
    stage = R08_RUN_GRID ? "production_authorised" : "prepare",
    extra = Dict(
        "candidate_names" => collect(first.(models)),
        "actual_oos_fixture_count" => length(canonical_ids),
        "canonical_match_ids" => canonical_ids,
        "preflight" => [Dict(string(k) => v for (k, v) in pairs(row)) for row in preflight_rows],
        "incident_registry_snapshot_hash" => incident_snapshot_hash,
        "config_hashes" => registered.fit_hashes,
    ))
println("  preflight : ", length(boundaries), " folds × ", L08_CHAINS,
        " chains × ", length(models), " models = ", length(boundaries) * L08_CHAINS * length(models), " native tasks")
println("  coverage  : ", length(canonical_ids), " exact common OOS fixtures")
println("  manifest  : ", manifest_path)

# %%
# ==============================================================================
# 5. Persisted-recipe preflight and training
# ==============================================================================
run_ids = Dict{String,Any}()
if !R08_RUN_GRID
    println("\nPREPARE ONLY passed.  No MCMC was launched.")
    for (name, _) in models
        completed = l08_completed_run_id(db, registered.fit_hashes[name])
        println("  ", rpad(name, 38), " completed exact recipe: ", something(completed, "none"))
    end
else
    lowercase(get(ENV, "L08_STRICT_SMOKE_ALL4", "false")) in ("1", "true", "yes") || error(
        "production blocked: run r08_smoke.jl until all four candidates pass, then set L08_STRICT_SMOKE_ALL4=true deliberately")
    for (name, model) in models
        completed = l08_completed_run_id(db, registered.fit_hashes[name])
        if completed !== nothing
            println("\nSKIP $name: completed exact config hash at $completed")
            run_ids[name] = completed
            continue
        end
        println("\n", "-"^110)
        println(" GRID $name · ", Dates.now())
        println("-"^110)
        config = configs[name]
        fit = fit_model(config;
            feature_sets = feature_sets[name],
            oos_fixtures = oos_by_name[name],
            thresholds = L08_THRESHOLDS,
            checkpoint_dir = joinpath(config.save_dir, "checkpoints"),
            cleanup_checkpoints = false,
            quiet = false)
        length(fit.folds) == L08_EXPECTED_FOLDS || error("$name returned incomplete folds")
        l08_assert_fit_coverage(name, fit, canonical_ids)
        l08_assert_promotion(name, fit.diagnostics)
        run_id = save_fit(fit, db)
        reloaded = load_fit(db, run_id)
        reloaded.latents.match_ids == fit.latents.match_ids || error("$name fit latent IDs failed round-trip")
        run_ids[name] = run_id
        println("  persisted run: ", run_id)
    end
end

# %%
# ==============================================================================
# 6. Final report
# ==============================================================================
println("\n", "="^110)
println(" PREPARED CANDIDATES")
println("="^110)
for row in preflight_rows
    @printf(" %-38s folds %2d | OOS %4d | incidents %5d | refusals %4d | tasks %3d\n",
        row.model, row.folds, row.oos_fixtures, row.incident_rows,
        row.filtration_refusals, row.queue_tasks)
end
println("Canonical fixture set: ", length(canonical_ids), " matches")
println("Finished: ", Dates.now())
