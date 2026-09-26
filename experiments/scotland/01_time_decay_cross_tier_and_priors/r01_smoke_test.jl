# ==============================================================================
# Scottish cross-tier time-decay — Stage-1 seven-gate smoke test
# ==============================================================================
#
# This is a one-fold, five-candidate production preflight for the Stage-1 Poisson
# models. It does not claim predictive superiority and it does not launch the
# 40-fold grid. Every candidate must pass all seven gates before production is
# eligible. Failed convergence blocks promotion but later independent gates still
# run when their inputs exist, so the durable report remains diagnostically useful.
#
# Filtration / comparability contract:
#   * m00 fits Scottish Lower only; m01 fits all-SPFL rows strictly before the
#     same lower-tier held-out bin.
#   * Stage 1 uses the Step-4 two-season history window, not a wider implicit one.
#   * NUTS is exactly two chains × 100 retained draws. This runner never increases
#     that budget to make an audit pass.
#
# Persistence:
#   * canonical recipes are registered and the `configs.config_hash` preflight is
#     checked before sampling;
#   * successful Fits and portfolios round-trip through `mcmc_experiments`;
#   * reports are immutable per execution timestamp and record failures honestly.
#
# Run only on mcmc-beast after this branch has been synchronised:
#   include("experiments/scotland/01_time_decay_cross_tier_and_priors/r01_smoke_test.jl")
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using SHA
using Statistics
using Test
using ThreadPinning

if !isdefined(@__MODULE__, :CrossTierPoissonModel)
    include("l01_cross_tier_loader.jl")
end
if !isdefined(@__MODULE__, :CrossTierSmokeGate)
    include("l02_smoke_helpers.jl")
end
if !isdefined(@__MODULE__, :CrossTierProductionManifest)
    include("l04_production_helpers.jl")
end

# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const CT01_EXPERIMENT = "scotland_cross_tier_time_decay"
const CT01_TARGET_SEASONS = ["24/25", "25/26"]
const CT01_HISTORY_SEASONS = 2
const CT01_SAMPLES = 100
const CT01_CHAINS = 2
const CT01_WARMUP = 100
const CT01_MAX_DEPTH = 10
const CT01_PREPARE_ONLY = lowercase(get(ENV, "CT01_PREPARE_ONLY", "false")) in ("1", "true", "yes")
const CT01_OUTPUT_ROOT = joinpath(@__DIR__, "results", "smoke")
const CT01_TAGS = ["todo028", "stage1", "poisson", "smoke", "two_season_history"]
const CT01_SOURCE_FILES = [
    joinpath(@__DIR__, "l01_cross_tier_loader.jl"),
    joinpath(@__DIR__, "l02_smoke_helpers.jl"),
    @__FILE__,
    joinpath(pwd(), "src", "Data", "fetchers", "segments.jl"),
]
const CT01_SOURCE_DIGEST = bytes2hex(SHA.sha256(join(read.(CT01_SOURCE_FILES, String), "\u001e")))

ct01_sampler() = NUTSConfig(
    n_samples = CT01_SAMPLES,
    n_warmup = CT01_WARMUP,
    n_chains = CT01_CHAINS,
    max_depth = CT01_MAX_DEPTH,
)

ct01_book_spec() = BookSpec(
    markets = Data.MarketConfig([
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
    ]),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = BakerMcHale(),
    exec = ExecutionConfig(
        commission = PerBetCommission(0.02),
        budget = 0.99,
        min_selection_stake = 0.001,
    ),
)

ct01_policy_spec() = PolicySpec(
    trust = FlatTrust(0.25),
    risk = SlateDrawdown(20.0),
    cap = FixedCap(0.25),
    grouping = DailySlate(),
)

# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const CT01_RUN_STAMP = Dates.format(now(), "yyyymmdd_HHMMSS")
const CT01_OUTPUT_DIR = joinpath(CT01_OUTPUT_ROOT, CT01_RUN_STAMP)
mkpath(CT01_OUTPUT_DIR)

println("\n", "="^96)
println(" TODO028 STAGE-1 — CROSS-TIER POISSON SEVEN-GATE SMOKE")
println("="^96)
println("  Retained draws: $CT01_CHAINS chains × $CT01_SAMPLES samples")
println("  History window: $CT01_HISTORY_SEASONS seasons")
println("  Output:         $CT01_OUTPUT_DIR")
println("  Mode:           ", CT01_PREPARE_ONLY ? "prepare-only (no sampling)" : "full smoke")
println("  Source digest:  $CT01_SOURCE_DIGEST")

# %%
# ===================================================================
# 4. Data snapshot and temporal splits
# ===================================================================

function ct01_load_datastores(candidates)
    stores = Dict{DataType, Data.DataStore}()
    for candidate in candidates
        key = typeof(candidate.segment)
        haskey(stores, key) && continue
        stores[key] = Data.load_datastore_cached(candidate.segment; max_age_hours = 100_000)
        println("  $(nameof(key)): $(nrow(stores[key].matches)) matches")
    end
    return stores
end

# %%
# ===================================================================
# 5. Engine / model construction
# ===================================================================

function ct01_candidates()
    candidates = cross_tier_smoke_candidates(
        target_seasons = CT01_TARGET_SEASONS,
        history_seasons = CT01_HISTORY_SEASONS,
        sampler = ct01_sampler(),
        execution = SequentialExecution(),
    )
    names = String[candidate.name for candidate in candidates]
    expected = ["m00_control", "m01_all_spfl_league_offsets",
                "m02_all_spfl_hierarchical_tiers", "m03_prior_structural_offset",
                "m04_prior_market_derived"]
    missing = filter(name -> !(name in names), expected)
    isempty(missing) || println(
        "  Pending loader candidates (not silently passed): ", join(missing, ", "),
        ". This run will report them as BLOCKED before sampling available candidates.")
    length(candidates) + length(missing) == length(expected) || error(
        "loader candidate manifest has duplicate or unexpected names: $(join(names, ", "))")
    return candidates, missing
end

# %%
# ===================================================================
# 6. Feature construction and preflight gates
# ===================================================================

function ct01_one_fold(candidate, ds)
    prepared = cross_tier_smoke_prepare_one_fold(candidate, ds)
    required = (:boundary, :meta, :feature_set, :oos, :cutoff, :fitted_ids,
                :heldout_ids, :prior_metrics)
    all(name -> hasproperty(prepared, name), required) || error(
        "$(candidate.name): loader preparation omitted required smoke fields")
    nrow(prepared.oos) > 0 || error("$(candidate.name): smoke fold has no held-out fixtures")

    fitted_ids = Set(Int.(prepared.fitted_ids))
    heldout_ids = Set(Int.(prepared.heldout_ids))
    heldout_ids == Set(Int.(prepared.oos.match_id)) || error(
        "$(candidate.name): loader held-out IDs differ from OOS fixtures")
    isempty(intersect(fitted_ids, heldout_ids)) || error(
        "$(candidate.name): fitted and held-out match IDs overlap")
    any(ismissing, prepared.oos.match_date) && error(
        "$(candidate.name): held-out fixture has missing kickoff")
    cutoff = Date(prepared.cutoff)
    cutoff == minimum(Date.(prepared.oos.match_date)) || error(
        "$(candidate.name): loader cutoff is not the first held-out kickoff")
    fitted = DataFrames.subset(ds.matches, :match_id => ByRow(in(fitted_ids)))
    nrow(fitted) == length(fitted_ids) || error(
        "$(candidate.name): fitted boundary cannot resolve every match ID")
    any(ismissing, fitted.match_date) && error("$(candidate.name): fitted fixture has missing kickoff")
    all(Date.(fitted.match_date) .< cutoff) || error(
        "$(candidate.name): filtration violation; fitted kickoff must be < held-out cutoff $cutoff")
    all(Int.(prepared.oos.tournament_id) .∈ Ref([56, 57])) || error(
        "$(candidate.name): held-out fixtures must be Scottish Lower only")
    return merge(prepared, (; cutoff, fitted_ids, heldout_ids,
                            n_fitted = nrow(fitted), n_heldout = nrow(prepared.oos)))
end

"Verify lower-likelihood controls share the same fitted rows and A arms add only upper-tier rows."
function ct01_assert_candidate_panels!(prepared_by_name, ds)
    lower_names = ("m00_control", "m03_prior_structural_offset", "m04_prior_market_derived")
    reference = prepared_by_name["m00_control"]
    reference_ids = reference.fitted_ids
    for name in lower_names[2:end]
        prepared_by_name[name].fitted_ids == reference_ids || error(
            "$name fitted IDs differ from m00_control; lower-likelihood controls are not comparable")
        prepared_by_name[name].heldout_ids == reference.heldout_ids || error(
            "$name held-out IDs differ from m00_control")
    end
    lower_fitted = DataFrames.subset(ds.matches, :match_id => ByRow(in(reference_ids)))
    all(Int.(lower_fitted.tournament_id) .∈ Ref([56, 57])) || error(
        "m00/B1/B2 likelihood includes a non-lower-tier fixture")
    for name in ("m01_all_spfl_league_offsets", "m02_all_spfl_hierarchical_tiers")
        candidate = prepared_by_name[name]
        candidate.heldout_ids == reference.heldout_ids || error(
            "$name held-out IDs differ from lower-control panel")
        candidate_fitted = DataFrames.subset(ds.matches, :match_id => ByRow(in(candidate.fitted_ids)))
        tier_by_id = Dict(Int(row.match_id) => Int(row.tournament_id) for row in eachrow(candidate_fitted))
        all_lower_ids = Set(id for id in candidate.fitted_ids if tier_by_id[id] in (56, 57))
        all_lower_ids == reference_ids || error(
            "$name lower-tier fitted IDs do not reproduce m00/B1/B2 training panel")
    end
    return nothing
end

function ct01_fit_config(candidate)
    cfg = candidate.fit_config
    return Training.FitConfig(
        name = cfg.name,
        model = cfg.model,
        splitter = cfg.splitter,
        sampler = cfg.sampler,
        execution = SequentialExecution(),
        tags = [CT01_TAGS; "source_sha256:" * CT01_SOURCE_DIGEST],
        description = "TODO028 Stage-1 one-fold seven-gate smoke for $(candidate.name); " *
                      "source_sha256=$CT01_SOURCE_DIGEST.",
        save_dir = CT01_OUTPUT_DIR,
    )
end

# %%
# ===================================================================
# 7. Checkpoint preparation
# ===================================================================

function ct01_register_preflight!(db, candidate, config, book_spec, policy_spec)
    recipe_candidate = merge(candidate, (; fit_config = config))
    description = "TODO028 Stage-1 pure-Poisson smoke recipe: $(candidate.name)."
    preflight = ct_smoke_register_and_preflight!(db, recipe_candidate, description, CT01_TAGS)
    save_book_spec(db, candidate.name * "_smoke_book", book_spec;
                   description = "Stage-1 smoke 1X2 and O/U 2.5 BookSpec.", tags = CT01_TAGS)
    save_policy_spec(db, candidate.name * "_smoke_policy", policy_spec;
                     description = "Stage-1 smoke 25% trust, 20% slate drawdown and 25% cap.",
                     tags = CT01_TAGS)
    return preflight
end

# %%
# ===================================================================
# 8. Training
# ===================================================================

function ct01_train_one_fold(config, prepared)
    return fit_model(
        config;
        feature_sets = [(prepared.feature_set, prepared.meta)],
        oos_fixtures = Any[prepared.oos],
        thresholds = ConvergenceThresholds(
            max_rhat = 1.05,
            min_ess = 100.0,
            max_divergence_rate = eps(),
            min_bfmi = 0.30,
            max_treedepth_rate = 0.05,
        ),
        checkpoint_dir = joinpath(CT01_OUTPUT_DIR, config.name, "checkpoints"),
        cleanup_checkpoints = false,
        quiet = false,
    )
end

# %%
# ===================================================================
# 9. Convergence diagnostics, inference and persistence gates
# ===================================================================

function ct01_run_candidate(db, candidate, ds, book_spec, policy_spec, prepared)
    started_at = now()
    gates = CrossTierSmokeGate[]
    config = ct01_fit_config(candidate)
    tape_result = nothing
    fit = nothing
    run_id = nothing

    # Hash registration/preflight is deliberately outside a catch-all gate wrapper: failure
    # must make sampling impossible, never merely colour Gate 1 red before Gate 2 proceeds.
    preflight = try
        ct01_register_preflight!(db, candidate, config, book_spec, policy_spec)
    catch err
        push!(gates, CrossTierSmokeGate(
            name = "Preflight — canonical recipe and hash",
            status = :failed,
            detail = sprint(showerror, err, catch_backtrace()),
        ))
        append!(gates, [ct_smoke_abstained("Gate $gate — unavailable after preflight",
                                           "sampling prohibited because preflight failed") for gate in 2:7])
        report = CrossTierSmokeReport(; candidate = candidate.name, started_at, completed_at = now(),
                                      gates, production_blocked = true)
        ct_smoke_write_report(report, CT01_OUTPUT_DIR)
        return report
    end

    # Build and verify correctness outside the performance-status wrapper. A gradient
    # mismatch means no diagnostic NUTS may run; a latency/allocation miss may continue
    # only as explicitly non-promotable diagnostics after preflight succeeded.
    tape_correctness = ct_smoke_gate("Gate 1 — compiled ReverseDiff tape") do
        tape_result = ct_smoke_tape_gate(candidate.model, prepared.feature_set)
        "compiled tape AD correctness passed", (; instructions = tape_result.instructions)
    end
    if tape_correctness.status !== :passed
        push!(gates, tape_correctness)
        append!(gates, [ct_smoke_abstained("Gate $gate — unavailable after tape correctness failure",
                                           "sampling prohibited because compiled-tape correctness failed") for gate in 2:7])
        report = CrossTierSmokeReport(; candidate = candidate.name, started_at, completed_at = now(),
                                      gates, production_blocked = true)
        ct_smoke_write_report(report, CT01_OUTPUT_DIR)
        return report
    end
    push!(gates, ct_smoke_gate("Gate 1 — compiled ReverseDiff tape") do
        detail, metrics = ct_smoke_tape_gate_result(tape_result)
        prior = prepared.prior_metrics
        "$detail; fitted=$(prepared.n_fitted), held-out=$(prepared.n_heldout), " *
        "cutoff=$(prepared.cutoff), fitted_ids=$(join(sort!(collect(prepared.fitted_ids)), ",")), " *
        "heldout_ids=$(join(sort!(collect(prepared.heldout_ids)), ",")), prior_metrics=$(repr(prior))",
        merge(metrics, (; n_fitted = prepared.n_fitted, n_heldout = prepared.n_heldout,
                         cutoff = string(prepared.cutoff), prior_metrics = repr(prior)))
    end)

    if CT01_PREPARE_ONLY
        append!(gates, [ct_smoke_abstained("Gate $gate — prepare-only",
                                           "prepare-only mode stops before sampling") for gate in 2:7])
        report = CrossTierSmokeReport(; candidate = candidate.name, started_at, completed_at = now(),
                                      gates, production_blocked = true)
        ct_smoke_write_report(report, CT01_OUTPUT_DIR)
        return report
    end

    if preflight.existing_run_id === nothing
        push!(gates, ct_smoke_gate("Gate 2 — NUTS 2×100") do
            fit = ct01_train_one_fold(config, prepared)
            detail, metrics = ct_smoke_chain_shape(fit)
            gates[1].status === :passed || (detail *= " (diagnostic-only; Gate 1 performance failed)")
            detail, metrics
        end)
    else
        fit = load_fit(db, preflight.existing_run_id)
        run_id = preflight.existing_run_id
        push!(gates, ct_smoke_gate("Gate 2 — NUTS 2×100") do
            detail, metrics = ct_smoke_chain_shape(fit)
            "reused completed run $run_id — $detail", metrics
        end)
    end

    push!(gates, fit === nothing ?
        ct_smoke_abstained("Gate 3 — six-part convergence audit",
                           "not available: Gate 2 did not produce a Fit") :
        ct_smoke_gate("Gate 3 — six-part convergence audit") do
            ct_smoke_convergence_gate(fit)
        end)

    push!(gates, fit === nothing ?
        ct_smoke_abstained("Gate 4 — CountLatents extraction",
                           "not available: Gate 2 did not produce a Fit") :
        ct_smoke_gate("Gate 4 — CountLatents extraction") do
            ct_smoke_latent_gate(fit)
        end)

    push!(gates, fit === nothing || !(fit.latents isa Models.CountLatents) ?
        ct_smoke_abstained("Gate 5 — score grid and market pricing",
                           "not available: CountLatents were not produced") :
        ct_smoke_gate("Gate 5 — score grid and market pricing") do
            ct_smoke_score_grid_gate(fit)
        end)

    push!(gates, fit === nothing || !(fit.latents isa Models.CountLatents) ?
        ct_smoke_abstained("Gate 6 — save/load Fit parity",
                           "not available: PostgreSQL requires a completed Fit with CountLatents") :
        ct_smoke_gate("Gate 6 — save/load Fit parity") do
            run_id = save_fit(fit, db)
            recovered = load_fit(db, run_id)
                actual_hash = config_hash(fit, db)
            actual_hash in (preflight.base_hash, preflight.failed_hash) || error(
                "save_fit recipe hash $actual_hash was not one of the two preflight hashes")
            ct_smoke_fit_parity(fit, recovered)
            "save_fit/load_fit exact parity", (; run_id = string(run_id), config_hash = actual_hash)
        end)

    push!(gates, run_id === nothing || !fit.diagnostics.passed ?
        ct_smoke_abstained("Gate 7 — portfolio persistence parity",
                           run_id === nothing ? "not available: Gate 6 did not produce a persisted Fit run ID" :
                           "not run: strict convergence failed; portfolio promotion is prohibited") :
        ct_smoke_gate("Gate 7 — portfolio persistence parity") do
            fit.diagnostics.passed || error(
                "portfolio promotion is blocked by failed convergence: " *
                join(fit.diagnostics.failures, " | "))
            result, _, _ = run_portfolio_simulation(
                book_spec, policy_spec, fit, ds.odds, ds;
                bootstrap = false,
                require_converged = true,
                quiet = true,
            )
            portfolio_id = save_portfolio_db(
                result, run_id, db;
                book_spec,
                policy_spec,
                metadata = (; runner = "r01_smoke_test", candidate = candidate.name),
            )
            ct_smoke_portfolio_parity(result, load_portfolio_db(portfolio_id, db))
            "portfolio simulation and persisted artefact exact parity", (; portfolio_id = string(portfolio_id))
        end)

    report = CrossTierSmokeReport(;
        candidate = candidate.name,
        started_at,
        completed_at = now(),
        gates,
        production_blocked = ct_smoke_production_blocked(gates),
    )
    report_stem = ct_smoke_write_report(report, CT01_OUTPUT_DIR)
    report_path = report_stem * ".md"
    if !report.production_blocked
        manifest_path = ct02_write_smoke_promotion_manifest(
            candidate, prepared, ds, report, report_path;
            source_digest = CT01_SOURCE_DIGEST,
            loader_path = joinpath(@__DIR__, "l01_cross_tier_loader.jl"),
            helper_path = joinpath(@__DIR__, "l02_smoke_helpers.jl"),
            output_dir = CT01_OUTPUT_DIR,
        )
        println("  Promotion manifest: $manifest_path")
    end
    return report
end

# %%
# ===================================================================
# 10. Final report
# ===================================================================

function ct01_print_summary(reports)
    println("\n", "="^96)
    println(" STAGE-1 SMOKE SUMMARY")
    println("="^96)
    for report in reports
        status = report.production_blocked ? "BLOCKED" : "ELIGIBLE"
        println("  $(rpad(report.candidate, 38)) $status")
        for gate in report.gates
            println("    $(rpad(gate.name, 45)) ", uppercase(String(gate.status)))
        end
    end
    return nothing
end

function ct01_run_smoke()
    candidates, missing = ct01_candidates()
    isempty(missing) || error("loader must expose all five Stage-1 candidates before smoke sampling")
    all(candidate -> candidate.segment isa Data.ScottishAll, candidates) || error(
        "all candidates must consume one Data.ScottishAll snapshot; likelihood scoping is enforced by fitted IDs")
    datastores = ct01_load_datastores(candidates)
    length(datastores) == 1 || error("smoke must use one shared ScottishAll DataStore snapshot")
    shared_ds = only(values(datastores))
    prepared_by_name = Dict(candidate.name => ct01_one_fold(candidate, shared_ds) for candidate in candidates)
    ct01_assert_candidate_panels!(prepared_by_name, shared_ds)
    db = PostgresStorage(CT01_EXPERIMENT)
    book_spec = ct01_book_spec()
    policy_spec = ct01_policy_spec()
    reports = CrossTierSmokeReport[]

    for candidate in candidates
        println("\n--- $(candidate.name) ---")
        ds = shared_ds
        prepared = prepared_by_name[candidate.name]
        push!(reports, ct01_run_candidate(db, candidate, ds, book_spec, policy_spec, prepared))
    end

    for name in missing
        gates = [ct_smoke_abstained("Gate $gate — unavailable candidate",
                                    "loader has not yet implemented $name") for gate in 1:7]
        report = CrossTierSmokeReport(;
            candidate = name,
            started_at = now(),
            completed_at = now(),
            gates,
            production_blocked = true,
        )
        ct_smoke_write_report(report, CT01_OUTPUT_DIR)
        push!(reports, report)
    end

    ct01_print_summary(reports)
    any(report -> report.production_blocked, reports) && error(
        "Stage-1 production remains blocked: inspect durable reports under $CT01_OUTPUT_DIR.")
    return reports
end

@testset "TODO028 Stage-1 seven-gate smoke" begin
    reports = ct01_run_smoke()
    @test length(reports) == 5
    @test all(report -> length(report.gates) == 7, reports)
    if CT01_PREPARE_ONLY
        @test all(report -> report.production_blocked, reports)
        @test all(report -> report.gates[1].status in (:passed, :failed), reports)
    else
        @test all(report -> !report.production_blocked, reports)
    end
end
