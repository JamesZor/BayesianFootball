# ==============================================================================
# Task 013 loader — MultiScaleGRW × PlayerLineupPillar ablation ladder
# ==============================================================================
#
# Definitions only. `r01_smoke.jl`, `r02_production_grid.jl` and
# `r03_extend_2627.jl` execute.
#
# THE LADDER. Four models, one dynamics component, one splitter, one sampler:
#
#   m00_baseline_grw               GRW                      Poisson
#   m05_wealth_grw                 GRW + wealth             Joint Gamma-Poisson
#   m10_lineup_grw                 GRW + lineup             Poisson
#   m12_joint_hybrid_synergy_grw   GRW + wealth + lineup    Joint Gamma-Poisson
#
# `m10 − m00` isolates the lineup pillar under a Poisson likelihood; `m12 − m05`
# isolates it under the two-arm joint likelihood. Every non-dynamics component is
# the EXACT Experiment 06 recipe (`l60_loader.jl`), so `m12_joint_hybrid_synergy_grw`
# differs from the Gen 4 production `m12_joint_hybrid_synergy` in exactly one
# slot: `TimeDecayDynamics(180.0)` → `MultiScaleGRW()`.
#
# WHY THE SPLITTER IS NOT `CVConfig(window_seasons = 3)`. The work package sketches
# that constructor, but the canonical 40-fold / 710-fixture grid every control was
# scored on is `GroupedCVConfig(history_seasons = 2, dynamics_col = :match_biweek)`
# (`l60_splitter`, `l01_splitter` in Task 007). The GRW's micro step IS the
# match-biweek, so a different splitter would change the walk's time geometry as
# well as the fixture set. Comparability wins.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using DynamicPPL
using ForwardDiff
using LibPQ
using LinearAlgebra
using LogDensityProblems
using MCMCChains
using Printf
using Random
using ReverseDiff
using SHA
using Statistics
using UUIDs

const GPH_PG = BayesianFootball.Models.PreGame
const GPH_FEATURES = BayesianFootball.Features
const GPH_INF = BayesianFootball.Training.Inference

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

"""
    GPHConfig

Every number that governs the study, in one place. Runners read `config.samples`,
never `ENV` directly.

`persist_stride` exists because of the artefact path, not the science. A 43-fold GRW
fit at 1,600 draws serialises to ~290 MB compressed (Task 007, measured); 4,000 draws
would be ~725 MB, whose hex text form exceeds PostgreSQL's 1 GB field limit. The
convergence audit runs on EVERY draw; only the persisted panel — and the latents
reconstructed from it — keep every `persist_stride`-th draw.
"""
Base.@kwdef struct GPHConfig
    experiment::String = "scottish_lower_grw_player_hybrid"
    smoke_experiment::String = "smoke_grw_player"
    save_root::String = joinpath(@__DIR__, "results")

    target_seasons::Vector{String} = ["24/25", "25/26"]
    extension_seasons::Vector{String} = ["24/25", "25/26", "26/27"]
    expected_folds::Int = 40
    expected_extended_folds::Int = 43
    expected_oos::Int = 710

    smoke_folds::Int = 2
    smoke_samples::Int = 100
    smoke_warmup::Int = 50
    smoke_chains::Int = 2

    samples::Int = 1000
    warmup::Int = 500
    chains::Int = 4
    accept_rate::Float64 = 0.80
    max_depth::Int = 10
    max_concurrent_tasks::Int = 16

    persist_stride::Int = 2
    gradient_replays::Int = 200

    # The six-part audit (`ConvergenceThresholds` defaults). The strict R̂ ≤ 1.01 of
    # Task 007 is reported beside it as advisory, not gated.
    max_rhat::Float64 = 1.05
    strict_rhat::Float64 = 1.01
    min_ess::Float64 = 400.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05
end

const GPH_MODEL_NAMES = [
    "m00_baseline_grw",
    "m05_wealth_grw",
    "m10_lineup_grw",
    "m12_joint_hybrid_synergy_grw",
]

const GPH_DESCRIPTIONS = Dict(
    "m00_baseline_grw" =>
        "Poisson baseline with two-speed MultiScaleGRW team attack and defence.",
    "m05_wealth_grw" =>
        "Two-arm joint Gamma-Poisson with MultiScaleGRW and age-adjusted production wealth.",
    "m10_lineup_grw" =>
        "Poisson MultiScaleGRW with shots-RAPM starters and fixed 0.10 bench weight.",
    "m12_joint_hybrid_synergy_grw" =>
        "Gen 4 hybrid with MultiScaleGRW: joint likelihood, production wealth, shots-RAPM lineup.",
)

const GPH_TAGS = [
    "scottish-lower", "24/25", "25/26", "multiscale-grw", "player-lineup",
    "todo013", "reversediff",
]

"""
A persisted benchmark posterior, pinned by immutable UUID.

UUIDs, not names: `m12_joint_hybrid_synergy` resolves to two completed runs in its
namespace (the 43-fold Exp 06 run and a 40-fold 2026-09-10 rerun). The pins below are
the runs the published Exp 06 README and the 2026/27 T−25 report both used.
"""
struct GPHControl
    label::String
    experiment::String
    run_id::UUID
    name::String
    description::String
end

const GPH_CONTROLS = [
    GPHControl("m05_joint_td_raw", "scottish_lower_joint_player_2426",
               UUID("ed541a7c-01e2-447e-a771-783517728d47"), "m05_joint_production_wealth",
               "Team TimeDecay(180) + production wealth + joint likelihood (Exp 06 control)"),
    GPHControl("m12_hybrid_td_raw", "scottish_lower_joint_player_2426",
               UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef"), "m12_joint_hybrid_synergy",
               "Gen 4 production hybrid: TimeDecay(180) + lineup + wealth + joint"),
    GPHControl("m05_joint_grw_raw", "scottish_lower_multiscale_grw_2426",
               UUID("f870dbb7-9df0-4dae-a84a-cf570cf8113e"), "m05_joint_production_wealth_grw",
               "Task 007 team-level GRW + wealth + joint (800+800 ×2, thinned 1-in-4)"),
]

# ==============================================================================
# 2. Components — the Experiment 06 recipes, verbatim except for dynamics
# ==============================================================================

"The two-speed walk at its graduated defaults — identical to Task 007's priors."
gph_dynamics() = MultiScaleGRW()

gph_joint_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

gph_production_wealth() = ProductionWealthCovariate(
    feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
    prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    role = SupremacyRole(),
)

# `fit_on = :history` freezes the ridge fit on each fold's history block, so a target
# fixture never contributes to the ratings that price it. RAPM is never sampled.
gph_shots_pillar() = PlayerLineupPillar(
    feature = GPH_FEATURES.ShotsPlusMinusFeature(
        w_sim = 0.0,
        λ = 1000.0,
        half_life_days = 730.0,
        fit_on = :history,
    ),
    aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10),
    w_att_prior = Normal(0.0, 0.3),
    w_def_prior = Normal(0.0, 0.3),
)

"All four ladder models, in ladder order."
function gph_models()
    m00 = CountModelBuilder(:m00_baseline_grw) |>
        add(GlobalInterception()) |>
        add(gph_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(PoissonObservation()) |>
        build

    m05 = CountModelBuilder(:m05_wealth_grw) |>
        add(GlobalInterception()) |>
        add(gph_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(gph_production_wealth()) |>
        add(gph_joint_observation()) |>
        build

    m10 = CountModelBuilder(:m10_lineup_grw) |>
        add(GlobalInterception()) |>
        add(gph_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(gph_shots_pillar()) |>
        add(PoissonObservation()) |>
        build

    # Same add order as Exp 06 `m12`: pillar, then wealth, then observation.
    m12 = CountModelBuilder(:m12_joint_hybrid_synergy_grw) |>
        add(GlobalInterception()) |>
        add(gph_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(gph_shots_pillar()) |>
        add(gph_production_wealth()) |>
        add(gph_joint_observation()) |>
        build

    return Tuple{String,Any}[
        (GPH_MODEL_NAMES[1], m00),
        (GPH_MODEL_NAMES[2], m05),
        (GPH_MODEL_NAMES[3], m10),
        (GPH_MODEL_NAMES[4], m12),
    ]
end

# ==============================================================================
# 3. Split, sampler, execution, thresholds
# ==============================================================================

"The canonical pooled 56/57 match-biweek walk-forward split over `seasons`."
gph_splitter(seasons::Vector{String}) = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = copy(seasons),
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    end_dynamics = nothing,
    stop_early = true,
)

gph_smoke_sampler(c::GPHConfig) = QueuedNUTSConfig(
    n_samples = c.smoke_samples,
    n_warmup = c.smoke_warmup,
    n_chains = c.smoke_chains,
    accept_rate = c.accept_rate,
    max_depth = c.max_depth,
    show_progress = false,
)

gph_production_sampler(c::GPHConfig) = QueuedNUTSConfig(
    n_samples = c.samples,
    n_warmup = c.warmup,
    n_chains = c.chains,
    accept_rate = c.accept_rate,
    max_depth = c.max_depth,
    show_progress = false,
)

gph_execution(c::GPHConfig) = QueuedExecution(max_concurrent_tasks = c.max_concurrent_tasks)

gph_thresholds(c::GPHConfig) = ConvergenceThresholds(
    max_rhat = c.max_rhat,
    min_ess = c.min_ess,
    max_divergence_rate = c.max_divergence_rate,
    min_bfmi = c.min_bfmi,
    max_treedepth_rate = c.max_treedepth_rate,
)

function gph_fit_configs(c::GPHConfig, models, splitter, sampler;
                         name_suffix::AbstractString = "")
    return Dict(name => FitConfig(
        name = name * name_suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = gph_execution(c),
        tags = copy(GPH_TAGS),
        description = GPH_DESCRIPTIONS[name],
        save_dir = joinpath(c.save_root, name * name_suffix),
    ) for (name, model) in models)
end

gph_load_data() = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)

function gph_database(experiment::AbstractString)
    db = PostgresStorage(experiment)
    ensure_schema!(db)
    return db
end

"Register every canonical component and assembled recipe in `config_registry`."
function gph_register!(db, models, splitter, sampler, configs)
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for (name, model) in models
        model_ids[name] = save_model(db, name, model;
                                     description = GPH_DESCRIPTIONS[name], tags = GPH_TAGS)
        fit_hashes[name] = save_config(db, name * "_fit", configs[name];
                                       description = GPH_DESCRIPTIONS[name] * " Task 013 recipe.",
                                       tags = GPH_TAGS)
    end
    splitter_id = save_splitter(db, "scottish_lower_grw_player_40fold", splitter;
        description = "Pooled 56/57, two history seasons, match-biweek walk-forward over 24/25 and 25/26.",
        tags = GPH_TAGS)
    sampler_id = save_sampler(db, "queued_nuts_4x1000_w500_a080", sampler;
        description = "ReverseDiff queued NUTS: 4 chains, 500 warmup, 1000 retained, target acceptance 0.80.",
        tags = GPH_TAGS)
    return (; model_ids, fit_hashes, splitter_id, sampler_id)
end

# ==============================================================================
# 4. Run-level config truth
# ==============================================================================

"""
    gph_run_hash(db, config) -> String

`save_fit`'s deduplication hash for a recipe, computed before any sampling.

Mirrors `Training.config_hash(fit, storage)` field for field, including its tag
filter, so a match here is a match there.
"""
function gph_run_hash(db, config::FitConfig)
    tags = filter(config.tags) do tag
        !any(prefix -> startswith(tag, prefix), ("time:", "folds_failed:", "latents:"))
    end
    canonical = join((db.experiment_name, config.name,
                      string(config.model), string(config.splitter),
                      string(config.sampler), string(config.execution),
                      join(tags, ""), config.description), "")
    return bytes2hex(SHA.sha256(canonical))
end

"The completed run UUID persisted for exactly this recipe, or `nothing`."
function gph_completed_run(db, config::FitConfig)
    conn = GPH_INF._db_connect(db)
    try
        rows = GPH_INF._db_rows(conn, """
            SELECT r.run_id
            FROM configs AS c
            JOIN runs AS r ON r.run_id = c.config_id
            WHERE c.config_hash = \$1 AND r.status = 'completed'
            LIMIT 1;
        """, (gph_run_hash(db, config),))
        return nrow(rows) == 0 ? nothing : UUID(string(rows.run_id[1]))
    finally
        close(conn)
    end
end

"The newest completed run in `db` carrying `name`, or `nothing`."
function gph_run_by_name(db, name::AbstractString)
    conn = GPH_INF._db_connect(db)
    try
        rows = GPH_INF._db_rows(conn, """
            SELECT run_id FROM runs
            WHERE experiment_name = \$1 AND name = \$2 AND status = 'completed'
            ORDER BY id DESC LIMIT 1;
        """, (db.experiment_name, String(name)))
        return nrow(rows) == 0 ? nothing : UUID(string(rows.run_id[1]))
    finally
        close(conn)
    end
end

# ==============================================================================
# 5. Folds, features, gradient audit
# ==============================================================================

"""
    gph_fold_inputs(ds, splitter, model; limit) -> (; boundaries, feature_sets, oos)

Build each fold's features and held-out fixture frame ONCE.

The same objects feed sampling, latent extraction, and — after thinning — latent
re-extraction, so the fixtures a persisted latent describes are by construction the
fixtures the chain was conditioned to predict.
"""
function gph_fold_inputs(ds, splitter, model; limit::Union{Nothing,Int} = nothing)
    boundaries = Data.create_id_boundaries(ds, splitter)
    selected = limit === nothing ? boundaries : boundaries[1:min(limit, length(boundaries))]
    feature_sets = GPH_FEATURES.create_features(selected, ds, model, splitter)
    oos = Any[Data.get_next_matches(ds, feature_sets[i], splitter)
              for i in eachindex(feature_sets)]
    return (; boundaries = selected, feature_sets, oos)
end

"""
    gph_filtration_report(ds, inputs) -> DataFrame

The filtration facts the runner prints before training, one row per fold.

`train` is the boundary's history block plus the target-season matches already
observed; `oos` is the next match-biweek. The contract asserted here is the one
the style guide requires — no fixture is both conditioned on and priced, and the
last training kickoff precedes the first held-out one.
"""
function gph_filtration_report(ds, inputs)
    date_of = Dict(Int(r.match_id) => Date(r.match_date) for r in eachrow(ds.matches))
    rows = NamedTuple[]
    for (i, fs) in enumerate(inputs.feature_sets)
        b = first(inputs.boundaries[i])
        d = first(fs).data
        train_ids = vcat(b.history_match_ids, b.target_match_ids)
        oos = inputs.oos[i]
        oos_ids = oos === nothing ? Int[] : Int.(oos.match_id)
        overlap = length(intersect(Set(train_ids), Set(oos_ids)))
        overlap == 0 || error("fold $i: $overlap fixtures are both trained on and held out")
        last_train = maximum(date_of[m] for m in train_ids)
        first_oos = isempty(oos_ids) ? missing : minimum(date_of[m] for m in oos_ids)
        push!(rows, (; fold = i,
                       n_train = length(train_ids),
                       n_oos = length(oos_ids),
                       n_history = Int(d[:n_history_steps]),
                       n_target = Int(d[:n_target_steps]),
                       n_teams = Int(d[:n_teams]),
                       last_train,
                       first_oos,
                       ordered = ismissing(first_oos) || last_train < first_oos))
    end
    return DataFrame(rows)
end

gph_relative_error(left, right) = norm(left - right) / max(norm(left), norm(right), 1.0)

"""
    gph_gradient_audit(model, feature_set; replays, seed) -> NamedTuple

Compile one fold's ReverseDiff tape and check it four ways:

1. log density and compiled gradient are finite;
2. compiled tape == fresh ReverseDiff (≤ 1e-8 relative);
3. ReverseDiff == ForwardDiff (≤ 1e-6 relative);
4. the compiled tape is still right at three perturbed points (≤ 1e-8) — a tape that
   recorded a data-dependent branch passes (2) and fails here.

Allocation per warmed gradient call and best-of-N gradient time are measured, not
gated: Task 007 established the installed ReverseDiff stack allocates ~35–130 KB per
call even for the TimeDecay control, so a literal zero-allocation gate would fail
every model in the repository.
"""
function gph_gradient_audit(model, feature_set; replays::Int = 200, seed::Int = 20260911)
    turing_model = GPH_PG.build_turing_model(model, first(feature_set))
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    objective = values -> LogDensityProblems.logdensity(density, values)

    log_density = objective(θ)
    isfinite(log_density) || error("gradient audit: non-finite log density $log_density")

    raw_tape = ReverseDiff.GradientTape(objective, θ)
    tape = ReverseDiff.compile(raw_tape)
    gradient = similar(θ)
    ReverseDiff.gradient!(gradient, tape, θ)
    all(isfinite, gradient) || error("gradient audit: compiled gradient is non-finite")

    fresh = ReverseDiff.gradient(objective, θ)
    forward = ForwardDiff.gradient(objective, θ)
    compiled_fresh_error = gph_relative_error(gradient, fresh)
    compiled_forward_error = gph_relative_error(gradient, forward)
    compiled_fresh_error <= 1.0e-8 || error(
        "compiled/fresh ReverseDiff relative error $compiled_fresh_error > 1e-8")
    compiled_forward_error <= 1.0e-6 || error(
        "ReverseDiff/ForwardDiff relative error $compiled_forward_error > 1e-6")

    worst_perturbed_error = 0.0
    coordinates = collect(eachindex(θ))
    for delta in (0.001, -0.002, 0.003)
        perturbed = θ .+ delta .* sin.(coordinates)
        compiled = similar(perturbed)
        ReverseDiff.gradient!(compiled, tape, perturbed)
        err = gph_relative_error(ReverseDiff.gradient(objective, perturbed), compiled)
        worst_perturbed_error = max(worst_perturbed_error, err)
    end
    worst_perturbed_error <= 1.0e-8 || error(
        "compiled tape changes under perturbation: relative error $worst_perturbed_error")

    for _ in 1:20
        ReverseDiff.gradient!(gradient, tape, θ)
    end
    allocated_bytes = @allocated ReverseDiff.gradient!(gradient, tape, θ)

    best_ns = typemax(UInt64)
    for _ in 1:replays
        started = time_ns()
        ReverseDiff.gradient!(gradient, tape, θ)
        best_ns = min(best_ns, time_ns() - started)
    end

    return (;
        n_parameters = length(θ),
        tape_instructions = length(raw_tape.tape),
        gradient_ms = Float64(best_ns) / 1.0e6,
        allocated_bytes,
        compiled_fresh_error,
        compiled_forward_error,
        worst_perturbed_error,
        log_density,
    )
end

# ==============================================================================
# 6. Sampling, thinning, persistence
# ==============================================================================

"Sample every fold in `inputs` under `fit_config`, checkpointing each fold as it lands."
function gph_sample(fit_config::FitConfig, inputs, c::GPHConfig; checkpoint_dir = nothing)
    return fit_model(fit_config;
                     feature_sets = inputs.feature_sets,
                     oos_fixtures = inputs.oos,
                     thresholds = gph_thresholds(c),
                     checkpoint_dir = checkpoint_dir,
                     cleanup_checkpoints = false,
                     quiet = false)
end

function _gph_thin_chain(chain::Chains, stride::Int)
    return Chains(
        parent(chain.value)[1:stride:end, :, :],
        names(chain),
        Dict(:parameters => names(chain, :parameters),
             :internals => names(chain, :internals));
        start = 1,
    )
end

"""
    gph_thin_for_persistence(fit, inputs, stride) -> Fit

The same run, carrying every `stride`-th retained draw and latents rebuilt from those
draws.

`fit.diagnostics` is carried over untouched: the audit was computed on every draw and
is the stronger statement. Latents are re-extracted rather than subsampled so that
`load_fit` reconstructs a container whose draws come from the persisted chain — the
same invariant Task 007's extension path relied on, and the one `extend_fit` needs
when it appends folds sampled at the thinned draw count.
"""
function gph_thin_for_persistence(fit, inputs, stride::Int)
    stride >= 1 || error("persist stride must be ≥ 1; got $stride")
    stride == 1 && return fit
    folds = FoldFit[FoldFit(f.fold, _gph_thin_chain(f.chain, stride), f.meta)
                    for f in fit.folds]
    latents, note = extract_run_latents(fit.config.model, folds, inputs.oos, inputs.feature_sets)
    latents === nothing && error("thinned latent extraction failed: $note")
    Set(latents.match_ids) == Set(fit.latents.match_ids) || error(
        "thinned latents cover a different fixture set than the full-chain latents")
    return Fit(fit.config, folds, latents, fit.diagnostics, fit.metadata, fit.save_path)
end

"""
    gph_latent_audit(fit) -> NamedTuple

Posterior rate draws must be finite and strictly positive, and every fixture's mean
and variance must be finite. A single NaN here becomes a NaN log-loss for the whole
run; a zero variance means a fixture priced from a point mass.
"""
function gph_latent_audit(fit)
    lat = fit.latents
    lat isa CountLatents || error("latents are $(typeof(lat)), expected CountLatents")
    allunique(lat.match_ids) || error("duplicate OOS match IDs in latents")
    for (side, draws) in (("home", lat.lambda_home), ("away", lat.lambda_away))
        all(isfinite, draws) || error("non-finite λ_$side draws")
        all(>(0.0), draws) || error("non-positive λ_$side draws")
    end
    μ_h = vec(mean(lat.lambda_home; dims = 2))
    μ_a = vec(mean(lat.lambda_away; dims = 2))
    v_h = vec(var(lat.lambda_home; dims = 2))
    v_a = vec(var(lat.lambda_away; dims = 2))
    all(isfinite, vcat(μ_h, μ_a, v_h, v_a)) || error("non-finite latent mean or variance")
    all(>(0.0), vcat(v_h, v_a)) || error("a fixture has zero posterior rate variance")
    return (; n_matches = length(lat.match_ids), n_draws = size(lat.lambda_home, 2),
              mean_lambda_h = mean(μ_h), mean_lambda_a = mean(μ_a),
              min_sd = sqrt(minimum(vcat(v_h, v_a))),
              max_sd = sqrt(maximum(vcat(v_h, v_a))))
end

"""
    gph_save_and_verify(db, fit) -> UUID

Persist `fit` and prove the artefact reloads to the same object.

Refuses when the recipe is already persisted: `save_fit` would otherwise return the
OLD run's UUID without writing anything, and the round-trip below would then compare
this fit against a different run's draws.
"""
function gph_save_and_verify(db, fit)
    existing = gph_completed_run(db, fit.config)
    existing === nothing || error(
        "recipe $(fit.config.name) is already persisted as run $existing; " *
        "load it instead of saving a second copy")
    run_id = save_fit(fit, db)
    reloaded = load_fit(db, run_id)
    length(reloaded.folds) == length(fit.folds) || error("round-trip fold count differs")
    reloaded.latents.match_ids == fit.latents.match_ids || error("round-trip match IDs differ")
    reloaded.latents.lambda_home == fit.latents.lambda_home || error("round-trip λ_home differs")
    reloaded.latents.lambda_away == fit.latents.lambda_away || error("round-trip λ_away differs")
    reloaded.diagnostics.max_rhat == fit.diagnostics.max_rhat || error("round-trip R̂ differs")
    for (a, b) in zip(fit.folds, reloaded.folds)
        parent(a.chain.value) == parent(b.chain.value) || error(
            "round-trip chain differs on fold $(a.fold)")
    end
    return run_id
end

function gph_assert_coverage(name::AbstractString, fit; folds::Int, oos::Int)
    length(fit.folds) == folds || error("$name has $(length(fit.folds)) folds; expected $folds")
    n_matches(fit.latents) == oos || error(
        "$name has $(n_matches(fit.latents)) OOS fixtures; expected $oos")
    allunique(fit.latents.match_ids) || error("$name OOS match IDs are not unique")
    return nothing
end

"One convergence row for a report. `strict_rhat_pass` is advisory."
function gph_convergence_row(name::AbstractString, fit, c::GPHConfig; run_id = nothing)
    d = fit.diagnostics
    return (;
        model = String(name),
        folds = length(fit.folds),
        oos = fit.latents === nothing ? 0 : n_matches(fit.latents),
        draws = fit.latents === nothing ? 0 : n_draws(fit.latents),
        max_rhat = d.max_rhat,
        worst_rhat_fold = d.worst_rhat_fold,
        min_ess_bulk = d.min_ess_bulk,
        min_ess_tail = d.min_ess_tail,
        n_divergent = d.n_divergent,
        n_transitions = d.n_transitions,
        divergence_rate = d.divergence_rate,
        treedepth_rate = d.treedepth_rate,
        min_bfmi = d.min_bfmi,
        passed = d.passed,
        strict_rhat_pass = d.max_rhat <= c.strict_rhat,
        failures = join(d.failures, "; "),
        wall_min = fit.metadata.elapsed_seconds / 60,
        run_id = run_id === nothing ? "" : string(run_id),
    )
end

# ==============================================================================
# 7. Extension to 2026/27
# ==============================================================================

"""
    gph_extend!(db, run_id, ds, c) -> Fit

Sample only the fold positions `fold_results` does not hold, under the 24/25 → 26/27
splitter, and update the run in place (same UUID).

`extend_fit` matches the existing per-chain draw count (the thinned 500), so the
appended folds are sampled at 500 warmup + 500 retained and every fold in the
extended container carries the same 2,000 draws.
"""
function gph_extend!(db, run_id::UUID, ds, c::GPHConfig)
    return extend_fit(db, string(run_id), ds;
                      splitter = gph_splitter(c.extension_seasons),
                      execution = gph_execution(c))
end

# ==============================================================================
# 8. Report formatting
# ==============================================================================

gph_num(v; digits = 4) = (v isa Real && isfinite(v)) ? @sprintf("%.*f", digits, v) : "n/a"
gph_signed(v; digits = 5) = (v isa Real && isfinite(v)) ? @sprintf("%+.*f", digits, v) : "n/a"

"A GitHub-flavoured markdown table from a DataFrame, with per-column formatters."
function gph_markdown_table(df::AbstractDataFrame; formats = Dict{Symbol,Function}())
    cols = propertynames(df)
    io = IOBuffer()
    println(io, "| ", join(string.(cols), " | "), " |")
    println(io, "|", join([eltype(df[!, c]) <: Real ? "---:" : "---" for c in cols], "|"), "|")
    for row in eachrow(df)
        cells = String[]
        for col in cols
            v = row[col]
            f = get(formats, col, nothing)
            push!(cells, f !== nothing ? f(v) :
                         v isa AbstractFloat ? gph_num(v) :
                         v isa Missing ? "—" : string(v))
        end
        println(io, "| ", join(cells, " | "), " |")
    end
    return String(take!(io))
end
