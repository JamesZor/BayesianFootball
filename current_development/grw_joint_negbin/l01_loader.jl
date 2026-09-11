# ==============================================================================
# Task 014 loader — JointGammaNegBinObservation ablation ladder
# ==============================================================================
#
# Definitions only. `r01_smoke.jl`, `r02_production_grid.jl`, `r04_evaluate.jl` and
# `r05_portfolio.jl` execute.
#
# THE LADDER. Four models. Every component except the GOALS DENSITY is the Task 013
# recipe verbatim, so each rung differs from its control in exactly one slot:
#
#   m00_baseline_grw_negbin           GRW                   NegBin
#   m05_wealth_grw_negbin             GRW + wealth          Joint Gamma-NegBin
#   m10_lineup_grw_negbin             GRW + lineup          NegBin
#   m12_joint_hybrid_synergy_negbin   GRW + wealth + lineup Joint Gamma-NegBin
#
#   control                           same, with Poisson / Joint Gamma-Poisson
#
# WHAT THE CONTRAST BUYS. `m05_negbin − m05` and `m12_negbin − m12` isolate the
# negative binomial under the two-arm joint likelihood; `m00_negbin − m00` and
# `m10_negbin − m10` isolate it under a single-arm one. Because the Gamma arm, the
# `obs.ν` / `obs.log_κ` block and their priors are shared by construction (one
# `_joint_gamma_poisson_params` submodel serves both observations), a difference in
# these contrasts is the goal density and nothing else.
#
# THE HONEST PRIOR. Experiment 02 measured r̂ ≈ 26.0–26.5 on this league — about 5%
# excess variance — and Δ LogLoss ≈ +0.0001 on 1X2. This study is not a rerun of that.
# It asks whether the tail mass a NegBin moves (more at 0, less at 1–2, more at 4+)
# pays on the markets that READ the tail: Over/Under 1.5/2.5/3.5/4.5 and BTTS. The
# 1X2 column is carried as a control, not as the finding.
#
# WHY THE SPLITTER IS NOT `CVConfig(window_seasons = 3)`. Same reason as Task 013: the
# canonical 40-fold / 710-fixture grid every control was scored on is
# `GroupedCVConfig(history_seasons = 2, dynamics_col = :match_biweek)`, and the GRW's
# micro step IS the match-biweek. Comparability wins.
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

const GJN_PG = BayesianFootball.Models.PreGame
const GJN_BUILDER = BayesianFootball.Models.PreGame.Builder
const GJN_FEATURES = BayesianFootball.Features
const GJN_INF = BayesianFootball.Training.Inference

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

"""
    GJNConfig

Every number that governs the study, in one place. Runners read `config.samples`,
never `ENV` directly.

`persist_stride` exists because of the artefact path, not the science. The convergence
audit runs on EVERY draw; only the persisted panel — and the latents reconstructed
from it — keep every `persist_stride`-th draw. Task 013 measured a 40-fold GRW fit at
4,000 draws as too large for PostgreSQL's 1 GB field limit in hex text form.
"""
Base.@kwdef struct GJNConfig
    experiment::String = "scottish_lower_grw_joint_negbin"
    smoke_experiment::String = "smoke_grw_joint_negbin"
    control_experiment::String = "scottish_lower_grw_player_hybrid"
    save_root::String = joinpath(@__DIR__, "results")

    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos::Int = 710

    smoke_folds::Int = 2
    # Task 013 established that the work-package smoke budget (2 x (50 + 100)) fails
    # R̂ and ESS as a BUDGET ARTEFACT, not as a model defect, and that the gate is only
    # informative at the production sampler. This gate therefore runs at production.
    smoke_samples::Int = 1000
    smoke_warmup::Int = 500
    smoke_chains::Int = 4

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

const GJN_MODEL_NAMES = [
    "m00_baseline_grw_negbin",
    "m05_wealth_grw_negbin",
    "m10_lineup_grw_negbin",
    "m12_joint_hybrid_synergy_negbin",
]

const GJN_DESCRIPTIONS = Dict(
    "m00_baseline_grw_negbin" =>
        "NegBin baseline with two-speed MultiScaleGRW team attack and defence.",
    "m05_wealth_grw_negbin" =>
        "Two-arm joint Gamma-NegBin with MultiScaleGRW and age-adjusted production wealth.",
    "m10_lineup_grw_negbin" =>
        "NegBin MultiScaleGRW with shots-RAPM starters and fixed 0.10 bench weight.",
    "m12_joint_hybrid_synergy_negbin" =>
        "Gen 4 hybrid with MultiScaleGRW and a NegBin goals arm: joint likelihood, production wealth, shots-RAPM lineup.",
)

const GJN_TAGS = [
    "scottish-lower", "24/25", "25/26", "multiscale-grw", "player-lineup",
    "negbin", "joint-gamma-negbin", "todo014", "reversediff",
]

"""
A persisted benchmark posterior, pinned by immutable UUID.

UUIDs, not names: a name can resolve to more than one completed run in its namespace.
These four are the Task 013 runs published in `current_development/grw_player_hybrid/README.md`
§3 — the direct Poisson-arm counterpart of each NegBin rung.

COVERAGE CAVEAT. `r03_extend_2627.jl` extended all four IN PLACE (same UUIDs) to 43
folds / 769 fixtures. This study's runs are 40 folds / 710. Every paired comparison in
`r04_evaluate.jl` therefore intersects on match IDs before scoring, and asserts the
intersection is the expected 710.
"""
struct GJNControl
    label::String
    experiment::String
    run_id::UUID
    name::String
    description::String
end

const GJN_CONTROLS = [
    GJNControl("m00_poisson", "scottish_lower_grw_player_hybrid",
               UUID("158d2a80-7ea3-4d6c-b3ab-be62bcf1bc11"), "m00_baseline_grw",
               "Task 013 Poisson baseline: GRW, no covariates"),
    GJNControl("m05_poisson", "scottish_lower_grw_player_hybrid",
               UUID("b0961bc4-c40c-4dbe-9c05-57df7ae0839e"), "m05_wealth_grw",
               "Task 013 joint Gamma-Poisson: GRW + production wealth"),
    GJNControl("m10_poisson", "scottish_lower_grw_player_hybrid",
               UUID("b13c8fb9-ce34-4210-aa3f-9d2ed493c286"), "m10_lineup_grw",
               "Task 013 Poisson: GRW + shots-RAPM lineup"),
    GJNControl("m12_poisson", "scottish_lower_grw_player_hybrid",
               UUID("3a9a4c7e-378b-45d0-a2d2-c8b69b46786b"), "m12_joint_hybrid_synergy_grw",
               "Task 013 joint Gamma-Poisson hybrid: GRW + lineup + wealth"),
]

"NegBin rung → its Task 013 Poisson control, by label. The four paired contrasts."
const GJN_PAIRS = [
    ("m00_baseline_grw_negbin",         "m00_poisson"),
    ("m05_wealth_grw_negbin",           "m05_poisson"),
    ("m10_lineup_grw_negbin",           "m10_poisson"),
    ("m12_joint_hybrid_synergy_negbin", "m12_poisson"),
]

# ==============================================================================
# 2. Components — the Task 013 recipes, verbatim except for the goals density
# ==============================================================================

"The two-speed walk at its graduated defaults — identical to Task 007 and Task 013."
gjn_dynamics() = MultiScaleGRW()

"""
The dispersion block, stated once so all four rungs share one prior.

`log r ~ Normal(3.1, 0.4)` is `GlobalDispersion`'s own default and the prior
Experiment 02 sampled under; it puts ~90% of the mass on `r ∈ [10, 47]`, i.e. mild
overdispersion. Reusing it is what makes `r̂` here comparable with that study's.
"""
gjn_dispersion() = GlobalDispersion(log_r = Normal(3.1, 0.4))

"The single-arm NegBin goals density — the Poisson rungs' counterpart."
gjn_negbin_observation() = NegativeBinomialObservation(dispersion = gjn_dispersion())

"""
The two-arm joint observation with a NegBin goals arm.

Every field except `dispersion` is `gph_joint_observation()` from Task 013 verbatim —
the same `MatchProxyXGFeature(k = 25.0, fallback = :none)`, the same shape and
log-kappa priors. That is the whole design of the contrast.
"""
gjn_joint_negbin_observation() = JointGammaNegBinObservation(
    dispersion = gjn_dispersion(),
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

"The Task 013 Poisson-arm joint observation, for the parity and control builds."
gjn_joint_poisson_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

gjn_production_wealth() = ProductionWealthCovariate(
    feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
    prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    role = SupremacyRole(),
)

# `fit_on = :history` freezes the ridge fit on each fold's history block, so a target
# fixture never contributes to the ratings that price it. RAPM is never sampled.
gjn_shots_pillar() = PlayerLineupPillar(
    feature = GJN_FEATURES.ShotsPlusMinusFeature(
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
function gjn_models()
    m00 = CountModelBuilder(:m00_baseline_grw_negbin) |>
        add(GlobalInterception()) |>
        add(gjn_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(gjn_negbin_observation()) |>
        build

    m05 = CountModelBuilder(:m05_wealth_grw_negbin) |>
        add(GlobalInterception()) |>
        add(gjn_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(gjn_production_wealth()) |>
        add(gjn_joint_negbin_observation()) |>
        build

    m10 = CountModelBuilder(:m10_lineup_grw_negbin) |>
        add(GlobalInterception()) |>
        add(gjn_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(gjn_shots_pillar()) |>
        add(gjn_negbin_observation()) |>
        build

    # Same add order as Task 013 `m12`: pillar, then wealth, then observation.
    m12 = CountModelBuilder(:m12_joint_hybrid_synergy_negbin) |>
        add(GlobalInterception()) |>
        add(gjn_dynamics()) |>
        add(GlobalHomeAdvantage()) |>
        add(gjn_shots_pillar()) |>
        add(gjn_production_wealth()) |>
        add(gjn_joint_negbin_observation()) |>
        build

    return Tuple{String,Any}[
        (GJN_MODEL_NAMES[1], m00),
        (GJN_MODEL_NAMES[2], m05),
        (GJN_MODEL_NAMES[3], m10),
        (GJN_MODEL_NAMES[4], m12),
    ]
end

# ==============================================================================
# 3. Split, sampler, execution, thresholds
# ==============================================================================

"The canonical pooled 56/57 match-biweek walk-forward split over `seasons`."
gjn_splitter(seasons::Vector{String}) = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = copy(seasons),
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    end_dynamics = nothing,
    stop_early = true,
)

gjn_smoke_sampler(c::GJNConfig) = QueuedNUTSConfig(
    n_samples = c.smoke_samples,
    n_warmup = c.smoke_warmup,
    n_chains = c.smoke_chains,
    accept_rate = c.accept_rate,
    max_depth = c.max_depth,
    show_progress = false,
)

gjn_production_sampler(c::GJNConfig) = QueuedNUTSConfig(
    n_samples = c.samples,
    n_warmup = c.warmup,
    n_chains = c.chains,
    accept_rate = c.accept_rate,
    max_depth = c.max_depth,
    show_progress = false,
)

gjn_execution(c::GJNConfig) = QueuedExecution(max_concurrent_tasks = c.max_concurrent_tasks)

gjn_thresholds(c::GJNConfig) = ConvergenceThresholds(
    max_rhat = c.max_rhat,
    min_ess = c.min_ess,
    max_divergence_rate = c.max_divergence_rate,
    min_bfmi = c.min_bfmi,
    max_treedepth_rate = c.max_treedepth_rate,
)

function gjn_fit_configs(c::GJNConfig, models, splitter, sampler;
                         name_suffix::AbstractString = "")
    return Dict(name => FitConfig(
        name = name * name_suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = gjn_execution(c),
        tags = copy(GJN_TAGS),
        description = GJN_DESCRIPTIONS[name],
        save_dir = joinpath(c.save_root, name * name_suffix),
    ) for (name, model) in models)
end

gjn_load_data() = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)

function gjn_database(experiment::AbstractString)
    db = PostgresStorage(experiment)
    ensure_schema!(db)
    return db
end

"Register every canonical component and assembled recipe in `config_registry`."
function gjn_register!(db, models, splitter, sampler, configs)
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for (name, model) in models
        model_ids[name] = save_model(db, name, model;
                                     description = GJN_DESCRIPTIONS[name], tags = GJN_TAGS)
        fit_hashes[name] = save_config(db, name * "_fit", configs[name];
                                       description = GJN_DESCRIPTIONS[name] * " Task 014 recipe.",
                                       tags = GJN_TAGS)
    end
    splitter_id = save_splitter(db, "scottish_lower_grw_negbin_40fold", splitter;
        description = "Pooled 56/57, two history seasons, match-biweek walk-forward over 24/25 and 25/26.",
        tags = GJN_TAGS)
    sampler_id = save_sampler(db, "queued_nuts_4x1000_w500_a080", sampler;
        description = "ReverseDiff queued NUTS: 4 chains, 500 warmup, 1000 retained, target acceptance 0.80.",
        tags = GJN_TAGS)
    return (; model_ids, fit_hashes, splitter_id, sampler_id)
end

# ==============================================================================
# 4. Run-level config truth
# ==============================================================================

"""
    gjn_run_hash(db, config) -> String

`save_fit`'s deduplication hash for a recipe, computed before any sampling.

Mirrors `Training.config_hash(fit, storage)` field for field, including its tag
filter, so a match here is a match there.
"""
function gjn_run_hash(db, config::FitConfig)
    tags = filter(config.tags) do tag
        !any(prefix -> startswith(tag, prefix), ("time:", "folds_failed:", "latents:"))
    end
    canonical = join((db.experiment_name, config.name,
                      string(config.model), string(config.splitter),
                      string(config.sampler), string(config.execution),
                      join(tags, ""), config.description), "")
    return bytes2hex(SHA.sha256(canonical))
end

"The completed run UUID persisted for exactly this recipe, or `nothing`."
function gjn_completed_run(db, config::FitConfig)
    conn = GJN_INF._db_connect(db)
    try
        rows = GJN_INF._db_rows(conn, """
            SELECT r.run_id
            FROM configs AS c
            JOIN runs AS r ON r.run_id = c.config_id
            WHERE c.config_hash = \$1 AND r.status = 'completed'
            LIMIT 1;
        """, (gjn_run_hash(db, config),))
        return nrow(rows) == 0 ? nothing : UUID(string(rows.run_id[1]))
    finally
        close(conn)
    end
end

"The newest completed run in `db` carrying `name`, or `nothing`."
function gjn_run_by_name(db, name::AbstractString)
    conn = GJN_INF._db_connect(db)
    try
        rows = GJN_INF._db_rows(conn, """
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
# 5. Folds, features, gradient audit, likelihood parity
# ==============================================================================

"""
    gjn_fold_inputs(ds, splitter, model; limit) -> (; boundaries, feature_sets, oos)

Build each fold's features and held-out fixture frame ONCE.

The same objects feed sampling, latent extraction, and — after thinning — latent
re-extraction, so the fixtures a persisted latent describes are by construction the
fixtures the chain was conditioned to predict.
"""
function gjn_fold_inputs(ds, splitter, model; limit::Union{Nothing,Int} = nothing)
    boundaries = Data.create_id_boundaries(ds, splitter)
    selected = limit === nothing ? boundaries : boundaries[1:min(limit, length(boundaries))]
    feature_sets = GJN_FEATURES.create_features(selected, ds, model, splitter)
    oos = Any[Data.get_next_matches(ds, feature_sets[i], splitter)
              for i in eachindex(feature_sets)]
    return (; boundaries = selected, feature_sets, oos)
end

"""
    gjn_filtration_report(ds, inputs) -> DataFrame

The filtration facts the runner prints before training, one row per fold.

The contract asserted here is the one the style guide requires — no fixture is both
conditioned on and priced, and the last training kickoff precedes the first held-out one.
"""
function gjn_filtration_report(ds, inputs)
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

gjn_relative_error(left, right) = norm(left - right) / max(norm(left), norm(right), 1.0)

"""
    gjn_gradient_audit(model, feature_set; replays, seed) -> NamedTuple

Compile one fold's ReverseDiff tape and check it four ways:

1. log density and compiled gradient are finite;
2. compiled tape == fresh ReverseDiff (≤ 1e-8 relative);
3. ReverseDiff == ForwardDiff (≤ 1e-6 relative);
4. the compiled tape is still right at three perturbed points (≤ 1e-8) — a tape that
   recorded a data-dependent branch passes (2) and fails here.

Allocation per warmed gradient call and best-of-N gradient time are measured, not
gated: Task 007 established the installed ReverseDiff stack allocates ~35–180 KB per
call even for the TimeDecay control, so a literal zero-allocation gate would fail
every model in the repository. The work package's "zero heap allocations inside the
gradient tape" is reported against that measured baseline rather than asserted.
"""
function gjn_gradient_audit(model, feature_set; replays::Int = 200, seed::Int = 20260911)
    turing_model = GJN_PG.build_turing_model(model, first(feature_set))
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
    compiled_fresh_error = gjn_relative_error(gradient, fresh)
    compiled_forward_error = gjn_relative_error(gradient, forward)
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
        err = gjn_relative_error(ReverseDiff.gradient(objective, perturbed), compiled)
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

"""
    gjn_parity_models() -> Vector{Tuple{String,Any}}

Two TimeDecay builds of the joint observation — Poisson arm and NegBin arm — for the
`equations.jl` log-joint parity check.

WHY TIMEDECAY AND NOT GRW. `cb_equation_data` refuses any dynamics but
`TimeDecayDynamics`: the reference is a second implementation of the log-joint written
from the equations, and nobody has written the GRW state-space half of it. That is a
pre-existing limit of the reference, not of this component.

It still checks the thing that matters here. The observation block is the ONLY part of
the model this task changed, `_observe` receives `η_h`/`η_a` as opaque vectors and
cannot see which dynamics produced them, and the reference's `cb_loglik` is dispatched
on the observation alone. A parity pass on TimeDecay is therefore a parity pass on the
new likelihood; what it does not cover is the GRW predictor, which Task 007 and Task
013 already checked and which this task did not touch.
"""
function gjn_parity_models()
    base(name, obs) = CountModelBuilder(name) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(gjn_production_wealth()) |>
        add(obs) |>
        build
    return Tuple{String,Any}[
        ("joint_gamma_poisson_td", base(:parity_joint_poisson, gjn_joint_poisson_observation())),
        ("joint_gamma_negbin_td",  base(:parity_joint_negbin,  gjn_joint_negbin_observation())),
        ("negbin_td",              base(:parity_negbin,        gjn_negbin_observation())),
    ]
end

"""
    gjn_parity_check(model, feature_set; perturbations, seed) -> NamedTuple

Engine log density vs the `equations.jl` reference log-joint, at a prior draw and at
`perturbations` displaced points.

This is the correctness gate that matters most for Task 014. `engine.jl` hand-expands

    log NegBin(y; r, κ·μ) = log Γ(y+r) − log Γ(r) − log Γ(y+1)
                            + r·(log r − log(r+λ)) + y·(log λ − log(r+λ))

for the tape's sake, and the two ways that expansion can be wrong — κ leaking into the
Gamma arm, or `r` written against `η` instead of `ζ = η + log κ` — both still sample
cleanly and both still look like a posterior. Neither survives this comparison, because
the reference builds `NegativeBinomial` and `Gamma` objects and calls `logpdf`.

COMPARED THROUGH `LogDensityFunction`, not `getlogjoint`. The engine's observation term
arrives by `Turing.@addlogprob!`, and the reference is written in the model's own
unlinked space; `LogDensityProblems.logdensity(density, θ)` is the quantity that means
the same thing on both sides, and is what the Experiment 06 parity harness
(`r64_assert_reference_parity`) compares. The displaced points matter as much as the
draw: agreeing at one point can happen by accident of a symmetric error, and agreeing
along a direction cannot.
"""
function gjn_parity_check(model, feature_set; perturbations = (0.01, -0.02, 0.035),
                          seed::Int = 20260914)
    # `inputs.feature_sets[i]` is a (FeatureSet, SplitMetaData) tuple; the engine wants the first.
    fs = first(feature_set)
    turing_model = GJN_PG.build_turing_model(model, fs)
    data = GJN_BUILDER.cb_equation_data(model, fs)

    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    f = x -> LogDensityProblems.logdensity(density, x)

    engine_values = Float64[]
    reference_values = Float64[]
    worst_abs = 0.0
    worst_rel = 0.0

    points = Vector{Float64}[θ]
    for δ in perturbations
        push!(points, θ .+ δ .* cos.(collect(eachindex(θ))))
    end

    for point in points
        vi = DynamicPPL.unflatten(varinfo, point)
        params = GJN_BUILDER.cb_params_from_varinfo(model, vi)
        engine = f(point)
        reference = GJN_BUILDER.cb_logjoint(model, params, data)
        isfinite(engine) && isfinite(reference) ||
            error("parity: non-finite log density (engine $engine, reference $reference)")
        push!(engine_values, engine)
        push!(reference_values, reference)
        worst_abs = max(worst_abs, abs(engine - reference))
        worst_rel = max(worst_rel, abs(engine - reference) / max(abs(engine), 1.0))
    end

    return (; n_points = length(points), worst_abs, worst_rel,
              engine = engine_values, reference = reference_values)
end

# ==============================================================================
# 6. Sampling, thinning, persistence
# ==============================================================================

"Sample every fold in `inputs` under `fit_config`, checkpointing each fold as it lands."
function gjn_sample(fit_config::FitConfig, inputs, c::GJNConfig; checkpoint_dir = nothing)
    return fit_model(fit_config;
                     feature_sets = inputs.feature_sets,
                     oos_fixtures = inputs.oos,
                     thresholds = gjn_thresholds(c),
                     checkpoint_dir = checkpoint_dir,
                     cleanup_checkpoints = false,
                     quiet = false)
end

function _gjn_thin_chain(chain::Chains, stride::Int)
    return Chains(
        parent(chain.value)[1:stride:end, :, :],
        names(chain),
        Dict(:parameters => names(chain, :parameters),
             :internals => names(chain, :internals));
        start = 1,
    )
end

"""
    gjn_thin_for_persistence(fit, inputs, stride) -> Fit

The same run, carrying every `stride`-th retained draw and latents rebuilt from those
draws.

`fit.diagnostics` is carried over untouched: the audit was computed on every draw and
is the stronger statement. Latents are re-extracted rather than subsampled so that
`load_fit` reconstructs a container whose draws come from the persisted chain.
"""
function gjn_thin_for_persistence(fit, inputs, stride::Int)
    stride >= 1 || error("persist stride must be ≥ 1; got $stride")
    stride == 1 && return fit
    folds = FoldFit[FoldFit(f.fold, _gjn_thin_chain(f.chain, stride), f.meta)
                    for f in fit.folds]
    latents, note = extract_run_latents(fit.config.model, folds, inputs.oos, inputs.feature_sets)
    latents === nothing && error("thinned latent extraction failed: $note")
    Set(latents.match_ids) == Set(fit.latents.match_ids) || error(
        "thinned latents cover a different fixture set than the full-chain latents")
    return Fit(fit.config, folds, latents, fit.diagnostics, fit.metadata, fit.save_path)
end

"""
    gjn_latent_audit(fit) -> NamedTuple

Posterior rate draws must be finite and strictly positive, and every fixture's mean and
variance must be finite. A single NaN here becomes a NaN log-loss for the whole run.

EXTRA FOR THIS TASK. A NegBin-family container also carries `observation_params.r_h` /
`r_a`, and the score grid divides by `r + λ`. An `r` that reached 0 or Inf would give a
silently degenerate grid rather than an error, so it is checked here, and reported: `r`
IS the finding this study is about.
"""
function gjn_latent_audit(fit)
    lat = fit.latents
    lat isa CountLatents || error("latents are $(typeof(lat)), expected CountLatents")
    allunique(lat.match_ids) || error("duplicate OOS match IDs in latents")
    for (side, draws) in (("home", lat.λ_home), ("away", lat.λ_away))
        all(isfinite, draws) || error("non-finite λ_$side draws")
        all(>(0.0), draws) || error("non-positive λ_$side draws")
    end
    μ_h = vec(mean(lat.λ_home; dims = 2))
    μ_a = vec(mean(lat.λ_away; dims = 2))
    v_h = vec(var(lat.λ_home; dims = 2))
    v_a = vec(var(lat.λ_away; dims = 2))
    all(isfinite, vcat(μ_h, μ_a, v_h, v_a)) || error("non-finite latent mean or variance")
    all(>(0.0), vcat(v_h, v_a)) || error("a fixture has zero posterior rate variance")

    obs = lat.observation_params
    obs === nothing && error(
        "a NegBin ladder run produced a Poisson latent container (observation_params === nothing); " *
        "the observation did not route to the NegBin score grid")
    for (side, draws) in (("r_h", obs.r_h), ("r_a", obs.r_a))
        all(isfinite, draws) || error("non-finite $side draws")
        all(>(0.0), draws) || error("non-positive $side draws")
    end

    return (; n_matches = length(lat.match_ids), n_draws = size(lat.λ_home, 2),
              mean_lambda_h = mean(μ_h), mean_lambda_a = mean(μ_a),
              min_sd = sqrt(minimum(vcat(v_h, v_a))),
              max_sd = sqrt(maximum(vcat(v_h, v_a))),
              mean_r = mean(vcat(vec(obs.r_h), vec(obs.r_a))),
              min_r = minimum(vcat(vec(obs.r_h), vec(obs.r_a))),
              max_r = maximum(vcat(vec(obs.r_h), vec(obs.r_a))))
end

"""
    gjn_market_probabilities(grid) -> NamedTuple

The market partitions of one fixture's `(12 × 12 × draws)` score tensor, posterior-averaged.

1X2, every totals line and BTTS are three partitions of ONE tensor — that is the property
the whole pricing stack rests on, and computing them here the same way keeps this gate
honest about what it is checking. `mass` is the tensor's own total, which is
`P(H ≤ 11)·P(A ≤ 11)` rather than 1: the NegBin grid leaves truncation mass on the floor
(`kernels.jl` §3 records that deliberately), so a `mass` slightly below 1 is correct and a
`mass` far below 1 means the tail has escaped the grid.
"""
function gjn_market_probabilities(grid::Array{Float64,3})
    n, _, draws = size(grid)
    home = away = draw = 0.0
    over15 = over25 = over35 = over45 = 0.0
    btts = 0.0
    mass = 0.0
    for k in 1:draws, j in 1:n, i in 1:n
        p = grid[i, j, k]
        mass += p
        h, a = i - 1, j - 1
        h > a ? (home += p) : h < a ? (away += p) : (draw += p)
        total = h + a
        total > 1 && (over15 += p)
        total > 2 && (over25 += p)
        total > 3 && (over35 += p)
        total > 4 && (over45 += p)
        (h > 0 && a > 0) && (btts += p)
    end
    inv = 1.0 / draws
    return (; mass = mass * inv,
              home = home * inv, draw = draw * inv, away = away * inv,
              over15 = over15 * inv, over25 = over25 * inv,
              over35 = over35 * inv, over45 = over45 * inv,
              btts = btts * inv)
end

"""
    gjn_grid_gate(fit; n_fixtures) -> DataFrame

Prove the dispersion actually reaches the pricing tensor.

A NegBin rung that silently priced on the double-Poisson grid would pass every other gate
in this file: it would sample, converge, extract finite latents and round-trip through
Postgres. It would simply not be the model the study claims to have run. So for each of
the first `n_fixtures` held-out fixtures this builds the model's OWN grid and, beside it,
the double-Poisson grid at the SAME `λ` draws, and reports the market-by-market gap.

Every `d_*` must be non-trivial somewhere, and `d_btts` must be strictly NEGATIVE: a
negative binomial at a fixed mean adds mass at zero on each side, so P(both score) can only
fall. A `d_btts` at machine zero means `r_h`/`r_a` never reached `compute_score_grid!`.

WHY BTTS IS THE DISCRIMINATOR AND A TOTALS LINE IS NOT. On a totals line the extra mass at
zero pushes the total DOWN while the fatter right tail pushes it UP, and the two nearly
cancel — measured at `r̂ ≈ 30` on this league, |Δ| is 0.0131 on BTTS, 0.0055 on O/U 2.5 and
0.0001 on O/U 3.5. BTTS reads the extra zeros unopposed and is the only partition that does.

O/U 1.5 is carried alongside 2.5 because the production Option B basket stakes BOTH — Under
2.5 at full trust and Over 1.5 at `1/1.4` — so those two lines are where this component can
actually reach a stake.

The 1X2 column is the control: the same mechanism that moves the totals markets is expected
to leave the result market nearly untouched, and seeing that here — before any proper score —
is what makes the eventual evaluation legible.
"""
function gjn_grid_gate(fit; n_fixtures::Int = 6)
    lat = fit.latents
    lat.observation_params === nothing && error(
        "grid gate: latents carry no observation_params, so this run prices on the " *
        "double-Poisson grid — the NegBin routing is broken")

    poisson_twin = CountLatents(lat.match_ids, lat.λ_home, lat.λ_away, nothing)
    ws = GridWorkspace()
    S_negbin = alloc_score_grid(lat)
    S_poisson = alloc_score_grid(poisson_twin)

    rows = NamedTuple[]
    for i in 1:min(n_fixtures, n_matches(lat))
        compute_score_grid!(S_negbin, ws, lat, i)
        compute_score_grid!(S_poisson, ws, poisson_twin, i)
        nb = gjn_market_probabilities(S_negbin)
        po = gjn_market_probabilities(S_poisson)
        push!(rows, (; fixture = lat.match_ids[i],
                       lambda_h = mean(view(lat.λ_home, i, :)),
                       lambda_a = mean(view(lat.λ_away, i, :)),
                       r = mean(view(lat.observation_params.r_h, i, :)),
                       mass = nb.mass,
                       p_home = nb.home, p_draw = nb.draw, p_away = nb.away,
                       d_home = nb.home - po.home,
                       d_over15 = nb.over15 - po.over15,
                       d_over25 = nb.over25 - po.over25,
                       d_over35 = nb.over35 - po.over35,
                       d_over45 = nb.over45 - po.over45,
                       d_btts = nb.btts - po.btts,
                       p_over15 = nb.over15, p_over25 = nb.over25, p_btts = nb.btts))
    end
    return DataFrame(rows)
end

"""
    gjn_save_and_verify(db, fit) -> UUID

Persist `fit` and prove the artefact reloads to the same object.

Refuses when the recipe is already persisted: `save_fit` would otherwise return the OLD
run's UUID without writing anything, and the round-trip below would then compare this
fit against a different run's draws.

The dispersion round-trip is checked explicitly. `r_h`/`r_a` ride in
`CountLatents.observation_params`, a slot the Poisson ladder never populates, so it is
the one part of this container that no previous task's persistence path exercised.
"""
function gjn_save_and_verify(db, fit)
    existing = gjn_completed_run(db, fit.config)
    existing === nothing || error(
        "recipe $(fit.config.name) is already persisted as run $existing; " *
        "load it instead of saving a second copy")
    run_id = save_fit(fit, db)
    reloaded = load_fit(db, run_id)
    length(reloaded.folds) == length(fit.folds) || error("round-trip fold count differs")
    reloaded.latents.match_ids == fit.latents.match_ids || error("round-trip match IDs differ")
    reloaded.latents.λ_home == fit.latents.λ_home || error("round-trip λ_home differs")
    reloaded.latents.λ_away == fit.latents.λ_away || error("round-trip λ_away differs")
    reloaded.diagnostics.max_rhat == fit.diagnostics.max_rhat || error("round-trip R̂ differs")

    a_obs = fit.latents.observation_params
    b_obs = reloaded.latents.observation_params
    b_obs === nothing && error("round-trip dropped observation_params: the reloaded run " *
                               "would price on the double-Poisson grid")
    a_obs.r_h == b_obs.r_h || error("round-trip r_h differs")
    a_obs.r_a == b_obs.r_a || error("round-trip r_a differs")

    for (a, b) in zip(fit.folds, reloaded.folds)
        parent(a.chain.value) == parent(b.chain.value) || error(
            "round-trip chain differs on fold $(a.fold)")
    end
    return run_id
end

function gjn_assert_coverage(name::AbstractString, fit; folds::Int, oos::Int)
    length(fit.folds) == folds || error("$name has $(length(fit.folds)) folds; expected $folds")
    n_matches(fit.latents) == oos || error(
        "$name has $(n_matches(fit.latents)) OOS fixtures; expected $oos")
    allunique(fit.latents.match_ids) || error("$name OOS match IDs are not unique")
    return nothing
end

"One convergence row for a report. `strict_rhat_pass` is advisory."
function gjn_convergence_row(name::AbstractString, fit, c::GJNConfig; run_id = nothing)
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

"""
    gjn_dispersion_summary(fit) -> NamedTuple

The posterior of `r`, pooled over folds, read off the chains rather than the latents.

This is the number Experiment 02 reported as `r̂ ≈ 26.0–26.5`, and the first thing to
look at when a NegBin rung scores like its Poisson control: an `r` in the hundreds is
a Poisson in all but name, and says the latent state already absorbed the dispersion.
"""
function gjn_dispersion_summary(fit)
    draws = Float64[]
    for f in fit.folds
        sym = Symbol("disp.log_r")
        sym in names(f.chain) || continue
        append!(draws, exp.(vec(Array(f.chain[sym]))))
    end
    isempty(draws) && return (; n_draws = 0, mean_r = NaN, median_r = NaN,
                                q05 = NaN, q95 = NaN)
    return (; n_draws = length(draws),
              mean_r = mean(draws), median_r = median(draws),
              q05 = quantile(draws, 0.05), q95 = quantile(draws, 0.95))
end

# ==============================================================================
# 7. Report formatting
# ==============================================================================

gjn_num(v; digits = 4) = (v isa Real && isfinite(v)) ? @sprintf("%.*f", digits, v) : "n/a"
gjn_signed(v; digits = 5) = (v isa Real && isfinite(v)) ? @sprintf("%+.*f", digits, v) : "n/a"

"A GitHub-flavoured markdown table from a DataFrame, with per-column formatters."
function gjn_markdown_table(df::AbstractDataFrame; formats = Dict{Symbol,Function}())
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
                         v isa AbstractFloat ? gjn_num(v) :
                         v isa Missing ? "—" : string(v))
        end
        println(io, "| ", join(cells, " | "), " |")
    end
    return String(take!(io))
end
