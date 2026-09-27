# experiments/scotland/03_dynamics_scope_matrix/candidates.jl

using BayesianFootball
using Distributions

const EXPERIMENT = "scottish_dynamics_scope_matrix_w1"
const TODO = 31
const QUESTION = "How do dynamics scope and observation interact across the Scottish pyramid?"
const DIMENSION = "dynamics_scope_observation"
const STATUS = "screened"
const DECISION = "MAP screen completed; validity is limited for learned scales and the NUTS grid is pending."
const README = joinpath(@__DIR__, "README.md")
const CONTROL = "td_lower_joint"

function make_joint_observation()
    return JointGammaPoissonObservation(
        feature = Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
        shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
        log_kappa_prior = Normal(0.0, 0.2),
    )
end

const SCOPES = [
    Data.DataScope(
        name = "lower",
        train_tournaments = [56, 57],
        cups = :none,
        target_tournaments = [56, 57],
        monitor_tournaments = Int[],
        clock_tournaments = [56, 57],
        target_seasons = ["24/25", "25/26"],
        history_seasons = 2,
        dynamics_col = :match_biweek
    ),
    Data.DataScope(
        name = "spfl",
        train_tournaments = [54, 55, 56, 57],
        cups = :none,
        target_tournaments = [56, 57],
        monitor_tournaments = [54, 55],
        clock_tournaments = [56, 57],
        target_seasons = ["24/25", "25/26"],
        history_seasons = 2,
        dynamics_col = :match_biweek
    ),
    Data.DataScope(
        name = "spfl_cups",
        train_tournaments = [54, 55, 56, 57],
        cups = :senior_spfl_ties,
        target_tournaments = [56, 57],
        monitor_tournaments = [54, 55],
        clock_tournaments = [56, 57],
        target_seasons = ["24/25", "25/26"],
        history_seasons = 2,
        dynamics_col = :match_biweek
    ),
]

const DYNAMICS = [
    (:td, () -> TimeDecayDynamics(days_half_life = 180.0)),
    (:grw, () -> MultiScaleGRW()),
]

const OBSERVATIONS = [
    (:poisson, () -> PoissonObservation()),
    (:joint, () -> make_joint_observation()),
]

const DEFAULT_SAMPLER = Samplers.QueuedNUTSConfig(
    n_samples = 1_000,
    n_warmup = 500,
    n_chains = 4,
    accept_rate = 0.65,
    max_depth = 10,
    initialisation = nothing,
    show_progress = false,
    silence_initial_stepsize = true
)

function build_w1_candidates()
    candidates = Harness.Candidate[]
    for (dyn_prefix, dyn_fn) in DYNAMICS
        for scope in SCOPES
            for (obs_prefix, obs_fn) in OBSERVATIONS
                name = "$(dyn_prefix)_$(scope.name)_$(obs_prefix)"
                role = name == CONTROL ? :control : :candidate
                model = CountModelBuilder(Symbol(name)) |>
                    add(GlobalInterception()) |>
                    add(dyn_fn()) |>
                    add(GlobalHomeAdvantage()) |>
                    add(obs_fn()) |>
                    build
                c = Harness.Candidate(
                    name = name,
                    model = model,
                    scope = scope,
                    sampler = DEFAULT_SAMPLER,
                    role = role,
                    hypothesis = "W1 factorial arm: dynamics=$(dyn_prefix), scope=$(scope.name), observation=$(obs_prefix)"
                )
                push!(candidates, c)
            end
        end
    end
    return candidates
end

const CANDIDATES = build_w1_candidates()
