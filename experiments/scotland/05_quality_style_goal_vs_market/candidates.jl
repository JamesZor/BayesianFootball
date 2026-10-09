# experiments/scotland/05_quality_style_goal_vs_market/candidates.jl
#
# Goal arms of the quality/style experiment. The market arm (C0 on inverted Betfair closes)
# is not a builder model; it is fitted per fold by `market_arm.jl` and scored through the
# harness via `scoring_adapter.jl`.

using BayesianFootball
using Distributions

const EXPERIMENT = "scottish_lower_quality_style_2426"
const TODO = 40
const QUESTION = "Does a quality/style rotation of team strength, with market-informed r, improve 40-fold goal forecasts, and how close does it come to a market-odds QS model?"
const DIMENSION = "dynamics_quality_style"
const STATUS = "running"
const DECISION = "Pending: smoke then 40-fold grid."
const README = joinpath(@__DIR__, "README.md")
const CONTROL = "control_grw"

const SCOPE = Data.DataScope(
    name = "lower",
    train_tournaments = [56, 57],
    cups = :none,
    target_tournaments = [56, 57],
    monitor_tournaments = Int[],
    clock_tournaments = [56, 57],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
)

# W2 sampler (experiment 04 DEFAULT_SAMPLER), unchanged.
const DEFAULT_SAMPLER = Samplers.QueuedNUTSConfig(
    n_samples = 1_000,
    n_warmup = 500,
    n_chains = 4,
    accept_rate = 0.65,
    max_depth = 10,
    initialisation = nothing,
    show_progress = false,
    silence_initial_stepsize = true,
)

const ARMS = [
    ("control_grw", :control, () -> MultiScaleGRW(),
     "Experiment control: independent attack/defence multiscale GRW."),
    ("control_td", :candidate, () -> TimeDecayDynamics(days_half_life = 180.0),
     "Secondary control: 180-day time decay."),
    ("qs_market_r", :candidate, () -> QualityStyleGRW(),
     "QS GRW with r priors centred on the market estimate (r ≈ 0.3)."),
    ("qs_weak_r", :candidate,
     () -> QualityStyleGRW(r₀ = LogNormal(0, 1), rₛ = LogNormal(0, 1), rₖ = LogNormal(0, 1)),
     "QS GRW with weak r priors (median 1 = independent attack/defence, equal scales)."),
]

function build_candidates()
    return [Harness.Candidate(
                name = name,
                model = CountModelBuilder(Symbol(name)) |>
                    add(GlobalInterception()) |>
                    add(dynamics()) |>
                    add(GlobalHomeAdvantage()) |>
                    add(PoissonObservation()) |>
                    build,
                scope = SCOPE,
                sampler = DEFAULT_SAMPLER,
                role = role,
                hypothesis = hypothesis)
            for (name, role, dynamics, hypothesis) in ARMS]
end

const CANDIDATES = build_candidates()
