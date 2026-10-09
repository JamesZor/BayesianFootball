# Wave 2: QS on joint goals+proxy-xG, and GRW/QS observing full-book market log rates.
# Same lower scope and W2 sampler as wave 1. No ROI/staking or reference refits.
using BayesianFootball
using Distributions
Base.include(@__MODULE__, joinpath(@__DIR__, "l01_market_table.jl"))

const EXPERIMENT = "scottish_lower_qs_wave2_2426"
const TODO = 41
const QUESTION = "Does QS help joint goals+proxy-xG forecasts, and can builder market-rate observations match saved market C0?"
const DIMENSION = "quality_style_and_observation"
const STATUS = "active"
const DECISION = "Pending four smoke gates and manager grid approval; no predictive result yet."
const README = joinpath(@__DIR__, "README.md")
const CONTROL = "grw_joint"

const SCOPE = Data.DataScope(
    name = "lower",
    train_tournaments = [56, 57], cups = :none,
    target_tournaments = [56, 57], monitor_tournaments = Int[],
    clock_tournaments = [56, 57], target_seasons = ["24/25", "25/26"],
    history_seasons = 2, dynamics_col = :match_biweek,
)
const DEFAULT_SAMPLER = Samplers.QueuedNUTSConfig(
    n_samples = 1_000, n_warmup = 500, n_chains = 4,
    accept_rate = 0.65, max_depth = 10, initialisation = nothing,
    show_progress = false, silence_initial_stepsize = true,
)
const MARKET_FEATURE = Wave2MarketTable.feature()

# Experiment 04 make_joint_observation(league_deltas=false), unchanged.
make_joint_observation() = JointGammaPoissonObservation(
    feature = Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2), kappa = SharedKappa(),
)
weak_qs() = QualityStyleGRW(r₀ = LogNormal(0, 1), rₛ = LogNormal(0, 1), rₖ = LogNormal(0, 1))

const ARMS = [
    ("grw_joint", :control, () -> MultiScaleGRW(), make_joint_observation,
     "Joint goals+proxy-xG control with independent attack/defence GRW."),
    ("qs_joint", :candidate, weak_qs, make_joint_observation,
     "Does weak-r QS improve the production-like joint observation?"),
    ("grw_marketobs", :candidate, () -> MultiScaleGRW(),
     () -> MarketRateObservation(feature = MARKET_FEATURE),
     "Can GRW observing full-book log rates match saved market C0 and joint goals+xG?"),
    ("qs_marketobs", :candidate, weak_qs,
     () -> MarketRateObservation(feature = MARKET_FEATURE),
     "Can weak-r QS observing full-book log rates match saved C0 and identify micro r?"),
]
const CANDIDATES = [Harness.Candidate(
    name = name,
    model = CountModelBuilder(Symbol(name)) |>
        add(GlobalInterception()) |> add(dynamics()) |> add(GlobalHomeAdvantage()) |>
        add(observation()) |> build,
    scope = SCOPE, sampler = DEFAULT_SAMPLER, role = role, hypothesis = hypothesis,
) for (name, role, dynamics, observation, hypothesis) in ARMS]
