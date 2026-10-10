# Wave 3: goals+proxy-xG and full-book market log rates fused on one latent, with learned static
# market bias (κ_D, δ_D, δ_M). Same lower scope, W2 sampler, joint config and frozen market table
# as wave 2. No ROI/staking or reference refits.
using BayesianFootball
using Distributions
# Wave 2's frozen-table adapter, unchanged; it loads wave 1's market module at top level as
# Main.QSMarketArm (namespace provenance ruling) and checks the table SHA256 before injecting.
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "06_qs_joint_and_market_observation", "l01_market_table.jl"))

const EXPERIMENT = "scottish_lower_qs_wave3_2426"
const TODO = 42
const QUESTION = "Does fusing goals+proxy-xG with market log rates, with learned static market bias, beat the market close?"
const DIMENSION = "observation_fusion"
const STATUS = "active"
const DECISION = "Smoke only; grid awaits manager approval."
const README = joinpath(@__DIR__, "README.md")
const CONTROL = "fusion_qs_bias"

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

# Experiment 04 make_joint_observation(league_deltas=false), unchanged from wave 2.
make_joint_observation() = JointGammaPoissonObservation(
    feature = Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2), kappa = SharedKappa(),
)
weak_qs() = QualityStyleGRW(r₀ = LogNormal(0, 1), rₛ = LogNormal(0, 1), rₖ = LogNormal(0, 1))

# Bias priors as the brief: κ_D ~ LogNormal(0, 0.2), δ_D ~ Normal(0, 0.1), δ_M ~ Normal(0, 0.1).
fusion_bias() = JointMarketFusionObservation(
    joint = make_joint_observation(), market = MarketRateObservation(feature = MARKET_FEATURE),
    kappa_D_prior = LogNormal(0.0, 0.2), delta_D_prior = Normal(0.0, 0.1),
    delta_M_prior = Normal(0.0, 0.1),
)
# Bias fixed off: κ_D = 1, δ_D = δ_M = 0, no sites (wave 2's market likelihood on the true rates).
fusion_nobias() = JointMarketFusionObservation(
    joint = make_joint_observation(), market = MarketRateObservation(feature = MARKET_FEATURE),
    kappa_D_prior = nothing, delta_D_prior = nothing, delta_M_prior = nothing,
)

const ARMS = [
    ("fusion_qs_bias", :control, weak_qs, fusion_bias,
     "Weak-r QS fusing goals+proxy-xG with market log rates and learned static market bias."),
    ("fusion_qs_nobias", :candidate, weak_qs, fusion_nobias,
     "Ablation: does any gain over the market come from the bias terms?"),
    ("fusion_grw_bias", :candidate, () -> MultiScaleGRW(), fusion_bias,
     "Does the QS latent matter once goals, xG and market rates are fused with bias?"),
]
const CANDIDATES = [Harness.Candidate(
    name = name,
    model = CountModelBuilder(Symbol(name)) |>
        add(GlobalInterception()) |> add(dynamics()) |> add(GlobalHomeAdvantage()) |>
        add(observation()) |> build,
    scope = SCOPE, sampler = DEFAULT_SAMPLER, role = role, hypothesis = hypothesis,
) for (name, role, dynamics, observation, hypothesis) in ARMS]
