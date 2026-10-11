# Stage 2/S3: one factor at a time on the frozen W2 panel. Default QS unchanged.
using BayesianFootball, Distributions, CSV, DataFrames, SHA
const EXPERIMENT = "scottish_lower_qs_transition_2326"
const TODO = 46
const QUESTION = "Which opt-in transition fix improves QS on W2's panel?"
const DIMENSION = "qs_transition"
const STATUS = "building"
const DECISION = "Stage 2 smokes only; S3 grids are not authorised in Stage 2."
const README = joinpath(@__DIR__, "REPORT_S2.md")
const CONTROL = "qs_control"
const SCOPE = Data.DataScope(name="lower", train_tournaments=[56,57], cups=:none,
    target_tournaments=[56,57], monitor_tournaments=Int[], clock_tournaments=[56,57],
    target_seasons=["23/24","24/25","25/26"], history_seasons=2, dynamics_col=:match_biweek)
const DEFAULT_SAMPLER = Samplers.QueuedNUTSConfig(n_samples=1000, n_warmup=500,
    n_chains=4, accept_rate=0.65, max_depth=10, initialisation=nothing,
    show_progress=false, silence_initial_stepsize=true)
const MARKET_PATH = joinpath(@__DIR__, "build", "market_rates.csv")
bytes2hex(open(SHA.sha256, MARKET_PATH)) == strip(read(joinpath(@__DIR__, "build", "market_rates.sha256"), String)) || error("frozen market table changed")
const MARKET_FEATURE = Features.MarketRatesFeature(CSV.read(MARKET_PATH, DataFrame))
weak_qs(; kwargs...) = QualityStyleGRW(; r₀=LogNormal(0,1), rₛ=LogNormal(0,1), rₖ=LogNormal(0,1), kwargs...)
function qs_recipe(name; dynamics=weak_qs(), tiers=nothing)
    builder = CountModelBuilder(Symbol(name)) |> add(GlobalInterception()) |>
        add(dynamics) |> add(GlobalHomeAdvantage())
    tiers === nothing || add!(builder, tiers)
    return builder |> add(MarketRateObservation(feature=MARKET_FEATURE)) |> build
end
if !isdefined(Main,:QSTransitionInputs)
    Base.include(Main,joinpath(@__DIR__,"l01_entrant_prior.jl"))
end
const ENTRANT_FEATURE = Main.QSTransitionInputs.EntrantPriorFeature()
const CANDIDATES = [
    Harness.Candidate(name="qs_control", model=qs_recipe("qs_control"), scope=SCOPE,
        sampler=DEFAULT_SAMPLER, role=:control, hypothesis="Unchanged wave-2 QS goals+market observation on W2 panel."),
    Harness.Candidate(name="qs_a2", model=qs_recipe("qs_a2"; tiers=PyramidTiers(
        anchor=:full_pyramid, carry=EstimatedCarry(), jump=EstimatedJump())), scope=SCOPE,
        sampler=DEFAULT_SAMPLER, role=:candidate, hypothesis="F3: one full-pyramid A2 term, 0.48 attack / 0.52 concession."),
    Harness.Candidate(name="qs_entrant", model=qs_recipe("qs_entrant";
        dynamics=weak_qs(entrant_feature=ENTRANT_FEATURE)),scope=SCOPE,
        sampler=DEFAULT_SAMPLER,role=:candidate,
        hypothesis="F1F2/D10: persistent, disjoint-endpoint entrant prior at the first target-season node; incumbents centre alone."),
    Harness.Candidate(name="qs_summer",model=qs_recipe("qs_summer";
        dynamics=weak_qs(target_season_step=true)),scope=SCOPE,
        sampler=DEFAULT_SAMPLER,role=:candidate,
        hypothesis="F4: independent target-season macro uncertainty, integrated at the opening fold."),
]
# F6 intentionally absent: cadence/clock are tied; see build/F6_DECISION.md.
