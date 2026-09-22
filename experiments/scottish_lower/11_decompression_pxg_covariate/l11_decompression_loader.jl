# Definitions only. Include this module before deserializing its prototype fits.
module DecompressionPXG

include(joinpath(@__DIR__, "../../../current_development/grw_player_hybrid/l01_loader.jl"))
include(joinpath(@__DIR__, "../../../current_development/grw_player_hybrid/l02_evaluation.jl"))
import CSV
import Serialization
import SpecialFunctions
import Turing

const B = GPH_PG.Builder
const PXG_MARKETS = Data.MarketConfig([
    Data.Market1X2(),
    Data.MarketOverUnder(2.5),
    Data.MarketBTTS(),
])

Base.@kwdef struct DecompressionConfig
    experiment::String = "scottish_lower_decompression"
    smoke_experiment::String = "smoke_scottish_lower_decompression"
    save_root::String = joinpath(@__DIR__, "results")
    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos::Int = 710
    smoke_folds::Vector{Int} = [1, 20, 40]
end

# ==============================================================================
# 1. Proxy-xG form covariate
# ==============================================================================

"""
    ProxyXGFormCovariate

The work package's coefficient parameterisation:

    η_h += 0.5 * w_pxg * Δpxg_form
    η_a -= 0.5 * w_pxg * Δpxg_form

`PxGFeature` already emits `Δpxg_form` from a strictly point-in-time walk. The
factor one-half lives in the design column, not in the prior or extraction, so
`pxg_form.w` is directly the effect on log-rate supremacy. Missing history is the
exact structural zero inherited from `PxGFeature`.
"""
Base.@kwdef struct ProxyXGFormCovariate{
    F<:GPH_FEATURES.PxGFeature,
    D<:Distributions.ContinuousUnivariateDistribution,
} <: GPH_PG.AbstractCovariateConfig
    feature::F = GPH_FEATURES.PxGFeature(
        lookback = 16,
        lookback_matches = 16,
        decay = :exponential,
        half_life_matches = 16.0,
        prior_weight = 3.0,
        min_matches = 2,
        k = 25.0,
        fallback = :none,
        scale = 1.0,
    )
    prior::D = Normal(0.60, 0.20)
end

GPH_PG.covariate_name(::ProxyXGFormCovariate) = :pxg_form
GPH_PG.covariate_role(::ProxyXGFormCovariate) = SupremacyRole()
GPH_PG.covariate_prior(c::ProxyXGFormCovariate) = c.prior
GPH_PG.covariate_features(c::ProxyXGFormCovariate) =
    GPH_FEATURES.AbstractFeatureConfig[c.feature]

function GPH_PG.covariate_column(::ProxyXGFormCovariate, fs)
    haskey(fs.data, :flat_pxg_supremacy) || error(
        "ProxyXGFormCovariate requires :flat_pxg_supremacy")
    return 0.5 .* Vector{Float64}(fs.data[:flat_pxg_supremacy])
end

function GPH_PG.covariate_oos(::ProxyXGFormCovariate, fs, df)
    if hasproperty(df, :pxg_supremacy)
        return 0.5 .* Float64.(df.pxg_supremacy)
    end
    bridge = get(fs.data, :pxg_supremacy_by_match_id, Dict{Int,Float64}())
    return Float64[0.5 * get(bridge, Int(row.match_id), 0.0) for row in eachrow(df)]
end

# ==============================================================================
# 2. Allocation-free candidate engine
# ==============================================================================

# The production composable engine is the mathematical reference. Its scalar
# broadcasts allocate on ReverseDiff replay in this package version. These local
# adapters preserve its sampled sites and equations while lifting every sampled
# scalar into a one-element tracked array. They do not modify src-owned methods.
struct ArrayClampGuard <: B.AbstractRateGuard
    lo::Vector{Float64}
    hi::Vector{Float64}
end
ArrayClampGuard() = ArrayClampGuard([-10.0], [10.0])
B.apply_guard(g::ArrayClampGuard, η) = clamp.(η, g.lo, g.hi)
B.guard_describe(g::ArrayClampGuard) = "array clamp to [$(only(g.lo)), $(only(g.hi))]"

array_scalar(x::Real) = [x]
array_scalar(x::AbstractArray) = x
function array_scalar(x::ReverseDiff.TrackedReal{V,D}) where {V,D}
    tape = ReverseDiff.tape(x)
    out = ReverseDiff.track([ReverseDiff.value(x)], D, tape)
    ReverseDiff.record!(tape, ReverseDiff.SpecialInstruction, array_scalar, x, out)
    return out
end
function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(array_scalar)})
    ReverseDiff.pull_value!(instruction.input)
    ReverseDiff.value(instruction.output)[1] = ReverseDiff.value(instruction.input)
    return nothing
end
function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(array_scalar)})
    ReverseDiff.increment_deriv!(instruction.input, ReverseDiff.deriv(instruction.output)[1])
    ReverseDiff.unseed!(instruction.output)
    return nothing
end

Turing.@model function array_interception(config::GlobalInterception)
    μ ~ config.μ
    return array_scalar(μ)
end

Turing.@model function array_home_advantage(config::GlobalHomeAdvantage)
    γ_global ~ config.γ_global
    return array_scalar(γ_global)
end

Turing.@model function array_time_decay(config::TimeDecayDynamics, center::Matrix{Float64},
                                        n_teams::Int)
    σ_a ~ config.σ_att
    σ_d ~ config.σ_def
    raw_a ~ Turing.filldist(Normal(), n_teams)
    raw_d ~ Turing.filldist(Normal(), n_teams)
    α = center * (raw_a .* array_scalar(σ_a))
    β = center * (raw_d .* array_scalar(σ_d))
    return (; α, β)
end

Turing.@model function array_pxg_form(c::ProxyXGFormCovariate, x::Vector{Float64})
    w ~ GPH_PG.covariate_prior(c)
    q = array_scalar(w) .* x
    return (; h = q, a = .-q)
end

Turing.@model function array_global_dispersion(config::GlobalDispersion)
    log_r ~ config.log_r
    bounded = B._cb_bound_dispersion_log.(array_scalar(log_r))
    return exp.(bounded)
end

Turing.@model function array_negbin_pxg_engine(config, z, center)
    inter ~ DynamicPPL.to_submodel(array_interception(config.interception))
    ha ~ DynamicPPL.to_submodel(array_home_advantage(config.home_advantage))
    dyn ~ DynamicPPL.to_submodel(array_time_decay(config.dynamics, center, z.n_teams))
    pxg ~ DynamicPPL.to_submodel(
        DynamicPPL.prefix(array_pxg_form(first(config.covariates), first(z.predictor_designs)),
                          Val(:pxg_form)),
        false,
    )
    disp ~ DynamicPPL.to_submodel(array_global_dispersion(config.observation.dispersion))

    η_h = B.apply_guard(config.guard,
        inter .+ ha .+ dyn.α[z.home_ids] .+ dyn.β[z.away_ids] .+ pxg.h)
    η_a = B.apply_guard(config.guard,
        inter .+ dyn.α[z.away_ids] .+ dyn.β[z.home_ids] .+ pxg.a)
    λ_h = exp.(η_h)
    λ_a = exp.(η_a)

    total_h = log.(disp .+ λ_h)
    total_a = log.(disp .+ λ_a)
    ll_h = SpecialFunctions.loggamma.(z.home_goals .+ disp) .-
           SpecialFunctions.loggamma.(disp) .- z.log_fact_h .+
           disp .* (log.(disp) .- total_h) .+
           z.home_goals .* (η_h .- total_h)
    ll_a = SpecialFunctions.loggamma.(z.away_goals .+ disp) .-
           SpecialFunctions.loggamma.(disp) .- z.log_fact_a .+
           disp .* (log.(disp) .- total_a) .+
           z.away_goals .* (η_a .- total_a)
    Turing.@addlogprob! sum(ll_h .* z.match_weights) + sum(ll_a .* z.match_weights)
end

const ArrayPXGNegBinModel = B.NegBinCountModel{
    GlobalInterception,
    TimeDecayDynamics,
    GlobalHomeAdvantage,
    Tuple{P},
    NegativeBinomialObservation{GlobalDispersion},
    ArrayClampGuard,
} where {P<:ProxyXGFormCovariate}

function GPH_PG.build_turing_model(model::ArrayPXGNegBinModel, fs)
    z = B.cb_design(model, fs)
    center = Matrix{Float64}(I, z.n_teams, z.n_teams) .- 1.0 / z.n_teams
    return array_negbin_pxg_engine(model, z, center)
end

# ==============================================================================
# 3. Three-arm recipe
# ==============================================================================

function pxg_covariate()
    return ProxyXGFormCovariate()
end

function joint_observation()
    return JointGammaPoissonObservation(
        feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
        shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
        log_kappa_prior = Normal(0.0, 0.2),
    )
end

function poisson_model(name::Symbol, observation)
    return CountModelBuilder(name) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(observation) |>
        build
end

function candidate_model(; optimized::Bool = true)
    reference = CountModelBuilder(:m03_negbin_pxg_covariate) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(pxg_covariate()) |>
        add(NegativeBinomialObservation(dispersion = GlobalDispersion())) |>
        build
    optimized || return reference
    return B.NegBinCountModel(reference.interception, reference.dynamics,
        reference.home_advantage, reference.covariates, reference.observation,
        ArrayClampGuard())
end

function models(; optimized::Bool = true)
    return [
        ("m01_poisson_time_decay",
         poisson_model(:m01_poisson_time_decay, PoissonObservation())),
        ("m02_joint_gamma_poisson",
         poisson_model(:m02_joint_gamma_poisson, joint_observation())),
        ("m03_negbin_pxg_covariate", candidate_model(; optimized)),
    ]
end

# ==============================================================================
# 4. Deterministic AD and mathematical parity
# ==============================================================================

function replay_performance(tape, gradient, θ)
    for _ in 1:30
        ReverseDiff.gradient!(gradient, tape, θ)
    end
    allocated_bytes = @allocated ReverseDiff.gradient!(gradient, tape, θ)
    best = minimum(@elapsed(ReverseDiff.gradient!(gradient, tape, θ)) for _ in 1:100)
    return (; allocated_bytes, gradient_ms = 1000 * best)
end

"Linked-space parity against the production composable engine, including warmup-scale probes."
function allocation_audit(model, reference, fs; seed::Int = 24)
    candidate = GPH_PG.build_turing_model(model, fs)
    baseline = GPH_PG.build_turing_model(reference, fs)
    Random.seed!(seed)
    candidate_vi = DynamicPPL.VarInfo(candidate)
    Random.seed!(seed)
    baseline_vi = DynamicPPL.VarInfo(baseline)
    string.(collect(keys(candidate_vi))) == string.(collect(keys(baseline_vi))) ||
        error("candidate/reference sampled-site layouts differ")
    candidate_vi[:] == baseline_vi[:] || error("prior initializations differ at a shared seed")
    candidate_vi = DynamicPPL.link!!(candidate_vi, candidate)
    baseline_vi = DynamicPPL.link!!(baseline_vi, baseline)
    θ = copy(candidate_vi[:])
    θ == baseline_vi[:] || error("linked parameter layouts differ")

    candidate_ld = DynamicPPL.LogDensityFunction(
        candidate, DynamicPPL.getlogjoint_internal, candidate_vi)
    baseline_ld = DynamicPPL.LogDensityFunction(
        baseline, DynamicPPL.getlogjoint_internal, baseline_vi)
    f = x -> LogDensityProblems.logdensity(candidate_ld, x)
    r = x -> LogDensityProblems.logdensity(baseline_ld, x)
    raw = ReverseDiff.GradientTape(f, θ)
    tape = ReverseDiff.compile(raw)
    gradient = similar(θ)
    worst_density = 0.0
    worst_gradient = 0.0
    for displacement in (0.0, 0.003, -0.8, 0.8, -3.0, 3.0)
        point = θ .+ displacement .* sin.(eachindex(θ))
        isfinite(f(point)) && isfinite(r(point)) ||
            error("non-finite density at linked displacement $displacement")
        density_error = abs(f(point) - r(point))
        density_error <= 1.0e-9 || error(
            "density parity failed at displacement $displacement: $density_error")
        ReverseDiff.gradient!(gradient, tape, point)
        candidates = (
            ReverseDiff.gradient(f, point),
            ForwardDiff.gradient(f, point),
            ForwardDiff.gradient(r, point),
        )
        gradient_error = maximum(gph_relative_error(gradient, other) for other in candidates)
        gradient_error <= 1.0e-8 || error(
            "gradient parity failed at displacement $displacement: $gradient_error")
        worst_density = max(worst_density, density_error)
        worst_gradient = max(worst_gradient, gradient_error)
    end
    performance = replay_performance(tape, gradient, θ)
    performance.allocated_bytes == 0 || error(
        "compiled candidate replay allocates $(performance.allocated_bytes) bytes")
    return (; n_parameters = length(θ), tape_instructions = length(raw.tape),
              performance..., worst_density, worst_gradient)
end

include(joinpath(@__DIR__, "l12_workflow.jl"))

end # module
