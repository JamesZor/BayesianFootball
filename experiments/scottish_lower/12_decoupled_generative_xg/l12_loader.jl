# Definitions only. Include this module before deserializing its prototype fits.
module DecoupledGenerativeXG

include(joinpath(@__DIR__, "../../../current_development/grw_player_hybrid/l01_loader.jl"))
include(joinpath(@__DIR__, "../../../current_development/grw_player_hybrid/l02_evaluation.jl"))
import CSV
import Serialization
import SpecialFunctions
import Turing

const B = GPH_PG.Builder
const FUNNEL_MARKETS = Data.MarketConfig([
    Data.Market1X2(),
    Data.MarketOverUnder(2.5),
    Data.MarketBTTS(),
])

Base.@kwdef struct FunnelConfig
    experiment::String = "scottish_lower_decoupled_xg"
    smoke_experiment::String = "smoke_scottish_lower_decoupled_xg"
    save_root::String = joinpath(@__DIR__, "results")
    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos::Int = 710
    smoke_folds::Vector{Int} = [1, 20, 40]
end

# ==============================================================================
# 1. Mathematical contract and four arms
# ==============================================================================

"The proxy-xG measurement and shared league conversion recipe."
shared_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
    kappa = SharedKappa(),
)

"The same chance layer with a zero-centred, non-centred team finishing hierarchy."
hierarchical_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
    kappa = HierarchicalKappa(
        σ_prior = truncated(Normal(0.0, 0.10), 0.0, Inf),
    ),
)

function standard_model(name::Symbol, observation)
    return CountModelBuilder(name) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |>
        add(observation) |>
        build
end

# The work package labels m02 a "parallel sensor" and m03 a "generative
# funnel", but their stated probability laws are identical. Both have
#
#   pxg | μ,ν ~ Gamma(ν, μ/ν),    goals | μ,κ ~ Poisson(κ μ),
#
# and in an ordinary joint posterior both likelihoods update μ. The separate arm
# remains in the benchmark so that this identity is tested rather than hidden.
function reference_models()
    return [
        ("m01_poisson_time_decay",
         standard_model(:m01_poisson_time_decay, PoissonObservation())),
        ("m02_joint_gamma_poisson",
         standard_model(:m02_joint_gamma_poisson, shared_observation())),
        ("m03_funnel_shared_kappa",
         standard_model(:m03_funnel_shared_kappa, shared_observation())),
        ("m04_funnel_hierarchical_kappa",
         standard_model(:m04_funnel_hierarchical_kappa, hierarchical_observation())),
    ]
end

# ==============================================================================
# 2. Allocation-free ReverseDiff engine
# ==============================================================================

# The component model is the mathematical reference. On ReverseDiff 1.17 its
# sampled scalars allocate during replay. These adapters lift every sampled scalar
# into a one-element TrackedArray without changing any site, prior, or equation.
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
function ReverseDiff.special_forward_exec!(
    instruction::ReverseDiff.SpecialInstruction{typeof(array_scalar)},
)
    ReverseDiff.pull_value!(instruction.input)
    ReverseDiff.value(instruction.output)[1] = ReverseDiff.value(instruction.input)
    return nothing
end
function ReverseDiff.special_reverse_exec!(
    instruction::ReverseDiff.SpecialInstruction{typeof(array_scalar)},
)
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

Turing.@model function array_time_decay(
    config::TimeDecayDynamics,
    center::Matrix{Float64},
    n_teams::Int,
)
    σ_a ~ config.σ_att
    σ_d ~ config.σ_def
    raw_a ~ Turing.filldist(Normal(), n_teams)
    raw_d ~ Turing.filldist(Normal(), n_teams)
    α = center * (raw_a .* array_scalar(σ_a))
    β = center * (raw_d .* array_scalar(σ_d))
    return (; α, β)
end

Turing.@model function array_shared_parameters(observation::B.SharedKappaJoint)
    ν ~ observation.shape_prior
    log_κ ~ observation.log_kappa_prior
    return (; ν = array_scalar(ν), log_κ = array_scalar(log_κ))
end

Turing.@model function array_hierarchical_parameters(
    observation::B.HierarchicalKappaJoint,
    center::Matrix{Float64},
    n_teams::Int,
)
    ν ~ observation.shape_prior
    log_κ ~ observation.log_kappa_prior
    σ_κ ~ observation.kappa.σ_prior
    κ_team_raw ~ Turing.filldist(Normal(), n_teams)
    # Matrix centring is algebraically identical to raw .- mean(raw), while
    # keeping the replay on ReverseDiff's preallocated array path.
    δ_κ = center * (κ_team_raw .* array_scalar(σ_κ))
    log_κ_team = array_scalar(log_κ) .+ δ_κ
    return (;
        ν = array_scalar(ν),
        log_κ = array_scalar(log_κ),
        σ_κ = array_scalar(σ_κ),
        log_κ_team,
    )
end

function array_predictors(config, z, center)
    return (; config, z, center)
end

Turing.@model function array_poisson_engine(config, z, center)
    inter ~ DynamicPPL.to_submodel(array_interception(config.interception))
    ha ~ DynamicPPL.to_submodel(array_home_advantage(config.home_advantage))
    dyn ~ DynamicPPL.to_submodel(array_time_decay(config.dynamics, center, z.n_teams))

    η_h = B.apply_guard(config.guard,
        inter .+ ha .+ dyn.α[z.home_ids] .+ dyn.β[z.away_ids])
    η_a = B.apply_guard(config.guard,
        inter .+ dyn.α[z.away_ids] .+ dyn.β[z.home_ids])
    ll_h = z.home_goals .* η_h .- exp.(η_h) .- z.log_fact_h
    ll_a = z.away_goals .* η_a .- exp.(η_a) .- z.log_fact_a
    Turing.@addlogprob! sum(ll_h .* z.match_weights) + sum(ll_a .* z.match_weights)
end

Turing.@model function array_shared_funnel_engine(config, z, center)
    inter ~ DynamicPPL.to_submodel(array_interception(config.interception))
    ha ~ DynamicPPL.to_submodel(array_home_advantage(config.home_advantage))
    dyn ~ DynamicPPL.to_submodel(array_time_decay(config.dynamics, center, z.n_teams))
    obs ~ DynamicPPL.to_submodel(array_shared_parameters(config.observation))

    η_h = B.apply_guard(config.guard,
        inter .+ ha .+ dyn.α[z.home_ids] .+ dyn.β[z.away_ids])
    η_a = B.apply_guard(config.guard,
        inter .+ dyn.α[z.away_ids] .+ dyn.β[z.home_ids])

    ζ_h = η_h .+ obs.log_κ
    ζ_a = η_a .+ obs.log_κ
    ll_h = z.home_goals .* ζ_h .- exp.(ζ_h) .- z.log_fact_h
    ll_a = z.away_goals .* ζ_a .- exp.(ζ_a) .- z.log_fact_a
    goals_ll = sum(ll_h .* z.match_weights) + sum(ll_a .* z.match_weights)

    od = z.observation_data
    ν = obs.ν
    # Keep the Gamma density in narrow vector kernels. A single nine-argument
    # fused broadcast makes ReverseDiff allocate an O(rows) derivative cache on
    # every replay; these algebraically identical stages stay preallocated.
    ν_minus_one = ν .- 1.0
    log_norm = ν .* log.(ν)
    log_norm = log_norm .- SpecialFunctions.loggamma.(ν)
    shape_h = ν_minus_one .* od.log_pxg_h
    shape_a = ν_minus_one .* od.log_pxg_a
    scaled_x_h = (ν .* od.pxg_h) .* exp.(.-η_h)
    scaled_x_a = (ν .* od.pxg_a) .* exp.(.-η_a)
    scaled_eta_h = ν .* η_h
    scaled_eta_a = ν .* η_a
    g_h = shape_h .- scaled_x_h
    g_h = g_h .- scaled_eta_h
    g_h = g_h .+ log_norm
    g_a = shape_a .- scaled_x_a
    g_a = g_a .- scaled_eta_a
    g_a = g_a .+ log_norm
    proxy_ll = sum(g_h .* od.mask_weights) + sum(g_a .* od.mask_weights)
    Turing.@addlogprob! goals_ll + proxy_ll
end

Turing.@model function array_hierarchical_funnel_engine(config, z, center)
    inter ~ DynamicPPL.to_submodel(array_interception(config.interception))
    ha ~ DynamicPPL.to_submodel(array_home_advantage(config.home_advantage))
    dyn ~ DynamicPPL.to_submodel(array_time_decay(config.dynamics, center, z.n_teams))
    obs ~ DynamicPPL.to_submodel(
        array_hierarchical_parameters(config.observation, center, z.n_teams))

    η_h = B.apply_guard(config.guard,
        inter .+ ha .+ dyn.α[z.home_ids] .+ dyn.β[z.away_ids])
    η_a = B.apply_guard(config.guard,
        inter .+ dyn.α[z.away_ids] .+ dyn.β[z.home_ids])

    od = z.observation_data
    ζ_h = η_h .+ obs.log_κ_team[od.home_idx]
    ζ_a = η_a .+ obs.log_κ_team[od.away_idx]
    ll_h = z.home_goals .* ζ_h .- exp.(ζ_h) .- z.log_fact_h
    ll_a = z.away_goals .* ζ_a .- exp.(ζ_a) .- z.log_fact_a
    goals_ll = sum(ll_h .* z.match_weights) + sum(ll_a .* z.match_weights)

    ν = obs.ν
    ν_minus_one = ν .- 1.0
    log_norm = ν .* log.(ν)
    log_norm = log_norm .- SpecialFunctions.loggamma.(ν)
    shape_h = ν_minus_one .* od.log_pxg_h
    shape_a = ν_minus_one .* od.log_pxg_a
    scaled_x_h = (ν .* od.pxg_h) .* exp.(.-η_h)
    scaled_x_a = (ν .* od.pxg_a) .* exp.(.-η_a)
    scaled_eta_h = ν .* η_h
    scaled_eta_a = ν .* η_a
    g_h = shape_h .- scaled_x_h
    g_h = g_h .- scaled_eta_h
    g_h = g_h .+ log_norm
    g_a = shape_a .- scaled_x_a
    g_a = g_a .- scaled_eta_a
    g_a = g_a .+ log_norm
    proxy_ll = sum(g_h .* od.mask_weights) + sum(g_a .* od.mask_weights)
    Turing.@addlogprob! goals_ll + proxy_ll
end

const ArrayPoissonModel = B.PoissonCountModel{
    GlobalInterception,TimeDecayDynamics,GlobalHomeAdvantage,Tuple{},
    PoissonObservation,ArrayClampGuard,
}
const ArraySharedFunnelModel = B.PoissonCountModel{
    GlobalInterception,TimeDecayDynamics,GlobalHomeAdvantage,Tuple{},O,ArrayClampGuard,
} where {O<:B.SharedKappaJoint}
const ArrayHierarchicalFunnelModel = B.PoissonCountModel{
    GlobalInterception,TimeDecayDynamics,GlobalHomeAdvantage,Tuple{},O,ArrayClampGuard,
} where {O<:B.HierarchicalKappaJoint}

_center(n_teams) = Matrix{Float64}(I, n_teams, n_teams) .- 1.0 / n_teams

function GPH_PG.build_turing_model(model::ArrayPoissonModel, fs)
    z = B.cb_design(model, fs)
    return array_poisson_engine(model, z, _center(z.n_teams))
end
function GPH_PG.build_turing_model(model::ArraySharedFunnelModel, fs)
    z = B.cb_design(model, fs)
    return array_shared_funnel_engine(model, z, _center(z.n_teams))
end
function GPH_PG.build_turing_model(model::ArrayHierarchicalFunnelModel, fs)
    z = B.cb_design(model, fs)
    return array_hierarchical_funnel_engine(model, z, _center(z.n_teams))
end

function optimized_model(reference)
    return B.PoissonCountModel(
        reference.interception,
        reference.dynamics,
        reference.home_advantage,
        reference.covariates,
        reference.observation,
        ArrayClampGuard(),
    )
end

models(; optimized::Bool = true) = [
    (name, optimized ? optimized_model(model) : model)
    for (name, model) in reference_models()
]

"""
    funnel_score_grid!(grid, μ_h, μ_a, κ_h, κ_a)

Fill an existing 12×12 (or caller-sized) score tensor for one posterior draw.
Rows and columns represent goals `0:(size-1)`. The recurrence is the exact
Poisson mass calculation at `λ_h = κ_h μ_h`, `λ_a = κ_a μ_a`; no renormalisation
hides the finite-grid tail. Production inference applies this same count family
to every posterior rate draw through `CountLatents`.
"""
function funnel_score_grid!(grid::AbstractMatrix{Float64}, μ_h::Real, μ_a::Real,
                            κ_h::Real, κ_a::Real)
    λ_h = Float64(μ_h * κ_h)
    λ_a = Float64(μ_a * κ_a)
    λ_h > 0.0 && λ_a > 0.0 || error("funnel rates must be strictly positive")
    grid[1, 1] = exp(-λ_h - λ_a)
    for home in 2:size(grid, 1)
        grid[home, 1] = grid[home - 1, 1] * λ_h / (home - 1)
    end
    for away in 2:size(grid, 2)
        grid[1, away] = grid[1, away - 1] * λ_a / (away - 1)
    end
    for away in 2:size(grid, 2), home in 2:size(grid, 1)
        grid[home, away] = grid[home, away - 1] * λ_a / (away - 1)
    end
    return grid
end

function funnel_score_grid(μ_h::Real, μ_a::Real, κ_h::Real, κ_a::Real;
                           max_goals::Int = 12)
    max_goals >= 1 || error("max_goals must be positive")
    return funnel_score_grid!(Matrix{Float64}(undef, max_goals, max_goals),
                              μ_h, μ_a, κ_h, κ_a)
end

# ==============================================================================
# 3. Deterministic density, gradient, and allocation verification
# ==============================================================================

function replay_performance(tape, gradient, θ)
    for _ in 1:30
        ReverseDiff.gradient!(gradient, tape, θ)
    end
    allocated_bytes = typemax(Int)
    for _ in 1:10
        allocated_bytes = min(
            allocated_bytes,
            @allocated(ReverseDiff.gradient!(gradient, tape, θ)),
        )
    end
    best = minimum(@elapsed(ReverseDiff.gradient!(gradient, tape, θ)) for _ in 1:100)
    return (; allocated_bytes, gradient_ms = 1000 * best)
end

"Compare the optimized engine with the production builder at one linked θ layout."
function engine_audit(model, reference, fs; seed::Int = 25)
    candidate = GPH_PG.build_turing_model(model, fs)
    baseline = GPH_PG.build_turing_model(reference, fs)
    Random.seed!(seed)
    candidate_vi = DynamicPPL.VarInfo(candidate)
    Random.seed!(seed)
    baseline_vi = DynamicPPL.VarInfo(baseline)
    string.(collect(keys(candidate_vi))) == string.(collect(keys(baseline_vi))) ||
        error("optimized/reference sampled-site layouts differ")
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
        density_error <= 1.0e-8 || error(
            "density parity failed at displacement $displacement: $density_error")
        ReverseDiff.gradient!(gradient, tape, point)
        comparisons = (
            ReverseDiff.gradient(f, point),
            ForwardDiff.gradient(f, point),
            ForwardDiff.gradient(r, point),
        )
        gradient_error = maximum(gph_relative_error(gradient, other) for other in comparisons)
        gradient_error <= 1.0e-8 || error(
            "gradient parity failed at displacement $displacement: $gradient_error")
        worst_density = max(worst_density, density_error)
        worst_gradient = max(worst_gradient, gradient_error)
    end
    performance = replay_performance(tape, gradient, θ)
    performance.allocated_bytes == 0 || error(
        "compiled optimized replay allocates $(performance.allocated_bytes) bytes")
    return (;
        n_parameters = length(θ),
        tape_instructions = length(raw.tape),
        performance...,
        worst_density,
        worst_gradient,
    )
end

"Prove that the work package's m02 and m03 are the same posterior."
function shared_identity_audit(m02, m03, fs; seed::Int = 25)
    left = GPH_PG.build_turing_model(m02, fs)
    right = GPH_PG.build_turing_model(m03, fs)
    Random.seed!(seed)
    lvi = DynamicPPL.link!!(DynamicPPL.VarInfo(left), left)
    Random.seed!(seed)
    rvi = DynamicPPL.link!!(DynamicPPL.VarInfo(right), right)
    string.(collect(keys(lvi))) == string.(collect(keys(rvi))) ||
        error("m02/m03 site layouts differ")
    θ = copy(lvi[:])
    θ == rvi[:] || error("m02/m03 linked prior draws differ")
    lf = DynamicPPL.LogDensityFunction(left, DynamicPPL.getlogjoint_internal, lvi)
    rf = DynamicPPL.LogDensityFunction(right, DynamicPPL.getlogjoint_internal, rvi)
    f = x -> LogDensityProblems.logdensity(lf, x)
    r = x -> LogDensityProblems.logdensity(rf, x)
    worst_density = 0.0
    worst_gradient = 0.0
    for displacement in (0.0, 0.003, -0.8, 0.8, -3.0, 3.0)
        point = θ .+ displacement .* sin.(eachindex(θ))
        density_error = abs(f(point) - r(point))
        gradient_error = gph_relative_error(
            ForwardDiff.gradient(f, point), ForwardDiff.gradient(r, point))
        worst_density = max(worst_density, density_error)
        worst_gradient = max(worst_gradient, gradient_error)
    end
    worst_density == 0.0 || error("m02/m03 densities differ by $worst_density")
    worst_gradient == 0.0 || error("m02/m03 gradients differ by $worst_gradient")
    return (; n_parameters = length(θ), worst_density, worst_gradient)
end

include(joinpath(@__DIR__, "l13_workflow.jl"))

end # module
