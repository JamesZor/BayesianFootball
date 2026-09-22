# Definitions only. Include this module before deserializing its prototype fits.
module MomentumGRW

include(joinpath(@__DIR__, "../../../current_development/grw_player_hybrid/l01_loader.jl"))
include(joinpath(@__DIR__, "../../../current_development/grw_player_hybrid/l02_evaluation.jl"))
import Turing
import CSV
import Serialization
const B = GPH_PG.Builder

Base.@kwdef struct MomentumGRWConfig
    experiment::String = "scottish_lower_momentum_grw"
    smoke_experiment::String = "smoke_scottish_lower_momentum_grw"
    save_root::String = joinpath(@__DIR__, "results")
    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos::Int = 710
    smoke_folds::Vector{Int} = [1, 20, 40]
end

"""
Damped local-linear trend, with the exact macro/level priors of MultiScaleGRW.
Velocity starts at zero at the target-season boundary; a step is a match-biweek,
NOT an individual fixture. Attack and defence have separate persistence/scales.

Only innovations η₁,…,ηₖ₋₁ enter observed positions. ηₖ is integrated out of the
conditional-mean forecast, not sampled as a prior-only nuisance parameter.
For K ≤ 1 the entire velocity block is structurally absent. No-target folds are
exactly the first-order model, including its chain site names.
"""
Base.@kwdef struct MomentumMultiScaleGRW{L,P,A,D} <: GPH_PG.AbstractDynamicsConfig
    level::L = MultiScaleGRW()
    persistence::P = Beta(2, 2)
    attack_velocity::A = Gamma(2, 0.0075)
    defence_velocity::D = Gamma(2, 0.006)
end

B._cb_dynamics_supported(::MomentumMultiScaleGRW) = true
B._dynamics_weighting_valid(::MomentumMultiScaleGRW) = true
B._dynamics_weighting_detail(::MomentumMultiScaleGRW) =
    "unit likelihood weights; macro GRW and damped micro velocity; conditional-mean forecast"
B.dynamics_match_weights(::MomentumMultiScaleGRW, dates::Vector{Float64}) = ones(length(dates))
B._sites_dynamics(c::MomentumMultiScaleGRW) = vcat(B._sites_dynamics(c.level),
    [Symbol("dyn.$side.$site") for side in ("α", "β") for site in ("φ", "σᵥ", "z_velocity")])

struct MomentumDesign{G,V}
    grw::G
    # [j,k]: coefficient of η_j in v_(k-1), before accumulating position.
    lag::Matrix{Int}
    mask::Matrix{Float64}
    center::Matrix{Float64}
    active::V
end

function momentum_design(grw, n_teams::Int)
    k = grw.n_target
    lag = [max(t - 1 - j, 0) for j in 1:max(k-1, 0), t in 1:k]
    mask = [Float64(j < t) for j in 1:max(k-1, 0), t in 1:k]
    center = Matrix{Float64}(I, n_teams, n_teams) .- 1.0 / n_teams
    return MomentumDesign(grw, lag, mask, center, Val(k >= 2))
end
B.dynamics_design(c::MomentumMultiScaleGRW, fs, n::Int) =
    momentum_design(B.dynamics_design(c.level, fs, n), Int(fs.data[:n_teams]))

# Polynomial representation avoids (1-φ^n)/(1-φ), its cancellation near 1,
# and cumsum/hcat scratch. No sampled-value branch or recurrence on the tape.
# n is fixed design data, never a sampled value. The n=0 specialization avoids
# ReverseDiff's generic power pullback 0*x^-1 producing NaN at φ=0.
polynomial_power(x, n::Real) = n == 0 ? one(x) : x^Int(n)
# ReverseDiff's broadcast oracle dualizes even constant integer arguments.
# Exponents are immutable design data, so deliberately do not differentiate n.
polynomial_power(x, n::ForwardDiff.Dual) = polynomial_power(x, ForwardDiff.value(n))
# A scalar broadcast takes ReverseDiff's allocating tracker_∇broadcast path.
# Lift a scalar to a length-one TrackedArray WITHOUT changing the sampled site
# or distribution. The instruction owns its buffer; no global/shared scratch.
# Extends only our own operation, not Base or any package's mathematical methods.
array_scalar(x::Real) = [x]
array_scalar(x::AbstractArray) = x
function array_scalar(x::ReverseDiff.TrackedReal{V,D}) where {V,D}
    tp = ReverseDiff.tape(x)
    out = ReverseDiff.track([ReverseDiff.value(x)], D, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, array_scalar, x, out)
    return out
end
function ReverseDiff.special_forward_exec!(i::ReverseDiff.SpecialInstruction{typeof(array_scalar)})
    ReverseDiff.pull_value!(i.input)
    ReverseDiff.value(i.output)[1] = ReverseDiff.value(i.input)
    return nothing
end
function ReverseDiff.special_reverse_exec!(i::ReverseDiff.SpecialInstruction{typeof(array_scalar)})
    ReverseDiff.increment_deriv!(i.input, ReverseDiff.deriv(i.output)[1])
    ReverseDiff.unseed!(i.output)
    return nothing
end

velocity_kernel(φ, design::MomentumDesign) =
    polynomial_power.(array_scalar(φ), design.lag) .* design.mask
momentum_positions(z, σ, φ, design::MomentumDesign) =
    ((z .* array_scalar(σ)) * velocity_kernel(φ, design)) * design.grw.target_accumulator

Turing.@model function velocity_positions(prior, scale_prior, design, n_teams, ::Val{true})
    φ ~ prior
    σᵥ ~ scale_prior
    z_velocity ~ Turing.filldist(Normal(), n_teams, design.grw.n_target - 1)
    return momentum_positions(z_velocity, σᵥ, φ, design)
end
Turing.@model function velocity_positions(prior, scale_prior, design, n_teams, ::Val{false})
    return nothing
end
add_velocity(level, ::Nothing) = level
add_velocity(level, velocity) = level .+ velocity

# Same sites, order and priors as the src GRW; raw states only. The installed
# ReverseDiff lacks a mean(...; dims=1) rule, so centering is a constant matrix
# multiply in momentum_side rather than a scalarizing keyword reduction.
Turing.@model function raw_level(c, initial_scale, season_scale, target_scale,
                                 design, n_teams, ::Val{true})
    σ₀ ~ initial_scale
    σₛ ~ season_scale
    σₖ ~ target_scale
    z_init ~ Turing.filldist(c.level.z₀, n_teams)
    z_season ~ Turing.filldist(c.level.zₛ, n_teams, design.grw.n_history - 1)
    z_target ~ Turing.filldist(c.level.zₖ, n_teams, design.grw.n_target)
    initial = reshape(z_init .* array_scalar(σ₀), n_teams, 1) * design.grw.initial_accumulator
    season = (z_season .* array_scalar(σₛ)) * design.grw.season_accumulator
    target = (z_target .* array_scalar(σₖ)) * design.grw.target_accumulator
    return initial .+ season .+ target
end
Turing.@model function raw_level(c, initial_scale, season_scale, target_scale,
                                 design, n_teams, ::Val{false})
    σ₀ ~ initial_scale
    σₛ ~ season_scale
    z_init ~ Turing.filldist(c.level.z₀, n_teams)
    z_season ~ Turing.filldist(c.level.zₛ, n_teams, design.grw.n_history - 1)
    initial = reshape(z_init .* array_scalar(σ₀), n_teams, 1) * design.grw.initial_accumulator
    season = (z_season .* array_scalar(σₛ)) * design.grw.season_accumulator
    return initial .+ season
end

Turing.@model function momentum_side(c, side_scale, initial_scale, season_scale,
                                     target_scale, design, n_teams, ::Val{true})
    # No extra prefix: baseline sites remain α.σ₀ etc.
    level ~ DynamicPPL.to_submodel(raw_level(c, initial_scale, season_scale,
        target_scale, design, n_teams, Val(true)), false)
    velocity ~ DynamicPPL.to_submodel(velocity_positions(
        c.persistence, side_scale, design, n_teams, design.active), false)
    return design.center * add_velocity(level, velocity)
end
Turing.@model function momentum_side(c, side_scale, initial_scale, season_scale,
                                     target_scale, design, n_teams, ::Val{false})
    level ~ DynamicPPL.to_submodel(raw_level(c, initial_scale, season_scale,
        target_scale, design, n_teams, Val(false)), false)
    return design.center * level
end
Turing.@model function momentum_pair(c, design, n_teams)
    α ~ DynamicPPL.to_submodel(momentum_side(c, c.attack_velocity,
        c.level.α_σ₀, c.level.α_σₛ, c.level.α_σₖ, design, n_teams, design.grw.target_marker))
    β ~ DynamicPPL.to_submodel(momentum_side(c, c.defence_velocity,
        c.level.β_σ₀, c.level.β_σₛ, c.level.β_σₖ, design, n_teams, design.grw.target_marker))
    return (; α, β)
end
Turing.@model function B._cb_dynamics_effects(c::MomentumMultiScaleGRW,
        home_ids::Vector{Int}, away_ids::Vector{Int}, design::MomentumDesign, n_teams::Int)
    state ~ DynamicPPL.to_submodel(momentum_pair(c, design, n_teams), false)
    return (; att_h=state.α[design.grw.home_state_indices],
              def_a=state.β[design.grw.away_state_indices],
              att_a=state.α[design.grw.away_state_indices],
              def_h=state.β[design.grw.home_state_indices])
end

site(chain, prefix, indices...) = vec(Array(chain[GPH_PG._grw_chain_symbol(chain, prefix, indices...)]))

"Independent scalar recurrence reconstruction; terminal innovation marginalized."
function reconstruct_side(chain, prefix, n_teams, n_history, n_target)
    states = GPH_PG._grw_reconstruct_trajectory(chain, prefix, n_teams, n_history, n_target)
    n_draws = size(states, 3)
    forecast_velocity = zeros(n_teams, n_draws)
    n_target < 2 && return (; states, velocity=forecast_velocity)
    φ = site(chain, "$prefix.φ")
    σ = site(chain, "$prefix.σᵥ")
    z = Array{Float64}(undef, n_teams, n_target-1, n_draws)
    for k in 1:n_target-1, team in 1:n_teams
        z[team, k, :] = site(chain, "$prefix.z_velocity", team, k)
    end
    for s in 1:n_draws
        velocity = zeros(n_teams)
        position = zeros(n_teams)
        for k in 1:n_target
            position .+= velocity
            states[:, n_history+k, s] .+= position .- mean(position)
            velocity .*= φ[s]
            if k < n_target
                velocity .+= σ[s] .* z[:, k, s]
            end
        end
        forecast_velocity[:, s] = velocity .- mean(velocity)
    end
    return (; states, velocity=forecast_velocity)
end
function B._cb_extract_dynamics(chain::Chains, ::MomentumMultiScaleGRW, prefix::String, n_teams::Int)
    counts = GPH_PG.grw_step_counts(chain, prefix)
    a = reconstruct_side(chain, "$prefix.α", n_teams, counts.n_history, counts.n_target)
    b = reconstruct_side(chain, "$prefix.β", n_teams, counts.n_history, counts.n_target)
    return (; α=a.states, β=b.states, velocity_α=a.velocity, velocity_β=b.velocity)
end
function B._cb_oos_dynamics(::MomentumMultiScaleGRW, draw, lineup_map, match_id::Int,
                            home_index::Int, away_index::Int, n_samples::Int)
    forecast(states, velocity, team) = team > 0 ?
        vec(states[team, end, :]) .+ vec(velocity[team, :]) : zeros(n_samples)
    return (; att_h=forecast(draw.α, draw.velocity_α, home_index),
              def_a=forecast(draw.β, draw.velocity_β, away_index),
              att_a=forecast(draw.α, draw.velocity_α, away_index),
              def_h=forecast(draw.β, draw.velocity_β, home_index))
end

# Allocation-free implementation of the SAME clamp. A distinct prototype type
# selects our minimal engine without replacing methods on any src-owned type.
struct ArrayClampGuard <: B.AbstractRateGuard
    lo::Vector{Float64}
    hi::Vector{Float64}
end
ArrayClampGuard() = ArrayClampGuard([-10.0], [10.0])
B.apply_guard(g::ArrayClampGuard, η) = clamp.(η, g.lo, g.hi)
B.guard_describe(g::ArrayClampGuard) = "array clamp to [$(only(g.lo)), $(only(g.hi))]"

Turing.@model function array_interception(c)
    μ ~ c.μ
    return array_scalar(μ)
end
Turing.@model function array_home(c)
    γ_global ~ c.γ_global
    return array_scalar(γ_global)
end

# Optimized controls are algebraically unchanged. Their sample sites and order
# match src, so equivalence is checked at identical θ, not by resampling.
Turing.@model function first_order_side(c, initial_scale, season_scale, target_scale, d, n)
    raw ~ DynamicPPL.to_submodel(raw_level(c, initial_scale, season_scale,
        target_scale, d, n, d.grw.target_marker), false)
    return d.center * raw
end
Turing.@model function optimized_states(c::MultiScaleGRW, d, n)
    cfg = MomentumMultiScaleGRW(level=c)
    α ~ DynamicPPL.to_submodel(first_order_side(cfg,c.α_σ₀,c.α_σₛ,c.α_σₖ,d,n))
    β ~ DynamicPPL.to_submodel(first_order_side(cfg,c.β_σ₀,c.β_σₛ,c.β_σₖ,d,n))
    return (; α, β)
end
Turing.@model function optimized_states(c::MomentumMultiScaleGRW, d, n)
    states ~ DynamicPPL.to_submodel(momentum_pair(c,d,n), false)
    return states
end
Turing.@model function optimized_states(c::TimeDecayDynamics, center, n)
    σ_a ~ c.σ_att
    σ_d ~ c.σ_def
    raw_a ~ Turing.filldist(Normal(),n)
    raw_d ~ Turing.filldist(Normal(),n)
    α = center * (raw_a .* array_scalar(σ_a))
    β = center * (raw_d .* array_scalar(σ_d))
    return (; α, β)
end

optimized_design(c::MomentumMultiScaleGRW, fs, z) = z.dynamics_data
optimized_design(c::MultiScaleGRW, fs, z) = momentum_design(z.dynamics_data,z.n_teams)
optimized_design(c::TimeDecayDynamics, fs, z) =
    Matrix{Float64}(I,z.n_teams,z.n_teams) .- 1.0/z.n_teams
state_indices(d::MomentumDesign,z) = (d.grw.home_state_indices,d.grw.away_state_indices)
state_indices(d::Matrix,z) = (z.home_ids,z.away_ids)

Turing.@model function array_poisson_engine(config, z, design, hidx, aidx)
    inter ~ DynamicPPL.to_submodel(array_interception(config.interception))
    ha ~ DynamicPPL.to_submodel(array_home(config.home_advantage))
    dyn ~ DynamicPPL.to_submodel(optimized_states(config.dynamics,design,z.n_teams))
    η_h = B.apply_guard(config.guard, inter .+ ha .+ dyn.α[hidx] .+ dyn.β[aidx])
    η_a = B.apply_guard(config.guard, inter .+ dyn.α[aidx] .+ dyn.β[hidx])
    ll_h = z.home_goals .* η_h .- exp.(η_h) .- z.log_fact_h
    ll_a = z.away_goals .* η_a .- exp.(η_a) .- z.log_fact_a
    Turing.@addlogprob! sum(ll_h .* z.match_weights) + sum(ll_a .* z.match_weights)
end
const ArrayPoissonModel = B.PoissonCountModel{GlobalInterception,T,GlobalHomeAdvantage,
    Tuple{},PoissonObservation,ArrayClampGuard} where {T<:Union{TimeDecayDynamics,MultiScaleGRW,MomentumMultiScaleGRW}}
function GPH_PG.build_turing_model(model::ArrayPoissonModel, fs)
    z = B.cb_design(model,fs)
    design = optimized_design(model.dynamics,fs,z)
    hidx,aidx = state_indices(design,z)
    return array_poisson_engine(model,z,design,hidx,aidx)
end

function poisson_model(name, dynamics; optimized=true)
    model = CountModelBuilder(Symbol(name)) |>
        add(GlobalInterception()) |> add(GlobalHomeAdvantage()) |>
        add(dynamics) |> add(PoissonObservation()) |> add(B.ClampGuard()) |> build
    # Builder validation hardcodes the two src guard types. Validate the exact
    # recipe above, then replace only the representation of its default bounds.
    optimized || return model
    return B.PoissonCountModel(model.interception, model.dynamics,
        model.home_advantage, model.covariates, model.observation, ArrayClampGuard())
end
models(; optimized=true) = [
    ("m01_poisson_time_decay", poisson_model("m01_poisson_time_decay", TimeDecayDynamics(days_half_life=180.0); optimized)),
    ("m02_poisson_grw_1st_order", poisson_model("m02_poisson_grw_1st_order", MultiScaleGRW(); optimized)),
    ("m03_poisson_momentum_grw", poisson_model("m03_poisson_momentum_grw", MomentumMultiScaleGRW(); optimized)),
]

"Linked-space tape audit with original-engine parity and warmup-scale probes."
function allocation_audit(model, reference, fs; seed=22)
    tm = GPH_PG.build_turing_model(model,fs)
    ref = GPH_PG.build_turing_model(reference,fs)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(tm)
    Random.seed!(seed)
    rv = DynamicPPL.VarInfo(ref)
    string.(collect(keys(vi))) == string.(collect(keys(rv))) || error("site layouts differ")
    vi[:] == rv[:] || error("prior draw layouts differ at shared RNG seed")
    vi = DynamicPPL.link!!(vi,tm)
    rv = DynamicPPL.link!!(rv,ref)
    θ = copy(vi[:])
    θ == rv[:] || error("linked parameter layouts differ")
    ld = DynamicPPL.LogDensityFunction(tm,DynamicPPL.getlogjoint_internal,vi)
    refld = DynamicPPL.LogDensityFunction(ref,DynamicPPL.getlogjoint_internal,rv)
    f = x -> LogDensityProblems.logdensity(ld,x)
    r = x -> LogDensityProblems.logdensity(refld,x)
    raw = ReverseDiff.GradientTape(f,θ)
    tape = ReverseDiff.compile(raw)
    g = similar(θ)
    worst_density = 0.0
    worst_gradient = 0.0
    for δ in (0.0, 0.003, -0.8, 0.8, -3.0, 3.0)
        p = θ .+ δ .* sin.(eachindex(θ))
        isfinite(f(p)) && isfinite(r(p)) || error("nonfinite density at displacement $δ")
        density_error = abs(f(p)-r(p))
        density_error <= 1e-9 || error("density parity failed: $density_error")
        ReverseDiff.gradient!(g,tape,p)
        fresh = ReverseDiff.gradient(f,p)
        oracle = ForwardDiff.gradient(f,p)
        reference_gradient = ForwardDiff.gradient(r,p)
        err = maximum(gph_relative_error(g,other) for other in (fresh,oracle,reference_gradient))
        err <= 1e-8 || error("gradient parity failed at displacement $δ: $err")
        worst_density = max(worst_density,density_error)
        worst_gradient = max(worst_gradient,err)
    end
    perf = replay_performance(tape,g,θ)
    perf.allocated_bytes == 0 || error("compiled replay allocates $(perf.allocated_bytes) bytes")
    return (; n_parameters=length(θ), tape_instructions=length(raw.tape),
              perf..., worst_density, worst_gradient)
end

# Function barrier: @allocated at top-level can include dynamic dispatch boxing.
function replay_performance(tape,g,θ)
    for _ in 1:30
        ReverseDiff.gradient!(g,tape,θ)
    end
    allocated_bytes = @allocated ReverseDiff.gradient!(g,tape,θ)
    best = minimum(@elapsed(ReverseDiff.gradient!(g,tape,θ)) for _ in 1:100)
    return (; allocated_bytes, gradient_ms=1000best)
end

include(joinpath(@__DIR__, "l11_workflow.jl"))

end # module
