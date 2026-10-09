# Independent quality/style innovations on MultiScaleGRW's state-count geometry.
# q moves log-rate supremacy; s moves log-rate intensity. Attack α = q+s,
# defence β = s-q (the defence sign is concedes-more).

"""
    QualityStyleGRW(; τ₀, τₛ, τₖ, r₀, rₛ, rₖ, z₀, zₛ, zₖ)

Non-centred, zero-centred league-wide quality/style random walks. At each scale,
`σ_q = τ / sqrt(1+r²)` and `σ_s = r*σ_q`; hence both log-rate sides have
marginal innovation variance `τ²`. `r` is one pooled ratio per time scale.
The default ratios are the predeclared market-centred priors. The weak-r arm
passes `LogNormal(0, 1)` for all three `r` keywords.
"""
struct QualityStyleGRW <: AbstractDynamicsConfig
    z₀::ContinuousUnivariateDistribution
    zₛ::ContinuousUnivariateDistribution
    zₖ::ContinuousUnivariateDistribution
    τ₀::ContinuousUnivariateDistribution
    τₛ::ContinuousUnivariateDistribution
    τₖ::ContinuousUnivariateDistribution
    r₀::ContinuousUnivariateDistribution
    rₛ::ContinuousUnivariateDistribution
    rₖ::ContinuousUnivariateDistribution
end

function QualityStyleGRW(;
    z₀::ContinuousUnivariateDistribution = Normal(0, 1),
    zₛ::ContinuousUnivariateDistribution = Normal(0, 1),
    zₖ::ContinuousUnivariateDistribution = Normal(0, 1),
    τ₀::ContinuousUnivariateDistribution = Gamma(2, 0.08),
    τₛ::ContinuousUnivariateDistribution = Gamma(2, 0.0425),
    τₖ::ContinuousUnivariateDistribution = Gamma(2, 0.0135),
    r₀::ContinuousUnivariateDistribution = LogNormal(log(0.3), 0.35),
    rₛ::ContinuousUnivariateDistribution = LogNormal(log(0.4), 0.6),
    rₖ::ContinuousUnivariateDistribution = LogNormal(log(0.3), 0.35),
)
    return QualityStyleGRW(z₀, zₛ, zₖ, τ₀, τₛ, τₖ, r₀, rₛ, rₖ)
end

"Compute one centred non-centred walk with observed target steps."
@model function _qs_trajectory(z_initial_prior, z_season_prior, z_target_prior,
                               σ₀, σₛ, σₖ,
                               initial_accumulator::Matrix{Float64},
                               season_accumulator::Matrix{Float64},
                               target_accumulator::Matrix{Float64},
                               n_teams::Int, n_season::Int, n_target::Int)
    z_init ~ filldist(z_initial_prior, n_teams)
    z_season ~ filldist(z_season_prior, n_teams, n_season)
    z_target ~ filldist(z_target_prior, n_teams, n_target)
    initial_states = reshape(z_init .* σ₀, n_teams, 1) * initial_accumulator
    season_states = (z_season .* σₛ) * season_accumulator
    target_states = (z_target .* σₖ) * target_accumulator
    raw = initial_states .+ season_states .+ target_states
    return _grw_centre(raw, n_teams)
end

"No unused target sites when the training fold has no observed target step."
@model function _qs_trajectory_no_target(z_initial_prior, z_season_prior,
                                         σ₀, σₛ,
                                         initial_accumulator::Matrix{Float64},
                                         season_accumulator::Matrix{Float64},
                                         n_teams::Int, n_season::Int)
    z_init ~ filldist(z_initial_prior, n_teams)
    z_season ~ filldist(z_season_prior, n_teams, n_season)
    initial_states = reshape(z_init .* σ₀, n_teams, 1) * initial_accumulator
    season_states = (z_season .* σₛ) * season_accumulator
    return _grw_centre(initial_states .+ season_states, n_teams)
end

"One pooled τ and r per scale, with two independent non-centred q/s walks."
@model function _qs_pair(config::QualityStyleGRW,
                         initial::Matrix{Float64}, season::Matrix{Float64},
                         target::Matrix{Float64}, n_teams::Int, n_season::Int,
                         n_target::Int, ::Val{true})
    τ₀ ~ config.τ₀
    τₛ ~ config.τₛ
    τₖ ~ config.τₖ
    r₀ ~ config.r₀
    rₛ ~ config.rₛ
    rₖ ~ config.rₖ
    σ_q₀ = τ₀ / sqrt(1 + r₀^2)
    σ_qₛ = τₛ / sqrt(1 + rₛ^2)
    σ_qₖ = τₖ / sqrt(1 + rₖ^2)
    q ~ to_submodel(_qs_trajectory(config.z₀, config.zₛ, config.zₖ,
                                    σ_q₀, σ_qₛ, σ_qₖ, initial, season, target,
                                    n_teams, n_season, n_target))
    s ~ to_submodel(_qs_trajectory(config.z₀, config.zₛ, config.zₖ,
                                    r₀ * σ_q₀, rₛ * σ_qₛ, rₖ * σ_qₖ,
                                    initial, season, target,
                                    n_teams, n_season, n_target))
    return (; α = q .+ s, β = s .- q, q, s)
end

@model function _qs_pair(config::QualityStyleGRW,
                         initial::Matrix{Float64}, season::Matrix{Float64},
                         ::Matrix{Float64}, n_teams::Int, n_season::Int,
                         ::Int, ::Val{false})
    τ₀ ~ config.τ₀
    τₛ ~ config.τₛ
    r₀ ~ config.r₀
    rₛ ~ config.rₛ
    σ_q₀ = τ₀ / sqrt(1 + r₀^2)
    σ_qₛ = τₛ / sqrt(1 + rₛ^2)
    q ~ to_submodel(_qs_trajectory_no_target(config.z₀, config.zₛ,
                                              σ_q₀, σ_qₛ, initial, season,
                                              n_teams, n_season))
    s ~ to_submodel(_qs_trajectory_no_target(config.z₀, config.zₛ,
                                              r₀ * σ_q₀, rₛ * σ_qₛ, initial, season,
                                              n_teams, n_season))
    return (; α = q .+ s, β = s .- q, q, s)
end

@model function build_dynamics(config::QualityStyleGRW, n_teams::Int,
                               n_history::Int, n_target::Int)
    acc = grw_accumulators(n_history, n_target)
    state ~ to_submodel(_qs_pair(config, acc.initial, acc.season, acc.target,
                                  n_teams, size(acc.season, 1), n_target,
                                  n_target == 0 ? Val(false) : Val(true)), false)
    return state
end

"Reconstruct q or s from a chain, using pooled τ and r sites at each scale."
function _qs_reconstruct_trajectory(chain::Chains, prefix::String, n_teams::Int,
                                    n_history::Int, n_target::Int; style::Bool = false)
    n_samples = size(chain, 1) * size(chain, 3)
    base = first(split(prefix, "."))
    scale(label) = begin
        τ = vec(Array(chain[_grw_chain_symbol(chain, "$base.τ$label")]))
        r = vec(Array(chain[_grw_chain_symbol(chain, "$base.r$label")]))
        σ_q = τ ./ sqrt.(1 .+ r.^2)
        reshape(style ? r .* σ_q : σ_q, n_samples, 1, 1)
    end
    z_init = Array{Float64}(undef, n_samples, n_teams, 1)
    for team in 1:n_teams
        z_init[:, team, 1] = vec(Array(chain[_grw_chain_symbol(chain, "$prefix.z_init", team)]))
    end
    z_season = Array{Float64}(undef, n_samples, n_teams, n_history - 1)
    for step in 1:(n_history - 1), team in 1:n_teams
        z_season[:, team, step] = vec(Array(chain[_grw_chain_symbol(chain, "$prefix.z_season", team, step)]))
    end
    z_target = Array{Float64}(undef, n_samples, n_teams, n_target)
    for step in 1:n_target, team in 1:n_teams
        z_target[:, team, step] = vec(Array(chain[_grw_chain_symbol(chain, "$prefix.z_target", team, step)]))
    end
    increments = n_target == 0 ? cat(z_init .* scale("₀"), z_season .* scale("ₛ"); dims = 3) :
        cat(z_init .* scale("₀"), z_season .* scale("ₛ"), z_target .* scale("ₖ"); dims = 3)
    acc = grw_accumulators(n_history, n_target)
    # Multiply by the same accumulator maps as the in-model walk: zero-width
    # season/target matrices are safe, unlike a state-count off-by-one cumsum.
    raw = reshape(increments[:, :, 1], n_samples, n_teams, 1) .* reshape(acc.initial, 1, 1, :)
    for step in 1:(n_history - 1)
        raw .+= reshape(increments[:, :, step + 1], n_samples, n_teams, 1) .*
                 reshape(acc.season[step, :], 1, 1, :)
    end
    for step in 1:n_target
        raw .+= reshape(increments[:, :, n_history + step], n_samples, n_teams, 1) .*
                 reshape(acc.target[step, :], 1, 1, :)
    end
    return permutedims(raw .- mean(raw, dims = 2), (2, 3, 1))
end

function qs_step_counts(chain::Chains, prefix::String)
    available = String.(names(chain))
    n_target = count(name -> startswith(name, "$prefix.q.z_target[1,"), available)
    n_history = count(name -> startswith(name, "$prefix.q.z_season[1,"), available) + 1
    return (; n_history, n_target)
end

function extract_dynamics(chain::Chains, ::QualityStyleGRW, prefix::String,
                          n_teams::Int, n_history::Int, n_target::Int)
    q = _qs_reconstruct_trajectory(chain, "$prefix.q", n_teams, n_history, n_target)
    s = _qs_reconstruct_trajectory(chain, "$prefix.s", n_teams, n_history, n_target; style = true)
    return (; α = q .+ s, β = s .- q, q, s)
end

"One shared next-biweek innovation per posterior draw, centred across teams."
function _qs_oos_innovations(chain::Chains, config::QualityStyleGRW,
                             prefix::String, n_teams::Int, seed::Int, n_target::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    rng = Random.MersenneTwister(seed)
    τ = n_target == 0 ? rand(rng, config.τₖ, n_samples) :
        vec(Array(chain[_grw_chain_symbol(chain, "$prefix.τₖ")]))
    r = n_target == 0 ? rand(rng, config.rₖ, n_samples) :
        vec(Array(chain[_grw_chain_symbol(chain, "$prefix.rₖ")]))
    σ_q = τ ./ sqrt.(1 .+ r.^2)
    zq = randn(rng, n_samples, n_teams)
    zs = randn(rng, n_samples, n_teams)
    q = (zq .- mean(zq, dims = 2)) .* σ_q
    s = (zs .- mean(zs, dims = 2)) .* (r .* σ_q)
    return (; oos_q = q, oos_s = s)
end
