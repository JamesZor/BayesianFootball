# Opt-in forms preserve QualityStyleGRW's established serialized representation.
struct TargetSeasonQualityStyleGRW <: AbstractDynamicsConfig
    base::QualityStyleGRW
end
struct EntrantQualityStyleGRW{F} <: AbstractDynamicsConfig
    base::QualityStyleGRW
    entrant_feature::F
end

"Dense entry-node maps. Historical entrant rows are zero; only incumbents centre."
struct QSEntryDesign
    incumbent_projection::Matrix{Float64}
    entry_incidence::Matrix{Float64}
    entry_projection::Matrix{Float64}
    entry_accumulator::Matrix{Float64}
    incumbent_initial::Matrix{Float64}
    incumbent_season::Matrix{Float64}
    incumbent_target::Matrix{Float64}
    entry_target::Matrix{Float64}
    q_mean::Vector{Float64}
    q_sd::Vector{Float64}
    s_mean::Vector{Float64}
    s_sd::Vector{Float64}
end

# A separate entry site is used at its actual node, never at the old window start.
# At an opening fold its training-likelihood contribution is identically zero,
# but its prior draw remains an in-model site, carried into the first target node.
@model function _qs_entry_walk(z₀, zₛ, zₖ, σ₀, σₛ, σₖ,
                               design::QSEntryDesign, means, sds,
                               n_teams::Int, n_season::Int, n_target::Int, ::Val{true})
    z_init ~ filldist(z₀, n_teams)
    z_season ~ filldist(zₛ, n_teams, n_season)
    z_target ~ filldist(zₖ, n_teams, n_target)
    z_entry ~ filldist(Normal(0.0,1.0), length(means))
    entry = means .+ sds .* z_entry
    raw = reshape(z_init .* σ₀, n_teams, 1) * design.incumbent_initial .+
          (z_season .* σₛ) * design.incumbent_season .+
          (z_target .* σₖ) * design.incumbent_target
    states = design.incumbent_projection * raw .+
             design.entry_incidence * (reshape(entry, length(means), 1) * design.entry_accumulator) .+
             design.entry_projection * ((z_target .* σₖ) * design.entry_target)
    return (; states, entry)
end

@model function _qs_entry_walk(z₀, zₛ, zₖ, σ₀, σₛ, σₖ,
                               design::QSEntryDesign, means, sds,
                               n_teams::Int, n_season::Int, n_target::Int, ::Val{false})
    z_init ~ filldist(z₀, n_teams)
    z_season ~ filldist(zₛ, n_teams, n_season)
    z_entry ~ filldist(Normal(0.0,1.0), length(means))
    entry = means .+ sds .* z_entry
    raw = reshape(z_init .* σ₀, n_teams, 1) * design.incumbent_initial .+
          (z_season .* σₛ) * design.incumbent_season
    return (; states=design.incumbent_projection * raw, entry)
end

@model function _qs_entry_pair(config::QualityStyleGRW, design::QSEntryDesign,
                               n_teams::Int, n_season::Int, n_target::Int, marker)
    τ₀ ~ config.τ₀
    τₛ ~ config.τₛ
    τₖ ~ config.τₖ
    r₀ ~ config.r₀
    rₛ ~ config.rₛ
    rₖ ~ config.rₖ
    σ_q₀ = τ₀ / sqrt(1+r₀^2)
    σ_qₛ = τₛ / sqrt(1+rₛ^2)
    σ_qₖ = τₖ / sqrt(1+rₖ^2)
    q ~ to_submodel(_qs_entry_walk(config.z₀,config.zₛ,config.zₖ,σ_q₀,σ_qₛ,σ_qₖ,
        design,design.q_mean,design.q_sd,n_teams,n_season,n_target,marker))
    s ~ to_submodel(_qs_entry_walk(config.z₀,config.zₛ,config.zₖ,r₀*σ_q₀,rₛ*σ_qₛ,rₖ*σ_qₖ,
        design,design.s_mean,design.s_sd,n_teams,n_season,n_target,marker))
    return (; α=q.states .+ s.states, β=s.states .- q.states,
              q=q.states, s=s.states, entry_q=q.entry, entry_s=s.entry)
end
