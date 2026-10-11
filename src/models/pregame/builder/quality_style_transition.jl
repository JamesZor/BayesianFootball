# Opt-in QS transitions. The legacy QualityStyleGRW methods remain untouched.
const TransitionQS = Union{CB_PG.TargetSeasonQualityStyleGRW,CB_PG.EntrantQualityStyleGRW}
_cb_dynamics_supported(::TransitionQS) = true
_dynamics_weighting_valid(::TransitionQS) = true
_dynamics_weighting_detail(c::TransitionQS) = _dynamics_weighting_detail(c.base)
dynamics_match_weights(c::TransitionQS,dates::Vector{Float64}) = dynamics_match_weights(c.base,dates)
_sites_dynamics(c::TransitionQS) = _sites_dynamics(c.base)
_sites_dynamics(c::CB_PG.EntrantQualityStyleGRW) = vcat(_sites_dynamics(c.base),
    [Symbol("dyn.q.z_entry"),Symbol("dyn.s.z_entry")])
dynamics_features(c::CB_PG.EntrantQualityStyleGRW) = CB_Features.AbstractFeatureConfig[c.entrant_feature]

function dynamics_design(c::CB_PG.TargetSeasonQualityStyleGRW,fs,n::Int)
    return dynamics_design(CB_PG.MultiScaleGRW(target_season_step=true),fs,n)
end
@model function _cb_dynamics_effects(c::CB_PG.TargetSeasonQualityStyleGRW,
    home_ids::Vector{Int},away_ids::Vector{Int},d::GRWDynamicsDesign,n::Int)
    state ~ to_submodel(CB_PG._qs_pair(c.base,d.initial_accumulator,d.season_accumulator,
        d.target_accumulator,n,size(d.season_accumulator,1),d.n_target,d.target_marker),false)
    return (; att_h=state.α[d.home_state_indices],def_a=state.β[d.away_state_indices],
              att_a=state.α[d.away_state_indices],def_h=state.β[d.home_state_indices])
end

function _cb_extract_dynamics(chain::Chains,c::CB_PG.TargetSeasonQualityStyleGRW,prefix::String,n::Int,fs)
    counts = CB_PG.qs_step_counts(chain,prefix)
    nh = counts.n_history - Int(counts.n_target>0)
    q = CB_PG._qs_reconstruct_trajectory(chain,"$prefix.q",n,nh,counts.n_target;target_season_step=true)
    s = CB_PG._qs_reconstruct_trajectory(chain,"$prefix.s",n,nh,counts.n_target;style=true,target_season_step=true)
    seed = Int(get(fs.data,:grw_oos_seed,0))
    micro = CB_PG._qs_oos_innovations(chain,c.base,prefix,n,seed,counts.n_target)
    counts.n_target>0 && return merge((; α=q.+s,β=s.-q,q,s),micro)
    ns = size(chain,1)*size(chain,3)
    τ = vec(Array(chain[Symbol("$prefix.τₛ")]))
    r = vec(Array(chain[Symbol("$prefix.rₛ")]))
    σ = τ ./ sqrt.(1 .+ r.^2)
    rng = CB_PG.Random.MersenneTwister(2seed+1)
    zq,zs = CB_PG.Random.randn(rng,ns,n),CB_PG.Random.randn(rng,ns,n)
    macroq = (zq .- mean(zq,dims=2)) .* σ
    macros = (zs .- mean(zs,dims=2)) .* (r.*σ)
    return (; α=q.+s,β=s.-q,q,s,oos_q=micro.oos_q.+macroq,oos_s=micro.oos_s.+macros)
end
_cb_oos_dynamics(c::CB_PG.TargetSeasonQualityStyleGRW,args...) = _cb_oos_dynamics(c.base,args...)

struct QSEntryDynamicsDesign{G}
    grw::G
    entry::CB_PG.QSEntryDesign
end

function dynamics_design(c::CB_PG.EntrantQualityStyleGRW,fs,n_matches::Int)
    g = dynamics_design(c.base,fs,n_matches)
    p = fs.data[:qs_entry_prior]
    n = Int(fs.data[:n_teams])
    ids = Int.(p.indices)
    isempty(ids) && return g
    length(unique(ids))==length(ids) && all(i -> 1<=i<=n,ids) || error("invalid QS entry indices")
    all(v -> length(v)==length(ids), (p.q_mean,p.q_sd,p.s_mean,p.s_sd)) || error("QS entry vectors mismatch")
    all(isfinite,vcat(p.q_mean,p.q_sd,p.s_mean,p.s_sd)) && all(>(0),vcat(p.q_sd,p.s_sd)) || error("invalid QS entry moments")
    # Entry nodes cannot be used for any historical likelihood row.
    history_rows = findall(t -> t<=g.n_history,fs.data[:time_indices])
    fitted_history = union(Set(fs.data[:flat_home_ids][history_rows]),Set(fs.data[:flat_away_ids][history_rows]))
    isempty(intersect(fitted_history,Set(ids))) || error("QS entrant has historical fitted rows")
    inc = setdiff(1:n,ids)
    isempty(inc) && error("QS entry centring requires incumbents")
    projection = zeros(n,n)
    for i in inc
        projection[i,i] = 1.0
        projection[i,inc] .-= 1/length(inc)
    end
    incidence = zeros(n,length(ids))
    entry_projection = zeros(n,n)
    for (j,i) in enumerate(ids)
        incidence[i,j]=1.0
        entry_projection[i,i]=1.0
    end
    entry_acc = zeros(1,g.n_rounds)
    entry_acc[:,(g.n_history+1):end] .= 1.0
    entry_target = copy(g.target_accumulator)
    g.n_target>0 && (entry_target[1,:] .= 0.0) # first target state IS the entry node
    entry = CB_PG.QSEntryDesign(projection,incidence,entry_projection,entry_acc,
        g.initial_accumulator,g.season_accumulator,g.target_accumulator,entry_target,
        Float64.(p.q_mean),Float64.(p.q_sd),Float64.(p.s_mean),Float64.(p.s_sd))
    return QSEntryDynamicsDesign(g,entry)
end

@model function _cb_dynamics_effects(c::CB_PG.EntrantQualityStyleGRW,
    home_ids::Vector{Int},away_ids::Vector{Int},d::QSEntryDynamicsDesign,n::Int)
    g = d.grw
    state ~ to_submodel(CB_PG._qs_entry_pair(c.base,d.entry,n,size(g.season_accumulator,1),
        g.n_target,g.target_marker),false)
    return (; att_h=state.α[g.home_state_indices],def_a=state.β[g.away_state_indices],
              att_a=state.α[g.away_state_indices],def_h=state.β[g.home_state_indices])
end
_cb_dynamics_effects(c::CB_PG.EntrantQualityStyleGRW,h,a,d::GRWDynamicsDesign,n) =
    _cb_dynamics_effects(c.base,h,a,d,n)

function _cb_extract_dynamics(chain::Chains,c::CB_PG.EntrantQualityStyleGRW,prefix::String,n::Int,fs)
    d = dynamics_design(c,fs,length(fs.data[:flat_home_ids]))
    d isa GRWDynamicsDesign && return _cb_extract_dynamics(chain,c.base,prefix,n,fs)
    g,p = d.grw,fs.data[:qs_entry_prior]
    ns = size(chain,1)*size(chain,3)
    sites(label,count) = hcat([vec(Array(chain[CB_PG._grw_chain_symbol(chain,label,i)])) for i in 1:count]...)
    scale(label,style) = begin
        τ = vec(Array(chain[Symbol("$prefix.τ$label")]))
        r = vec(Array(chain[Symbol("$prefix.r$label")]))
        σ = τ ./ sqrt.(1 .+ r.^2)
        reshape(style ? r.*σ : σ,ns,1)
    end
    function trajectory(side,means,sds,style)
        entry = reshape(means,1,:) .+ sites("$prefix.$side.z_entry",length(means)).*reshape(sds,1,:)
        init = sites("$prefix.$side.z_init",n).*scale("₀",style)
        raw = reshape(init,ns,n,1).*reshape(g.initial_accumulator,1,1,:)
        for j in 1:size(g.season_accumulator,1)
            z = hcat([vec(Array(chain[CB_PG._grw_chain_symbol(chain,"$prefix.$side.z_season",i,j)])) for i in 1:n]...)
            raw .+= reshape(z.*scale("ₛ",style),ns,n,1).*reshape(g.season_accumulator[j,:],1,1,:)
        end
        # Use exactly the in-model matrix maps/order for reconstruction parity.
        states = Array{Float64}(undef,n,g.n_rounds,ns)
        for draw in 1:ns
            states[:,:,draw] = d.entry.incumbent_projection * raw[draw,:,:]
            states[:,:,draw] .+= d.entry.entry_incidence * (reshape(entry[draw,:],:,1)*d.entry.entry_accumulator)
        end
        for j in 1:g.n_target
            z = hcat([vec(Array(chain[CB_PG._grw_chain_symbol(chain,"$prefix.$side.z_target",i,j)])) for i in 1:n]...)
            step = z.*scale("ₖ",style)
            for draw in 1:ns
                states[:,:,draw] .+= d.entry.incumbent_projection * (reshape(step[draw,:],:,1)*reshape(g.target_accumulator[j,:],1,:))
                states[:,:,draw] .+= d.entry.entry_projection * (reshape(step[draw,:],:,1)*reshape(d.entry.entry_target[j,:],1,:))
            end
        end
        return states,entry
    end
    q,eq = trajectory("q",p.q_mean,p.q_sd,false)
    s,es = trajectory("s",p.s_mean,p.s_sd,true)
    seed = Int(get(fs.data,:grw_oos_seed,0))
    # Forecast the next target period: incumbent-centred micro; entrant micro only
    # after the first target state. At the opener use the sampled entry sites.
    τ,r = vec(Array(chain[Symbol("$prefix.τₖ")])),vec(Array(chain[Symbol("$prefix.rₖ")]))
    σ = τ ./ sqrt.(1 .+ r.^2)
    rng = CB_PG.Random.MersenneTwister(seed)
    zq,zs = CB_PG.Random.randn(rng,ns,n),CB_PG.Random.randn(rng,ns,n)
    mq,ms = zq.*σ,(zs.*σ).*r
    oosq,ooss = mq*transpose(d.entry.incumbent_projection),ms*transpose(d.entry.incumbent_projection)
    if g.n_target==0
        oosq[:,p.indices] .= eq
        ooss[:,p.indices] .= es
    else
        oosq[:,p.indices] .= mq[:,p.indices]
        ooss[:,p.indices] .= ms[:,p.indices]
    end
    return (; α=q.+s,β=s.-q,q,s,oos_q=oosq,oos_s=ooss)
end
_cb_oos_dynamics(c::CB_PG.EntrantQualityStyleGRW,args...) = _cb_oos_dynamics(c.base,args...)
