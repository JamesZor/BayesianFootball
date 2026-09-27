# Fixed-scale SPFL-tier intercept and match-competition finishing deltas.
# This file extends JointGammaPoissonObservation only when CompetitionKappa is
# selected; the default SharedKappa path remains structurally unchanged.

struct LeagueDeltaFeature <: CB_Features.AbstractFeatureConfig end

_competition_level(tournament::Int) = tournament in CB_Features.Data.SCOTTISH_SPFL_TOURNAMENTS ?
    tournament : 0 # all accepted senior cups share one level

function CB_Features.add_feature!(F_data::Dict, ::LeagueDeltaFeature,
                                  ordered_ids, team_map::Dict,
                                  ds::CB_Features.Data.DataStore)
    wanted = Set(Int.(ordered_ids))
    row_by_id = Dict(Int(row.match_id) => row for row in eachrow(ds.matches)
                     if Int(row.match_id) in wanted)
    rows = [row_by_id[Int(id)] for id in ordered_ids]
    tiers = _full_tier_table(ds)

    home_tournaments = Int[tiers[(String(row.home_team), String(row.season))] for row in rows]
    away_tournaments = Int[tiers[(String(row.away_team), String(row.season))] for row in rows]
    tier_levels = sort!(unique!(vcat(copy(home_tournaments), copy(away_tournaments))))
    length(tier_levels) >= 2 || error(
        "CompetitionKappa intercept deltas require at least 2 SPFL tiers in the fitted scope; " *
        "found $(tier_levels)")
    tier_map = Dict(level => index for (index, level) in enumerate(tier_levels))

    competitions = Int[_competition_level(Int(row.tournament_id)) for row in rows]
    competition_levels = sort!(unique!(copy(competitions)))
    length(competition_levels) >= 2 || error(
        "CompetitionKappa finishing deltas require at least 2 match competitions in the fitted scope; " *
        "found $(competition_levels)")
    competition_map = Dict(level => index for (index, level) in enumerate(competition_levels))

    F_data[:league_delta_tier_home_idx] = Int[tier_map[t] for t in home_tournaments]
    F_data[:league_delta_tier_away_idx] = Int[tier_map[t] for t in away_tournaments]
    F_data[:league_delta_competition_idx] = Int[competition_map[c] for c in competitions]
    F_data[:league_delta_tier_levels] = tier_levels
    F_data[:league_delta_competition_levels] = competition_levels

    bridge = Dict{Int,NamedTuple}()
    for row in eachrow(ds.matches)
        season = String(row.season)
        home_key = (String(row.home_team), season)
        away_key = (String(row.away_team), season)
        haskey(tiers, home_key) && haskey(tiers, away_key) || continue
        home_tier = tiers[home_key]
        away_tier = tiers[away_key]
        competition = _competition_level(Int(row.tournament_id))
        haskey(tier_map, home_tier) && haskey(tier_map, away_tier) &&
            haskey(competition_map, competition) || continue
        bridge[Int(row.match_id)] = (;
            tier_home_idx = tier_map[home_tier], tier_away_idx = tier_map[away_tier],
            competition_idx = competition_map[competition])
    end
    F_data[:league_delta_oos_bridge] = bridge
    return nothing
end

_joint_observation_features(::CompetitionKappa, o) =
    CB_Features.AbstractFeatureConfig[o.feature, LeagueDeltaFeature()]

function _joint_mode_design(::CompetitionKappa, o, feature_set, n_matches::Int,
                            match_weights::Vector{Float64})
    # Build the unchanged proxy-xG block through the shared method, then replace
    # its neutral league indices with the fold-local, schedule-derived maps.
    base = _joint_observation_design(
        o, feature_set, n_matches, match_weights)
    d = feature_set.data
    required = (:league_delta_tier_home_idx, :league_delta_tier_away_idx,
                :league_delta_competition_idx, :league_delta_tier_levels,
                :league_delta_competition_levels)
    all(key -> haskey(d, key), required) || error(
        "CompetitionKappa needs LeagueDeltaFeature data; build through required_features")
    tier_h = Vector{Int}(d[:league_delta_tier_home_idx])
    tier_a = Vector{Int}(d[:league_delta_tier_away_idx])
    competition = Vector{Int}(d[:league_delta_competition_idx])
    for (name, values) in (("tier_home", tier_h), ("tier_away", tier_a),
                           ("competition", competition))
        length(values) == n_matches || error("CompetitionKappa $name length mismatch")
    end
    n_tiers = length(d[:league_delta_tier_levels])
    n_competitions = length(d[:league_delta_competition_levels])
    n_tiers >= 2 || error("CompetitionKappa requires at least 2 tiers; got $n_tiers")
    n_competitions >= 2 || error(
        "CompetitionKappa requires at least 2 competitions; got $n_competitions")
    return JointGammaPoissonDesign(
        base.pxg_h, base.pxg_a, base.log_pxg_h, base.log_pxg_a,
        base.mask_weights, base.home_idx, base.away_idx, base.n_observed,
        tier_h, tier_a, competition, n_tiers, n_competitions)
end

@model function _joint_competition_kappa_params(o::CompetitionKappaJoint,
                                                n_tiers::Int,
                                                n_competitions::Int)
    ν ~ o.shape_prior
    log_κ ~ o.log_kappa_prior
    intercept_raw ~ filldist(Normal(0.0, 1.0), n_tiers)
    kappa_raw ~ filldist(Normal(0.0, 1.0), n_competitions)
    # Centre, THEN scale, as two unfused broadcasts. Fused, `scale .* (raw .- mean(raw))` carries a
    # constant and a tracked scalar into one kernel and takes ReverseDiff's allocating
    # `tracker_∇broadcast` (AD guide §10.5); unfused, each step is a preallocated binary kernel and
    # the arithmetic is the same operations in the same order.
    intercept_centred = intercept_raw .- mean(intercept_raw)
    δ_intercept = o.kappa.intercept_scale .* intercept_centred
    kappa_centred = kappa_raw .- mean(kappa_raw)
    δ_kappa = o.kappa.kappa_scale .* kappa_centred
    return (; ν, log_κ, δ_intercept, δ_kappa)
end

@model function _observe(o::CompetitionKappaJoint,
                         η_h, η_a,
                         yh::Vector{Int}, ya::Vector{Int}, wts::Vector{Float64},
                         lfh::Vector{Float64}, lfa::Vector{Float64},
                         n_teams::Int, n_months::Int, od::JointGammaPoissonDesign)
    obs ~ to_submodel(_joint_competition_kappa_params(
        o, od.n_tiers, od.n_competitions))
    ν = obs.ν

    # Tier intercept belongs to the latent log intensity and is therefore seen by
    # both the Gamma proxy arm and goals arm. It is keyed by the attacking side.
    ξ_h = η_h .+ obs.δ_intercept[od.tier_home_idx]
    ξ_a = η_a .+ obs.δ_intercept[od.tier_away_idx]

    # The competition delta is shared by both sides and enters goals only. `+ log κ` is its own
    # (unfused) step: a tracked scalar inside the fused three-term sum would take the allocating
    # `tracker_∇broadcast` adjoint. Same additions, same order.
    ζ_h_league = ξ_h .+ obs.log_κ
    ζ_a_league = ξ_a .+ obs.log_κ
    ζ_h = ζ_h_league .+ obs.δ_kappa[od.competition_idx]
    ζ_a = ζ_a_league .+ obs.δ_kappa[od.competition_idx]
    ll_h = yh .* ζ_h .- exp.(ζ_h) .- lfh
    ll_a = ya .* ζ_a .- exp.(ζ_a) .- lfa
    goals_ll = sum(ll_h .* wts) + sum(ll_a .* wts)

    # The Gamma arm reads ξ — the tier intercept, not κ — exactly as the shared mode reads η.
    proxy_ll = _gamma_proxy_ll(ν, ξ_h, ξ_a, od)
    return goals_ll + proxy_ll
end

function _cb_extract_observation(o::CompetitionKappaJoint, chain, n_teams, feature_set)
    ν = _chain_draws(chain, Symbol("obs.ν"))
    log_κ = _chain_draws(chain, Symbol("obs.log_κ"))
    intercept_raw = _tier_chain_matrix(chain, "obs.intercept_raw")
    kappa_raw = _tier_chain_matrix(chain, "obs.kappa_raw")
    δ_intercept = o.kappa.intercept_scale .*
        (intercept_raw .- mean(intercept_raw, dims = 2))
    δ_kappa = o.kappa.kappa_scale .* (kappa_raw .- mean(kappa_raw, dims = 2))
    return (; ν, κ = exp.(log_κ), log_κ, δ_intercept, δ_kappa,
              bridge = get(feature_set.data, :league_delta_oos_bridge,
                           Dict{Int,NamedTuple}()))
end

_cb_rate_index(::CompetitionKappaJoint, row) = Int(row.match_id)

function _cb_rates(::CompetitionKappaJoint, μ_h, μ_a, obs_nt,
                   h_idx, a_idx, match_id::Int)
    index = get(obs_nt.bridge, match_id, nothing)
    index === nothing && error("CompetitionKappa has no OOS league design for match $match_id")
    true_μ_h = exp.(obs_nt.δ_intercept[:, index.tier_home_idx]) .* μ_h
    true_μ_a = exp.(obs_nt.δ_intercept[:, index.tier_away_idx]) .* μ_a
    finishing = obs_nt.κ .* exp.(obs_nt.δ_kappa[:, index.competition_idx])
    λ_h = finishing .* true_μ_h
    λ_a = finishing .* true_μ_a
    return (; λ_h, λ_a, μ_h = true_μ_h, μ_a = true_μ_a,
              κ = obs_nt.κ, κ_competition = finishing, ν = obs_nt.ν,
              true_xg_h = true_μ_h, true_xg_a = true_μ_a)
end
