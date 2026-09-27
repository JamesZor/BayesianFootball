# Pyramid-aware predictor for the composable count-model builder.
# All schedule, anchor and transition-event decisions are made here, outside the
# Turing tape. The @model block below sees only dense Float64 designs.

abstract type AbstractTierCarry end
struct NoCarry <: AbstractTierCarry end
Base.@kwdef struct EstimatedCarry{D<:ContinuousUnivariateDistribution} <: AbstractTierCarry
    prior::D = Beta(1.5, 4.0)
end
struct FixedCarry <: AbstractTierCarry
    value::Float64
end
FixedCarry(value::Real) = FixedCarry(Float64(value))

abstract type AbstractTierJump end
struct NoJump <: AbstractTierJump end
Base.@kwdef struct EstimatedJump{D<:ContinuousUnivariateDistribution} <: AbstractTierJump
    scale_prior::D = Gamma(2.0, 0.08)
end

const DEFAULT_TRANSITION_PRIORS = (
    (from = 55, to = 56, α_mean = 0.12, α_sd = 0.20, β_mean = -0.16, β_sd = 0.26),
    (from = 56, to = 57, α_mean = 0.03, α_sd = 0.20, β_mean = -0.04, β_sd = 0.24),
    (from = 57, to = 56, α_mean = 0.00, α_sd = 0.21, β_mean =  0.01, β_sd = 0.31),
)

struct FixedTransitionPriors{T<:Tuple} <: AbstractTierJump
    table::T
end
FixedTransitionPriors() = FixedTransitionPriors(DEFAULT_TRANSITION_PRIORS)

"Fold-local schedule design requested by `PyramidTiers`."
Base.@kwdef struct PyramidTierFeature{J<:AbstractTierJump} <: CB_Features.AbstractFeatureConfig
    anchor::Symbol = :first_in_window
    jump::J = NoJump()
end

"""
    PyramidTiers(; tier_term=true, anchor=:first_in_window, carry=NoCarry(), jump=NoJump())

Pyramid tier shrinkage target and persistent club-season transition events. Tier
T=1..4 corresponds to tournaments 54..57, with `τ_T = -step*(T-1)`.
The component contributes attack at 0.48 and concession at 0.52. `:full_pyramid`
uses the unscoped SPFL schedule table retained by `Data.apply_scope`.
"""
Base.@kwdef struct PyramidTiers{
    S<:ContinuousUnivariateDistribution,
    O<:Union{Nothing,ContinuousUnivariateDistribution},
    C<:AbstractTierCarry,
    J<:AbstractTierJump,
} <: AbstractPredictorTerm
    tier_term::Bool = true
    anchor::Symbol = :first_in_window
    step_prior::S = truncated(Normal(0.47, 0.16), 0.0, Inf)
    attack_share::Float64 = 0.48
    old_firm_prior::O = truncated(Normal(1.15, 0.10), 0.0, Inf)
    carry::C = NoCarry()
    jump::J = NoJump()
end

struct PyramidTierDesign{M<:Union{Val{true},Val{false}}}
    anchor_home::Vector{Float64}
    anchor_away::Vector{Float64}
    delta_home::Vector{Float64}
    delta_away::Vector{Float64}
    old_firm_home::Vector{Float64}
    old_firm_away::Vector{Float64}
    events_home::Matrix{Float64}
    events_away::Matrix{Float64}
    jump_α_mean::Vector{Float64}
    jump_α_sd::Vector{Float64}
    jump_β_mean::Vector{Float64}
    jump_β_sd::Vector{Float64}
    tier_marker::M
end

predictor_name(::PyramidTiers) = :pyramid_tiers
predictor_features(c::PyramidTiers) =
    CB_Features.AbstractFeatureConfig[PyramidTierFeature(anchor = c.anchor, jump = c.jump)]

_tier_event_directions(::Union{NoJump,EstimatedJump}) = nothing
_tier_event_directions(j::FixedTransitionPriors) = Set((row.from, row.to) for row in j.table)

function _tier_prior_row(::Union{NoJump,EstimatedJump}, from::Int, to::Int)
    return (α_mean = 0.0, α_sd = 1.0, β_mean = 0.0, β_sd = 1.0)
end
function _tier_prior_row(j::FixedTransitionPriors, from::Int, to::Int)
    index = findfirst(row -> row.from == from && row.to == to, j.table)
    index === nothing && error("no fixed transition prior for tournament direction $from→$to")
    row = j.table[index]
    return (α_mean = Float64(row.α_mean), α_sd = Float64(row.α_sd),
            β_mean = Float64(row.β_mean), β_sd = Float64(row.β_sd))
end

function _tier_shift_season(season::String, offset::Int)
    match = Base.match(r"^(\d{2})/(\d{2})$", season)
    match === nothing && return nothing
    start = parse(Int, match.captures[1])
    finish = parse(Int, match.captures[2])
    return lpad(string(mod(start + offset, 100)), 2, '0') * "/" *
           lpad(string(mod(finish + offset, 100)), 2, '0')
end
_tier_previous_season(season::String) = _tier_shift_season(season, -1)
_tier_next_season(season::String) = _tier_shift_season(season, 1)

_tier_number(tournament::Int) = tournament - 53
_tier_coefficient(tournament::Int) = -Float64(_tier_number(tournament) - 1)
_is_old_firm(club::String) = club == "celtic" || club == "rangers"

function _full_tier_table(ds)
    return DataFrames.metadata(ds.matches, "full_club_season_tiers",
        CB_Features.Data.club_season_tiers(ds.matches))
end

function _tier_anchor_table(anchor::Symbol, ds)
    anchor in (:first_in_window, :full_pyramid) || error(
        "PyramidTiers.anchor must be :first_in_window or :full_pyramid; got :$anchor")
    return anchor === :full_pyramid ? _full_tier_table(ds) :
           CB_Features.Data.club_season_tiers(ds.matches)
end

function _tier_window_seasons(F_data, ordered_rows)
    seasons = String.(ordered_rows.season)
    target = get(F_data, :target_season, nothing)
    if target === nothing && !isempty(seasons)
        # The first fold has no fitted target row. Its target season is still a
        # schedule fact and is the season immediately after the history window.
        target = _tier_next_season(maximum(seasons))
    end
    target === nothing || push!(seasons, String(target))
    return sort!(unique!(seasons))
end

function _tier_events(full_tiers, window_seasons, clubs, jump)
    directions = _tier_event_directions(jump)
    events = NamedTuple[]
    for club in sort!(collect(clubs))
        available = [season for season in window_seasons if haskey(full_tiers, (club, season))]
        isempty(available) && continue
        first_window = first(available)
        for season in available
            season == first_window && continue
            previous = _tier_previous_season(season)
            previous === nothing && continue
            old_key = (club, previous)
            new_key = (club, season)
            haskey(full_tiers, old_key) || continue # entered-SPFL: no event
            from = full_tiers[old_key]
            to = full_tiers[new_key]
            from == to && continue
            directions === nothing || ((from, to) in directions) || continue
            prior = _tier_prior_row(jump, from, to)
            push!(events, (; club, season, from, to, prior...))
        end
    end
    sort!(events; by = event -> (event.season, event.club))
    return events
end

function _tier_side_row(club, season, anchor_by_club, full_tiers, events)
    key = (String(club), String(season))
    haskey(full_tiers, key) || error(
        "Pyramid tier table has no scheduled SPFL tier for $(key[1]) in $(key[2])")
    current = full_tiers[key]
    anchor = get(anchor_by_club, key[1], current)
    incidence = Float64[event.club == key[1] && event.season <= key[2] for event in events]
    return (; anchor = _tier_coefficient(anchor),
              delta = _tier_coefficient(current) - _tier_coefficient(anchor),
              old_firm = _is_old_firm(key[1]) ? 1.0 : 0.0,
              incidence)
end

function CB_Features.add_feature!(F_data::Dict, config::PyramidTierFeature,
                                  ordered_ids, team_map::Dict,
                                  ds::CB_Features.Data.DataStore)
    wanted = Set(Int.(ordered_ids))
    ordered_rows = ds.matches[in.(Int.(ds.matches.match_id), Ref(wanted)), :]
    row_by_id = Dict(Int(row.match_id) => row for row in eachrow(ordered_rows))
    rows = [row_by_id[Int(id)] for id in ordered_ids]
    window_seasons = _tier_window_seasons(F_data, ordered_rows)
    full_tiers = _full_tier_table(ds)
    anchor_tiers = _tier_anchor_table(config.anchor, ds)

    # Include clubs from the full declared window, not only already-played rows.
    # Their league membership is a schedule fact and is the only future datum used.
    window_clubs = Set{String}()
    for ((club, season), _) in full_tiers
        season in window_seasons && push!(window_clubs, club)
    end
    anchor_by_club = Dict{String,Int}()
    for club in window_clubs
        seasons = [season for season in window_seasons if haskey(anchor_tiers, (club, season))]
        isempty(seasons) || (anchor_by_club[club] = anchor_tiers[(club, first(seasons))])
    end
    events = _tier_events(full_tiers, window_seasons, window_clubs, config.jump)

    home = [_tier_side_row(row.home_team, row.season, anchor_by_club, full_tiers, events)
            for row in rows]
    away = [_tier_side_row(row.away_team, row.season, anchor_by_club, full_tiers, events)
            for row in rows]
    n_events = length(events)
    event_matrix(values) = n_events == 0 ? zeros(Float64, length(values), 0) :
        reduce(vcat, permutedims.(getproperty.(values, :incidence)))

    F_data[:pyramid_anchor_home] = Float64[getproperty(x, :anchor) for x in home]
    F_data[:pyramid_anchor_away] = Float64[getproperty(x, :anchor) for x in away]
    F_data[:pyramid_delta_home] = Float64[getproperty(x, :delta) for x in home]
    F_data[:pyramid_delta_away] = Float64[getproperty(x, :delta) for x in away]
    F_data[:pyramid_old_firm_home] = Float64[getproperty(x, :old_firm) for x in home]
    F_data[:pyramid_old_firm_away] = Float64[getproperty(x, :old_firm) for x in away]
    F_data[:pyramid_events_home] = event_matrix(home)
    F_data[:pyramid_events_away] = event_matrix(away)
    F_data[:pyramid_events] = events

    bridge = Dict{Int,NamedTuple}()
    for row in eachrow(ds.matches)
        String(row.season) in window_seasons || continue
        h = _tier_side_row(row.home_team, row.season, anchor_by_club, full_tiers, events)
        a = _tier_side_row(row.away_team, row.season, anchor_by_club, full_tiers, events)
        bridge[Int(row.match_id)] = (; home = h, away = a)
    end
    F_data[:pyramid_oos_bridge] = bridge
    return nothing
end

function predictor_design(c::PyramidTiers, feature_set, n_matches::Int)
    d = feature_set.data
    keys_needed = (:pyramid_anchor_home, :pyramid_anchor_away,
                   :pyramid_delta_home, :pyramid_delta_away,
                   :pyramid_old_firm_home, :pyramid_old_firm_away,
                   :pyramid_events_home, :pyramid_events_away, :pyramid_events)
    all(key -> haskey(d, key), keys_needed) || error(
        "PyramidTiers feature design is incomplete; build through required_features")
    c.anchor in (:first_in_window, :full_pyramid) || error("invalid PyramidTiers anchor :$(c.anchor)")
    0.0 <= c.attack_share <= 1.0 || error("PyramidTiers.attack_share must be in [0,1]")
    c.carry isa FixedCarry && !(0.0 <= c.carry.value <= 1.0) && error(
        "FixedCarry value must be in [0,1]; got $(c.carry.value)")
    events = d[:pyramid_events]
    α_mean = Float64[event.α_mean for event in events]
    α_sd = Float64[event.α_sd for event in events]
    β_mean = Float64[event.β_mean for event in events]
    β_sd = Float64[event.β_sd for event in events]
    design = PyramidTierDesign(
        Vector{Float64}(d[:pyramid_anchor_home]), Vector{Float64}(d[:pyramid_anchor_away]),
        Vector{Float64}(d[:pyramid_delta_home]), Vector{Float64}(d[:pyramid_delta_away]),
        Vector{Float64}(d[:pyramid_old_firm_home]), Vector{Float64}(d[:pyramid_old_firm_away]),
        Matrix{Float64}(d[:pyramid_events_home]), Matrix{Float64}(d[:pyramid_events_away]),
        α_mean, α_sd, β_mean, β_sd, Val(c.tier_term))
    for vector in (design.anchor_home, design.anchor_away, design.delta_home,
                   design.delta_away, design.old_firm_home, design.old_firm_away)
        length(vector) == n_matches || error("PyramidTiers design length mismatch")
    end
    size(design.events_home) == size(design.events_away) || error(
        "PyramidTiers home/away event incidence shapes differ")
    size(design.events_home, 1) == n_matches || error("PyramidTiers event design row mismatch")
    return design
end

@model function _tier_step(::Val{true}, prior)
    step ~ prior
    return step
end
@model function _tier_step(::Val{false}, prior)
    return 0.0
end
@model function _tier_carry(::NoCarry)
    return 0.0
end
@model function _tier_carry(c::FixedCarry)
    return c.value
end
@model function _tier_carry(c::EstimatedCarry)
    κ ~ c.prior
    return κ
end
@model function _tier_old_firm(::Val{false}, prior)
    return 0.0
end
@model function _tier_old_firm(::Val{true}, ::Nothing)
    return 0.0
end
@model function _tier_old_firm(::Val{true}, prior::ContinuousUnivariateDistribution)
    offset ~ prior
    return offset
end
@model function _tier_jump(::NoJump, design::PyramidTierDesign)
    return nothing
end
@model function _tier_jump(j::EstimatedJump, design::PyramidTierDesign)
    σ ~ j.scale_prior
    raw_α ~ filldist(Normal(0.0, 1.0), size(design.events_home, 2))
    raw_β ~ filldist(Normal(0.0, 1.0), size(design.events_home, 2))
    return (; α = σ .* raw_α, β = σ .* raw_β)
end
@model function _tier_jump(::FixedTransitionPriors, design::PyramidTierDesign)
    raw_α ~ filldist(Normal(0.0, 1.0), size(design.events_home, 2))
    raw_β ~ filldist(Normal(0.0, 1.0), size(design.events_home, 2))
    return (; α = design.jump_α_mean .+ design.jump_α_sd .* raw_α,
              β = design.jump_β_mean .+ design.jump_β_sd .* raw_β)
end

_tier_add_jump(base, ::Nothing, incidence, side) = base
_tier_add_jump(base, jump, incidence, ::Val{:α}) = base .+ incidence * jump.α
_tier_add_jump(base, jump, incidence, ::Val{:β}) = base .+ incidence * jump.β

@model function _predictor_term(c::PyramidTiers, design::PyramidTierDesign)
    step ~ to_submodel(_tier_step(design.tier_marker, c.step_prior), false)
    carry ~ to_submodel(_tier_carry(c.carry), false)
    old_firm ~ to_submodel(
        _tier_old_firm(design.tier_marker, c.old_firm_prior), false)
    jump ~ to_submodel(_tier_jump(c.jump, design), false)

    B_home = step .* (design.anchor_home .+ carry .* design.delta_home) .+
             old_firm .* design.old_firm_home
    B_away = step .* (design.anchor_away .+ carry .* design.delta_away) .+
             old_firm .* design.old_firm_away
    defence_share = 1.0 - c.attack_share
    tier_h = c.attack_share .* B_home .- defence_share .* B_away
    tier_a = c.attack_share .* B_away .- defence_share .* B_home
    attack_h = _tier_add_jump(tier_h, jump, design.events_home, Val(:α))
    attack_a = _tier_add_jump(tier_a, jump, design.events_away, Val(:α))
    h = _tier_add_jump(attack_h, jump, design.events_away, Val(:β))
    a = _tier_add_jump(attack_a, jump, design.events_home, Val(:β))
    return (; h, a)
end

predictor_sites(c::PyramidTiers) = _tier_predictor_sites(c, Val(c.tier_term))
function _tier_predictor_sites(c::PyramidTiers, marker)
    sites = Symbol[]
    marker isa Val{true} && push!(sites, Symbol("pyramid_tiers.step"))
    c.carry isa EstimatedCarry && push!(sites, Symbol("pyramid_tiers.carry.κ"))
    (marker isa Val{false} || c.old_firm_prior === nothing) ||
        push!(sites, Symbol("pyramid_tiers.old_firm.offset"))
    if c.jump isa EstimatedJump
        append!(sites, [Symbol("pyramid_tiers.jump.σ"), Symbol("pyramid_tiers.jump.raw_α"),
                        Symbol("pyramid_tiers.jump.raw_β")])
    elseif c.jump isa FixedTransitionPriors
        append!(sites, [Symbol("pyramid_tiers.jump.raw_α"), Symbol("pyramid_tiers.jump.raw_β")])
    end
    return sites
end

_chain_draws(chain, name::Symbol) = vec(Array(chain[name]))
function _tier_chain_matrix(chain, base::String)
    names_available = String.(names(chain))
    n = count(name -> startswith(name, base * "["), names_available)
    samples = size(chain, 1) * size(chain, 3)
    out = Matrix{Float64}(undef, samples, n)
    for i in 1:n
        out[:, i] = _chain_draws(chain, _grw_chain_symbol(chain, base, i))
    end
    return out
end
_tier_extract_step(chain, prefix, ::Val{true}) = _chain_draws(chain, Symbol("$prefix.step"))
_tier_extract_step(chain, prefix, ::Val{false}) = zeros(size(chain, 1) * size(chain, 3))
_tier_extract_carry(chain, prefix, ::NoCarry) = zeros(size(chain, 1) * size(chain, 3))
_tier_extract_carry(chain, prefix, c::FixedCarry) = fill(c.value, size(chain, 1) * size(chain, 3))
_tier_extract_carry(chain, prefix, ::EstimatedCarry) = _chain_draws(chain, Symbol("$prefix.carry.κ"))
_tier_extract_old(chain, prefix, ::Val{false}, prior) = zeros(size(chain, 1) * size(chain, 3))
_tier_extract_old(chain, prefix, ::Val{true}, ::Nothing) = zeros(size(chain, 1) * size(chain, 3))
_tier_extract_old(chain, prefix, ::Val{true}, prior) =
    _chain_draws(chain, Symbol("$prefix.old_firm.offset"))
_tier_extract_jump(chain, prefix, ::NoJump) = nothing
function _tier_extract_jump(chain, prefix, ::EstimatedJump)
    σ = _chain_draws(chain, Symbol("$prefix.jump.σ"))
    return (; α = σ .* _tier_chain_matrix(chain, "$prefix.jump.raw_α"),
              β = σ .* _tier_chain_matrix(chain, "$prefix.jump.raw_β"))
end
function _tier_extract_jump(chain, prefix, jump::FixedTransitionPriors)
    raw_α = _tier_chain_matrix(chain, "$prefix.jump.raw_α")
    raw_β = _tier_chain_matrix(chain, "$prefix.jump.raw_β")
    # Event-specific locations/scales are carried in the OOS bridge as well; the
    # chain alone retains one raw draw per ordered event.
    return (; raw_α, raw_β, fixed = jump)
end

function predictor_extract(chain::Chains, c::PyramidTiers, prefix::String)
    return (; step = _tier_extract_step(chain, prefix, Val(c.tier_term)),
              carry = _tier_extract_carry(chain, prefix, c.carry),
              old_firm = _tier_extract_old(chain, prefix, Val(c.tier_term), c.old_firm_prior),
              jump = _tier_extract_jump(chain, prefix, c.jump),
              attack_share = c.attack_share)
end

function _tier_oos_jump(draw, source, ::NoJump)
    samples = length(draw.step)
    return (; att_h = zeros(samples), att_a = zeros(samples),
              def_h = zeros(samples), def_a = zeros(samples))
end
function _tier_oos_jump(draw, source, ::EstimatedJump)
    return (; att_h = draw.jump.α * source.home.incidence,
              att_a = draw.jump.α * source.away.incidence,
              def_h = draw.jump.β * source.home.incidence,
              def_a = draw.jump.β * source.away.incidence)
end
function _tier_oos_jump(draw, source, jump::FixedTransitionPriors)
    events = source.events
    α_mean = Float64[event.α_mean for event in events]
    α_sd = Float64[event.α_sd for event in events]
    β_mean = Float64[event.β_mean for event in events]
    β_sd = Float64[event.β_sd for event in events]
    α = α_mean' .+ draw.jump.raw_α .* α_sd'
    β = β_mean' .+ draw.jump.raw_β .* β_sd'
    return (; att_h = α * source.home.incidence, att_a = α * source.away.incidence,
              def_h = β * source.home.incidence, def_a = β * source.away.incidence)
end

function predictor_oos(c::PyramidTiers, draw, bridge, row)
    source = get(bridge, Int(row.match_id), nothing)
    source === nothing && error("PyramidTiers has no OOS schedule bridge for match $(row.match_id)")
    B_home = draw.step .* (source.home.anchor .+ draw.carry .* source.home.delta) .+
             draw.old_firm .* source.home.old_firm
    B_away = draw.step .* (source.away.anchor .+ draw.carry .* source.away.delta) .+
             draw.old_firm .* source.away.old_firm
    defence_share = 1.0 - c.attack_share
    jump = _tier_oos_jump(draw, source, c.jump)
    return (; h = c.attack_share .* B_home .- defence_share .* B_away .+
                 jump.att_h .+ jump.def_a,
              a = c.attack_share .* B_away .- defence_share .* B_home .+
                 jump.att_a .+ jump.def_h)
end

function _cb_predictor_oos_source(::PyramidTiers, feature_set, df, lineup_map)
    events = get(feature_set.data, :pyramid_events, NamedTuple[])
    raw = get(feature_set.data, :pyramid_oos_bridge, Dict{Int,NamedTuple}())
    return Dict(id => merge(value, (; events)) for (id, value) in raw)
end
