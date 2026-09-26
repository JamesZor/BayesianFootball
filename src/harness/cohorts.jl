# Cohort definitions for the versioned experiment scorecard.

const HARNESS_LEAGUE_TIERS = Set((54, 55, 56, 57))
const HARNESS_DIRECTIONS =
    (:relegated_into_L1, :promoted_into_L1, :l1_l2, :any)

"""
    club_season_tiers(ds_leagues) -> Dict{Tuple{String,String},Int}

Map each club-season to its Scottish league tournament (54--57). A club appearing in
more than one league in the same season is ambiguous and is refused rather than assigned
according to row order.
"""
function club_season_tiers(ds_leagues)
    required = (:home_team, :away_team, :season, :tournament_id)
    all(c -> hasproperty(ds_leagues.matches, c), required) ||
        error("club_season_tiers: matches require columns $(collect(required))")

    tiers = Dict{Tuple{String,String},Int}()
    for r in eachrow(ds_leagues.matches)
        tier = Int(r.tournament_id)
        tier in HARNESS_LEAGUE_TIERS || continue
        season = String(r.season)
        for club in (String(r.home_team), String(r.away_team))
            key = (club, season)
            prior = get(tiers, key, tier)
            prior == tier || error(
                "club_season_tiers: $club appears in tiers $prior and $tier in $season")
            tiers[key] = tier
        end
    end
    return tiers
end

function _previous_season(season::AbstractString)
    m = match(r"^(\d{2}|\d{4})/(\d{2}|\d{4})$", season)
    m === nothing && error("transition_cohort: unsupported season label '$season'")
    left = m.captures[1]
    width = length(left)
    right = m.captures[2]
    start = parse(Int, left) - 1
    finish = mod(parse(Int, right) - 1, 10^length(right))
    width == 2 && (start = mod(start, 100))
    return lpad(string(start), width, '0') * "/" * lpad(string(finish), length(right), '0')
end

function _transition_matches(direction::Symbol, previous::Int, current::Int)
    direction === :relegated_into_L1 && return previous == 55 && current == 56
    direction === :promoted_into_L1 && return previous == 57 && current == 56
    direction === :l1_l2 && return (previous == 56 && current == 57) ||
                                       (previous == 57 && current == 56)
    direction === :any && return previous != current
    return false
end

"""
    transition_cohort(ds, panel_ids, tiers; first_n, direction)

Return `(match_ids, clubs)`, where `match_ids` is the requested transition cohort and
`clubs[match_id]` names the transitioning club or clubs in that fixture. The first-N
counter is taken over all of a club's league fixtures in its new tier and is applied
before intersecting with `panel_ids`.
"""
function transition_cohort(ds, panel_ids, tiers;
                           first_n::Int, direction::Symbol)
    first_n in (10, 20) || throw(ArgumentError("first_n must be 10 or 20; got $first_n"))
    direction in HARNESS_DIRECTIONS || throw(ArgumentError(
        "direction must be one of $(HARNESS_DIRECTIONS); got :$direction"))

    panel = Set{Int}(Int(m) for m in panel_ids)
    rows = filter(r -> Int(r.tournament_id) in HARNESS_LEAGUE_TIERS,
                  ds.matches)
    sort_columns = :match_date in propertynames(rows) ? [:match_date, :match_id] : [:match_id]
    sort!(rows, sort_columns)

    transitioned = Set{Tuple{String,String}}()
    for ((club, season), current) in tiers
        previous = get(tiers, (club, _previous_season(season)), 0)
        previous == 0 && continue
        _transition_matches(direction, previous, current) && push!(transitioned, (club, season))
    end

    appearances = Dict{Tuple{String,String},Int}()
    ids = Set{Int}()
    clubs = Dict{Int,Vector{String}}()
    for r in eachrow(rows)
        mid = Int(r.match_id)
        season = String(r.season)
        tier = Int(r.tournament_id)
        for club in (String(r.home_team), String(r.away_team))
            key = (club, season)
            get(tiers, key, 0) == tier || continue
            count = get(appearances, key, 0) + 1
            appearances[key] = count
            key in transitioned || continue
            count <= first_n || continue
            mid in panel || continue
            push!(ids, mid)
            push!(get!(clubs, mid, String[]), club)
        end
    end
    foreach(unique!, values(clubs))
    return (; match_ids = ids, clubs)
end

"""Fixture IDs whose de-vigged 1X2 close contains a selection at least `threshold`."""
function favourites(odds::AbstractDataFrame; threshold::Real = 0.60)
    0.0 <= threshold <= 1.0 || throw(ArgumentError("threshold must be in [0,1]"))
    required = (:match_id, :market_name, :prob_fair_close)
    all(c -> hasproperty(odds, c), required) ||
        error("favourites: odds require columns $(collect(required))")
    return Set{Int}(Int(r.match_id) for r in eachrow(odds)
                    if lowercase(String(r.market_name)) == "1x2" &&
                       !ismissing(r.prob_fair_close) && isfinite(r.prob_fair_close) &&
                       r.prob_fair_close >= threshold)
end
