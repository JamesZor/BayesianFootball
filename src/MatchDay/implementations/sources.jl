# src/MatchDay/implementations/sources.jl
#
# Fixture sources, identity resolvers, lineup sources.

export SofaScoreEvents, ExplicitFixtures, MatchMetaCrosswalk, LiveNameMatch, ResolverChain,
       ProvisionalDB, LastHistorical, JsonPin, SourceChain, BBCLineupSource,
       parse_bbc_lineup, team_name_score, match_event_scores

# ===================================================================
# Fixture sources
# ===================================================================

"""
    SofaScoreEvents(; horizon = Hour(36))

Unstarted fixtures from `sofascore.events` whose kick-off falls in `[as_of, as_of + horizon)`.

A horizon rather than a calendar day, deliberately: the prototype used
`start_timestamp >= EXTRACT(EPOCH FROM CURRENT_DATE)`, which puts a late kick-off on the wrong
side of a UTC midnight the moment the DB and the fixture disagree about the day.
"""
Base.@kwdef struct SofaScoreEvents <: AbstractFixtureSource
    horizon::Period = Hour(36)
end

function fixtures(s::SofaScoreEvents, segment, as_of::DateTime)
    t_ids = Data.tournament_ids(segment)
    lo = Int(round(datetime2unix(as_of)))
    hi = Int(round(datetime2unix(as_of + s.horizon)))
    df = _query(FIXTURES_SQL, (t_ids, lo, hi))
    return Fixture[Fixture(Int(r.match_id), String(r.home_team), String(r.away_team),
                           unix2datetime(r.start_timestamp), Int(r.tournament_id))
                   for r in eachrow(df)]
end

"""
    ExplicitFixtures(fixtures)

A fixed list. The replay and test source -- it is how a past match day is re-run without
depending on what `sofascore.events` says today.
"""
struct ExplicitFixtures <: AbstractFixtureSource
    list::Vector{Fixture}
end

fixtures(s::ExplicitFixtures, _segment, as_of::DateTime) =
    Fixture[f for f in s.list if f.kickoff >= as_of]

# ===================================================================
# Identity resolvers
# ===================================================================

"""
    MatchMetaCrosswalk(; require_verified = true, markets = nothing)

Looks the fixture up in `betfair.match_meta`. **Not a matcher** -- see `AbstractIdentityResolver`.

Failure modes, all reported rather than thrown:
* `:absent_from_crosswalk` -- no row. The resolution job has not seen this fixture. This is the
  common case for anything recent: the job stopped around 2026-06-27, after which resolution is
  0%, having been 100% before.
* `:not_verified` -- a row exists but `is_verified` is false.
* `:no_markets` -- the event resolved but `betfair_live.market_metadata` has no markets for it.
"""
Base.@kwdef struct MatchMetaCrosswalk <: AbstractIdentityResolver
    require_verified::Bool = true
end

function resolve(r::MatchMetaCrosswalk, f::Fixture)
    df = _query(IDENTITY_SQL, (f.m_id,))
    isempty(df) && return Unresolved(f, :absent_from_crosswalk)

    verified = any(skipmissing(df.is_verified))
    (r.require_verified && !verified) && return Unresolved(f, :not_verified)

    ev = String(first(skipmissing(df.betfair_event_id)))
    mk = Dict{String,String}()
    for row in eachrow(df)
        (ismissing(row.market_id) || ismissing(row.market_type)) && continue
        mk[String(row.market_type)] = String(row.market_id)
    end
    isempty(mk) && return Unresolved(f, :no_markets)
    return Resolved(f, ev, mk, verified)
end

# -------------------------------------------------------------------
# LiveNameMatch -- the fallback for when the crosswalk job has not run
# -------------------------------------------------------------------
#
# `AbstractIdentityResolver`'s docstring argues against a fuzzy matcher, on the grounds that it
# would be a second source of truth papering over an operational gap. That argument stands, and
# this resolver does NOT retire it: `MatchMetaCrosswalk` remains the authority and this is only
# ever reached through a `ResolverChain` after it has failed. What has changed is the measured
# cost of having no fallback at all -- the crosswalk job stopped 2026-06-27, so on 2026-08-07 it
# held 0 rows for all 9 of that evening's fixtures and the gate refused the entire card.
#
# The design point that makes this safe is the MARGIN, not the score. Matching "is this the same
# club?" on strings is unreliable in the abstract; matching "which of the 5 events kicking off at
# 18:45 in this window is this fixture?" is a much smaller question, and it is answerable because
# a wrong pairing scores near zero rather than near the right answer. The resolver therefore
# refuses on ambiguity instead of taking the best available guess.

"Alias map for the abbreviations the exchange uses and SofaScore's slugs do not."
const _TEAM_ALIAS = Dict("utd" => "united", "ath" => "athletic", "acad" => "academical",
                         "wands" => "wanderers", "rvrs" => "rovers", "caley" => "caledonian")

"Tokens that carry no identifying information and appear on only one of the two feeds."
const _TEAM_DROP = Set(["fc", "afc", "cf", "club", "the"])

function _team_tokens(s::AbstractString)
    out = String[]
    for p in split(lowercase(s), r"[^a-z0-9]+"; keepempty = false)
        p = get(_TEAM_ALIAS, p, p)
        p in _TEAM_DROP && continue
        push!(out, p)
    end
    return out
end

"""
    team_name_score(a, b) -> Float64 in [0, 1]

How strongly two spellings of a club name agree, after alias substitution and dropping
non-identifying tokens.

The tiers, each earned by a real pair observed on 2026-08-07:

| score | rule | example |
|---|---|---|
| 1.00 | equal after normalisation | `"Galway Utd"` / `galway-united` |
| 0.90 | one is a prefix of the other | `"Partick"` / `partick-thistle` |
| 0.85 | one side is the other's initialism | `"UCD"` / `university-college-dublin` |
| 0.5+ | partial token overlap, scaled by Jaccard | `"Cork City"` / `cobh-ramblers` -> 0.0 |
| 0.00 | no shared token | |

The initialism tier is not decoration: UCD v Wexford is unmatchable by any substring rule, and
it is a fixture in tonight's card.
"""
function team_name_score(a::AbstractString, b::AbstractString)
    ta, tb = _team_tokens(a), _team_tokens(b)
    (isempty(ta) || isempty(tb)) && return 0.0
    ja, jb = join(ta), join(tb)
    ja == jb && return 1.0
    (startswith(ja, jb) || startswith(jb, ja)) && return 0.9
    (length(ta) == 1 && ta[1] == join(first.(tb))) && return 0.85
    (length(tb) == 1 && tb[1] == join(first.(ta))) && return 0.85
    sa, sb = Set(ta), Set(tb)
    inter = length(intersect(sa, sb))
    inter == 0 && return 0.0
    return 0.5 + 0.4 * inter / length(union(sa, sb))
end

"""
    LiveNameMatch(; window = Minute(90), min_score = 0.75, min_margin = 0.25)

Resolve a fixture against `betfair_live.market_metadata` by kick-off window plus team name,
refusing whenever the answer is not unambiguous.

* `window` -- how far `open_date` may sit from `Fixture.kickoff`. Candidates outside it are never
  considered, which is what keeps the name comparison a 5-way question rather than a 500-way one.
* `min_score` -- floor on the mean of the home and away name scores.
* `min_margin` -- how far the best candidate must beat the runner-up. **This is the real safety
  property.** Measured on the 2026-08-07 card: 9/9 correct, worst score 0.875, worst margin
  0.633. A margin threshold of 0.25 therefore sits an order of magnitude clear of the observed
  worst case, and a genuinely ambiguous card fails closed as `:ambiguous_name_match`.

Failure modes, all reported rather than thrown: `:no_live_event_in_window`,
`:weak_name_match`, `:ambiguous_name_match`, `:no_markets`.
"""
Base.@kwdef struct LiveNameMatch <: AbstractIdentityResolver
    window::Period      = Minute(90)
    min_score::Float64  = 0.75
    min_margin::Float64 = 0.25
end

"""
    match_event_scores(r::LiveNameMatch, f::Fixture) -> DataFrame

Every candidate event and its score, best first. The audit view behind `resolve` -- call it when
a fixture comes back `:ambiguous_name_match` to see what it was confused between.
"""
function match_event_scores(r::LiveNameMatch, f::Fixture)
    df = _query(LIVE_EVENTS_SQL, (f.kickoff - r.window, f.kickoff + r.window))
    isempty(df) && return DataFrame(event_id = String[], home_team = String[],
                                    away_team = String[], score = Float64[])
    rows = NamedTuple[]
    for g in groupby(df, :event_id)
        bh, ba = String(first(g.home_team)), String(first(g.away_team))
        s = (team_name_score(bh, f.home) + team_name_score(ba, f.away)) / 2
        push!(rows, (event_id = String(first(g.event_id)), home_team = bh, away_team = ba,
                     score = s))
    end
    return sort!(DataFrame(rows), :score, rev = true)
end

function resolve(r::LiveNameMatch, f::Fixture)
    df = _query(LIVE_EVENTS_SQL, (f.kickoff - r.window, f.kickoff + r.window))
    isempty(df) && return Unresolved(f, :no_live_event_in_window)

    best, best_score, runner_up = nothing, -1.0, -1.0
    for g in groupby(df, :event_id)
        s = (team_name_score(String(first(g.home_team)), f.home) +
             team_name_score(String(first(g.away_team)), f.away)) / 2
        if s > best_score
            runner_up, best_score, best = best_score, s, g
        elseif s > runner_up
            runner_up = s
        end
    end

    best_score < r.min_score && return Unresolved(f, :weak_name_match)
    # A lone candidate has no runner-up to beat; `runner_up` is still -1.0 there, so the margin
    # test passes trivially, which is the intended behaviour.
    (best_score - runner_up) < r.min_margin && return Unresolved(f, :ambiguous_name_match)

    mk = Dict{String,String}()
    for row in eachrow(best)
        (ismissing(row.market_id) || ismissing(row.market_type)) && continue
        mk[String(row.market_type)] = String(row.market_id)
    end
    isempty(mk) && return Unresolved(f, :no_markets)

    # `verified = false` is deliberate and load-bearing: it is how a name-matched identity stays
    # distinguishable downstream from one the crosswalk job actually confirmed.
    return Resolved(f, String(first(best.event_id)), mk, false)
end

"""
    ResolverChain(resolvers...)

First success wins; if all fail, the **first** reason is reported, because it is the one from
the most authoritative source.
"""
struct ResolverChain{T<:Tuple} <: AbstractIdentityResolver
    resolvers::T
end
ResolverChain(rs::AbstractIdentityResolver...) = ResolverChain(rs)

function resolve(c::ResolverChain, f::Fixture)
    first_fail = nothing
    for r in c.resolvers
        out = resolve(r, f)
        out isa Resolved && return out
        first_fail === nothing && (first_fail = out)
    end
    return first_fail === nothing ? Unresolved(f, :no_resolver) : first_fail
end

# ===================================================================
# Lineup sources
# ===================================================================

"""
    BBCLineupSource(; ds = nothing, timeout_seconds = 3.0, max_retries = 1,
                     user_agent = ...)

Native BBC Sport lineup source.  BBC's CDN is deliberately kept outside the DataStore: the
source is an execution-time adapter, while the resolved players and event crosswalk are written
back to `betdb.bbc` for the next call.  A failed lookup is a normal `nothing` result, so a
`SourceChain` can continue to its configured fallback.
"""
Base.@kwdef struct BBCLineupSource <: AbstractLineupSource
    ds::Any = nothing
    timeout_seconds::Float64 = 3.0
    max_retries::Int = 1
    user_agent::String = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
end

const BBC_CDN_BASE = "https://web-cdn.api.bbci.co.uk/wc-poll-data/container/"
const BBC_TOURNAMENT_URNS = Dict(
    54 => "urn:bbc:sportsdata:football:tournament:scottish-premiership",
    55 => "urn:bbc:sportsdata:football:tournament:scottish-championship",
    56 => "urn:bbc:sportsdata:football:tournament:scottish-league-one",
    57 => "urn:bbc:sportsdata:football:tournament:scottish-league-two",
    1  => "urn:bbc:sportsdata:football:tournament:premier-league",
    2  => "urn:bbc:sportsdata:football:tournament:championship",
    3  => "urn:bbc:sportsdata:football:tournament:league-one",
    84 => "urn:bbc:sportsdata:football:tournament:league-two",
)

struct _BBCRawPlayer
    bbc_player_id::String
    name::String
    surname::String
    shirt_number::Union{Nothing,Int}
    position::Symbol
    substitute::Bool
    captain::Bool
end

struct _BBCPlayerResolution
    player_id::Int
    sofascore_name::String
    outcome::String
end

struct _BBCPersistedPlayer
    bbc_player_id::String
    player_id::Int
    bbc_name::String
    sofascore_name::String
    substitute::Bool
    shirt_number::Union{Nothing,Int}
    position::Symbol
    map_outcome::String
    team_slug::String
    captain::Bool
end

# JSON3 objects intentionally go through this small adapter.  It handles both JSON3.Object and
# ordinary Dict payloads, and makes a schema variation a failed source lookup rather than an
# exception escaping the slate loop.
function _bbc_get(value, key::Symbol, default = nothing)
    value === nothing && return default
    try
        return haskey(value, key) ? get(value, key, default) : default
    catch
        return default
    end
end

_bbc_string(value, default = "") =
    value === nothing || ismissing(value) ? default : String(value)

function _bbc_event_id(value)
    (value === nothing || ismissing(value)) && return nothing
    text = String(value)
    isempty(text) && return nothing
    prefix = "urn:bbc:sportsdata:football:event:"
    return startswith(text, prefix) ? text[length(prefix) + 1:end] : text
end

function _bbc_http_json(s::BBCLineupSource, endpoint::AbstractString,
                        query::Vector{Pair{String,String}}; base_url::String = BBC_CDN_BASE)
    attempts = max(0, s.max_retries) + 1
    for attempt in 1:attempts
        try
            response = HTTP.request("GET", base_url * endpoint;
                                    headers = ["User-Agent" => s.user_agent], query = query,
                                    connect_timeout = max(1, ceil(Int, s.timeout_seconds)),
                                    # HTTP.jl's timeout keywords are integer-valued in the
                                    # pinned release; retain sub-second configuration by rounding
                                    # up rather than failing the request at dispatch time.
                                    readtimeout = max(1, ceil(Int, s.timeout_seconds)), retry = false,
                                    status_exception = false)
            status = Int(response.status)
            if status >= 500
                attempt < attempts && continue
                @warn "BBCLineupSource: CDN returned HTTP $status" endpoint
                return nothing
            elseif status >= 400
                @warn "BBCLineupSource: CDN returned HTTP $status" endpoint
                return nothing
            end
            return JSON3.read(String(response.body))
        catch e
            # HTTP 1.11/1.12 exposes transport failures through Base.IOError,
            # HTTP.Exceptions.RequestError and TimeoutError (there is no HTTP.IOError
            # binding in the pinned release).  Retrying these and only these preserves the
            # requested fail-soft behaviour without retrying malformed JSON indefinitely.
            retryable = e isa Base.IOError || e isa HTTP.TimeoutError ||
                        e isa HTTP.Exceptions.RequestError || e isa HTTP.Exceptions.ConnectError
            if retryable && attempt < attempts
                continue
            end
            @warn "BBCLineupSource: request failed" endpoint exception = e
            return nothing
        end
    end
    return nothing
end

function _bbc_date(value)
    text = _bbc_string(value)
    length(text) >= 10 || return nothing
    try
        return Date(text[1:10])
    catch
        return nothing
    end
end

function _bbc_fixture_events(value)
    events = Any[]
    function walk(node)
        node === nothing && return
        if node isa AbstractDict || node isa JSON3.Object
            home = _bbc_get(node, :home)
            away = _bbc_get(node, :away)
            raw_event_id = _bbc_get(node, :id)
            raw_event_id === nothing && (raw_event_id = _bbc_get(node, :urn))
            event_id = _bbc_event_id(raw_event_id)
            start = _bbc_get(node, :startDateTime)
            if home !== nothing && away !== nothing && event_id !== nothing && start !== nothing
                push!(events, node)
                return
            end
            for value_ in values(node)
                walk(value_)
            end
        elseif node isa AbstractVector || node isa JSON3.Array
            for value_ in node
                walk(value_)
            end
        end
    end
    walk(value)
    return events
end

function _bbc_team_value(team, key::Symbol)
    name = _bbc_get(team, :name)
    value = _bbc_get(team, key)
    value !== nothing && return _bbc_string(value)
    name === nothing && return ""
    return _bbc_string(_bbc_get(name, key))
end

function _bbc_team_slug(team)
    urn = _bbc_string(_bbc_get(team, :urn))
    marker = "urn:bbc:sportsdata:football:team:"
    return startswith(urn, marker) ? urn[length(marker) + 1:end] : urn
end

function _bbc_map_score(fixture_name::String, team, maps::DataFrame)
    score = team_name_score(fixture_name, _bbc_team_value(team, :fullName))
    score = max(score, team_name_score(fixture_name, _bbc_team_value(team, :shortName)))
    slug = _bbc_team_slug(team)
    for row in eachrow(maps)
        _bbc_string(row.bbc_slug) == slug || continue
        sofa = _bbc_string(row.sofascore_slug)
        !isempty(sofa) && (score = max(score, team_name_score(fixture_name, sofa)))
        bbc_name = :bbc_name in propertynames(row) ? _bbc_string(row.bbc_name) : ""
        !isempty(bbc_name) && (score = max(score, team_name_score(fixture_name, bbc_name)))
    end
    return score
end

function _bbc_discover_event(s::BBCLineupSource, f::Fixture)
    urn = get(BBC_TOURNAMENT_URNS, f.tournament_id, nothing)
    urn === nothing && return nothing
    day = string(Date(f.kickoff))
    payload = _bbc_http_json(s, "sport-data-scores-fixtures",
        ["selectedStartDate" => day, "selectedEndDate" => day,
         "todayDate" => day, "urn" => urn, "useSdApi" => "false"])
    payload === nothing && return nothing

    maps = try
        _query("SELECT bbc_slug, sofascore_slug, bbc_name FROM bbc.team_map")
    catch e
        @warn "BBCLineupSource: team map lookup failed" exception = e
        DataFrame(bbc_slug = String[], sofascore_slug = String[], bbc_name = String[])
    end
    candidates = Any[]
    for event in _bbc_fixture_events(payload)
        event_date = _bbc_date(_bbc_get(event, :startDateTime))
        event_date === nothing && continue
        abs(Dates.value(event_date - Date(f.kickoff))) <= 1 || continue
        home = _bbc_get(event, :home); away = _bbc_get(event, :away)
        hs = _bbc_map_score(f.home, home, maps)
        as = _bbc_map_score(f.away, away, maps)
        (hs >= 0.75 && as >= 0.75) || continue
        push!(candidates, (event = event, score = hs + as))
    end
    isempty(candidates) && return nothing
    sort!(candidates, by = x -> x.score, rev = true)
    if length(candidates) > 1 && candidates[1].score - candidates[2].score < 0.25
        @warn "BBCLineupSource: fixture discovery was ambiguous" match_id = f.m_id
        return nothing
    end
    raw_event_id = _bbc_get(candidates[1].event, :id)
    raw_event_id === nothing && (raw_event_id = _bbc_get(candidates[1].event, :urn))
    event_id = _bbc_event_id(raw_event_id)
    event_id === nothing && return nothing
    _bbc_persist_event!(f, event_id)
    return event_id
end

function _bbc_event_for(s::BBCLineupSource, f::Fixture)
    try
        rows = _query("SELECT bbc_event_id FROM bbc.match_meta WHERE match_id = \$1", (f.m_id,))
        if !isempty(rows) && !ismissing(rows.bbc_event_id[1])
            event_id = _bbc_event_id(rows.bbc_event_id[1])
            event_id !== nothing && return event_id
        end
    catch e
        # Discovery can still work when the optional crosswalk row is absent or the DB is
        # temporarily unavailable; the later persistence attempt is itself fail-soft.
        @warn "BBCLineupSource: event-id lookup failed" match_id = f.m_id exception = e
    end
    return _bbc_discover_event(s, f)
end

"""Return the surname used by the shirt+surname fallback."""
function _bbc_surname(name::AbstractString)
    words = split(strip(name))
    isempty(words) && return ""
    return lowercase(replace(last(words), r"[^A-Za-z0-9]" => ""))
end

function _bbc_edit_distance(a::AbstractString, b::AbstractString)
    aa, bb = collect(a), collect(b)
    previous = collect(0:length(bb))
    for (i, ca) in enumerate(aa)
        current = Vector{Int}(undef, length(bb) + 1); current[1] = i
        for j in eachindex(bb)
            current[j + 1] = min(current[j] + 1, previous[j + 1] + 1,
                                  previous[j] + (ca == bb[j] ? 0 : 1))
        end
        previous = current
    end
    return previous[end]
end

function _bbc_name_score(a::AbstractString, b::AbstractString)
    aa, bb = _bbc_surname(a), _bbc_surname(b)
    (isempty(aa) || isempty(bb)) && return 0.0
    aa == bb && return 1.0
    (occursin(aa, bb) || occursin(bb, aa)) && return 0.9
    return 1.0 - _bbc_edit_distance(aa, bb) / max(length(aa), length(bb))
end

function _bbc_tier2_id(s::BBCLineupSource, raw::_BBCRawPlayer, f::Fixture, side::String)
    s.ds === nothing && return nothing
    hasproperty(s.ds, :lineups) || return nothing
    lineups = s.ds.lineups
    (:match_id in propertynames(lineups) && :player_id in propertynames(lineups) &&
     :shirt_number in propertynames(lineups) && :player_name in propertynames(lineups) &&
     :team_side in propertynames(lineups)) || return nothing
    raw.shirt_number === nothing && return nothing

    matches = hasproperty(s.ds, :matches) ? s.ds.matches : nothing
    has_match_identity = matches !== nothing &&
                         (:match_id in propertynames(matches) &&
                          :home_team in propertynames(matches) &&
                          :away_team in propertynames(matches))
    teams = Dict{Int,Tuple{String,String}}()
    if has_match_identity
        for row in eachrow(matches)
            teams[Int(row.match_id)] = (String(row.home_team), String(row.away_team))
        end
    end

    target_team = side == "home" ? f.home : f.away
    candidates = NamedTuple{(:id, :name, :score),Tuple{Int,String,Float64}}[]
    for row in eachrow(lineups)
        ismissing(row.shirt_number) && continue
        Int(row.shirt_number) == raw.shirt_number || continue
        row_side = String(row.team_side)
        if has_match_identity
            historical = get(teams, Int(row.match_id), nothing)
            historical === nothing && continue
            historical_team = row_side == "home" ? historical[1] : historical[2]
            team_name_score(target_team, historical_team) >= 0.75 || continue
        else
            # A minimal test/store without match identity can only distinguish the side.
            row_side == side || continue
        end
        ismissing(row.player_id) && continue
        name = ismissing(row.player_name) ? "" : String(row.player_name)
        score = max(_bbc_name_score(raw.name, name), _bbc_name_score(raw.surname, name))
        score >= 0.60 || continue
        push!(candidates, (id = Int(row.player_id), name = name, score = score))
    end
    isempty(candidates) && return nothing

    by_id = Dict{Int,NamedTuple{(:id, :name, :score),Tuple{Int,String,Float64}}}()
    for candidate in candidates
        old = get(by_id, candidate.id, nothing)
        (old === nothing || candidate.score > old.score) && (by_id[candidate.id] = candidate)
    end
    ranked = sort!(collect(values(by_id)), by = x -> x.score, rev = true)
    (length(ranked) == 1 || ranked[1].score - ranked[2].score >= 0.20) || return nothing
    return _BBCPlayerResolution(ranked[1].id, ranked[1].name, "fuzzy")
end

function _bbc_synthetic_id(bbc_player_id::String)
    # Keep the reserved value away from zero (the existing BBC fallback uses zero as "unmapped").
    return -Int(mod(hash(bbc_player_id), UInt(999_999_999)) + UInt(1))
end

function _bbc_map_dict(player_map)
    out = Dict{String,_BBCPlayerResolution}()
    player_map === nothing && return out
    if player_map isa AbstractDict
        for (key, value) in player_map
            value === nothing && continue
            if value isa _BBCPlayerResolution
                out[String(key)] = value
            elseif value isa NamedTuple && hasproperty(value, :id)
                out[String(key)] = _BBCPlayerResolution(
                    Int(value.id), hasproperty(value, :name) ? String(value.name) : "", "db")
            else
                out[String(key)] = _BBCPlayerResolution(Int(value), "", "db")
            end
        end
    elseif player_map isa DataFrame
        for row in eachrow(player_map)
            id = :sofascore_player_id in propertynames(row) ? row.sofascore_player_id : missing
            ismissing(id) && continue
            name = (:sofascore_name in propertynames(row) && !ismissing(row.sofascore_name)) ?
                   String(row.sofascore_name) : ""
            out[String(row.bbc_player_id)] = _BBCPlayerResolution(Int(id), name, "db")
        end
    end
    return out
end

function _bbc_raw_player(raw, substitute::Bool)
    # Explicitly strip the player prefix; event and player URNs have different namespaces.
    player_prefix = "urn:bbc:sportsdata:football:player:"
    raw_urn = _bbc_string(_bbc_get(raw, :urn))
    startswith(raw_urn, player_prefix) || return nothing
    player_id = raw_urn[length(player_prefix) + 1:end]
    isempty(player_id) && return nothing
    name = _bbc_get(raw, :name)
    last = _bbc_string(_bbc_get(name, :last))
    first = _bbc_string(_bbc_get(name, :first))
    display = _bbc_string(_bbc_get(raw, :displayName))
    full_name = if !isempty(first) || !isempty(last)
        strip(first * " " * last)
    elseif !isempty(display)
        display
    else
        _bbc_string(_bbc_get(name, :short), "Unknown")
    end
    number = _bbc_get(raw, :shirtNumber)
    shirt = try
        number === nothing ? nothing : Int(number)
    catch
        tryparse(Int, _bbc_string(number))
    end
    raw_captain = _bbc_get(raw, :isCaptain)
    captain = raw_captain === nothing || ismissing(raw_captain) ? false : Bool(raw_captain)
    return _BBCRawPlayer(player_id, full_name, last, shirt,
                         clean_position(_bbc_string(_bbc_get(raw, :position), "M")),
                         substitute, captain)
end

function _bbc_players(team, substitute::Bool)
    players = _bbc_get(_bbc_get(team, :players), substitute ? :substitutes : :starters)
    players isa AbstractVector || players isa JSON3.Array || return _BBCRawPlayer[]
    out = _BBCRawPlayer[]
    for raw in players
        player = _bbc_raw_player(raw, substitute)
        player === nothing || push!(out, player)
    end
    return out
end

function _bbc_resolve_player(s::BBCLineupSource, raw::_BBCRawPlayer, f::Fixture,
                             side::String, mapped::Dict{String,_BBCPlayerResolution})
    resolution = get(mapped, raw.bbc_player_id, nothing)
    resolution === nothing && (resolution = _bbc_tier2_id(s, raw, f, side))
    resolution === nothing && (resolution = _BBCPlayerResolution(
        _bbc_synthetic_id(raw.bbc_player_id), "", "synthetic"))
    return resolution
end

"""
    parse_bbc_lineup(payload, fixture, as_of; ds = nothing, player_map = nothing)

Parse a BBC `match-lineups` payload without performing I/O.  This is the deterministic seam used
by unit tests and by callers that already have a response.  `player_map` may be a DataFrame from
`bbc.player_map` or a `Dict{String,Int}`.  Unknown players receive a negative synthetic ID.
"""
function parse_bbc_lineup(payload, f::Fixture, as_of::DateTime;
                          ds = nothing, player_map = nothing)
    data = payload
    if payload isa AbstractString
        data = try
            JSON3.read(payload)
        catch
            return nothing
        end
    end
    home_team, away_team = _bbc_get(data, :homeTeam), _bbc_get(data, :awayTeam)
    (home_team === nothing || away_team === nothing) && return nothing
    home_raw = vcat(_bbc_players(home_team, false), _bbc_players(home_team, true))
    away_raw = vcat(_bbc_players(away_team, false), _bbc_players(away_team, true))
    (count(x -> !x.substitute, home_raw) >= 11 &&
     count(x -> !x.substitute, away_raw) >= 11) || return nothing

    source = BBCLineupSource(ds = ds)
    mapped = _bbc_map_dict(player_map)
    function convert(raws, side)
        out = Player[]
        for raw in raws
            resolved = _bbc_resolve_player(source, raw, f, side, mapped)
            position = resolved.outcome == "synthetic" ? :M : raw.position
            push!(out, Player(resolved.player_id, raw.name, position, raw.substitute))
        end
        return out
    end
    return Lineup(convert(home_raw, "home"), convert(away_raw, "away"), true, :bbc, as_of)
end

function _bbc_player_map(s::BBCLineupSource, ids::Vector{String})
    isempty(ids) && return DataFrame(bbc_player_id = String[], sofascore_player_id = Union{Missing,Int}[],
                                     sofascore_name = Union{Missing,String}[])
    try
        return _query("SELECT bbc_player_id, sofascore_player_id, sofascore_name FROM bbc.player_map " *
                      "WHERE bbc_player_id = ANY(\$1)", (ids,))
    catch e
        @warn "BBCLineupSource: player map lookup failed" exception = e
        return DataFrame(bbc_player_id = String[], sofascore_player_id = Union{Missing,Int}[],
                         sofascore_name = Union{Missing,String}[])
    end
end

function _bbc_persist_event!(f::Fixture, event_id::String)
    try
        c = _conn()
        try
            LibPQ.execute(c, """
                INSERT INTO bbc.match_meta (match_id, bbc_event_id, status, retry_count)
                VALUES (\$1, \$2, \$3, 0)
                ON CONFLICT (match_id) DO UPDATE SET
                    bbc_event_id = EXCLUDED.bbc_event_id, status = EXCLUDED.status,
                    last_updated = now()
            """, (f.m_id, event_id, "EVENT_DISCOVERED"))
        finally
            close(c)
        end
    catch e
        @warn "BBCLineupSource: could not persist discovered event" match_id = f.m_id exception = e
    end
    return nothing
end

function _bbc_persist_lineup!(f::Fixture, event_id::String, players)
    try
        c = _conn()
        try
            LibPQ.execute(c, "BEGIN")
            for side in (players.home, players.away)
                for p in side
                    LibPQ.execute(c, """
                    INSERT INTO bbc.player_map
                        (bbc_player_id, sofascore_player_id, sofascore_name, bbc_name,
                         bbc_team_slug, is_verified)
                    VALUES (\$1, \$2, \$3, \$4, \$5, false)
                    ON CONFLICT (bbc_player_id) DO UPDATE SET
                        sofascore_player_id = COALESCE(EXCLUDED.sofascore_player_id,
                                                      bbc.player_map.sofascore_player_id),
                        sofascore_name = COALESCE(EXCLUDED.sofascore_name,
                                                 bbc.player_map.sofascore_name),
                        bbc_name = EXCLUDED.bbc_name,
                        bbc_team_slug = EXCLUDED.bbc_team_slug,
                        updated_at = now()
                    """, (p.bbc_player_id,
                          p.map_outcome == "synthetic" ? missing : p.player_id,
                          isempty(p.sofascore_name) ? missing : p.sofascore_name,
                          p.bbc_name, p.team_slug))
                end
            end
            LibPQ.execute(c, "DELETE FROM bbc.match_lineup WHERE match_id = \$1", (f.m_id,))
            for (is_home, side) in ((true, players.home), (false, players.away))
                for p in side
                    LibPQ.execute(c, """
                        INSERT INTO bbc.match_lineup
                            (match_id, bbc_player_id, is_home_team, is_substitute,
                             shirt_number, position, is_captain, bbc_name,
                             sofascore_player_id, map_outcome)
                        VALUES (\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10)
                    """, (f.m_id, p.bbc_player_id, is_home, p.substitute,
                          something(p.shirt_number, missing), String(p.position), p.captain,
                          p.bbc_name, p.player_id, p.map_outcome))
                end
            end
            LibPQ.execute(c, """
                UPDATE bbc.match_meta
                SET has_lineup = true, status = 'SUCCESS', last_updated = now()
                WHERE match_id = \$1 AND bbc_event_id = \$2
            """, (f.m_id, event_id))
            LibPQ.execute(c, "COMMIT")
        catch
            try LibPQ.execute(c, "ROLLBACK") catch end
            rethrow()
        finally
            close(c)
        end
    catch e
        @warn "BBCLineupSource: could not persist lineup" match_id = f.m_id exception = e
    end
    return nothing
end

function _bbc_lineup_from_event(s::BBCLineupSource, f::Fixture, as_of::DateTime,
                                event_id::String; base_url::String = BBC_CDN_BASE)
    try
        payload = _bbc_http_json(s, "match-lineups", Pair{String,String}[
            "urn" => "urn:bbc:sportsdata:football:event:" * event_id]; base_url)
        payload === nothing && return nothing
        ids = String[]
        # Read the IDs before the pure parser so a DB outage still permits synthetic fallback.
        for team_key in (:homeTeam, :awayTeam)
            team = _bbc_get(payload, team_key)
            for substitute in (false, true)
                for raw in _bbc_players(team, substitute)
                    push!(ids, raw.bbc_player_id)
                end
            end
        end
        mapped = _bbc_map_dict(_bbc_player_map(s, unique(ids)))
        parsed = parse_bbc_lineup(payload, f, as_of; ds = s.ds, player_map = mapped)
        parsed === nothing && (@warn "BBCLineupSource: incomplete lineup" match_id = f.m_id; return nothing)
        # Preserve transport fields that the public `Player` deliberately does not carry.
        persisted = (_BBCPersistedPlayer[], _BBCPersistedPlayer[])
        for (team, side, output) in ((_bbc_get(payload, :homeTeam), "home", persisted[1]),
                                     (_bbc_get(payload, :awayTeam), "away", persisted[2]))
            team_slug = _bbc_team_slug(team)
            raws = vcat(_bbc_players(team, false), _bbc_players(team, true))
            for raw in raws
                resolved = _bbc_resolve_player(s, raw, f, side, mapped)
                position = resolved.outcome == "synthetic" ? :M : raw.position
                push!(output, _BBCPersistedPlayer(
                    raw.bbc_player_id, resolved.player_id, raw.name,
                    resolved.sofascore_name, raw.substitute, raw.shirt_number,
                    position, resolved.outcome, team_slug, raw.captain))
            end
        end
        _bbc_persist_lineup!(f, event_id, (home = persisted[1], away = persisted[2]))
        return parsed
    catch e
        @warn "BBCLineupSource: lineup lookup failed" match_id = f.m_id exception = e
        return nothing
    end
end

function lineup(s::BBCLineupSource, f::Fixture, as_of::DateTime)
    try
        event_id = _bbc_event_for(s, f)
        event_id === nothing && return nothing
        return _bbc_lineup_from_event(s, f, as_of, event_id)
    catch e
        @warn "BBCLineupSource: event discovery failed" match_id = f.m_id exception = e
        return nothing
    end
end

"""
    ProvisionalDB()

`sofascore.lineup_provisional`, filtered to rows scraped at or before `as_of`.

Health warning, measured: `confirmed` has never been true for any match in this table. Every
scrape so far has run 4.4-5.8 hours before kick-off and SofaScore publishes the confirmed XI
about an hour out, so what this returns is a *predicted* XI. The scraper is correct; it has
simply never been invoked inside the window where the answer changes. Treat
`kickoff - scraped_at` as the usable signal, not `confirmed`.
"""
struct ProvisionalDB <: AbstractLineupSource end

function lineup(::ProvisionalDB, f::Fixture, as_of::DateTime)
    df = _query(LINEUP_SQL, (f.m_id, as_of))
    isempty(df) && return nothing

    home = Player[]; away = Player[]
    for r in eachrow(df)
        p = Player(Int(r.player_id),
                   ismissing(r.player_name) ? "Unknown" : String(r.player_name),
                   clean_position(ismissing(r.position) ? "M" : String(r.position)),
                   coalesce(r.substitute, false))
        push!(coalesce(r.is_home_team, true) ? home : away, p)
    end
    (isempty(home) || isempty(away)) && return nothing

    return Lineup(home, away, any(coalesce.(df.confirmed, false)), :provisional,
                  maximum(DateTime.(df.scraped_at)))
end

"""
    LastHistorical(ds)

Each team's most recent completed XI from `ds.lineups`. The floor of the chain: always answers,
never fresh. `compare_matchday_lineups` in the prototype measured how far this moves the model's
positional-sum inputs versus a provisional XI, and that comparison is still worth porting.
"""
struct LastHistorical <: AbstractLineupSource
    ds::Any
end
LastHistorical() = LastHistorical(nothing)

function lineup(s::LastHistorical, f::Fixture, as_of::DateTime)
    s.ds === nothing && return nothing
    h = _last_xi(s.ds, f.home, as_of)
    a = _last_xi(s.ds, f.away, as_of)
    (isempty(h) || isempty(a)) && return nothing
    return Lineup(h, a, false, :last_historical, as_of)
end

function _last_xi(ds, team::AbstractString, as_of::DateTime)
    m = ds.matches
    rows = findall(i -> (m.home_team[i] == team || m.away_team[i] == team) &&
                        DateTime(m.match_date[i]) <= as_of, 1:nrow(m))
    isempty(rows) && return Player[]
    i = rows[argmax(m.match_date[rows])]
    mid, side = m.match_id[i], m.home_team[i] == team ? "home" : "away"

    lu = ds.lineups
    sel = findall(j -> lu.match_id[j] == mid && String(lu.team_side[j]) == side, 1:nrow(lu))
    return Player[Player(Int(lu.player_id[j]),
                         ismissing(lu.player_name[j]) ? "Unknown" : String(lu.player_name[j]),
                         clean_position(ismissing(lu.position[j]) ? "M" : String(lu.position[j])),
                         coalesce(lu.is_substitute[j], false)) for j in sel]
end

"""
    JsonPin(dir)

A manually pinned XI at `<dir>/<match_id>.json`, in SofaScore's own response shape. Top of the
chain so a human can override every automated source for one fixture.
"""
struct JsonPin <: AbstractLineupSource
    dir::String
end

function lineup(s::JsonPin, f::Fixture, ::DateTime)
    path = joinpath(s.dir, "$(f.m_id).json")
    isfile(path) || return nothing
    data = try
        JSON3.read(read(path, String))
    catch e
        @warn "JsonPin: unparseable lineup file" path exception = e
        return nothing
    end
    (haskey(data, :home) && haskey(data, :away)) || return nothing
    pull(side) = Player[Player(Int(p.player.id), String(p.player.name),
                               clean_position(String(p.position)), Bool(p.substitute))
                        for p in side.players]
    return Lineup(pull(data.home), pull(data.away),
                  get(data, :confirmed, false), :json_pin, unix2datetime(0))
end

"""
    SourceChain(sources...)

First source that answers wins. This is the prototype's tiering, preserved verbatim in order --
manual pin, then announced XI, then last completed XI -- because each tier is strictly less
informative than the one above and the fallback never fails outright.

Distinct from `GateChain`, which is conjunctive. Two different combinators, deliberately.
"""
struct SourceChain{T<:Tuple} <: AbstractLineupSource
    sources::T
end
SourceChain(ss::AbstractLineupSource...) = SourceChain(ss)

function lineup(c::SourceChain, f::Fixture, as_of::DateTime)
    for s in c.sources
        out = lineup(s, f, as_of)
        out === nothing || return out
    end
    return nothing
end

"Normalise a raw position label to `:G`, `:D`, `:M`, `:F`. Unknown labels become `:M`."
function clean_position(pos::AbstractString)
    p = uppercase(strip(pos))
    (p in ("G", "GK") || occursin("GOALKEEPER", p)) && return :G
    (p in ("D", "DF") || occursin("DEFENDER", p) || occursin("BACK", p)) && return :D
    (p in ("F", "FW", "A") || occursin("FORWARD", p) ||
     occursin("STRIKER", p) || occursin("WINGER", p) ||
     occursin("ATTACKER", p)) && return :F
    return :M
end
