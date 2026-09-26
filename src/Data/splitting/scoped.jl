# Scoped training and evaluation over one canonical walk-forward clock.

const SCOTTISH_SPFL_TOURNAMENTS = (54, 55, 56, 57)
const SCOTTISH_CUP_TOURNAMENTS = (73, 982, 1520)
const DATA_SCOPE_CUP_POLICIES = (:none, :senior_spfl_ties)

"A declarative data policy wrapped around one canonical walk-forward clock."
Base.@kwdef struct DataScope
    name::String
    train_tournaments::Vector{Int} = [56, 57]
    cups::Symbol = :none
    target_tournaments::Vector{Int} = [56, 57]
    monitor_tournaments::Vector{Int} = Int[]
    clock_tournaments::Vector{Int} = [56, 57]
    target_seasons::Vector{String} = ["24/25", "25/26"]
    history_seasons::Int = 2
    dynamics_col::Symbol = :match_biweek

    function DataScope(name, train_tournaments, cups, target_tournaments,
                       monitor_tournaments, clock_tournaments, target_seasons,
                       history_seasons, dynamics_col)
        train = unique(Int.(train_tournaments))
        target = unique(Int.(target_tournaments))
        monitor = unique(Int.(monitor_tournaments))
        clock = unique(Int.(clock_tournaments))
        isempty(train) && throw(ArgumentError("DataScope train_tournaments must not be empty"))
        isempty(target) && throw(ArgumentError("DataScope target_tournaments must not be empty"))
        isempty(clock) && throw(ArgumentError("DataScope clock_tournaments must not be empty"))
        issubset(Set(target), Set(train)) || throw(ArgumentError(
            "DataScope target_tournaments must be a subset of train_tournaments"))
        issubset(Set(clock), Set(target)) || throw(ArgumentError(
            "DataScope clock_tournaments must be a subset of target_tournaments"))
        isempty(intersect(Set(target), Set(monitor))) || throw(ArgumentError(
            "DataScope monitor_tournaments must be disjoint from target_tournaments"))
        cups in DATA_SCOPE_CUP_POLICIES || throw(ArgumentError(
            "DataScope cups must be :none or :senior_spfl_ties; got :$cups"))
        isempty(target_seasons) && throw(ArgumentError("DataScope target_seasons must not be empty"))
        history_seasons >= 0 || throw(ArgumentError("DataScope history_seasons must be non-negative"))
        dynamics_col in keys(CALENDAR_DYNAMICS_WIDTH_WEEKS) || throw(ArgumentError(
            "DataScope dynamics_col must be one of $(sort!(collect(keys(CALENDAR_DYNAMICS_WIDTH_WEEKS)))); got :$dynamics_col"))
        return new(String(name), train, cups, target, monitor, clock,
                   String.(target_seasons), Int(history_seasons), Symbol(dynamics_col))
    end
end

"A canonical grouped walk-forward split whose fitted rows are widened by its scope."
struct ScopedWalkForwardCV <: AbstractSplitter
    scope::DataScope
end

_scope_grouped(scope::DataScope) = GroupedCVConfig(
    tournament_groups = [copy(scope.clock_tournaments)],
    target_seasons = copy(scope.target_seasons),
    history_seasons = scope.history_seasons,
    dynamics_col = scope.dynamics_col,
    warmup_period = 0,
    end_dynamics = nothing,
    stop_early = true,
)

"Map club-season membership from SPFL league rows only."
function club_season_tiers(matches::AbstractDataFrame)
    required = (:home_team, :away_team, :season, :tournament_id)
    all(column -> hasproperty(matches, column), required) || error(
        "club_season_tiers: matches require columns $(collect(required))")
    tiers = Dict{Tuple{String,String},Int}()
    for row in eachrow(matches)
        tier = Int(row.tournament_id)
        tier in SCOTTISH_SPFL_TOURNAMENTS || continue
        season = String(row.season)
        for club in (String(row.home_team), String(row.away_team))
            key = (club, season)
            prior = get(tiers, key, tier)
            prior == tier || error(
                "club_season_tiers: $club appears in tiers $prior and $tier in $season")
            tiers[key] = tier
        end
    end
    return tiers
end

club_season_tiers(ds_leagues) = club_season_tiers(ds_leagues.matches)

_football_season(date::Date) = let year_start = month(date) >= 7 ? year(date) : year(date) - 1
    lpad(string(year_start % 100), 2, '0') * "/" *
        lpad(string((year_start + 1) % 100), 2, '0')
end

function _scope_copy_domain(frame::DataFrame, kept_ids::Set{Int})
    copied = copy(frame)
    :match_id in propertynames(copied) || return copied
    return copied[in.(Int.(copied.match_id), Ref(kept_ids)), :]
end

function _neutral_tie_ids()
    path = joinpath(@__DIR__, "..", "data", "scottish_neutral_venue_ties.csv")
    isfile(path) || error(
        "Required scoped-cup exclusion file is missing: $path. Restore the git-tracked " *
        "Data input before using cups=:senior_spfl_ties.")
    rows = CSV.read(path, DataFrame)
    :match_id in propertynames(rows) || error("neutral-venue CSV requires match_id")
    return Set(Int.(rows.match_id))
end

"""
    apply_scope(ds, scope) -> DataStore

Return a detached store containing only the scope's training, target, monitor and accepted
cup fixtures. Cup season labels are rewritten to football seasons before point-in-time SPFL
membership is evaluated. The input store and every cached DataFrame remain untouched.
"""
function apply_scope(ds::DataStore, scope::DataScope)
    matches = copy(ds.matches)
    required = (:match_id, :tournament_id, :season, :match_date, :home_team, :away_team)
    all(column -> column in propertynames(matches), required) || error(
        "apply_scope: matches require columns $(collect(required))")

    is_cup = in.(Int.(matches.tournament_id), Ref(Set(SCOTTISH_CUP_TOURNAMENTS)))
    seasons = String.(matches.season)
    for index in findall(is_cup)
        seasons[index] = _football_season(Date(matches.match_date[index]))
    end
    matches.season = seasons

    allowed = union(Set(scope.train_tournaments), Set(scope.target_tournaments),
                    Set(scope.monitor_tournaments))
    keep = in.(Int.(matches.tournament_id), Ref(allowed)) .& .!is_cup
    if scope.cups === :senior_spfl_ties
        tiers = club_season_tiers(matches)
        column_neutral_ids = :neutral_venue in propertynames(matches) ?
            Set(Int(matches.match_id[i]) for i in eachindex(matches.match_id)
                if coalesce(matches.neutral_venue[i], false)) : Set{Int}()
        # The CSV also contains four audited scoreability exclusions, so it remains
        # authoritative even if a native neutral-venue column is added later.
        neutral_ids = union(_neutral_tie_ids(), column_neutral_ids)
        for index in findall(is_cup)
            season = seasons[index]
            home = (String(matches.home_team[index]), season)
            away = (String(matches.away_team[index]), season)
            keep[index] = haskey(tiers, home) && haskey(tiers, away) &&
                          !(Int(matches.match_id[index]) in neutral_ids)
        end
    end

    kept_matches = matches[keep, :]
    kept_ids = Set(Int.(kept_matches.match_id))
    return DataStore(ds.segment, kept_matches,
        _scope_copy_domain(ds.statistics, kept_ids),
        _scope_copy_domain(ds.odds, kept_ids),
        _scope_copy_domain(ds.lineups, kept_ids),
        _scope_copy_domain(ds.incidents, kept_ids),
        _scope_copy_domain(ds.betfair_odds, kept_ids),
        _scope_copy_domain(ds.bbc, kept_ids),
        _scope_copy_domain(ds.bbc_events, kept_ids))
end

function _scope_step_map(matches::AbstractDataFrame, scope::DataScope, season)
    clock_map = _effective_step_map(
        matches, scope.clock_tournaments, season, scope.dynamics_col)
    isempty(clock_map) && return Dict{Int,Int}()
    clock_rows = matches[in.(Int.(matches.tournament_id), Ref(Set(scope.clock_tournaments))) .&
                         coalesce.(matches.season .== season, false), :]
    anchor = minimum(_week_ending_sunday.(Date.(clock_rows.match_date)))
    width = CALENDAR_DYNAMICS_WIDTH_WEEKS[scope.dynamics_col]
    result = Dict{Int,Int}()
    for row in eachrow(matches[coalesce.(matches.season .== season, false), :])
        elapsed = div(Dates.value(_week_ending_sunday(Date(row.match_date)) - anchor), 7)
        # Widened July rows may precede the first clock fixture. Collapse them into step 1,
        # exactly as pcx_align_time! did; the cross-check below is intentionally limited to
        # clock-tournament rows, whose canonical effective clock has no off-anchor rows.
        result[Int(row.match_id)] = max(1, cld(1 + elapsed, width))
    end
    for (match_id, step) in clock_map
        get(result, match_id, 0) == step || error(
            "Scoped clock mismatch for clock-tournament match $match_id: " *
            "calendar=$(get(result, match_id, 0)), canonical=$step")
    end
    return result
end

function get_next_matches(ds::DataStore, meta::GroupedSplitMetaData,
                          splitter::ScopedWalkForwardCV)::AbstractDataFrame
    scope = splitter.scope
    clock_steps = _effective_step_map(
        ds.matches, scope.clock_tournaments, meta.target_season, scope.dynamics_col)
    observed = sort!(unique!(collect(values(clock_steps))))
    position = findfirst(>(meta.time_step), observed)
    isnothing(position) && return ds.matches[Int[], :]
    predict_step = observed[position]
    all_steps = _scope_step_map(ds.matches, scope, meta.target_season)
    held_tournaments = union(Set(scope.target_tournaments), Set(scope.monitor_tournaments))
    return subset(ds.matches,
        :tournament_id => ByRow(tournament -> Int(tournament) in held_tournaments),
        :season => ByRow(isequal(meta.target_season)),
        :match_id => ByRow(match_id -> get(all_steps, Int(match_id), 0) == predict_step))
end

get_next_matches(ds::DataStore, pair::Tuple{Any,<:AbstractSplitMetaData},
                 splitter::ScopedWalkForwardCV) =
    get_next_matches(ds, pair[2], splitter)

function create_id_boundaries(ds::DataStore, splitter::ScopedWalkForwardCV)
    scope = splitter.scope
    canonical = create_id_boundaries(ds, _scope_grouped(scope))
    output = Vector{Tuple{SplitBoundary,GroupedSplitMetaData}}()
    matches = ds.matches
    match_ids = Int.(matches.match_id)
    season_by_id = Dict(Int(row.match_id) => String(row.season) for row in eachrow(matches))
    kickoffs = Dict(Int(row.match_id) => _match_kickoff(row) for row in eachrow(matches))
    cup_training = scope.cups === :senior_spfl_ties ?
                   Set(SCOTTISH_CUP_TOURNAMENTS) : Set{Int}()
    training_tournaments = union(Set(scope.train_tournaments), cup_training)

    for (boundary, meta) in canonical
        heldout = get_next_matches(ds, meta, splitter)
        if isempty(heldout)
            # Preserve positional alignment with the canonical grouped splitter.
            push!(output, (boundary, meta))
            continue
        end
        cutoff = minimum(_match_kickoff(row) for row in eachrow(heldout))
        canonical_history = Set(Int.(boundary.history_match_ids))
        canonical_target = Set(Int.(boundary.target_match_ids))
        canonical_ids = union(canonical_history, canonical_target)
        history_seasons = Set(String(matches.season[i]) for i in eachindex(match_ids)
                              if match_ids[i] in canonical_history)

        eligible = [Int(matches.match_id[i]) for i in axes(matches, 1)
            if Int(matches.tournament_id[i]) in training_tournaments &&
               Date(kickoffs[Int(matches.match_id[i])]) < Date(cutoff) &&
               !(Int(matches.match_id[i]) in canonical_ids)]
        added_history = Int[id for id in eligible if season_by_id[id] in history_seasons]
        added_target = Int[id for id in eligible if season_by_id[id] == meta.target_season]

        widened = SplitBoundary(boundary.fold_id, boundary.target_step,
            vcat(Int.(boundary.history_match_ids), added_history),
            vcat(Int.(boundary.target_match_ids), added_target))
        fitted = vcat(widened.history_match_ids, widened.target_match_ids)
        length(unique(fitted)) == length(fitted) || error(
            "Scoped fold $(boundary.fold_id) contains duplicate fitted match IDs")
        isempty(intersect(Set(fitted), Set(Int.(heldout.match_id)))) || error(
            "Scoped fold $(boundary.fold_id) overlaps its held-out fixtures")
        _assert_temporal_safety(matches, fitted, Int.(heldout.match_id);
            group_ids = scope.clock_tournaments, season = meta.target_season,
            train_step = meta.time_step, predict_step = "next scoped bin")
        push!(output, (widened, meta))
    end
    return output
end
