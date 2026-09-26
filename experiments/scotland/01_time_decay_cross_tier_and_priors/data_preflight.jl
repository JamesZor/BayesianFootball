using BayesianFootball
using DataFrames
using Dates

const D = BayesianFootball.Data
const TOURNAMENTS = [54, 55, 56, 57]
const DOMAIN_NAMES = (:odds, :lineups, :bbc, :bbc_events)

function coverage_table(df::DataFrame, match_ids::Set{Int})
    nrow(df) == 0 && return (rows = 0, matches = 0, coverage = 0.0, outside_matches = 0)
    ids = Set(Int.(skipmissing(df.match_id)))
    matched = intersect(ids, match_ids)
    return (rows = nrow(df), matches = length(matched),
            coverage = length(matched) / length(match_ids),
            outside_matches = length(setdiff(ids, match_ids)))
end

function coverage_by_tournament(df::DataFrame, matches::DataFrame)
    rows = NamedTuple[]
    for tournament_id in TOURNAMENTS
        match_ids = Set(Int.(matches.match_id[matches.tournament_id .== tournament_id]))
        domain = nrow(df) == 0 ? DataFrame() : df[map(id -> !ismissing(id) && Int(id) in match_ids,
                                                      df.match_id), :]
        coverage = coverage_table(domain, match_ids)
        push!(rows, (; tournament_id, coverage...))
    end
    return DataFrame(rows)
end

function domain_report(ds, matches::DataFrame)
    match_ids = Set(Int.(matches.match_id))
    return NamedTuple{DOMAIN_NAMES}(map(DOMAIN_NAMES) do name
        df = getfield(ds, name)
        (; aggregate = coverage_table(df, match_ids),
           by_tournament = coverage_by_tournament(df, matches))
    end)
end

function canonical_continuity(matches::DataFrame)
    required = ("home_team", "away_team", "tournament_id")
    all(in(names(matches)), required) || error("matches lacks canonical team or tournament columns")
    sides = vcat(DataFrame(team = matches.home_team, tournament_id = matches.tournament_id),
                 DataFrame(team = matches.away_team, tournament_id = matches.tournament_id))
    by_team = combine(groupby(sides, :team),
                      :tournament_id => (x -> length(unique(skipmissing(x)))) => :tiers)
    return (multi_tier_teams = count(>(1), by_team.tiers), teams = nrow(by_team))
end

function print_domain_report(name::Symbol, report)
    total = report.aggregate
    println("$(name) rows=$(total.rows) matches=$(total.matches) coverage=$(round(total.coverage; digits=4)) outside_matches=$(total.outside_matches)")
    show(stdout, MIME("text/plain"), report.by_tournament); println()
end

function preflight()
    @assert D.tournament_ids(D.ScottishAll()) == TOURNAMENTS
    ds = D.load_datastore_cached(D.ScottishAll(); force = true)
    matches = ds.matches
    match_ids = Set(Int.(matches.match_id))
    observed_tournaments = sort(unique(Int.(skipmissing(matches.tournament_id))))
    duplicate_match_ids = nrow(matches) - length(match_ids)
    @assert observed_tournaments == TOURNAMENTS
    @assert !isempty(matches)

    domains = domain_report(ds, matches)
    timestamps = (
        match_date_missing = count(ismissing, matches.match_date),
        match_date_min = minimum(skipmissing(matches.match_date)),
        match_date_max = maximum(skipmissing(matches.match_date)),
    )
    continuity = canonical_continuity(matches)

    println("SCOTTISH_ALL_PREFLIGHT julia_version=$(VERSION)")
    println("matches=$(nrow(matches)) unique_match_ids=$(length(match_ids)) duplicate_match_ids=$(duplicate_match_ids) tournaments=$(join(observed_tournaments, ','))")
    println("tier_counts=")
    show(stdout, MIME("text/plain"), combine(groupby(matches, :tournament_id), nrow => :matches)); println()
    for name in DOMAIN_NAMES
        print_domain_report(name, getfield(domains, name))
    end
    println("timestamps missing=$(timestamps.match_date_missing) min=$(timestamps.match_date_min) max=$(timestamps.match_date_max)")
    println("canonical_continuity multi_tier_teams=$(continuity.multi_tier_teams) teams=$(continuity.teams)")
    return (; ds, domains, timestamps, continuity, duplicate_match_ids)
end

preflight()
