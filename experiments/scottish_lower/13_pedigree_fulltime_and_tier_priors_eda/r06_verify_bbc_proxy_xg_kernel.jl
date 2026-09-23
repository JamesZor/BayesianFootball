# r06 independent parity check: execute the PROJECT Julia parser and empirical-Bayes
# ShotXGModel over the identical fixture IDs, then compare its sums with the Python extraction.
# Same-window empirical-Bayes estimation only; no MCMC or database writes.
# BF_DB_URL is consumed only by LibPQ and never printed.

using BayesianFootball
using CSV
using DataFrames
using LibPQ

const R06_DIR = @__DIR__
const R06_DATA = joinpath(R06_DIR, "data")
const R06_RESULTS = joinpath(R06_DIR, "results")
const R06_MATCH_PATH = joinpath(R06_DATA, "r06_bbc_proxy_xg_match.csv")
const R06_JULIA_PATH = joinpath(R06_DATA, "r06_bbc_proxy_xg_match_julia_kernel.csv")
const R06_PARITY_PATH = joinpath(R06_RESULTS, "r06_bbc_proxy_xg_kernel_parity.csv")
const R06_SHOT_EVENTS = ["goal", "attempt_missed", "attempt_saved", "attempt_blocked", "post",
                          "penalty_missed", "penalty_saved"]

"""Fetch the exact event fields used by `Features.build_shots`, for already-fixed r06 fixture IDs."""
function r06_fetch_events(conn::LibPQ.Connection, match_ids::Vector{Int})
    # A raw Julia string preserves PostgreSQL's regex end-anchor `$` and libpq `$1/$2`
    # placeholders literally. Escaping `$` in an interpolated Julia string is error-prone.
    sql = raw"""
    SELECT lt.match_id, lt.post_index, lt.event_type,
           CASE
             WHEN regexp_replace(lt.team, '-fc$', '') =
                  regexp_replace(mm.bbc_home_slug, '-fc$', '') THEN true
             WHEN regexp_replace(lt.team, '-fc$', '') =
                  regexp_replace(mm.bbc_away_slug, '-fc$', '') THEN false
             ELSE NULL
           END AS is_home,
           lt.text
    FROM bbc.live_text AS lt
    JOIN bbc.match_meta AS mm ON mm.match_id = lt.match_id
    WHERE lt.match_id = ANY($1) AND lt.event_type = ANY($2)
    ORDER BY lt.match_id, lt.post_index
    """
    return DataFrame(LibPQ.execute(conn, sql, [match_ids, R06_SHOT_EVENTS]))
end

"""Build the `build_shots` parser/model input without constructing a DataStore."""
function r06_project_shots(events::DataFrame)
    parsed = BayesianFootball.Features.parse_shot.(String.(events.event_type), events.text)
    shots = DataFrame(
        match_id = Int.(events.match_id),
        is_home = events.is_home,
        zone = [x.zone for x in parsed],
        body_part = [x.body_part for x in parsed],
        context = [x.context for x in parsed],
        is_penalty = [x.is_penalty for x in parsed],
        parsed = [x.parsed for x in parsed],
    )
    shots.is_goal = String.(events.event_type) .== "goal"
    return shots
end

function r06_julia_proxy_rows(fixture::DataFrame, events::DataFrame)
    shots = r06_project_shots(events)
    model = BayesianFootball.Features.fit_shot_xg(shots; k = 25.0)
    predicted = BayesianFootball.Features.predict_xg(model, shots)
    python_shots = CSV.read(joinpath(R06_DATA, "r06_bbc_proxy_xg_shot_descriptors.csv"),
                            DataFrame; truestrings = ["True"], falsestrings = ["False"])
    parser_equal = nrow(python_shots) == nrow(shots) &&
        python_shots.match_id == events.match_id && python_shots.post_index == events.post_index &&
        all(isequal(python_shots[!, c], String.(shots[!, c])) for c in (:zone, :body_part, :context)) &&
        all(isequal(python_shots[!, c], shots[!, c]) for c in (:is_penalty, :parsed))
    CSV.write(joinpath(R06_RESULTS, "r06_bbc_proxy_xg_parser_parity.csv"),
              DataFrame(event_rows = [nrow(events)], pass = [parser_equal]))
    parser_equal || error("r06 row-level Python/project parser parity failed")
    totals = Dict{Int,Tuple{Float64,Float64}}()
    resolved = Dict{Int,Tuple{Int,Int}}()
    for (i, shot) in enumerate(eachrow(shots))
        ismissing(shot.is_home) && continue
        match_id = Int(shot.match_id)
        home_xg, away_xg = get(totals, match_id, (0.0, 0.0))
        home_n, away_n = get(resolved, match_id, (0, 0))
        if shot.is_home
            totals[match_id] = (home_xg + predicted[i], away_xg)
            resolved[match_id] = (home_n + 1, away_n)
        else
            totals[match_id] = (home_xg, away_xg + predicted[i])
            resolved[match_id] = (home_n, away_n + 1)
        end
    end
    rows = NamedTuple[]
    for row in eachrow(fixture)
        match_id = Int(row.match_id)
        home_xg, away_xg = get(totals, match_id, (0.0, 0.0))
        home_n, away_n = get(resolved, match_id, (0, 0))
        push!(rows, (
            match_id,
            julia_proxy_xg_home = home_n > 0 ? home_xg : missing,
            julia_proxy_xg_away = away_n > 0 ? away_xg : missing,
            julia_proxy_xg_available_home = home_n > 0 ? 1 : 0,
            julia_proxy_xg_available_away = away_n > 0 ? 1 : 0,
        ))
    end
    return DataFrame(rows), model
end

function r06_compare(python::DataFrame, julia::DataFrame)
    fixture_ids_equal = nrow(python) == nrow(julia) == length(unique(python.match_id)) ==
                        length(unique(julia.match_id)) && Set(python.match_id) == Set(julia.match_id)
    joined = innerjoin(
        select(python, :match_id, :proxy_xg_home, :proxy_xg_away,
               :proxy_xg_available_home, :proxy_xg_available_away),
        julia;
        on = :match_id,
    )
    # Python persists six decimal places, so half a unit in the final decimal is the exact
    # serialization bound. Coverage and missingness must be bit-identical.
    availability_equal = (joined.proxy_xg_available_home .== joined.julia_proxy_xg_available_home) .&
                         (joined.proxy_xg_available_away .== joined.julia_proxy_xg_available_away)
    missingness_equal = isequal(ismissing.(joined.proxy_xg_home), ismissing.(joined.julia_proxy_xg_home)) &&
                        isequal(ismissing.(joined.proxy_xg_away), ismissing.(joined.julia_proxy_xg_away))
    covered_home = .!ismissing.(joined.proxy_xg_home) .& .!ismissing.(joined.julia_proxy_xg_home)
    covered_away = .!ismissing.(joined.proxy_xg_away) .& .!ismissing.(joined.julia_proxy_xg_away)
    home_error = any(covered_home) ? maximum(abs.(Float64.(joined.proxy_xg_home[covered_home]) .-
                                                  Float64.(joined.julia_proxy_xg_home[covered_home]))) : 0.0
    away_error = any(covered_away) ? maximum(abs.(Float64.(joined.proxy_xg_away[covered_away]) .-
                                                  Float64.(joined.julia_proxy_xg_away[covered_away]))) : 0.0
    return DataFrame([(
        fixture_rows = nrow(joined),
        fixture_ids_equal,
        missingness_equal,
        availability_equal_rows = sum(availability_equal),
        max_abs_home_difference = home_error,
        max_abs_away_difference = away_error,
        serialization_tolerance = 0.5e-6,
        pass = fixture_ids_equal && missingness_equal && sum(availability_equal) == nrow(joined) &&
               home_error <= 0.5e-6 && away_error <= 0.5e-6,
    )])
end

function main()
    fixture = CSV.read(R06_MATCH_PATH, DataFrame; types = Dict(:match_id => Int))
    dsn = get(ENV, "BF_DB_URL", nothing)
    dsn === nothing && error("BF_DB_URL must be set; its value is intentionally not printed.")
    conn = LibPQ.Connection(dsn)
    events = try
        LibPQ.execute(conn, "BEGIN READ ONLY")
        LibPQ.execute(conn, "SET LOCAL statement_timeout = '120s'")
        r06_fetch_events(conn, Int.(fixture.match_id))
    finally
        close(conn)
    end
    julia_rows, model = r06_julia_proxy_rows(fixture, events)
    CSV.write(R06_JULIA_PATH, julia_rows)
    parity = r06_compare(fixture, julia_rows)
    CSV.write(R06_PARITY_PATH, parity)
    parity.pass[1] || error("r06 Python/Julia proxy kernel parity failed; inspect $R06_PARITY_PATH")
    println("r06 Julia kernel parity passed over $(nrow(fixture)) fixtures and $(nrow(events)) events; " *
            "base rate=$(model.base_rate), penalty xG=$(model.penalty_xg).")
end

main()
