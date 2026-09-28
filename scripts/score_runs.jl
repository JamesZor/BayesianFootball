#!/usr/bin/env julia

using BayesianFootball
using CSV
using DataFrames
using Serialization
using UUIDs

const PG = BayesianFootball.Models.PreGame

# Read-side compatibility for pre-2026-09-03 three-parameter joint-observation artefacts.
function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   T::Type{<:PG.JointGammaPoissonObservation})
    T isa DataType && return invoke(Serialization.deserialize,
        Tuple{Serialization.AbstractSerializer,DataType}, s, T)
    fields = Any[]
    for _ in 1:3
        tag = Int32(read(s.io, UInt8)::UInt8)
        push!(fields, Serialization.handle_deserialize(s, tag))
    end
    return PG.JointGammaPoissonObservation(fields[1], fields[2], fields[3], PG.SharedKappa())
end

const DEFAULT_TARGET_SEASONS = ["24/25", "25/26"]
const DEFAULT_EXPECTED_FIXTURES = 710
const SCORE_RUNS_USAGE = """
usage: julia --project -t 16 scripts/score_runs.jl <runs.csv> [options]

options:
  --portfolio close_option_b|t25_calibrated
  --target-seasons 23/24,24/25,25/26
  --expected-fixtures N|none

The optional CSV `control` column names (by label or UUID) the paired control for
that row.  Without it, the single role=:control row remains the control for all arms.
"""

function load_run_refs(path)
    frame = CSV.read(path, DataFrame)
    required = [:label, :experiment, :run_id, :role]
    all(c -> c in propertynames(frame), required) ||
        error("run CSV requires columns $(required)")
    return Harness.RunRef[Harness.RunRef(r.label, r.experiment, r.run_id, r.role)
                          for r in eachrow(frame)]
end

function _control_groups(frame::AbstractDataFrame, refs::AbstractVector{Harness.RunRef})
    :control in propertynames(frame) || return [(; control = nothing, refs)]
    length(frame.label) == length(refs) || error("run CSV rows and references are misaligned")
    by_key = Dict{String,Harness.RunRef}()
    for ref in refs
        for key in (ref.label, string(ref.run_id))
            haskey(by_key, key) && error("run CSV has duplicate control key '$key'")
            by_key[key] = ref
        end
    end
    grouped = Dict{UUID,Vector{Harness.RunRef}}()
    for (row, ref) in zip(eachrow(frame), refs)
        value = row.control
        if ismissing(value) || isempty(strip(String(value)))
            ref.role === :control || error("$(ref.label): CSV control is required for candidate rows")
            get!(grouped, ref.run_id, Harness.RunRef[])
            continue
        end
        key = strip(String(value))
        haskey(by_key, key) || error("$(ref.label): control '$key' does not name a CSV label or UUID")
        control = by_key[key]
        control.role === :control || error("$(ref.label): control '$key' must have role=control")
        push!(get!(grouped, control.run_id, Harness.RunRef[]), ref)
    end
    groups = NamedTuple[]
    for control in refs
        control.role === :control || continue
        members = get(grouped, control.run_id, Harness.RunRef[])
        pushfirst!(members, control)
        unique!(members)
        push!(groups, (; control, refs = members))
    end
    isempty(groups) && error("CSV control column did not assign any candidate arms to a control")
    return groups
end

function load_run_groups(path)
    frame = CSV.read(path, DataFrame)
    refs = load_run_refs(path)
    return _control_groups(frame, refs)
end

function _parse_target_seasons(value)
    seasons = filter(!isempty, strip.(split(value, ',')))
    isempty(seasons) && error("--target-seasons requires at least one comma-separated season")
    return seasons
end

function _parse_expected_fixtures(value)
    lowercase(value) == "none" && return nothing
    count = tryparse(Int, value)
    count === nothing && error("--expected-fixtures must be a positive integer or none")
    count < 1 && error("--expected-fixtures must be a positive integer or none")
    return count
end

function parse_args(args)
    isempty(args) && error(SCORE_RUNS_USAGE)
    first(args) in ("-h", "--help") && return (; help = true)
    path = first(args)
    portfolio = nothing
    target_seasons = copy(DEFAULT_TARGET_SEASONS)
    expected_fixtures = DEFAULT_EXPECTED_FIXTURES
    i = 2
    while i <= length(args)
        if args[i] == "--portfolio"
            i + 1 <= length(args) || error("--portfolio requires a container")
            portfolio = Symbol(args[i + 1])
            i += 2
        elseif startswith(args[i], "--portfolio=")
            portfolio = Symbol(split(args[i], "=", limit = 2)[2])
            i += 1
        elseif args[i] == "--target-seasons"
            i + 1 <= length(args) || error("--target-seasons requires comma-separated seasons")
            target_seasons = _parse_target_seasons(args[i + 1])
            i += 2
        elseif startswith(args[i], "--target-seasons=")
            target_seasons = _parse_target_seasons(split(args[i], "=", limit = 2)[2])
            i += 1
        elseif args[i] == "--expected-fixtures"
            i + 1 <= length(args) || error("--expected-fixtures requires N or none")
            expected_fixtures = _parse_expected_fixtures(args[i + 1])
            i += 2
        elseif startswith(args[i], "--expected-fixtures=")
            expected_fixtures = _parse_expected_fixtures(split(args[i], "=", limit = 2)[2])
            i += 1
        else
            error("unknown argument: $(args[i])\n$SCORE_RUNS_USAGE")
        end
    end
    portfolio === nothing || portfolio in (:close_option_b, :t25_calibrated) ||
        error("--portfolio must be close_option_b or t25_calibrated")
    return (; help = false, path, portfolio, target_seasons, expected_fixtures)
end

"Include the prototype loaders whose experiments occur in `refs` (type definitions for load_fit)."
function include_run_loaders(refs)
    if any(ref -> ref.experiment == "scottish_pyramid_grw_cups", refs)
        include(joinpath(@__DIR__, "..", "current_development",
                         "grw_pyramid_cups", "l01_loader.jl"))
    end
    prototype_loaders = Dict(
        "scottish_lower_momentum_grw" =>
            ("10_momentum_multiscale_grw", "l10_momentum_grw_loader.jl"),
        "scottish_lower_decompression" =>
            ("11_decompression_pxg_covariate", "l11_decompression_loader.jl"),
        "scottish_lower_decoupled_xg" =>
            ("12_decoupled_generative_xg", "l12_loader.jl"))
    for (experiment, (folder, loader)) in prototype_loaders
        any(ref -> ref.experiment == experiment, refs) || continue
        include(joinpath(@__DIR__, "..", "experiments", "scottish_lower", folder, loader))
    end
    return nothing
end

"""
Score every CSV control group in memory; nothing is written. Returns `(; scores, failures)`.
`scripts/validate_klm_board_dryrun.jl` uses this to rehearse the runbook without writing.
"""
function score_csv_groups(groups, ds; target_seasons = DEFAULT_TARGET_SEASONS,
                          expected_fixtures = DEFAULT_EXPECTED_FIXTURES)
    tiers = Harness.club_season_tiers(ds)
    failures = NamedTuple[]
    frames = DataFrame[]
    # `include_run_loaders` defines prototype types during this invocation. Enter the latest
    # world before deserializing them (Julia 1.12 otherwise warns about stale bindings).
    for group in groups
        scores = Base.invokelatest(Harness.score_runs, group.refs; ds = ds, tiers = tiers,
                                   control = group.control,
                                   target_seasons = target_seasons,
                                   expected_fixtures = expected_fixtures,
                                   failures = failures)
        isempty(scores) || push!(frames, scores)
    end
    return (; scores = isempty(frames) ? DataFrame() : vcat(frames...), failures)
end

function main(path; portfolio = nothing, target_seasons = DEFAULT_TARGET_SEASONS,
              expected_fixtures = DEFAULT_EXPECTED_FIXTURES)
    groups = load_run_groups(path)
    refs = reduce(vcat, (group.refs for group in groups); init = Harness.RunRef[])
    unique!(refs)
    include_run_loaders(refs)

    # The pyramid store is a strict superset of the Lower panel and supplies the
    # 54/55 monitor fixtures for pooled fits.
    ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)
    db = PostgresStorage("harness")
    Harness.ensure_harness_schema!(db)

    if portfolio !== nothing
        summary = Base.invokelatest(Harness.portfolio_runs, refs;
            ds = ds, container = portfolio, db = db)
        show(stdout, MIME("text/plain"), summary)
        println("\nPortfolio-scored $(nrow(summary)) runs under $portfolio.")
        return 0
    end

    (; scores, failures) = score_csv_groups(groups, ds; target_seasons, expected_fixtures)
    Harness.write_scores!(db, scores)
    println("Scored $(length(unique(scores.run_id))) / $(length(refs)) runs; " *
            "wrote $(nrow(scores)) score rows.")
    if !isempty(failures)
        println(stderr, "RUN FAILURES ($(length(failures))):")
        for failure in failures
            println(stderr, "- $(failure.label) $(failure.run_id): $(failure.reason)")
        end
    end
    return isempty(failures) ? 0 : 2
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed = parse_args(ARGS)
    if parsed.help
        print(SCORE_RUNS_USAGE)
    else
        exit(main(parsed.path; portfolio = parsed.portfolio,
                  target_seasons = parsed.target_seasons,
                  expected_fixtures = parsed.expected_fixtures))
    end
end
