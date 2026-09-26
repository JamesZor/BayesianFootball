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

function load_run_refs(path)
    frame = CSV.read(path, DataFrame)
    required = [:label, :experiment, :run_id, :role]
    all(c -> c in propertynames(frame), required) ||
        error("run CSV requires columns $(required)")
    return Harness.RunRef[Harness.RunRef(r.label, r.experiment, r.run_id, r.role)
                          for r in eachrow(frame)]
end

function main(path)
    refs = load_run_refs(path)
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

    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
    leagues = Data.load_datastore_cached(Data.ScottishAll(); max_age_hours = 10_000)
    tiers = Harness.club_season_tiers(leagues)
    failures = NamedTuple[]
    scores = Harness.score_runs(refs; ds, tiers, failures)

    db = PostgresStorage("harness")
    Harness.ensure_harness_schema!(db)
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

length(ARGS) == 1 || error("usage: julia --project -t 16 scripts/score_runs.jl <runs.csv>")
exit(main(only(ARGS)))
