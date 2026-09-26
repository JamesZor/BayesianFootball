#!/usr/bin/env julia

using BayesianFootball
using CSV
using DataFrames

const DEFAULT_REGISTER_SEED = joinpath(@__DIR__, "..", "experiments", "register_seed.csv")

function main(path::AbstractString = DEFAULT_REGISTER_SEED)
    isfile(path) || error("Register seed CSV does not exist: $path")
    seed = CSV.read(path, DataFrame; missingstring = "")
    required = [:id, :date, :todo, :question, :dimension, :status, :decision, :run_ids, :readme]
    missing_columns = Symbol[column for column in required if column ∉ propertynames(seed)]
    isempty(missing_columns) || error(
        "Register seed is missing required columns $(join(string.(missing_columns), ", ")).")

    db = Training.PostgresStorage("harness")
    Harness.ensure_harness_schema!(db)
    for row in eachrow(seed)
        Harness.write_experiment!(db, row)
    end
    println("Seeded $(nrow(seed)) harness register rows from $(abspath(path)).")
    return nothing
end

main(length(ARGS) > 1 ? error("usage: seed_register.jl [register_seed.csv]") :
     (isempty(ARGS) ? DEFAULT_REGISTER_SEED : only(ARGS)))
