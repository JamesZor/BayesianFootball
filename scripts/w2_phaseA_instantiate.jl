#!/usr/bin/env julia
# Phase-A build audit only: instantiate and compile 15 candidates on two folds.
# This script does not sample and writes no database rows.
using BayesianFootball
using DynamicPPL
using LogDensityProblems
using ReverseDiff
using LinearAlgebra
using ThreadPinning
using Printf
using Turing
using Serialization

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

module W2PhaseACandidates
include(joinpath(@__DIR__, "..", "experiments", "scotland",
                 "04_pyramid_tier_transition_arms", "candidates.jl"))
end

const W2 = W2PhaseACandidates
const PG = BayesianFootball.Models.PreGame

function audit_model(model, feature_set)
    turing_model = PG.build_turing_model(model, feature_set)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    theta = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    objective = values -> LogDensityProblems.logdensity(density, values)
    raw = nothing
    compiled_tape = nothing
    compile_seconds = @elapsed begin
        raw = ReverseDiff.GradientTape(objective, theta)
        compiled_tape = ReverseDiff.compile(raw)
    end
    gradient = similar(theta)
    for _ in 1:5
        ReverseDiff.gradient!(gradient, compiled_tape, theta)
    end
    gradient_seconds = minimum(@elapsed(
        ReverseDiff.gradient!(gradient, compiled_tape, theta)) for _ in 1:20)
    all(isfinite, gradient) || error("non-finite compiled gradient")
    return (; parameters = length(theta), compile_seconds,
              gradient_ms = 1_000 * gradient_seconds,
              tape_instructions = length(raw.tape), turing_model)
end

function audit_oos(model, feature_set, heldout, turing_model)
    isempty(heldout) && error("OOS audit received no held-out fixtures")
    chain = Turing.sample(turing_model, Turing.Prior(), 2; progress = false)
    rates = PG.extract_parameters(model, heldout, feature_set, chain)
    all(row -> haskey(rates, Int(row.match_id)), eachrow(heldout)) ||
        error("OOS extraction missed a held-out fixture")
    all(values -> all(isfinite, values.λ_h) && all(isfinite, values.λ_a) &&
                  all(>(0.0), values.λ_h) && all(>(0.0), values.λ_a), values(rates)) ||
        error("OOS extraction produced invalid rates")
    buffer = IOBuffer()
    Serialization.serialize(buffer, chain)
    seekstart(buffer)
    reloaded = Serialization.deserialize(buffer)
    reloaded_rates = PG.extract_parameters(model, heldout, feature_set, reloaded)
    rates == reloaded_rates || error("OOS extraction changed after chain reload")
    return length(rates)
end

raw_ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishPyramid())
println("candidate,fold_label,source_fold,target_season,n_target,parameters,compile_seconds,gradient_ms,tape_instructions")
for candidate in W2.CANDIDATES
    ds = BayesianFootball.Data.apply_scope(raw_ds, candidate.scope)
    splitter = BayesianFootball.Data.ScopedWalkForwardCV(candidate.scope)
    boundaries = BayesianFootball.Data.create_id_boundaries(ds, splitter)
    first_index = 1
    first_2425 = findfirst(pair -> pair[2].target_season == "24/25" &&
                                  isempty(pair[1].target_match_ids), boundaries)
    first_2425 === nothing && error("$(candidate.name): no empty-target first 24/25 fold")
    for (label, index) in (("fold1", first_index), ("first_2425", first_2425))
        selected = [boundaries[index]]
        feature_set = first(BayesianFootball.Features.create_features(
            selected, ds, candidate.model, splitter))[1]
        result = audit_model(candidate.model, feature_set)
        heldout = BayesianFootball.Data.get_next_matches(
            ds, boundaries[index][2], splitter)
        audit_oos(candidate.model, feature_set, heldout, result.turing_model)
        @printf("%s,%s,%d,%s,%d,%d,%.6f,%.6f,%d\n",
            candidate.name, label, index, boundaries[index][2].target_season,
            feature_set.data[:n_target_steps], result.parameters,
            result.compile_seconds, result.gradient_ms, result.tape_instructions)
        flush(stdout)
    end
end
