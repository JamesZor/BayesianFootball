# Diagnostic only after the unchanged control failed its smoke allocation gate.
# No sampling, DB writes, threshold changes or package patches.
using BayesianFootball, CSV, DataFrames, DynamicPPL, Turing, ReverseDiff, Distributions
using LinearAlgebra
BLAS.set_num_threads(1)
allocation_rows = NamedTuple[]
for (label, candidate, raw) in (("wave2_legacy",legacy_candidate,legacy_ds),
                              ("w2_panel_control",S2C.CANDIDATES[1],s2ds))
    inputs = Harness._fold_inputs(candidate,raw; stage=:smoke)
    fs = first(first(inputs.feature_sets))
    m = Harness.tape_metrics(candidate.model,fs; seed=20260911,latency_reps=20)
    push!(allocation_rows,(; label, fold=inputs.source_indices[1], allocated_bytes=m.allocated_bytes,
        gradient_ms=m.gradient_ms,tape_instructions=m.tape_instructions,n_parameters=m.n_parameters,
        julia=string(VERSION),turing=string(pkgversion(Turing)),dynamicppl=string(pkgversion(DynamicPPL)),
        reversediff=string(pkgversion(ReverseDiff)),distributions=string(pkgversion(Distributions))))
    CSV.write(joinpath(@__DIR__,"allocation_instructions_$(label).csv"),DataFrame(m.allocating_instructions))
    println("ALLOCATION_DIAGNOSTIC label=$label bytes=$(m.allocated_bytes) instructions=$(length(m.allocating_instructions))")
end
CSV.write(joinpath(@__DIR__,"allocation_diagnostic.csv"),DataFrame(allocation_rows))
