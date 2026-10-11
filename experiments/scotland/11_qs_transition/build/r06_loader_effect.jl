# D11: distinguish source regression from legacy-loader/session side effects.
# Run after r05_matched_allocation.jl in the current-source REPL.
using CSV, DataFrames
const D11W2 = Module(:D11Wave2Recipes)
Base.include(D11W2,normpath(joinpath(@__DIR__,"..","..","06_qs_joint_and_market_observation","candidates.jl")))
d11loader_rows = NamedTuple[]
for (idx,pair) in zip((1,21,41),d11sets)
    m = BayesianFootball.Harness.tape_metrics(d11candidate.model,first(pair);seed=20260911,latency_reps=20)
    push!(d11loader_rows,(; source_label="current_after_wave2_loader",fold=idx,
        rows=length(first(pair).data[:ordered_match_ids]),parameters=m.n_parameters,
        instructions=m.tape_instructions,allocated_bytes=m.allocated_bytes,gradient_ms=m.gradient_ms))
    println("D11_LOADER_EFFECT fold=$idx bytes=$(m.allocated_bytes)")
end
CSV.write(joinpath(@__DIR__,"allocation_loader_effect.csv"),DataFrame(d11loader_rows))
