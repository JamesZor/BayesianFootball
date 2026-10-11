# Matched-fold diagnostic; run in old-src and current-src REPLs with the same Manifest.
# No sampling, DB writes, package mutation or gate alteration.
using BayesianFootball, CSV, DataFrames, Serialization, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
const D11C = Module(:D11Recipes)
Base.include(D11C,joinpath(@__DIR__,"..","candidates.jl"))
d11raw = Serialization.deserialize("/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls")
d11candidate = first(D11C.CANDIDATES)
d11ds = Data.apply_scope(d11raw,d11candidate.scope)
d11splitter = Data.ScopedWalkForwardCV(d11candidate.scope)
d11bounds = Data.create_id_boundaries(d11ds,d11splitter)
d11sets = Features.create_features(d11bounds[[1,21,41]],d11ds,d11candidate.model,d11splitter)
d11rows = NamedTuple[]
for (idx, pair) in zip((1,21,41),d11sets)
    m = Harness.tape_metrics(d11candidate.model,first(pair);seed=20260911,latency_reps=20)
    push!(d11rows,(; source_label=ENV["D11_LABEL"],fold=idx,
        rows=length(first(pair).data[:ordered_match_ids]),parameters=m.n_parameters,
        instructions=m.tape_instructions,allocated_bytes=m.allocated_bytes,gradient_ms=m.gradient_ms))
    println("D11_MATCHED source=$(ENV["D11_LABEL"]) fold=$idx bytes=$(m.allocated_bytes)")
end
CSV.write(joinpath(@__DIR__,"allocation_matched_$(ENV["D11_LABEL"]).csv"),DataFrame(d11rows))
