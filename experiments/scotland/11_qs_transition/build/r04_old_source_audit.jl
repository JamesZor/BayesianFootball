# D11: old wave-2 grid src with the current environment; self-contained current audit.
# Start s2 in allocation_sources/081ba5a6, whose src/Project come from git archive.
# The current Manifest is copied unchanged. No writes to other beast checkouts.
using LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
empty!(ARGS)
append!(ARGS,[joinpath(@__DIR__,"..","candidates.jl"),"--only","qs_control","--folds","opening",
    "--out",joinpath(@__DIR__,"allocation_selfcontained_$(get(ENV,"D11_LABEL","081ba5a6")).tsv")])
include("/root/BF_runs/qs_transition/scripts/tape_allocation_audit.jl")
