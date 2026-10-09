# Read-only audit of the failed market-observation persistence smoke, no sampling.
# %% 1. Runtime and diagnostic loader
using BayesianFootball, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("audit only in a 16-threaded beast REPL")
include(joinpath(@__DIR__, "l04_parity_audit.jl"))

# %% 2. Original frozen model and saved round trip
const W2_PARITY_CANDIDATES = Module(:Wave2ParityCandidates)
Base.include(W2_PARITY_CANDIDATES, joinpath(@__DIR__, "candidates.jl"))
w2_parity_candidate = only(filter(c -> c.name == "grw_marketobs", W2_PARITY_CANDIDATES.CANDIDATES))

# %% 3. Diagnosis receipts (read-only database; no fit/check mutation)
w2_parity_facts = Wave2ParityAudit.audit(w2_parity_candidate, W2_PARITY_CANDIDATES.EXPERIMENT;
    output = "/root/BF_runs/qs_experiment_w2_out/phase2")
