# Phase 2 correctness smoke, not a grid or predictive comparison.
# Fresh owned beast REPL per arm; same pinned ScottishLower snapshot as wave 1.
# The generic CLI loads ScottishPyramid, so use the unchanged wave-1 checked adapter.

# %% 1. Packages and runtime
using BayesianFootball, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("smoke only on a 16-threaded beast REPL")
include(joinpath(@__DIR__, "l03_smoke.jl"))

# %% 2. Declarative candidates and exactly one selected arm
const W2_SMOKE_CANDIDATES = Module(:Wave2Candidates)
Base.include(W2_SMOKE_CANDIDATES, joinpath(@__DIR__, "candidates.jl"))
w2_smoke_name = get(ENV, "QSX2_ONLY", "")
w2_smoke_candidate = only(filter(c -> c.name == w2_smoke_name, W2_SMOKE_CANDIDATES.CANDIDATES))
ENV["BF_LOG_DIR"] = "/root/BF_runs/logs/qs_experiment_w2"

# %% 3. Pinned data and canonical registry metadata
w2_smoke_ds = Main.QSMarketArm.checked_datastore()
w2_smoke_metadata = (; experiment = W2_SMOKE_CANDIDATES.EXPERIMENT,
    question = W2_SMOKE_CANDIDATES.QUESTION, dimension = W2_SMOKE_CANDIDATES.DIMENSION,
    readme = relpath(W2_SMOKE_CANDIDATES.README, pwd()))

# %% 4. Full harness smoke: folds 1 and 40, 2x(200+200), 16-chain performance probe
w2_smoke_summary = Wave2Smoke.run(w2_smoke_candidate, w2_smoke_ds, w2_smoke_metadata;
    output = "/root/BF_runs/qs_experiment_w2_out/phase2",
    log_dir = ENV["BF_LOG_DIR"])
println("PHASE2_SMOKE_DONE arm=", w2_smoke_name)
