# Phase 2 correctness smoke, not a grid or predictive comparison (wave 2's r03_smoke.jl pattern).
# Fresh owned beast REPL per arm; pinned ScottishLower snapshot via wave 1's checked adapter.

# %% 1. Packages and runtime
using BayesianFootball, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("smoke only on a 16-threaded beast REPL")
include(joinpath(@__DIR__, "l03_smoke.jl"))

# %% 2. Declarative candidates and exactly one selected arm
const W3_SMOKE_CANDIDATES = Module(:Wave3Candidates)
Base.include(W3_SMOKE_CANDIDATES, joinpath(@__DIR__, "candidates.jl"))
w3_smoke_name = get(ENV, "QSX3_ONLY", "")
w3_smoke_candidate = only(filter(c -> c.name == w3_smoke_name, W3_SMOKE_CANDIDATES.CANDIDATES))
ENV["BF_LOG_DIR"] = "/root/BF_runs/logs/qs_experiment_w3"

# %% 3. Pinned data and canonical registry metadata
w3_smoke_ds = Main.QSMarketArm.checked_datastore()
w3_smoke_metadata = (; experiment = W3_SMOKE_CANDIDATES.EXPERIMENT,
    question = W3_SMOKE_CANDIDATES.QUESTION, dimension = W3_SMOKE_CANDIDATES.DIMENSION,
    readme = relpath(W3_SMOKE_CANDIDATES.README, pwd()))

# %% 4. Full harness smoke: folds 1 and 40, 2x(200+200), 16-chain performance probe
w3_smoke_summary = Wave3Smoke.run(w3_smoke_candidate, w3_smoke_ds, w3_smoke_metadata;
    output = get(ENV, "QSX3_SMOKE_OUTPUT", "/root/BF_runs/qs_experiment_w3_out/phase2"),
    log_dir = ENV["BF_LOG_DIR"])
println("PHASE2_SMOKE_DONE arm=", w3_smoke_name)
