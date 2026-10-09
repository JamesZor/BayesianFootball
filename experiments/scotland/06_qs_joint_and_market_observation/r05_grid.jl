# Joint goals+xG and market-rate dynamics comparison; no ROI/staking.
# Frozen panel/scope/budget; explicit per-chain seeds and atomic fold checkpoints.
# No fold observes its own held-out measurements. A prescribed Rhat rerun is
# always selected, not compared with the original to pick the favourable fit.

# %% 1. Packages and runtime
using BayesianFootball, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("grid only in a 16-threaded beast REPL")
include(joinpath(@__DIR__, "l05_grid.jl"))

# %% 2. Frozen recipes and manager-authorised single selected arm
const W2_GRID_CANDIDATES = Module(:Wave2GridCandidates)
Base.include(W2_GRID_CANDIDATES, joinpath(@__DIR__, "candidates.jl"))
w2_grid_name = ENV["QSX2_ONLY"]
w2_grid_candidate = only(filter(c -> c.name == w2_grid_name, W2_GRID_CANDIDATES.CANDIDATES))
w2_grid_metadata = (; experiment = W2_GRID_CANDIDATES.EXPERIMENT)
ENV["BF_LOG_DIR"] = "/root/BF_runs/logs/qs_experiment_w2"

# %% 3. Pinned snapshot and canonical experiment storage
w2_grid_ds = Main.QSMarketArm.checked_datastore()
w2_grid_db = Training.PostgresStorage(w2_grid_metadata.experiment)
Training.ensure_schema!(w2_grid_db)
Harness.ensure_harness_schema!(w2_grid_db)

# %% 4. Durable initial grid, then exactly one prescribed seeded rerun per bad fold
# Seed schedule and queue infrastructure are visible in l05_grid.jl / DECISIONS_LOG.md.
# Harness performs latents/coverage/parity checks and its standard clustered scoring.
w2_grid_result = Wave2Grid.run(w2_grid_candidate, w2_grid_ds, w2_grid_db, w2_grid_metadata;
    output = "/root/BF_runs/qs_experiment_w2_out/phase3")
println("PHASE3_ARM_DONE arm=", w2_grid_name, " final_run=", w2_grid_result.run_id)
