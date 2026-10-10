# Wave 3 Phase 3 grid (wave 2's r05_grid.jl pattern): fusion of goals, xG and market rates.
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

# %% 2. Frozen recipes and manager-approved single selected arm
const W3_GRID_CANDIDATES = Module(:Wave3GridCandidates)
Base.include(W3_GRID_CANDIDATES, joinpath(@__DIR__, "candidates.jl"))
w3_grid_name = ENV["QSX3_ONLY"]
w3_grid_candidate = only(filter(c -> c.name == w3_grid_name, W3_GRID_CANDIDATES.CANDIDATES))
w3_grid_metadata = (; experiment = W3_GRID_CANDIDATES.EXPERIMENT)
ENV["BF_LOG_DIR"] = "/root/BF_runs/logs/qs_experiment_w3"
w3_grid_output = "/root/BF_runs/qs_experiment_w3_out/phase3"

# %% 3. Pinned snapshot and canonical experiment storage
w3_grid_ds = Main.QSMarketArm.checked_datastore()
w3_grid_db = Training.PostgresStorage(w3_grid_metadata.experiment)
Training.ensure_schema!(w3_grid_db)
Harness.ensure_harness_schema!(w3_grid_db)
# The final control is loaded/verified, never refit by this preflight.
w3_grid_control = w3_grid_candidate.role === :control ? nothing :
    Wave3Grid.checked_control(w3_grid_db, w3_grid_metadata.experiment, w3_grid_output)

# %% 4. Durable initial grid, then exactly one prescribed seeded rerun per bad fold
# Seed schedule and queue infrastructure are visible in l05_grid.jl / DECISIONS_LOG.md.
# Harness performs latents/coverage/parity checks and its standard clustered scoring.
w3_grid_result = Wave3Grid.run(w3_grid_candidate, w3_grid_ds, w3_grid_db, w3_grid_metadata;
    output = w3_grid_output, control = w3_grid_control)
println("PHASE3_ARM_DONE arm=", w3_grid_name, " final_run=", w3_grid_result.run_id)
