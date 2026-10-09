# Phase 4 goal-arm grids on the pinned ScottishLower snapshot, in manager priority order.
# Include in an owned persistent beast REPL. ENV:
#   QSX_ONLY     comma-separated arms (default: all, in ORDER)
#   QSX_CORES    first:last physical cores to pin (default 0:Threads.nthreads()-1)
# Resume is the harness's: a completed run with the same config hash is reloaded, and
# in-flight folds resume from data/checkpoints/<experiment>/<arm>.
using BayesianFootball, DataFrames, Dates, LinearAlgebra, Random, ThreadPinning, UUIDs
strip(read(`hostname`, String)) == "mcmc-beast" || error("grid only on mcmc-beast")
g4_cores = let r = split(get(ENV, "QSX_CORES", "0:$(Threads.nthreads() - 1)"), ":")
    parse(Int, r[1]):parse(Int, r[2])
end
length(g4_cores) == Threads.nthreads() || error("core range $(g4_cores) != $(Threads.nthreads()) threads")
pinthreads(collect(g4_cores))
BLAS.set_num_threads(1)
isdefined(Main, :QSMarketArm) || include(joinpath(@__DIR__, "market_arm.jl"))
const G4C = Module(:QSGridCandidates)
Base.include(G4C, joinpath(@__DIR__, "candidates.jl"))
ENV["BF_LOG_DIR"] = "/root/BF_runs/logs/qs_experiment"

# Manager priority: the primary comparison lands first.
const G4_ORDER = ["control_grw", "qs_market_r", "qs_weak_r", "control_td"]
# Turing draws from the task-local RNG; child tasks are seeded from their parent, so a
# fixed seed per arm before `grid` makes the queued sampling replayable (frozen here).
const G4_SEEDS = Dict("control_grw" => 4_001, "qs_market_r" => 4_002,
                      "qs_weak_r" => 4_003, "control_td" => 4_004)

g4_ds = QSMarketArm.checked_datastore()
g4_db = Training.PostgresStorage(G4C.EXPERIMENT)
Training.ensure_schema!(g4_db)
Harness.ensure_harness_schema!(g4_db)
g4_only = split(get(ENV, "QSX_ONLY", join(G4_ORDER, ",")), ",")
g4_lookup = Dict(c.name => c for c in G4C.CANDIDATES)
for name in filter(in(g4_only), G4_ORDER)
    c = g4_lookup[name]
    started = time()
    println("[GRID_START] candidate=$name threads=$(Threads.nthreads()) cores=$(g4_cores) at=$(now())")
    flush(stdout)
    Random.seed!(G4_SEEDS[name])
    res = Harness.grid(c; ds = g4_ds, experiment = G4C.EXPERIMENT, db = g4_db, control = G4C.CONTROL)
    hard = all(r -> r.status == "pass", filter(r -> r.severity == "hard", res.records))
    conv = filter(r -> r.check == "convergence", res.records)
    ll = filter(r -> r.model == name && r.subset == "target" && r.market == "1X2" &&
                     r.metric == "logloss", res.scores)
    println("[GRID] candidate=$name hard=$(hard ? "PASS" : "FAIL") hours=$(round((time() - started) / 3600; digits = 2)) ",
            "logloss_1x2=$(isempty(ll.value) ? "n/a" : first(ll.value)) ",
            "convergence=$(isempty(conv) ? "resumed" : first(conv).value) run_id=$(res.run_id)")
    flush(stdout)
end
println("GOAL_GRID_DONE")
