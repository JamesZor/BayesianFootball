# Phase 3 goal-arm smoke on the ONE pinned ScottishLower snapshot (not the ScottishPyramid
# cache scripts/run_candidates.jl loads). Include in an owned persistent beast REPL (-t 16).
using BayesianFootball, DataFrames, Dates, LinearAlgebra, ThreadPinning, UUIDs
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("smoke only on a 16-threaded beast REPL")
isdefined(Main, :QSMarketArm) || include(joinpath(@__DIR__, "market_arm.jl"))
const GSC = Module(:QSCandidates)
Base.include(GSC, joinpath(@__DIR__, "candidates.jl"))
ENV["BF_LOG_DIR"] = "/root/BF_runs/logs/qs_experiment"

g3_ds = QSMarketArm.checked_datastore()
g3_db = Training.PostgresStorage(GSC.EXPERIMENT)
Training.ensure_schema!(g3_db)
Harness.ensure_harness_schema!(g3_db)
only_names = split(get(ENV, "QSX_ONLY", join([c.name for c in GSC.CANDIDATES], ",")), ",")
for c in filter(c -> c.name in only_names, GSC.CANDIDATES)
    started = time()
    res = Harness.smoke(c; ds = g3_ds, experiment = GSC.EXPERIMENT, db = g3_db)
    hard = all(r -> r.status == "pass", filter(r -> r.severity == "hard", res.records))
    conv = filter(r -> r.check == "convergence", res.records)
    println("[SMOKE] candidate=$(c.name) hard=$(hard ? "PASS" : "FAIL") seconds=$(round(time() - started; digits = 1)) ",
            "convergence=$(isempty(conv) ? "n/a" : first(conv).value) ",
            Harness.perf_summary(res.perf), " run_id=$(res.run_id)")
    for r in res.records
        println("[CHECK] candidate=$(c.name) check=$(r.check) severity=$(r.severity) status=$(r.status)")
    end
    flush(stdout)
end
println("GOAL_SMOKE_DONE")
