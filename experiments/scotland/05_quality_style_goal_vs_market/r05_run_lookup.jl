# Read-only diagnostic: which grid runs does mcmc_experiments hold for this experiment,
# and does the harness config-hash lookup resolve each candidate? No writes.
using BayesianFootball, DataFrames
const RLC = Module(:QSLookupCandidates)
Base.include(RLC, joinpath(@__DIR__, "candidates.jl"))
rl_db = Training.PostgresStorage(RLC.EXPERIMENT)
rl_conn = Training.Inference._db_connect(rl_db)
try
    q = "SELECT r.name, r.status, r.run_id::text AS run_id, left(c.config_hash, 12) AS hash " *
        "FROM runs r JOIN configs c ON c.config_id = r.run_id " *
        "WHERE r.experiment_name = \$1 ORDER BY r.id"
    show(stdout, MIME("text/plain"), Training.Inference._db_rows(rl_conn, q, (rl_db.experiment_name,));
         allrows = true, allcols = true)
    println()
finally
    close(rl_conn)
end
for c in RLC.CANDIDATES
    cfg = Harness.fit_config(c; stage = :grid, experiment = RLC.EXPERIMENT)
    println("LOOKUP ", c.name, " => ", something(Harness.find_completed_run(rl_db, cfg), "nothing"))
end
println("RUN_LOOKUP_DONE")
