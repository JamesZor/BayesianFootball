# Read-only final grid audit: no data extraction, scoring, sampling or DB writes.
using BayesianFootball, CSV, DataFrames, LinearAlgebra, ThreadPinning
pinthreads(:cores); BLAS.set_num_threads(1)
include("/root/BF_runs/qs_experiment/experiments/scotland/06_qs_joint_and_market_observation/l05_grid.jl")
let
    experiment = "scottish_lower_qs_wave2_2426"
    db = Training.PostgresStorage(experiment)
    out = "/root/BF_runs/qs_experiment_w2_out/phase3"
    conn = Training.Inference._db_connect(db)
    rows = try
        Training.Inference._db_rows(conn, "SELECT run_id, name, experiment_name, status, git_commit FROM runs WHERE experiment_name = \$1 AND name IN ('grw_joint','qs_joint','grw_marketobs','qs_marketobs') ORDER BY name", (experiment,))
    finally
        close(conn)
    end
    nrow(rows) == 4 && Set(rows.name) == Set(Wave2Grid.ORDER) || error("new/duplicate/missing grid runs")
    all(==("completed"), rows.status) || error("incomplete DB run")
    CSV.write(joinpath(out, "final_db_runs.csv"), rows)
    checks = DataFrame[]
    summaries = NamedTuple[]
    for arm in Wave2Grid.ORDER
        summary = CSV.read(joinpath(out, "grid_summary_$arm.csv"), DataFrame)
        id = string(summary.final_run_id[1])
        id == string(only(eachrow(filter(:name => ==(arm), rows))).run_id) || error("summary DB UUID mismatch")
        strip(read(joinpath(out, "run_$arm.txt"), String)) == id || error("receipt mismatch")
        fit = Training.load_fit(db, id)
        length(fit.folds) == 40 && sort([f.fold for f in fit.folds]) == collect(1:40) || error("fold coverage")
        all(f -> size(f.chain, 1) == 1000 && size(f.chain, 3) == 4, fit.folds) || error("retained chain budget")
        records = Harness.read_checks(db; run_id = id, stage = "grid")
        hard = filter(:severity => ==("hard"), records)
        Set(hard.check) == Set(["filtration", "latents", "target_coverage", "fit_parity"]) &&
            all(==("pass"), hard.status) || error("missing/failed grid hard checks")
        push!(checks, records)
        diag = Wave2Grid.diagnostics(fit, arm, 0, id)
        all(diag.rhat_pass) && sum(diag.n_divergent) == 0 || error("unexpected reruns/divergences")
        push!(summaries, (; candidate = arm, run_id = id, folds = 40, chains = 4, retained_per_chain = 1000,
            max_rhat = maximum(diag.max_rhat), min_ess_bulk = minimum(diag.min_ess_bulk),
            min_ess_tail = minimum(diag.min_ess_tail), tail_review_folds = sum(.!diag.tail_ess_400),
            divergences = sum(diag.n_divergent), hard_passes = nrow(hard)))
    end
    CSV.write(joinpath(out, "final_grid_checks.csv"), vcat(checks...))
    CSV.write(joinpath(out, "final_grid_audit.csv"), DataFrame(summaries))
    println("FINAL_GRID_AUDIT_PASS runs=4 folds=160 chains=640 retained_per_chain=1000 hard_passes=16; no new joint runs")
end
