# Read-only recovery of the persisted fit after harness scoring failure.
let
    I = Training.Inference
    out = "/root/BF_runs/qs_experiment_w2_out/phase3"
    conn = I._db_connect(w2_grid_db)
    rows = try
        I._db_rows(conn, "SELECT run_id, name, status, git_commit FROM runs WHERE experiment_name = \$1 AND name = \$2 ORDER BY created_at DESC", ("scottish_lower_qs_wave2_2426", "qs_joint"))
    finally
        close(conn)
    end
    Wave2Grid.CSV.write(joinpath(out, "blocked_runs_qs_joint.csv"), rows)
    Wave2Grid.DataFrames.nrow(rows) == 1 || error("expected one persisted qs_joint grid fit")
    id = string(rows.run_id[1])
    fit = Training.load_fit(w2_grid_db, id)
    length(fit.folds) == 40 || error("persisted grid incomplete")
    diag = Wave2Grid.diagnostics(fit, "qs_joint", 0, id)
    Wave2Grid.CSV.write(joinpath(out, "blocked_diagnostics_qs_joint.csv"), diag)
    checks = Harness.read_checks(w2_grid_db; run_id = id, stage = "grid")
    Wave2Grid.CSV.write(joinpath(out, "blocked_checks_qs_joint.csv"), checks)
    println("BLOCKED_AUDIT run_id=", id, " folds=", length(fit.folds), " max_rhat=", maximum(diag.max_rhat), " min_bulk=", minimum(diag.min_ess_bulk), " min_tail=", minimum(diag.min_ess_tail), " divergences=", sum(diag.n_divergent), " rhat_reruns_due=", sum(.!diag.rhat_pass), " tail_reviews=", sum(.!diag.tail_ess_400))
end
