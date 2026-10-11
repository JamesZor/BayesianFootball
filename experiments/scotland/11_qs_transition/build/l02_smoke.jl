module S2Smoke
using BayesianFootball, CSV, DataFrames, Random
const H = BayesianFootball.Harness
const T = BayesianFootball.Training
function run(candidate, ds, experiment; output=@__DIR__)
    db = T.PostgresStorage(experiment)
    T.ensure_schema!(db)
    H.ensure_harness_schema!(db)
    grid = H.fit_config(candidate; stage=:grid, experiment)
    smoke = H.fit_config(candidate; stage=:smoke, experiment)
    H.find_completed_run(db, grid) === nothing || error("completed grid exists; do not refit")
    tags = ["harness", "todo:046", "stage:S2"]
    T.save_model(db, candidate.name, candidate.model; tags, description=candidate.hypothesis)
    T.save_splitter(db, "lower_2326", grid.splitter; tags)
    T.save_sampler(db, "s2_grid", grid.sampler; tags)
    T.save_sampler(db, "s2_smoke", smoke.sampler; tags)
    T.save_config(db, candidate.name*"_grid_recipe", grid; tags)
    T.save_config(db, candidate.name*"_smoke_template", smoke; tags)
    Random.seed!(20261046)
    started = time()
    res = H.smoke(candidate; ds, experiment, db, log_dir=joinpath(output,"logs"))
    wall_seconds = time()-started
    rows = [(; candidate=candidate.name, run_id=string(res.run_id), recipe_hash=r.recipe_hash,
        check=r.check, severity=r.severity, status=r.status, value=repr(r.value), detail=r.detail) for r in res.records]
    CSV.write(joinpath(output,"smoke_checks_$(candidate.name).csv"), DataFrame(rows))
    conv = only(filter(r -> r.check=="convergence", res.records))
    metrics = res.perf.metrics
    # Full-load probe, proportional warmup+sample work, with concurrency held fixed.
    projected_seconds = metrics.window_seconds *
        (60*candidate.sampler.n_chains*(candidate.sampler.n_warmup+candidate.sampler.n_samples)) /
        (metrics.chains*(metrics.n_warmup+metrics.n_samples))
    hard = all(r -> r.status=="pass", filter(r -> r.severity=="hard", res.records)) &&
        issubset(Set(H.SMOKE_REQUIRED_CHECKS),Set(r.check for r in res.records))
    summary = (; candidate=candidate.name, run_id=string(res.run_id), recipe_hash=H.recipe_hash(candidate),
        hard_pass=hard, wall_seconds, convergence_status=conv.status, conv.value...,
        projected_grid_seconds=projected_seconds, projected_full_fold_seconds=projected_seconds/60,
        probe_window_seconds=metrics.window_seconds)
    CSV.write(joinpath(output,"smoke_summary_$(candidate.name).csv"),DataFrame([summary]))
    println("S2_SMOKE ", repr(summary))
    hard || error("smoke hard checks failed")
    return summary
end
end
