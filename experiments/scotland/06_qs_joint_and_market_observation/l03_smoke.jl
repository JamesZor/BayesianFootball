# Fresh-beast smoke adapter: pinned input, canonical registration, durable evidence.
module Wave2Smoke
import BayesianFootball, CSV, DataFrames, Dates, Random
const H = BayesianFootball.Harness
const T = BayesianFootball.Training

"Register canonical templates and check completed grid hashes before any sampling."
function preflight(candidate, experiment, db)
    grid = H.fit_config(candidate; stage = :grid, experiment)
    smoke = H.fit_config(candidate; stage = :smoke, experiment)
    completed = H.find_completed_run(db, grid)
    completed === nothing || error("completed grid already exists for $(candidate.name): $completed; review rather than refit")
    tags = ["harness", "wave2", "todo:041"]
    T.save_model(db, candidate.name, candidate.model; tags, description = candidate.hypothesis)
    T.save_splitter(db, "lower", grid.splitter; tags)
    T.save_sampler(db, "w2_grid", grid.sampler; tags)
    T.save_sampler(db, "w2_smoke", smoke.sampler; tags)
    grid_hash = T.save_config(db, candidate.name * "_grid_recipe", grid; tags)
    smoke_hash = T.save_config(db, candidate.name * "_smoke_template", smoke; tags)
    # Harness assigns its invocation nonce internally and persists that exact fit later.
    println("[PREFLIGHT] candidate=", candidate.name, " completed_grid=none grid_registry_hash=",
        grid_hash, " smoke_template_registry_hash=", smoke_hash)
end

"Run all smoke hard checks, record convergence/performance, and stop on a hard failure."
function run(candidate, ds, metadata; output, log_dir)
    db = T.PostgresStorage(metadata.experiment)
    T.ensure_schema!(db)
    H.ensure_harness_schema!(db)
    preflight(candidate, metadata.experiment, db)
    order = ("grw_joint", "qs_joint", "grw_marketobs", "qs_marketobs")
    seed = 20261039 + findfirst(==(candidate.name), order)
    Random.seed!(seed)
    started = time()
    res = H.smoke(candidate; ds, experiment = metadata.experiment, db, log_dir)
    wall_seconds = time() - started
    mkpath(output)
    check_rows = [(; candidate = candidate.name, run_id = string(res.run_id),
        recipe_hash = r.recipe_hash, git_sha = r.git_sha, check = r.check,
        severity = r.severity, status = r.status, value = repr(r.value), detail = r.detail)
        for r in res.records]
    CSV.write(joinpath(output, "smoke_checks_$(candidate.name).csv"), DataFrames.DataFrame(check_rows))
    hard_rows = filter(r -> r.severity == "hard", res.records)
    required = Set(H.SMOKE_REQUIRED_CHECKS)
    required_present = issubset(required, Set(r.check for r in hard_rows))
    hard = required_present && all(r -> r.status == "pass", hard_rows)
    conv_row = only(filter(r -> r.check == "convergence", res.records))
    conv = conv_row.value
    metrics = res.perf === nothing ? nothing : res.perf.metrics
    # Largest-fold, full-concurrency probe extrapolation; not a promise about grid duration.
    eta_seconds = metrics === nothing ? NaN : metrics.window_seconds *
        (40 * candidate.sampler.n_chains * (candidate.sampler.n_warmup + candidate.sampler.n_samples)) /
        (metrics.chains * (metrics.n_warmup + metrics.n_samples))
    summary = (; candidate = candidate.name, run_id = string(res.run_id),
        recipe_hash = H.recipe_hash(candidate), git_sha = res.fit.metadata.git_commit,
        root_seed = seed, wall_seconds, hard_pass = hard,
        convergence_status = conv_row.status, max_rhat = conv.max_rhat,
        min_ess_bulk = conv.min_ess_bulk, min_ess_tail = conv.min_ess_tail,
        divergences = conv.divergences, divergence_rate = conv.divergence_rate,
        estimated_grid_seconds = eta_seconds,
        performance_status = res.perf === nothing ? "missing" : res.perf.status)
    CSV.write(joinpath(output, "smoke_summary_$(candidate.name).csv"), DataFrames.DataFrame([summary]))
    H.write_experiment!(db, (; id = metadata.experiment, date = Dates.today(),
        todo = 41, question = metadata.question, dimension = metadata.dimension,
        status = hard ? "active" : "blocked", decision = "Smoke only; grid awaits manager approval.",
        run_ids = string(res.run_id),
        run_commits = Dict(string(res.run_id) => (; candidate = candidate.name,
            git_sha = res.fit.metadata.git_commit)), readme = metadata.readme))
    for r in res.records
        println("[CHECK] candidate=", candidate.name, " check=", r.check,
            " severity=", r.severity, " status=", r.status, " value=", repr(r.value))
    end
    println("[SMOKE] ", summary, " ", H.perf_summary(res.perf))
    flush(stdout)
    hard || error("smoke hard checks fail/incomplete for $(candidate.name)")
    res.run_id === nothing && error("smoke fit has no persisted UUID")
    return summary
end
end
