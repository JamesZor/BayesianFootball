# src/harness/stages.jl
#
# Execution stages for the experiment harness:
#   - screen : in-memory MAP on all folds, scores written to harness_scores, no run persisted
#   - smoke  : 2 folds, 2×200 NUTS, all hard checks (SMOKE_REQUIRED_CHECKS, incl. tape_allocation),
#              diagnostics recorded, saved under <exp>_smoke
#   - grid   : requires passing smoke, resumes completed runs, per-fold checkpoints, full draws
#              at stride 1, score_runs against control, convergence flagged review.

const SCREEN_NAMESPACE_UUID = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

"A new smoke identity for every invocation, independent of the scientific recipe hash."
function _smoke_fit_config(candidate::Candidate, experiment::AbstractString)
    config = fit_config(candidate; stage = :smoke, experiment)
    # Never use a time: tag: `_db_recipe_tags` excludes those from config_hash.
    push!(config.tags, "smoke_nonce:" * string(uuid4()))
    return config
end

"Run in-memory MAP inference on all folds for a cohort of candidates and score them."
function screen(candidates::AbstractVector{<:Candidate};
                ds::Data.DataStore,
                experiment::AbstractString,
                control = nothing,
                target_seasons = ["24/25", "25/26"],
                bootstrap_B::Int = 10_000,
                db = nothing)
    if db !== nothing
        ensure_harness_schema!(db)
    end

    fits = Pair{RunRef, Training.Fit}[]
    errors = Tuple{String, String}[]
    check_records = NamedTuple[]

    for c in candidates
        validity_record = _screen_validity_record(c, experiment)
        push!(check_records, validity_record)
        base = (;
            run_id = validity_record.run_id,
            recipe_hash = recipe_hash(c),
            experiment = String(experiment),
            candidate = c.name,
            stage = "screen",
            git_sha = Training.git_commit_id()
        )
        try
            inputs = _fold_inputs(c, ds; stage = :screen)
            _run_hard_check!(check_records, base, "filtration", () -> _filtration_check(c, inputs))
            fit_cfg = fit_config(c; stage = :screen, experiment = experiment)
            fit = Training.fit_model(fit_cfg;
                feature_sets = inputs.feature_sets,
                oos_fixtures = inputs.oos,
                quiet = true)
            _run_hard_check!(check_records, base, "latents", () -> _latent_audit(fit; require_variance = false))
            ref = RunRef(c.name, experiment, validity_record.run_id, c.role)
            push!(fits, ref => fit)
        catch err
            detail = sprint(showerror, err)
            push!(errors, (c.name, detail))
            @error "Candidate $(c.name) failed in screen stage" exception = (err, catch_backtrace())
        end
    end

    if db !== nothing && !isempty(check_records)
        write_checks!(db, check_records)
    end

    isempty(fits) && error("All candidates failed in screen stage:\n" *
                           join(["- $name: $err" for (name, err) in errors], "\n"))

    effective_control = if control isa RunRef
        control
    elseif control !== nothing && any(first(p).label == control || string(first(p).run_id) == control for p in fits)
        control
    elseif any(first(p).role === :control for p in fits)
        nothing
    else
        first(fits)[1]
    end

    tiers = club_season_tiers(ds)
    scores = score_fits(fits; ds = ds, tiers = tiers, control = effective_control,
                        target_seasons = target_seasons, bootstrap_B = bootstrap_B)
    scores.stage .= "screen"

    if db !== nothing
        write_scores!(db, scores)
    end

    return (; scores, fits, errors)
end

"""
    smoke(candidate; ds, experiment, db = nothing, perf_probe = true, perf_chains = nthreads(),
          perf_warmup = 50, perf_samples = 50, log_dir = default_harness_log_dir(),
          proc_root = "/proc")

Run the 2-fold correctness and diagnostic gate for a candidate, then — unless `perf_probe = false`
— the load-realistic sampling-performance probe on the largest smoke fold (`perf_chains` chains of
`perf_warmup + perf_samples` iterations; a `review` row, never hard), writing its report to
`<log_dir>/smoke_perf/<experiment>/<candidate>.md`. Returns `(; fit, run_id, records, perf)`.
"""
function smoke(candidate::Candidate;
               ds::Data.DataStore,
               experiment::AbstractString,
               db = nothing,
               perf_probe::Bool = true,
               perf_chains::Int = Threads.nthreads(),
               perf_warmup::Int = 50,
               perf_samples::Int = 50,
               log_dir::AbstractString = default_harness_log_dir(),
               proc_root::AbstractString = "/proc")
    if db !== nothing
        ensure_harness_schema!(db)
    end

    inputs = _fold_inputs(candidate, ds; stage = :smoke)
    records = NamedTuple[]
    base = (;
        run_id = nothing,
        recipe_hash = recipe_hash(candidate),
        experiment = String(experiment),
        candidate = candidate.name,
        stage = "smoke",
        git_sha = Training.git_commit_id()
    )

    smoke_run_id = nothing
    fit = nothing
    perf = nothing
    try
        # 1. Gradient audit on fold 1
        grad_result = _run_hard_check!(records, base, "gradient") do
            _gradient_audit(candidate.model, first(inputs.feature_sets))
        end
        _run_diagnostic!(records, base, "gradient_telemetry", "info") do
            (; tape_bytes = grad_result.tape_bytes,
               gradient_ms = grad_result.gradient_ms,
               allocated_bytes = grad_result.allocated_bytes)
        end

        # 1b. Bytes per compiled gradient on every smoke fold, in NUTS's linked space. Hard:
        #     a fast gradient that allocates still starves 16 sampler threads through the GC.
        _run_hard_check!(records, base, "tape_allocation") do
            _tape_allocation_check(candidate.model, inputs.feature_sets)
        end

        # 2. Filtration check
        _run_hard_check!(records, base, "filtration") do
            _filtration_check(candidate, inputs)
        end

        # 3. Fit 2 folds
        fit_cfg = _smoke_fit_config(candidate, experiment)
        fit = Training.fit_model(fit_cfg;
            feature_sets = inputs.feature_sets,
            oos_fixtures = inputs.oos,
            quiet = true)

        # 4. Latent audit
        _run_hard_check!(records, base, "latents") do
            _latent_audit(fit; require_variance = true)
        end

        # 5. Score grid coherence
        _run_hard_check!(records, base, "score_grid_coherence") do
            _grid_diagnostics(fit.latents; check_coherence = true)
        end
        _run_diagnostic!(records, base, "score_grid_tail", "diagnostic") do
            _grid_diagnostics(fit.latents; check_coherence = false)
        end

        # 5b. Load-realistic sampling performance on the largest smoke fold: every sampler
        #     thread under the concurrency the grid will see. Review, never hard.
        if perf_probe
            perf = _smoke_sampling_performance!(records, base, candidate, inputs;
                experiment, n_chains = perf_chains, n_warmup = perf_warmup,
                n_samples = perf_samples, log_dir, proc_root)
        end

        # 6. Convergence diagnostic (recorded review, never throws)
        _run_diagnostic!(records, base, "convergence", "review") do
            _convergence_diagnostic(fit)
        end

        # 7. Persistence round-trip parity under <experiment>_smoke
        if db isa Training.PostgresStorage
            smoke_db = Training.PostgresStorage(String(experiment) * "_smoke")
            Training.ensure_schema!(smoke_db)
            smoke_run_id = Training.save_fit(fit, smoke_db)
            reloaded = Training.load_fit(smoke_db, smoke_run_id)
            _run_hard_check!(records, base, "fit_parity") do
                _fit_parity(fit, reloaded)
            end
        else
            push!(records, merge(base, (;
                check = "fit_parity",
                severity = "hard",
                status = "abstain",
                value = NamedTuple(),
                detail = "fit_parity skipped: db is not a PostgresStorage",
                at = now()
            )))
        end

        if db !== nothing
            stamped = NamedTuple[merge(r, (; run_id = smoke_run_id)) for r in records]
            write_checks!(db, stamped)
        end

        return (; fit, run_id = smoke_run_id, records, perf)
    catch err
        if db !== nothing
            try
                write_checks!(db, records)
            catch
            end
        end
        rethrow(err)
    end
end

"Run the full walk-forward grid for a candidate with resume, checkpoints, and scoring."
function grid(candidate::Candidate;
              ds::Data.DataStore,
              experiment::AbstractString,
              db,
              control = nothing,
              target_seasons = ["24/25", "25/26"],
              bootstrap_B::Int = 10_000)
    ensure_harness_schema!(db)

    # 1. Smoke-before-grid enforcement
    has_passing_smoke(db, candidate) || error(
        "Candidate $(candidate.name) has no passing smoke record in harness_checks for " *
        "recipe_hash $(recipe_hash(candidate)); run --stage smoke first.")

    fit_cfg = fit_config(candidate; stage = :grid, experiment = experiment)
    # The panel comes from the candidate's DataScope, not the W1 default.
    target_seasons = candidate.scope.target_seasons
    expected_fixtures = target_seasons == ["24/25", "25/26"] ? 710 :
                        target_seasons == ["23/24", "24/25", "25/26"] ? 1070 : nothing
    existing_run_id = find_completed_run(db, fit_cfg)

    saved_run_id = nothing
    saved_stride = 1
    fit = nothing
    records = NamedTuple[]
    base = (;
        run_id = nothing,
        recipe_hash = recipe_hash(candidate),
        experiment = String(experiment),
        candidate = candidate.name,
        stage = "grid",
        git_sha = Training.git_commit_id()
    )

    if existing_run_id !== nothing
        @info "Resuming existing completed run for $(candidate.name): $existing_run_id"
        fit = Training.load_fit(db, existing_run_id)
        saved_run_id = existing_run_id
    else
        inputs = _fold_inputs(candidate, ds; stage = :grid)
        try
            # Filtration check
            _run_hard_check!(records, base, "filtration") do
                _filtration_check(candidate, inputs)
            end

            # Sampling with checkpoints
            checkpoint_dir = joinpath("data", "checkpoints", experiment, candidate.name)
            fit = Training.fit_model(fit_cfg;
                feature_sets = inputs.feature_sets,
                oos_fixtures = inputs.oos,
                checkpoint_dir = checkpoint_dir,
                quiet = false)

            # Latents audit
            _run_hard_check!(records, base, "latents") do
                _latent_audit(fit; require_variance = true)
            end

            # Target coverage
            _run_hard_check!(records, base, "target_coverage") do
                _target_coverage(candidate, fit, inputs;
                    expected_folds = length(inputs.boundaries),
                    expected_target = expected_fixtures)
            end

            # Diagnostics: convergence (review), monitor coverage (diagnostic)
            _run_diagnostic!(records, base, "convergence", "review") do
                _convergence_diagnostic(fit)
            end
            _run_diagnostic!(records, base, "monitor_coverage", "diagnostic") do
                _monitor_coverage(candidate, fit, inputs, ds)
            end
            _run_diagnostic!(records, base, "git_telemetry", "info") do
                (; git_sha = Training.git_commit_id(),
                   dirty = endswith(Training.git_commit_id(), "-dirty"))
            end

            # No thinning: a failed save is a real error, not a request to discard draws.
            saved_run_id = Training.save_fit(fit, db; on_duplicate = :error)
            _run_diagnostic!(records, base, "persistence_stride", "info") do
                (; stride = saved_stride)
            end

            # Parity check
            reloaded = Training.load_fit(db, saved_run_id)
            _run_hard_check!(records, base, "fit_parity") do
                _fit_parity(fit, reloaded)
            end

            stamped = NamedTuple[merge(r, (; run_id = saved_run_id)) for r in records]
            write_checks!(db, stamped)
        catch err
            try
                write_checks!(db, records)
            catch
            end
            rethrow(err)
        end
    end

    # Scoring
    tiers = club_season_tiers(ds)
    candidate_ref = RunRef(candidate.name, experiment, saved_run_id, candidate.role)
    refs = RunRef[candidate_ref]

    control_ref = nothing
    if control isa RunRef
        control_ref = control
    elseif control isa AbstractString
        # Check if control is a UUID or a run name
        control_ref = try
            u = UUID(control)
            RunRef(candidate.role == :control ? candidate.name : "control", experiment, u, :control)
        catch
            u = Training.Inference._run_uuid(db, control)
            RunRef(String(control), experiment, u, :control)
        end
    elseif candidate.role === :control
        control_ref = candidate_ref
    end

    if control_ref !== nothing && control_ref.run_id != candidate_ref.run_id
        push!(refs, control_ref)
    end

    # Ensure prototype loader is included if control is from scottish_pyramid_grw_cups
    if control_ref !== nothing && control_ref.experiment == "scottish_pyramid_grw_cups"
        pyramid_loader = joinpath(@__DIR__, "..", "..", "current_development",
                                  "grw_pyramid_cups", "l01_loader.jl")
        if isfile(pyramid_loader)
            Base.include(Main, pyramid_loader)
        end
    end

    scores = Base.invokelatest(score_runs, refs;
        ds = ds, tiers = tiers, control = control_ref,
        target_seasons = target_seasons, expected_fixtures = expected_fixtures,
        bootstrap_B = bootstrap_B)
    write_scores!(db, scores)

    return (; fit, run_id = saved_run_id, records, scores)
end
