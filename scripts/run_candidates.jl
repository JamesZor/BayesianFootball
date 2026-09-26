#!/usr/bin/env julia

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using ThreadPinning
using UUIDs

function parse_args(args)
    candidates_file = nothing
    stage = nothing
    only_names = nothing
    container = :close_option_b

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--stage"
            i + 1 <= length(args) || error("--stage requires an argument (screen|smoke|grid|portfolio)")
            stage = Symbol(args[i + 1])
            i += 2
        elseif startswith(arg, "--stage=")
            stage = Symbol(split(arg, "=", limit = 2)[2])
            i += 1
        elseif arg == "--only"
            i + 1 <= length(args) || error("--only requires comma-separated candidate names")
            only_names = Set(String.(split(args[i + 1], ",")))
            i += 2
        elseif startswith(arg, "--only=")
            only_names = Set(String.(split(split(arg, "=", limit = 2)[2], ",")))
            i += 1
        elseif arg == "--container"
            i + 1 <= length(args) || error("--container requires close_option_b or t25_calibrated")
            container = Symbol(args[i + 1])
            i += 2
        elseif startswith(arg, "--container=")
            container = Symbol(split(arg, "=", limit = 2)[2])
            i += 1
        elseif startswith(arg, "-")
            error("Unknown option: $arg")
        else
            if candidates_file === nothing
                candidates_file = arg
            else
                error("Unexpected multiple candidates file arguments: $candidates_file and $arg")
            end
            i += 1
        end
    end

    candidates_file !== nothing || error("Usage: julia --project -t 16 scripts/run_candidates.jl <candidates.jl> --stage screen|smoke|grid|portfolio [--only name,...] [--container close_option_b|t25_calibrated]")
    stage in (:screen, :smoke, :grid, :portfolio) || error(
        "Stage must be one of: screen, smoke, grid, portfolio; got $stage")
    container in (:close_option_b, :t25_calibrated) || error(
        "Container must be close_option_b or t25_calibrated; got $container")

    return (; candidates_file, stage, only_names, container)
end

function main()
    parsed = parse_args(ARGS)
    candidates_path = abspath(parsed.candidates_file)
    isfile(candidates_path) || error("Candidates file not found: $candidates_path")

    stage = parsed.stage
    if stage in (:smoke, :grid)
        Threads.nthreads() >= 8 || error(
            "run_candidates.jl: stage $stage requires at least 8 threads; got $(Threads.nthreads())")
        try
            pinthreads(:cores)
        catch
        end
        LinearAlgebra.BLAS.set_num_threads(1)
    end

    CandidatesModule = Module(:CandidatesModule)
    Base.include(CandidatesModule, candidates_path)

    isdefined(CandidatesModule, :EXPERIMENT) || error("Candidate file must define EXPERIMENT::String")
    isdefined(CandidatesModule, :CANDIDATES) || error("Candidate file must define CANDIDATES::Vector{Candidate}")

    experiment = String(Base.invokelatest(getproperty, CandidatesModule, :EXPERIMENT))
    all_candidates = Base.invokelatest(getproperty, CandidatesModule, :CANDIDATES)
    candidates = parsed.only_names === nothing ?
        all_candidates :
        filter(c -> c.name in parsed.only_names, all_candidates)
    if parsed.stage === :portfolio && !any(c -> c.role === :control, candidates)
        controls = filter(c -> c.role === :control, all_candidates)
        length(controls) == 1 || error(
            "portfolio stage needs exactly one control in the candidates file; found $(length(controls))")
        push!(candidates, only(controls))
    end

    isempty(candidates) && error("No matching candidates selected to run")

    db = Training.PostgresStorage(experiment)
    Training.ensure_schema!(db)
    Harness.ensure_harness_schema!(db)

    println("Loading DataStore (ScottishPyramid)...")
    ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)

    run_ids = UUID[]

    get_mod_var(sym, default = nothing) = isdefined(CandidatesModule, sym) ?
        Base.invokelatest(getproperty, CandidatesModule, sym) : default

    if stage === :screen
        println("Running screen stage for $(length(candidates)) candidates...")
        control_name = get_mod_var(:CONTROL)
        res = Harness.screen(candidates; ds = ds, experiment = experiment, control = control_name, db = db)
        scores = res.scores

        for c in candidates
            sub_ll = subset(scores,
                :model => ByRow(==(c.name)),
                :subset => ByRow(==("target")),
                :market => ByRow(==("all")),
                :metric => ByRow(==("logloss")))
            ll_str = nrow(sub_ll) > 0 ? @sprintf("%.4f", sub_ll.value[1]) : "n/a"

            sub_sl = subset(scores,
                :model => ByRow(==(c.name)),
                :subset => ByRow(==("target")),
                :market => ByRow(==("1X2")),
                :metric => ByRow(==("compression_slope")))
            sl_str = nrow(sub_sl) > 0 ? @sprintf("%.4f", sub_sl.value[1]) : "n/a"

            println("[SUMMARY] candidate=$(rpad(c.name, 26)) hard=PASS review=- logloss=$ll_str slope=$sl_str run_id=-")
        end
    elseif stage === :smoke
        println("Running smoke stage for $(length(candidates)) candidates...")
        for c in candidates
            res = Harness.smoke(c; ds = ds, experiment = experiment, db = db)
            push!(run_ids, res.run_id)

            hard_pass = all(r -> r.status == "pass", filter(r -> r.severity == "hard", res.records))
            review_pass = all(r -> r.status == "pass", filter(r -> r.severity == "review", res.records))
            hard_str = hard_pass ? "PASS" : "FAIL"
            review_str = review_pass ? "PASS" : "FAIL"

            println("[SUMMARY] candidate=$(rpad(c.name, 26)) hard=$hard_str review=$review_str logloss=n/a slope=n/a run_id=$(res.run_id)")
        end
    elseif stage === :grid
        println("Running grid stage for $(length(candidates)) candidates...")
        control = get_mod_var(:CONTROL)
        for c in candidates
            res = Harness.grid(c; ds = ds, experiment = experiment, db = db, control = control)
            push!(run_ids, res.run_id)
            scores = res.scores

            sub_ll = subset(scores,
                :model => ByRow(==(c.name)),
                :subset => ByRow(==("target")),
                :market => ByRow(==("all")),
                :metric => ByRow(==("logloss")))
            ll_str = nrow(sub_ll) > 0 ? @sprintf("%.4f", sub_ll.value[1]) : "n/a"

            sub_sl = subset(scores,
                :model => ByRow(==(c.name)),
                :subset => ByRow(==("target")),
                :market => ByRow(==("1X2")),
                :metric => ByRow(==("compression_slope")))
            sl_str = nrow(sub_sl) > 0 ? @sprintf("%.4f", sub_sl.value[1]) : "n/a"

            hard_pass = all(r -> r.status == "pass", filter(r -> r.severity == "hard", res.records))
            review_pass = all(r -> r.status == "pass", filter(r -> r.severity == "review", res.records))
            hard_str = hard_pass ? "PASS" : "FAIL"
            review_str = review_pass ? "PASS" : "FAIL"

            println("[SUMMARY] candidate=$(rpad(c.name, 26)) hard=$hard_str review=$review_str logloss=$ll_str slope=$sl_str run_id=$(res.run_id)")
        end
    elseif stage === :portfolio
        println("Running finalist portfolio stage for $(length(candidates)) candidates...")
        refs = Harness.RunRef[]
        for c in candidates
            config = Harness.fit_config(c; stage = :grid, experiment = experiment)
            run_id = Harness.find_completed_run(db, config)
            run_id === nothing && error(
                "Candidate $(c.name) has no completed grid run with this config hash")
            push!(refs, Harness.RunRef(c.name, experiment, run_id, c.role))
            push!(run_ids, run_id)
        end
        summary = Harness.portfolio_runs(refs; ds = ds, container = parsed.container, db = db)
        for row in eachrow(summary)
            println("[SUMMARY] candidate=$(rpad(row.model, 26)) container=$(parsed.container) " *
                    @sprintf("return=%+.4f%% roi=%+.4f%% sharpe=%.4f n_panel=%d run_id=%s",
                             row.total_return_pct, row.roi_pct, row.sharpe_ann,
                             row.n_panel, string(row.run_id)))
        end
    end

    # Upsert experiment row in harness_experiments
    readme_path = isdefined(CandidatesModule, :README) ?
        String(Base.invokelatest(getproperty, CandidatesModule, :README)) :
        joinpath(dirname(candidates_path), "README.md")
    exp_row = (;
        id = experiment,
        date = today(),
        todo = get_mod_var(:TODO, 30),
        question = String(get_mod_var(:QUESTION, "Experiment $experiment")),
        dimension = String(get_mod_var(:DIMENSION, "matrix")),
        status = String(get_mod_var(:STATUS, stage === :screen ? "screened" : "completed")),
        decision = String(get_mod_var(:DECISION, "in_progress")),
        run_ids = join([string(id) for id in run_ids], ","),
        readme = relpath(readme_path, pwd())
    )
    Harness.write_experiment!(db, exp_row)

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
