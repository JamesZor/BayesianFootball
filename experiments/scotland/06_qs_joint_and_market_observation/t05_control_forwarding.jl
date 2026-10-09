# Offline wrapper regression: no datastore, database connection, or MCMC.
using Test, BayesianFootball, CSV, DataFrames, LinearAlgebra, ThreadPinning
pinthreads(:cores); BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l05_grid.jl"))
const TEST_EXPERIMENT = "scottish_lower_qs_wave2_2426"
const TEST_CONTROL = Harness.RunRef("grw_joint", TEST_EXPERIMENT, Wave2Grid.CONTROL_UUID, :control)

@testset "Wave 2 complete owned control preflight (offline)" begin
    mktempdir() do dir
        CSV.write(joinpath(dir, "grid_summary_grw_joint.csv"),
            DataFrame(candidate = ["grw_joint"], final_run_id = [Wave2Grid.CONTROL_UUID]))
        rows = DataFrame(run_id = [Wave2Grid.CONTROL_UUID], name = ["grw_joint"],
            experiment_name = [TEST_EXPERIMENT], status = ["completed"])
        fit = (; folds = [(; fold = i) for i in 1:40])
        ref = Wave2Grid.checked_control(nothing, TEST_EXPERIMENT, dir;
            read_run = (db, id) -> rows, load_fit = (db, id) -> fit)
        @test ref.run_id == TEST_CONTROL.run_id
        @test ref.role === :control
        for (column, value) in ((:experiment_name, "wrong_namespace"), (:status, "running"),
                                (:name, "wrong_control"), (:run_id, Wave2Grid.RESUME_QS_UUID))
            wrong = copy(rows)
            wrong[1, column] = value
            @test_throws ErrorException Wave2Grid.checked_control(nothing, TEST_EXPERIMENT, dir;
                read_run = (db, id) -> wrong, load_fit = (db, id) -> error("must fail before load"))
        end
        @test_throws ErrorException Wave2Grid.checked_control(nothing, TEST_EXPERIMENT, dir;
            read_run = (db, id) -> rows, load_fit = (db, id) -> (; folds = fit.folds[1:39]))
        CSV.write(joinpath(dir, "grid_summary_grw_joint.csv"),
            DataFrame(candidate = ["grw_joint"], final_run_id = [Wave2Grid.RESUME_QS_UUID]))
        @test_throws ErrorException Wave2Grid.checked_control(nothing, TEST_EXPERIMENT, dir;
            read_run = (db, id) -> error("must reject unapproved UUID before DB"))
    end
end

const TEST_SCORES = DataFrame(run_id = [TEST_CONTROL.run_id, TEST_CONTROL.run_id],
    model = ["grw_joint", "grw_joint"], stage = ["grid", "grid"],
    scorecard_version = [Harness.SCORECARD_VERSION, Harness.SCORECARD_VERSION],
    panel = ["56+57|24/25,25/26|n=710", "56+57|24/25,25/26|n=710"],
    subset = ["target", "target"], market = ["1X2", "1X2"], metric = ["logloss", "ece"],
    value = [0.6, 0.01], lo = [NaN, NaN], hi = [NaN, NaN],
    n_obs = [710, 710], n_fixtures = [710, 710], reference = ["frozen", "frozen"],
    control_run_id = [missing, missing])

@testset "Wave 2 control scoring comparison (offline)" begin
    comparison = Wave2Grid.control_score_comparison(TEST_SCORES, TEST_SCORES[2:-1:1, :])
    @test comparison.passed && comparison.exact
    @test comparison.rows == 2 && comparison.cells == 30
    rounded = copy(TEST_SCORES); rounded.value[1] += 5e-13
    @test Wave2Grid.control_score_comparison(TEST_SCORES, rounded).passed
    changed = copy(TEST_SCORES); changed.value[1] += 2e-12
    @test !Wave2Grid.control_score_comparison(TEST_SCORES, changed).passed
    changed = copy(TEST_SCORES); changed.n_obs[1] += 1
    @test !Wave2Grid.control_score_comparison(TEST_SCORES, changed).passed
    changed = copy(TEST_SCORES); changed.metric[1] = "different_metric"
    @test !Wave2Grid.control_score_comparison(TEST_SCORES, changed).passed
    @test_throws ErrorException Wave2Grid.control_score_comparison(TEST_SCORES, TEST_SCORES[1:1, :])
end

@testset "Wave 2 explicit forwarding: all candidate primary/rerun stages (offline)" begin
    for (arm, name) in enumerate(Wave2Grid.ORDER[2:end])
        mktempdir() do dir
            calls = Tuple{String,Any}[]
            fake_fit = (; diagnostics = (; folds = [(; fold = 1, max_rhat = 1.0,
                min_ess_bulk = 500.0, min_ess_tail = 500.0, divergence_rate = 0.0, n_divergent = 0)]))
            function fake_grid(candidate; ds, experiment, db, control)
                push!(calls, (candidate.name, control))
                return (; fit = fake_fit, run_id = Wave2Grid.RESUME_QS_UUID, records = NamedTuple[])
            end
            for attempt in 0:1
                candidate = (; name = name * (attempt == 0 ? "" : "__rhat_rerun"), role = :candidate)
                Wave2Grid.grid_stage(candidate, nothing, nothing, TEST_EXPERIMENT,
                    arm + 1, attempt, dir; control = TEST_CONTROL, grid = fake_grid,
                    read_scores = (db; run_id) -> copy(TEST_SCORES))
                @test calls[end] == (candidate.name, TEST_CONTROL)
                @test isfile(joinpath(dir, "run_$(candidate.name).txt"))
                @test CSV.read(joinpath(dir, "control_score_parity_$(candidate.name).csv"),
                    DataFrame).passed[1]
            end
            @test length(calls) == 2
            candidate = (; name = name * "__missing_control", role = :candidate)
            @test_throws ErrorException Wave2Grid.grid_stage(candidate, nothing, nothing,
                TEST_EXPERIMENT, arm + 1, 0, dir; grid = fake_grid)
            @test length(calls) == 2
            wrong_control = Harness.RunRef("grw_joint", "wrong_namespace", Wave2Grid.CONTROL_UUID, :control)
            @test_throws ErrorException Wave2Grid.grid_stage(candidate, nothing, nothing,
                TEST_EXPERIMENT, arm + 1, 0, dir; control = wrong_control, grid = fake_grid)
            @test length(calls) == 2
            reads = Ref(0)
            changed = copy(TEST_SCORES); changed.value[1] += 0.001
            function changing_scores(db; run_id)
                reads[] += 1
                return reads[] == 1 ? copy(TEST_SCORES) : changed
            end
            candidate = (; name = name * "__score_mismatch", role = :candidate)
            @test_throws ErrorException Wave2Grid.grid_stage(candidate, nothing, nothing,
                TEST_EXPERIMENT, arm + 1, 0, dir; control = TEST_CONTROL, grid = fake_grid,
                read_scores = changing_scores)
            @test !isfile(joinpath(dir, "run_$(candidate.name).txt"))
            @test !CSV.read(joinpath(dir, "control_score_parity_$(candidate.name).csv"),
                DataFrame).passed[1]
        end
    end
end
println("GRID_CONTROL_FORWARDING_PASS: offline mocks only, no sampling")
