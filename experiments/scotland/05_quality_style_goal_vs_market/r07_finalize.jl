# Final record only: verify this experiment's persisted score rows, upsert its decision,
# then regenerate the shared leaderboard/register using the existing project workflow.
# Include in an owned mcmc-beast REPL. No sampling and no betdb access.

# %% 1. Packages and completed recipe metadata
using BayesianFootball, CSV, DataFrames, Dates, UUIDs
strip(read(`hostname`, String)) == "mcmc-beast" || error("finalize only on mcmc-beast")
const F7C = Module(:QSFinalCandidates)
Base.include(F7C, joinpath(@__DIR__, "candidates.jl"))
const F7_RUNS = Dict(
    "control_grw" => UUID("a036d22a-ff32-404c-b801-5f928d8a89f4"),
    "control_td" => UUID("1dccb320-526c-4700-9258-134788a636ef"),
    "qs_market_r" => UUID("b18ae74b-9bc1-4cfa-b363-a640131adb2d"),
    "qs_weak_r" => UUID("21f2a9f9-b96f-4034-97de-767704a9d54a"),
)
f7_db = Training.PostgresStorage(F7C.EXPERIMENT)
f7_commits = Dict{String,Any}()
f7_expected = CSV.read(joinpath(@__DIR__, "results", "harness_scores_vs_control_grw.csv"), DataFrame)

# %% 2. Verify persisted v1.2 headline scores and preserve the original fit commits
for c in F7C.CANDIDATES
    id = F7_RUNS[c.name]
    fit = Training.load_fit(f7_db, id)
    fit.config.name == c.name || error("$id has the wrong run name")
    f7_commits[string(id)] = (; candidate = c.name, git_sha = fit.metadata.git_commit)
    actual = Harness.read_scores(f7_db; run_id = id, scorecard_version = "v1.2",
        subset = "target", market = "all", metric = "logloss")
    actual = filter(:stage => ==("grid"), actual)
    expected = filter(r -> r.model == c.name && r.subset == "target" &&
                          r.market == "all" && r.metric == "logloss", f7_expected)
    nrow(actual) == nrow(expected) == 1 || error("$id: missing/ambiguous score rows")
    # Absolute headline scores have a NULL control key; only paired delta rows carry it.
    isequal(actual.control_run_id, expected.control_run_id) || error("$id: control keys differ")
    only(actual.value) == only(expected.value) || error("$id: persisted score differs from CSV")
    println("FINALIZE score_verified arm=$(c.name) run=$id logloss=$(only(actual.value))")
end

# %% 3. Update this experiment's register row, without changing other records
Harness.write_experiment!(f7_db, (;
    id = F7C.EXPERIMENT, date = Date(2026, 10, 9), todo = F7C.TODO,
    question = F7C.QUESTION, dimension = F7C.DIMENSION,
    status = F7C.STATUS, decision = F7C.DECISION,
    run_ids = join(sort!(string.(collect(values(F7_RUNS)))), ","),
    run_commits = f7_commits,
    readme = "experiments/scotland/05_quality_style_goal_vs_market/README.md",
))
f7_record = filter(:id => ==(F7C.EXPERIMENT), Harness.read_experiments(f7_db))
nrow(f7_record) == 1 && only(f7_record.status) == "completed" || error("register update failed")
only(f7_record.todo) == 40 || error("register TODO is incorrect")
println("FINALIZE register_verified id=$(F7C.EXPERIMENT) status=completed todo=040")

# %% 4. Regenerate shared output with the canonical script (isolated names)
const F7_BOARD = Module(:QSFinalLeaderboard)
Base.include(F7_BOARD, normpath(joinpath(@__DIR__, "..", "..", "..", "scripts", "leaderboard.jl")))
Base.invokelatest(F7_BOARD.main)
f7_board = CSV.read(normpath(joinpath(@__DIR__, "..", "..", "LEADERBOARD.csv")), DataFrame)
for id in values(F7_RUNS)
    any(==(string(id)), string.(f7_board.run_id)) || error("$id missing from generated leaderboard")
end
println("FINALIZE leaderboard_verified goal_rows=4 market_c0=file_based_not_a_database_run")
println("FINALIZE_DONE")
