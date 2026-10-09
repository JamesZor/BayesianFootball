# Register/report materialisation only. No fits, reference writes, or score upserts.
# Read-only audit that Phase4 new-arm rows equal the existing Phase3 score rows;
# update this experiment's register metadata and regenerate the canonical boards.
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, ThreadPinning, UUIDs
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" || error("record on beast only")
isdefined(Main, :Wave2Grid) || include(joinpath(@__DIR__, "l05_grid.jl"))
const R8C = Module(:Wave2RecordCandidates)
Base.include(R8C, joinpath(@__DIR__, "candidates.jl"))
r8_db = Training.PostgresStorage(R8C.EXPERIMENT)
r8_out = "/root/BF_runs/qs_experiment_w2_out/phase4"
r8_scores = CSV.read(joinpath(r8_out, "harness_scores_vs_grw_joint.csv"), DataFrame)
r8_ids = CSV.read(joinpath(@__DIR__, "results", "phase3", "RUNS.csv"), DataFrame)
r8_audit = NamedTuple[]
for r in eachrow(r8_ids)
    saved = Harness.read_scores(r8_db; run_id=UUID(string(r.run_id)))
    fresh = filter(:model => ==(r.label), r8_scores)
    for df in (saved, fresh), col in (:lo, :hi)
        df[!, col] = Float64.(coalesce.(df[!, col], NaN))
    end
    comp = Wave2Grid.control_score_comparison(saved, fresh)
    push!(r8_audit, (; arm=r.label, comp...))
end
CSV.write(joinpath(r8_out, "phase3_phase4_score_parity.csv"), DataFrame(r8_audit))
all(r -> r.passed && r.exact, r8_audit) || error("Phase3/Phase4 score parity failed")
r8_status = get(ENV, "QSX2_RECORD_STATUS", "scored")
r8_status in ("scored", "completed") || error("unknown record status")
if r8_status == "completed"
    for arm in Wave2Grid.ORDER
        repro = CSV.read(joinpath(dirname(r8_out), "phase5", arm, "reproduction.csv"), DataFrame)
        nrow(repro) == 2 && all(repro.identical) && all(repro.max_abs .== 0) ||
            error("reproduction incomplete $arm")
    end
end
r8_decision = "No promotion: primary prescribed noncircular intervals cross zero; circular market-vs-joint intervals favour market arms. Market micro r informed near0.3; GRW ESS review flags retained."
r8_decision *= r8_status == "scored" ? " Frozen-seed reproduction pending." :
    " Four frozen-seed reproductions byte-identical."
Harness.write_experiment!(r8_db, (; id=R8C.EXPERIMENT, date=Date(2026,10,9), todo="041",
    question=R8C.QUESTION, dimension=R8C.DIMENSION, status=r8_status, decision=r8_decision,
    run_ids=join(string.(r8_ids.run_id), ","),
    run_commits=Dict(string(r.run_id) => String(r.sampling_git_sha) for r in eachrow(r8_ids)),
    readme="experiments/scotland/06_qs_joint_and_market_observation/README.md"))
CSV.write(joinpath(r8_out, "leaderboard.csv"),
    Harness.leaderboard(filter(:model => x -> x in Wave2Grid.ORDER, r8_scores)))
r8_board = Module(:Wave2CanonicalBoards)
Base.include(r8_board, joinpath(@__DIR__, "..", "..", "..", "scripts", "leaderboard.jl"))
Base.invokelatest(r8_board.main)
println("PHASE4_RECORD_PASS status=$r8_status arms=4 exact_phase3_scores=true")
