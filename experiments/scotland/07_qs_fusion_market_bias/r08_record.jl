# Wave 3 register/report materialisation (wave 2's r08_record.jl). No fits, reference writes or
# score upserts: a read-only audit that the Phase 4 rows of the three new arms equal their
# persisted Phase 3 rows, then this experiment's register row and the canonical boards.
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, ThreadPinning, UUIDs
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" || error("record on beast only")
isdefined(Main, :Wave3Grid) || include(joinpath(@__DIR__, "l05_grid.jl"))
const R8C = Module(:Wave3RecordCandidates)
Base.include(R8C, joinpath(@__DIR__, "candidates.jl"))
r8_db = Training.PostgresStorage(R8C.EXPERIMENT)
r8_out = "/root/BF_runs/qs_experiment_w3_out/phase4"
r8_scores = CSV.read(joinpath(r8_out, "harness_scores_vs_fusion_qs_bias.csv"), DataFrame)
r8_ids = CSV.read(joinpath(@__DIR__, "results", "phase3", "RUNS.csv"), DataFrame)
r8_audit = NamedTuple[]
for r in eachrow(r8_ids)
    saved = Harness.read_scores(r8_db; run_id = UUID(string(r.run_id)))
    fresh = filter(:model => ==(r.label), r8_scores)
    for df in (saved, fresh), col in (:lo, :hi)
        df[!, col] = Float64.(coalesce.(df[!, col], NaN))
    end
    comp = Wave3Grid.control_score_comparison(saved, fresh)
    push!(r8_audit, (; arm = r.label, comp...))
end
CSV.write(joinpath(r8_out, "phase3_phase4_score_parity.csv"), DataFrame(r8_audit))
all(r -> r.passed && r.exact, r8_audit) || error("Phase3/Phase4 score parity failed")
r8_status = get(ENV, "QSX3_RECORD_STATUS", "scored")
r8_status in ("scored", "completed") || error("unknown record status")
if r8_status == "completed"
    for arm in Wave3Grid.ORDER
        repro = CSV.read(joinpath(dirname(r8_out), "phase5", arm, "reproduction.csv"), DataFrame)
        nrow(repro) > 0 && all(repro.identical) && all(repro.max_abs .== 0) ||
            error("reproduction incomplete $arm")
    end
end
leakage = CSV.read(joinpath(@__DIR__, "results", "leakage_audit.csv"), DataFrame)
nrow(leakage) == 120 && all(leakage.active_in_target .== 0) &&
    all(leakage.active_at_or_after_target .== 0) && all(leakage.any_likelihood_target_rows .== 0) ||
    error("leakage audit incomplete")
# Written after the Phase 4 scores were reviewed; it states the prescribed classification only.
r8_decision = "No promotion: primary fusion_qs_bias − market close 1X2 LogLoss −0.000325, noncircular 90% [−0.00707, 0.00113], no detectable difference; bias terms give no detectable gain over fusion_qs_nobias (+0.000235). κ_D > 1 in all 40 folds (QS 1.17, GRW 1.33): the market is more dispersed than the fused latent, not compressed; δ_D and δ_M intervals include 0 in every fold. Leakage audit PASS."
r8_decision *= r8_status == "scored" ? " Frozen-seed reproduction pending." :
    " Frozen-seed reproductions byte-identical."
Harness.write_experiment!(r8_db, (; id = R8C.EXPERIMENT, date = Date(2026, 10, 10), todo = "042",
    question = R8C.QUESTION, dimension = R8C.DIMENSION, status = r8_status, decision = r8_decision,
    run_ids = join(string.(r8_ids.run_id), ","),
    run_commits = Dict(string(r.run_id) => String(r.sampling_git_sha) for r in eachrow(r8_ids)),
    readme = "experiments/scotland/07_qs_fusion_market_bias/README.md"))
CSV.write(joinpath(r8_out, "leaderboard.csv"),
    Harness.leaderboard(filter(:model => x -> x in Wave3Grid.ORDER, r8_scores)))
r8_board = Module(:Wave3CanonicalBoards)
Base.include(r8_board, joinpath(@__DIR__, "..", "..", "..", "scripts", "leaderboard.jl"))
Base.invokelatest(r8_board.main)
println("PHASE4_RECORD_PASS status=$r8_status arms=3 exact_phase3_scores=true")
