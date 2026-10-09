# Phase 3 market-arm smoke: three folds (first, season-boundary, last) at the full C0 budget.
# Include in an owned persistent mcmc-beast REPL (-t 16) AFTER r02_phase2_checks.jl.
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
isdefined(Main, :QSMarketArm) || include(joinpath(@__DIR__, "market_arm.jl"))
const MAS = QSMarketArm
const MAS_OUT = joinpath(get(ENV, "QSX_OUT", "/root/BF_runs/qs_experiment_out"), "market_smoke")
strip(read(`hostname`, String)) == "mcmc-beast" && Threads.nthreads() == 16 ||
    error("market fits only on a 16-threaded beast REPL")

s3_ds = MAS.checked_datastore()
s3_market = MAS.prepare_panel(s3_ds, joinpath(dirname(MAS_OUT), "market_panel"))
s3_folds = MAS.goal_folds(s3_ds)
s3_fixtures = Dict(Int(r.match_id) => r for r in eachrow(s3_folds.scoped.matches))
s3_rows = NamedTuple[]
for number in (1, 21, 40)
    fold = s3_folds.folds[number]
    r = MAS.run_fold(s3_market.panel, fold, s3_fixtures, MAS_OUT)
    push!(s3_rows, (; fold = number, seconds = r.seconds, fit_seconds = r.fit.seconds,
        rhat_max = maximum(r.diag.rhat), ess_bulk_min = minimum(r.diag.ess_bulk),
        ess_tail_min = minimum(r.diag.ess_tail), gate = all(r.diag.gate_pass),
        train_fixtures = MAS.MID.n_fixtures(r.training.panel), heldout = length(fold.ids)))
    CSV.write(joinpath(MAS_OUT, "smoke_summary.csv"), DataFrame(s3_rows))
end
println("MARKET_SMOKE mean_fold_s=", round(sum(r.seconds for r in s3_rows) / 3; digits = 1),
        " gates=", [r.gate for r in s3_rows])
println("MARKET_SMOKE_DONE")
