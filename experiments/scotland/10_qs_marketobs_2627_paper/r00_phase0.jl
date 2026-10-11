# Phase 0 (TODO 045): 26/27 DataStore and order book (read-only, pinned to files), coverage,
# odds frames and the mid close, then the gates: market-rate inversion (ruling 3), engine
# reproduction (ruling 9), m12 fold inventory (ruling 4) and the walk-forward dry runs.
# No sampling, no DB writes. Execute in the owned beast REPL (jl claude_p27): -t 16, BLAS 1, pinned.

using LinearAlgebra
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
pinthreads(:cores)
@assert Threads.nthreads() == 16

const P27_LOG_DIR = "/root/BF_runs/logs/qs_2627"
const P27_OUT0 = joinpath(@__DIR__, "results", "phase0")
const P27_W4 = joinpath(@__DIR__, "..", "08_qs_marketobs_backtest")
const P27_W5 = joinpath(@__DIR__, "..", "09_qs_marketobs_wave5_config")
mkpath(P27_LOG_DIR)
mkpath(P27_OUT0)

# Include at top level (Julia 1.12 world age): wave-1 market arm, wave-4 and wave-5 modules unchanged.
isdefined(Main, :QSMarketArm) ||
    Base.include(Main, joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "market_arm.jl"))
isdefined(Main, :QSBacktestPhase0) || include(joinpath(P27_W4, "l00_phase0_inventory.jl"))
isdefined(Main, :QSBacktestScenario) || include(joinpath(P27_W4, "l01_scenario_kelly.jl"))
isdefined(Main, :QSBacktestPhase3) || include(joinpath(P27_W4, "l02_phase3_grid.jl"))
isdefined(Main, :QSWave5) || include(joinpath(P27_W5, "l00_wave5.jl"))
include(joinpath(@__DIR__, "l00_p27.jl"))

function p27_phase0(out)
    Q = QSPaper2627
    BF = Q.BF
    DF = Q.DF
    CSV = Q.CSV
    write_csv(name, df) = CSV.write(joinpath(out, name), df)

    # 1. Data: the 26/27 DataStore (built once, pinned) and the 79 fixtures.
    ds, ds_sha = Q.datastore_2627()
    fx, clubs = Q.fixtures_2627(ds)
    write_csv("phase0_fixtures.csv", fx)

    # 2. Order book (pulled once, pinned), entry book at T-25m, odds frames, mid close, coverage.
    ob, ob_sha = Q.orderbook_2627(ds, fx.match_id)
    entry, refused = Q.entry_book(ob.snaps)
    write_csv("phase0_entry_book.csv", entry)
    write_csv("phase0_entry_refusals.csv", refused)
    for price in (:back, :mid, :ltp)
        f = Q.odds_frame(entry, price)
        write_csv("phase0_odds_$(price).csv", f)
        println("ODDS_FRAME price=", price, " rows=", DF.nrow(f), " fixtures=", length(unique(f.match_id)),
                " markets=", DF.nrow(unique(f[:, Q.MKEY])))
    end
    closebook, close_refused = Q.close_book(ob.snaps)
    write_csv("phase0_close_mid.csv", closebook)
    write_csv("phase0_close_refusals.csv", close_refused)
    println("CLOSE_MID rows=", DF.nrow(closebook), " fixtures=", length(unique(closebook.match_id)),
            " 1x2_fixtures=", length(unique(closebook.match_id[closebook.market_name .== "1X2"])))
    cov = Q.coverage_table(fx, ob, entry, closebook)
    write_csv("phase0_coverage.csv", cov)
    show(stdout, MIME"text/plain"(), cov; allrows = true, allcols = true)
    println()

    # 3. Gate, ruling 3: the inversion on the pinned odds_history closes; then the 26/27 rates.
    pinned = Q.pinned_datastore()
    inv_gate, ref = Q.inversion_gate(pinned)
    write_csv("phase0_inversion_gate.csv", inv_gate)
    rates, table = Q.rates_2627(ds, closebook)
    write_csv("phase0_rates_2627_inversion.csv", rates)
    write_csv("phase0_market_rates_2627.csv", table)
    ext = Q.table_with_2627(ref, table)
    write_csv("phase0_market_rates_extended.csv", ext)

    # 4. Gate, ruling 9: the replay code on wave 5's 25/26 T-25m inputs.
    eg = Q.engine_gate(joinpath(P27_W5, "FREEZE.md"))
    write_csv("phase0_engine_gate.csv", eg)

    # 5. qs_marketobs walk-forward dry run (positions 1-40, new positions, filtration; no sampling).
    qsfit = BF.Training.load_fit(BF.Training.PostgresStorage(Q.QS.ns), Q.UUIDs.UUID(Q.QS.run))
    qsmodel = Q.qs_model_2627(qsfit, ref, ext)
    scope = Q.scope_extended(qsfit.config.splitter.scope)
    qsplit = Q.Data.ScopedWalkForwardCV(scope)
    qplan = Q.walkforward_plan("qs_marketobs", qsfit, Q.Data.apply_scope(ds, scope), qsplit,
                               Set(Int.(qsfit.latents.match_ids)), Q.fitted_heldout(qsfit, pinned))
    full = Set(table.match_id[table.full_book])
    qfilt = Q.filtration(qplan, Set(scope.train_tournaments); full_book = full)
    write_csv("phase0_qs_positions.csv", qplan.positions)
    write_csv("phase0_qs_filtration.csv", qfilt)
    qcover = Set(Int(id) for o in qplan.oos for id in o.match_id if Int(id) in Set(fx.match_id))
    println("QS_PLAN new_positions=", join(qplan.delta, ","), " heldout_2627=", length(qcover), "/", Q.N_FIXTURES,
            " sampler=", qsfit.config.sampler, " execution=", qsfit.config.execution)

    # 6. m12 fold inventory (ruling 4) and its in-memory dry run.
    m12fit = BF.Training.load_fit(BF.Training.PostgresStorage(Q.M12.ns), Q.UUIDs.UUID(Q.M12.run))
    mconn = BF.Training.Inference._db_connect(BF.Training.PostgresStorage(Q.M12.ns))
    inv, covered, missing_days = try
        Q.m12_inventory(mconn, m12fit, ds, fx)
    finally
        close(mconn)
    end
    write_csv("phase0_m12_db_folds.csv", inv)
    by_fold = Q.fitted_heldout(m12fit, pinned)
    old = Set(Int.(pinned.matches.match_id[in.(coalesce.(pinned.matches.season, ""), Ref(["24/25", "25/26"])) .&
                                           in.(pinned.matches.tournament_id, Ref([56, 57]))]))
    persisted = intersect(Set(Int.(m12fit.latents.match_ids)), old)
    mplan = Q.walkforward_plan("m12", m12fit, ds, m12fit.config.splitter, persisted,
                               Dict(k => v for (k, v) in by_fold if k <= Q.N_PERSISTED))
    mfilt = Q.filtration(mplan, Set(Q.TOURNAMENTS))
    write_csv("phase0_m12_positions.csv", mplan.positions)
    write_csv("phase0_m12_filtration.csv", mfilt)
    mcover = Set(Int(id) for o in mplan.oos for id in o.match_id if Int(id) in Set(fx.match_id))
    println("M12_PLAN new_positions=", join(mplan.delta, ","), " heldout_2627=", length(mcover), "/", Q.N_FIXTURES,
            " sampler=", m12fit.config.sampler, " execution=", m12fit.config.execution,
            " db_uncovered_days=", join(string.(missing_days), " "))

    println("PINS datastore_sha256=", ds_sha, " orderbook_sha256=", ob_sha)
    println("PHASE0_DONE")
    return nothing
end

open(joinpath(P27_LOG_DIR, "phase0.log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                println("RUNTIME version=", VERSION, " threads=", Threads.nthreads(), " BLAS=", BLAS.get_num_threads(),
                        " package_root=", pkgdir(QSPaper2627.BF), " head=", readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
                Base.invokelatest(QSWave5.with_audit, () -> p27_phase0(P27_OUT0), P27_OUT0, "phase0",
                                  ["../10_qs_marketobs_2627_paper/l00_p27.jl", "../10_qs_marketobs_2627_paper/r00_phase0.jl",
                                   "l00_wave5.jl", "l01_phase0_gate.jl", "l02_phase1_select.jl", "l03_phase2_eval.jl",
                                   "../08_qs_marketobs_backtest/l00_phase0_inventory.jl",
                                   "../08_qs_marketobs_backtest/l01_scenario_kelly.jl",
                                   "../08_qs_marketobs_backtest/l02_phase3_grid.jl"])
            catch err
                # Never print credential-bearing connection exceptions or environment URLs.
                println("ERROR: PHASE0_STOP exception_type=", typeof(err))
                if err isa ErrorException || err isa AssertionError || err isa ArgumentError
                    msg = err.msg
                    occursin("postgres", lowercase(msg)) || println(msg)
                elseif err isa MethodError
                    println("method=", err.f, " argument_types=", map(typeof, err.args))
                end
                for frame in stacktrace(catch_backtrace())[1:min(end, 25)]
                    println(frame)
                end
            finally
                println("PHASE0_END")
                flush(io)
            end
        end
    end
end
println("Phase 0 stopped; inspect /root/BF_runs/logs/qs_2627/phase0.log")
