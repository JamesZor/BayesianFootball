# Phase 2 (TODO 045): the engine gate again at this head (ruling 9), then the 9 paper-trade cells
# (frozen / fallback / m12 canonical x back / mid / LTP) on the 26/27 days with a T-25m book, the
# per-bet ledger, the fill check (ruling 7), the drawdown watch, slate-clustered intervals and
# PAPER_LEDGER.csv. No sampling, no DB writes. Execute in the owned beast REPL (jl claude_p27).

using LinearAlgebra
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
pinthreads(:cores)

const P27_LOG_DIR = "/root/BF_runs/logs/qs_2627"
const P27_OUT2 = joinpath(@__DIR__, "results", "phase2")
const P27_W4 = joinpath(@__DIR__, "..", "08_qs_marketobs_backtest")
const P27_W5 = joinpath(@__DIR__, "..", "09_qs_marketobs_wave5_config")
mkpath(P27_OUT2)

isdefined(Main, :QSMarketArm) ||
    Base.include(Main, joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "market_arm.jl"))
isdefined(Main, :QSBacktestPhase0) || include(joinpath(P27_W4, "l00_phase0_inventory.jl"))
isdefined(Main, :QSBacktestScenario) || include(joinpath(P27_W4, "l01_scenario_kelly.jl"))
isdefined(Main, :QSBacktestPhase3) || include(joinpath(P27_W4, "l02_phase3_grid.jl"))
isdefined(Main, :QSWave5) || include(joinpath(P27_W5, "l00_wave5.jl"))
isdefined(Main, :QSPaper2627) || include(joinpath(@__DIR__, "l00_p27.jl"))
isdefined(Main, :QSPaper2627Sampling) || include(joinpath(@__DIR__, "l01_sampling.jl"))
isdefined(Main, :QSPaper2627Predictions) || include(joinpath(@__DIR__, "l02_predictions.jl"))
isdefined(Main, :QSPaper2627Paper) || include(joinpath(@__DIR__, "l03_paper.jl"))

function p27_phase2(out)
    Q, S, X = QSPaper2627, QSPaper2627Sampling, QSPaper2627Paper
    CSV, DF = X.CSV, X.DF
    w(name, df) = CSV.write(joinpath(out, name), df)

    # Ruling 9 at this head, before any 26/27 number.
    w("phase2_engine_gate.csv", Q.engine_gate(joinpath(P27_W5, "FREEZE.md")))

    inp = S.inputs()
    entry, odds = X.price_frames(inp)
    qs, m12 = X.model_2627("qs_marketobs"), X.model_2627("m12")
    cells = X.run_cells(qs, m12, odds, entry, inp)
    DF.nrow(cells.summary) == 9 || error("expected 9 cells, got $(DF.nrow(cells.summary))")

    unc = X.uncertainty(cells)
    summary = DF.leftjoin(cells.summary, DF.select(unc, DF.Not([:n_days, :n_bets, :clv_mean_pp])), on = [:price, :cell])
    w("phase2_summary.csv", summary)
    w("phase2_uncertainty.csv", unc)
    w("phase2_paired.csv", X.paired(cells))
    w("phase2_markets.csv", cells.markets)
    w("phase2_daily.csv", cells.daily)
    w("phase2_bets.csv", cells.bets)
    w("phase2_solve_diag.csv", cells.diag)
    w("phase2_books.csv", cells.books)
    w("phase2_bet_sets.csv", X.bet_sets(cells))
    w("phase2_drawdown_watch.csv",
      DF.select(summary, :price, :cell, :final_wealth, :mdd_pct, :mdd_open_pct, :dd_watch_pct, :dd_watch_breached))
    w("phase2_wave5_reference.csv", X.wave5_reference())

    ref = only(filter(r -> r.price == "back" && r.cell == "frozen_B1_l12", cells.summary)).final_wealth
    fill, per_bet = X.fill_check(cells, ref)
    w("phase2_fill.csv", fill)
    ledger = X.paper_ledger(cells, per_bet)
    CSV.write(joinpath(@__DIR__, "PAPER_LEDGER.csv"), ledger)
    println("PAPER_LEDGER rows=", DF.nrow(ledger), " fixtures=", length(unique(ledger.match_id)),
            " days=", length(unique(ledger.date)))

    show(stdout, MIME"text/plain"(), DF.select(summary, :price, :cell, :final_wealth, :wealth_lo95, :wealth_hi95,
                                              :mdd_open_pct, :n_bets, :n_slates, :roi_money_pct, :clv_mean_pp);
         allrows = true, allcols = true)
    println()
    println("PHASE2_DONE")
    return nothing
end

open(joinpath(P27_LOG_DIR, "phase2.log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                println("RUNTIME version=", VERSION, " threads=", Threads.nthreads(), " head=",
                        readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
                Base.invokelatest(QSWave5.with_audit, () -> p27_phase2(P27_OUT2), P27_OUT2, "phase2",
                                  ["../10_qs_marketobs_2627_paper/l00_p27.jl", "../10_qs_marketobs_2627_paper/l01_sampling.jl",
                                   "../10_qs_marketobs_2627_paper/l02_predictions.jl",
                                   "../10_qs_marketobs_2627_paper/l03_paper.jl",
                                   "../10_qs_marketobs_2627_paper/r04_phase2.jl"])
            catch err
                # Never print credential-bearing connection exceptions or environment URLs.
                println("ERROR: PHASE2_STOP exception_type=", typeof(err))
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
                println("PHASE2_END")
                flush(io)
            end
        end
    end
end
println("Phase 2 stopped; inspect /root/BF_runs/logs/qs_2627/phase2.log")
