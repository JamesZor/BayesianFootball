# Phase 1 (TODO 045): held-out probabilities from the two in-memory extensions, the m12 DB-fold
# check, the held-out XI summary and 1X2 log loss vs the order-book mid close (ruling 8).
# No sampling, no DB writes. Execute in the owned beast REPL (jl claude_p27).

using LinearAlgebra
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
pinthreads(:cores)

const P27_LOG_DIR = "/root/BF_runs/logs/qs_2627"
const P27_OUT1 = joinpath(@__DIR__, "results", "phase1")
const P27_W4 = joinpath(@__DIR__, "..", "08_qs_marketobs_backtest")
const P27_W5 = joinpath(@__DIR__, "..", "09_qs_marketobs_wave5_config")
mkpath(P27_OUT1)

isdefined(Main, :QSMarketArm) ||
    Base.include(Main, joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "market_arm.jl"))
isdefined(Main, :QSBacktestPhase0) || include(joinpath(P27_W4, "l00_phase0_inventory.jl"))
isdefined(Main, :QSBacktestScenario) || include(joinpath(P27_W4, "l01_scenario_kelly.jl"))
isdefined(Main, :QSBacktestPhase3) || include(joinpath(P27_W4, "l02_phase3_grid.jl"))
isdefined(Main, :QSWave5) || include(joinpath(P27_W5, "l00_wave5.jl"))
isdefined(Main, :QSPaper2627) || include(joinpath(@__DIR__, "l00_p27.jl"))
isdefined(Main, :QSPaper2627Sampling) || include(joinpath(@__DIR__, "l01_sampling.jl"))
isdefined(Main, :QSPaper2627Predictions) || include(joinpath(@__DIR__, "l02_predictions.jl"))

function p27_predictions(out)
    S = QSPaper2627Sampling
    R = QSPaper2627Predictions
    CSV, DF = R.CSV, R.DF
    inp = S.inputs()
    probs = Dict{String,Any}()
    for label in ("qs_marketobs", "m12")
        ext = R.load_extension(label)
        fold_of, cut_of = R.fold_maps(ext)
        pr = R.probabilities(ext.latents, label; fold_of, cutoff_of = cut_of)
        DF.nrow(pr) == R.Q.N_FIXTURES && Set(pr.match_id) == Set(inp.fx.match_id) || error("$label: not the 79 fixtures")
        CSV.write(joinpath(out, "phase1_$(label)_probabilities.csv"), pr)
        probs[label] = pr
        if label == "m12"
            j, shapes, summary = R.m12_db_check(ext, pr, inp.fx)
            CSV.write(joinpath(out, "phase1_m12_db_vs_memory.csv"), j)
            CSV.write(joinpath(out, "phase1_m12_chain_shapes.csv"), shapes)
            CSV.write(joinpath(out, "phase1_m12_db_check_summary.csv"), summary)
        end
    end
    xi = CSV.read(joinpath(out, "phase1_m12_heldout_xi.csv"), DF.DataFrame)
    println("M12_HELDOUT_XI fixtures=", DF.nrow(xi), " both_11_starters=", count((xi.home_starters .== 11) .& (xi.away_starters .== 11)),
            " in_lineup_map=", count(xi.in_lineup_map), " min_starters=", minimum(min.(xi.home_starters, xi.away_starters)))
    ll = R.logloss_rows(inp.fx, [(l, probs[l]) for l in ("qs_marketobs", "m12")], inp.closebook)
    CSV.write(joinpath(out, "phase1_logloss_by_fixture.csv"), ll)
    tab = R.logloss_table(ll)
    CSV.write(joinpath(out, "phase1_logloss.csv"), tab)
    show(stdout, MIME"text/plain"(), tab; allrows = true, allcols = true)
    println()
    println("PHASE1_PREDICTIONS_DONE")
    return nothing
end

open(joinpath(P27_LOG_DIR, "phase1_predictions.log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                println("RUNTIME version=", VERSION, " threads=", Threads.nthreads(), " head=",
                        readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
                Base.invokelatest(QSWave5.with_audit, () -> p27_predictions(P27_OUT1), P27_OUT1, "phase1_predictions",
                                  ["../10_qs_marketobs_2627_paper/l00_p27.jl", "../10_qs_marketobs_2627_paper/l01_sampling.jl",
                                   "../10_qs_marketobs_2627_paper/l02_predictions.jl",
                                   "../10_qs_marketobs_2627_paper/r03_phase1_predictions.jl"])
            catch err
                # Never print credential-bearing connection exceptions or environment URLs.
                println("ERROR: PHASE1_PREDICTIONS_STOP exception_type=", typeof(err))
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
                println("PHASE1_PREDICTIONS_END")
                flush(io)
            end
        end
    end
end
println("Phase 1 predictions stopped; inspect /root/BF_runs/logs/qs_2627/phase1_predictions.log")
