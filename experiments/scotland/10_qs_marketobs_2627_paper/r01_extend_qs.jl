# Phase 1 (TODO 045): in-memory walk-forward sampling of positions 41-44 for qs (manager-approved job).
# The fit's own model, sampler and seed rule (l01_sampling.jl SEED_RULES); one seeded rerun for R-hat > 1.05.
# Nothing persisted to any database: out/ext_<label>_2627.jls on the beast plus small CSVs.
# Execute in the owned beast REPL (jl claude_p27): -t 16, BLAS 1, pinned; nothing else heavy running.

using LinearAlgebra
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
pinthreads(:cores)
@assert Threads.nthreads() == 16

const P27_LOG_DIR = "/root/BF_runs/logs/qs_2627"
const P27_OUT1 = joinpath(@__DIR__, "results", "phase1")
const P27_W4 = joinpath(@__DIR__, "..", "08_qs_marketobs_backtest")
const P27_W5 = joinpath(@__DIR__, "..", "09_qs_marketobs_wave5_config")
mkpath(P27_LOG_DIR)
mkpath(P27_OUT1)

isdefined(Main, :QSMarketArm) ||
    Base.include(Main, joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "market_arm.jl"))
isdefined(Main, :QSBacktestPhase0) || include(joinpath(P27_W4, "l00_phase0_inventory.jl"))
isdefined(Main, :QSBacktestScenario) || include(joinpath(P27_W4, "l01_scenario_kelly.jl"))
isdefined(Main, :QSBacktestPhase3) || include(joinpath(P27_W4, "l02_phase3_grid.jl"))
isdefined(Main, :QSWave5) || include(joinpath(P27_W5, "l00_wave5.jl"))
isdefined(Main, :QSPaper2627) || include(joinpath(@__DIR__, "l00_p27.jl"))
isdefined(Main, :QSPaper2627Sampling) || include(joinpath(@__DIR__, "l01_sampling.jl"))

function p27_qs_extension(out)
    S = QSPaper2627Sampling
    inp = S.inputs()
    setup = S.qs_setup(inp)
    println("PLAN model=", setup.label, " positions=", join(setup.plan.delta, ","), " sampler=", setup.sampler,
            " seed_rule=", S.SEED_RULES[setup.label].source, " uptime=", readchomp(`uptime`))
    S.run_extension(setup, inp.ds, out)
    println("EXTEND_QS_DONE")
    return nothing
end

open(joinpath(P27_LOG_DIR, "extend_qs.log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                println("RUNTIME version=", VERSION, " threads=", Threads.nthreads(), " BLAS=", BLAS.get_num_threads(),
                        " package_root=", pkgdir(QSPaper2627.BF), " head=", readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
                Base.invokelatest(QSWave5.with_audit, () -> p27_qs_extension(P27_OUT1), P27_OUT1, "extend_qs",
                                  ["../10_qs_marketobs_2627_paper/l00_p27.jl", "../10_qs_marketobs_2627_paper/l01_sampling.jl",
                                   "../10_qs_marketobs_2627_paper/r01_extend_qs.jl"])
            catch err
                # Never print credential-bearing connection exceptions or environment URLs.
                println("ERROR: EXTEND_QS_STOP exception_type=", typeof(err))
                if err isa ErrorException || err isa AssertionError || err isa ArgumentError
                    msg = err.msg
                    occursin("postgres", lowercase(msg)) || println(msg)
                elseif err isa MethodError
                    println("method=", err.f, " argument_types=", map(typeof, err.args))
                elseif err isa CompositeException
                    for e in err.exceptions
                        println("task_exception=", typeof(e))
                    end
                end
                for frame in stacktrace(catch_backtrace())[1:min(end, 25)]
                    println(frame)
                end
            finally
                println("EXTEND_QS_END")
                flush(io)
            end
        end
    end
end
println("qs extension stopped; inspect /root/BF_runs/logs/qs_2627/extend_qs.log")
