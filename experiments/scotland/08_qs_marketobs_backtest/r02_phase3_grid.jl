# Phase 3 only: gate, then the 5 models x 4 snapshots x 5 policies betting grid.
# No fitting, tuning, database writes or live/replay ledger access. Outputs go to files.
# Execute in an owned persistent beast REPL: julia --project -t 16, BLAS 1, core-pinned.

using LinearAlgebra
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
pinthreads(:cores)
@assert Threads.nthreads() == 16

const QSBT3_LOG_DIR = "/root/BF_runs/logs/qs_backtest"
const QSBT3_RESULTS = joinpath(@__DIR__, "results", "phase3")
const QSBT3_PHASE0 = joinpath(@__DIR__, "results", "phase0_historical")
mkpath(QSBT3_LOG_DIR)

# Include at top level (Julia 1.12 world age); the Phase 3 module reads the other two.
isdefined(Main, :QSBacktestPhase0) || include(joinpath(@__DIR__, "l00_phase0_inventory.jl"))
isdefined(Main, :QSBacktestScenario) || include(joinpath(@__DIR__, "l01_scenario_kelly.jl"))
isdefined(Main, :QSBacktestPhase3) || include(joinpath(@__DIR__, "l02_phase3_grid.jl"))

open(joinpath(QSBT3_LOG_DIR, "phase3_grid.log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                println("RUNTIME version=", VERSION, " threads=", Threads.nthreads(),
                        " BLAS=", BLAS.get_num_threads(), " package_root=", pkgdir(Main.QSBacktestPhase3.BF))
                Base.invokelatest(QSBacktestPhase3.run, QSBT3_RESULTS, QSBT3_PHASE0)
            catch err
                # Never print credential-bearing connection exceptions or environment URLs.
                println("ERROR: PHASE3_STOP exception_type=", typeof(err))
                if err isa ErrorException || err isa AssertionError || err isa ArgumentError
                    msg = err isa AssertionError ? err.msg : err isa ArgumentError ? err.msg : err.msg
                    occursin("postgres", lowercase(msg)) || println(msg)
                elseif err isa MethodError
                    println("method=", err.f, " argument_types=", map(typeof, err.args))
                end
                for frame in stacktrace(catch_backtrace())[1:min(end, 25)]
                    println(frame)
                end
            finally
                println("PHASE3_END")
                flush(io)
            end
        end
    end
end
println("Phase 3 stopped; inspect /root/BF_runs/logs/qs_backtest/phase3_grid.log")
