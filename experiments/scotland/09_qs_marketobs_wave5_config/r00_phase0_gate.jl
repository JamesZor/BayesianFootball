# Phase 0 only: reproduce the two wave-4 T-25m cells, per-season MDDs, half/basket/seal checks.
# No fitting, tuning, database writes or live/replay ledger access. Outputs go to files.
# Execute in an owned persistent beast REPL: julia --project -t 16, BLAS 1, core-pinned.

using LinearAlgebra
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
pinthreads(:cores)
@assert Threads.nthreads() == 16

const W5_LOG_DIR = "/root/BF_runs/logs/qs_wave5"
const W5_OUT0 = joinpath(@__DIR__, "results", "phase0")
const W4_SRC = joinpath(@__DIR__, "..", "08_qs_marketobs_backtest")
mkpath(W5_LOG_DIR)

# Include at top level (Julia 1.12 world age); wave-4 modules unchanged, then wave 5.
isdefined(Main, :QSBacktestPhase0) || include(joinpath(W4_SRC, "l00_phase0_inventory.jl"))
isdefined(Main, :QSBacktestScenario) || include(joinpath(W4_SRC, "l01_scenario_kelly.jl"))
isdefined(Main, :QSBacktestPhase3) || include(joinpath(W4_SRC, "l02_phase3_grid.jl"))
isdefined(Main, :QSWave5) || include(joinpath(@__DIR__, "l00_wave5.jl"))

open(joinpath(W5_LOG_DIR, "phase0.log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                println("RUNTIME version=", VERSION, " threads=", Threads.nthreads(),
                        " BLAS=", BLAS.get_num_threads(), " package_root=", pkgdir(QSWave5.BF),
                        " head=", readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
                Base.invokelatest(QSWave5.with_audit, () -> QSWave5.phase0(W5_OUT0), W5_OUT0, "phase0",
                                  ["l00_wave5.jl", "l01_phase0_gate.jl", "r00_phase0_gate.jl",
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
println("Phase 0 stopped; inspect /root/BF_runs/logs/qs_wave5/phase0.log")
