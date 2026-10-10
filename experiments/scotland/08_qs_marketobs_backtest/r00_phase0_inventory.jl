# Phase 0 only: can the exact canonical MatchDay defaults reproduce experiment 07?
# No fitting, policy tuning, database writes, live/replay ledger access, or Phase 1 work.
# Execute in an owned persistent beast REPL: julia --project -t 16 -i

# ===================================================================
# 1. Packages and runtime contract
# ===================================================================
using LinearAlgebra
using ThreadPinning
LinearAlgebra.BLAS.set_num_threads(1)
pinthreads(:cores)
@assert Threads.nthreads() == 16

# ===================================================================
# 2. Fixed paths and read-only database sessions
# ===================================================================
const QSBT0_LOG_DIR = "/root/BF_runs/logs/qs_backtest"
const QSBT0_RESULTS = joinpath(@__DIR__, "results")
ENV["PGOPTIONS"] = "-c default_transaction_read_only=on"
mkpath(QSBT0_LOG_DIR)

# ===================================================================
# 3. Load saved OOS predictions, reproduce, then inventory snapshots
# ===================================================================
open(joinpath(QSBT0_LOG_DIR, "phase0_inventory.log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                include(joinpath(@__DIR__, "l00_phase0_inventory.jl"))
                Base.invokelatest(QSBacktestPhase0.run, QSBT0_RESULTS)
            catch err
                # Do not print credential-bearing connection exceptions or environment URLs.
                println("PHASE0_STOP exception_type=", typeof(err))
                for frame in stacktrace(catch_backtrace())
                    println(frame)
                end
            finally
                flush(io)
            end
        end
    end
end
println("Phase 0 stopped; inspect /root/BF_runs/logs/qs_backtest/phase0_inventory.log")
