# Phase 2 only: does joint daily scenario allocation pass deterministic mathematical checks?
# Synthetic fixtures only: no saved fits, MCMC, cache, SQL or financial backtest.
# The low-coverage T−6h and separately labelled T−24h90/360 books remain unpooled.
# Include in the owned beast persistent Julia REPL, -t16/core-pinned/BLAS1.

# %%
# ===================================================================
# 1. Packages and paired prototype (include once; do not orphan structs)
# ===================================================================
using BayesianFootball, ThreadPinning, LinearAlgebra, Test
pinthreads(:cores)
BLAS.set_num_threads(1)
if !isdefined(Main, :QSBacktestScenario)
    include(joinpath(@__DIR__, "l01_scenario_kelly.jl"))
end

# %%
# ===================================================================
# 2. Fixed verification configuration and files
# ===================================================================
const QSBT1_NATIVE = get(ENV, "QSBT1_NATIVE", "false") == "true"
const QSBT1_OUTPUT = joinpath(@__DIR__, "results", "phase2")
const QSBT1_LOG_DIR = "/root/BF_runs/logs/qs_backtest"
mkpath(QSBT1_LOG_DIR)

# %%
# ===================================================================
# 3. Pure checks and (after graduation) existing portfolio regression
# ===================================================================
open(joinpath(QSBT1_LOG_DIR, "phase2_" * (QSBT1_NATIVE ? "native" : "prototype") * ".log"), "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                Base.run(`python3 $(joinpath(@__DIR__, "check_read_paths.py")) l01_scenario_kelly.jl l01_scenario_kelly_tests.jl r01_scenario_kelly_check.jl`)
                println("NO_DATABASE_ACCESS — no touched tables; runtime DB audit not applicable")
                println("VERIFICATION_CONFIG scenarios=1000 seed=20261010 native=", QSBT1_NATIVE)
                Base.invokelatest(QSBacktestScenario.verify_scenario_allocator, QSBT1_OUTPUT;
                                  native = QSBT1_NATIVE, regression = QSBT1_NATIVE)
            catch e
                println("PHASE2_FAIL type=", typeof(e))
                e isa Test.TestSetException ? println(e) : Base.showerror(io, e)
                println(io)
                rethrow()
            finally
                flush(io)
            end
        end
    end
end
println("Verification complete; inspect phase2 log. No Phase3 execution.")
