# Phase C revision 7 full-book synthetic recovery prerequisite, NOT production.
# Question: does the approximate C1 collapsed fit recover known q/s/u/n?
# Use the pinned full-book fixture schedule and market line mix, with independently
# generated states, deviations and Dirichlet probabilities on the unchanged grid.
# No thin books, thresholds, cache refresh, database writes or package changes.
# All chains: 2000 warmup + 3000 retained, thin 1. No seed selection after recovery.
# Results: results/C/v7_recovery/ (or a fresh C05_RECOVERY_OUT directory).
# Binary serialization requires l05_pooled.jl BEFORE deserializing prototype types.
# This runner refuses non-beast hosts; include in an owned persistent Julia REPL.
# A failed recovery/convergence gate stops before any production fit/promotion.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, CSV, DataFrames, SHA, TOML, ThreadPinning
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__, "l05_pooled.jl"))
end
const C05R = PooledMarket

# %%
# ===================================================================
# 2. Configuration, model, frozen seeds and fresh output
# ===================================================================
const C05R_CONFIG = C05R.MM.scottish_lower_2425_2526()
const C05R_SEEDS = TOML.parsefile(joinpath(@__DIR__, "fullbook_seeds.toml"))
const C05R_OUT = get(ENV,"C05_RECOVERY_OUT",joinpath(@__DIR__,"results","C","v7_recovery"))
const C05R_TRUTH = log.([0.03,0.01,0.06,1000.0])
const C05R_ARM = C05R.FullBookRung(:C1)
isfile(joinpath(C05R_OUT,"C1_recovery.jls")) && error("immutable recovery fit exists; use a fresh output directory")
mkpath(C05R_OUT)

# %%
# ===================================================================
# 3. Runtime — sampling only on the beast
# ===================================================================
strip(read(`hostname`,String)) == "mcmc-beast" || error("posterior sampling requires mcmc-beast")
Threads.nthreads() == 16 || error("recovery requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)

# %%
# ===================================================================
# 4. Pinned population and book schedule (no data refresh)
# ===================================================================
c05r_cache = joinpath(pwd(),".cache","datastore_ScottishLower.jls")
open(io -> bytes2hex(SHA.sha256(io)),c05r_cache) ==
    "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" || error("wrong pinned cache")
c05r_ds = BayesianFootball.Data.load_datastore_cached(C05R_CONFIG.segment; max_age_hours=10^6)
c05r_panel = C05R.CM.TB.phase_b_panel(c05r_ds; config=C05R_CONFIG).panel
c05r_markets = C05R.fullbook_markets(c05r_ds,c05r_panel,C05R_CONFIG)
println("C1 synthetic population: $(length(c05r_markets)) full fixtures, $(c05r_panel.n_weeks) weeks")

# %%
# ===================================================================
# 5. Deterministic engine gates protect recovery
# ===================================================================
c05r_gates = C05R.fullbook_engine_gates()
CSV.write(joinpath(C05R_OUT,"recovery_engine_gates_c.csv"),c05r_gates)
all(c05r_gates.pass) || error("engine gates failed; no recovery sampling")

# %%
# ===================================================================
# 6. Synthetic generation, prescribed chains, convergence and recovery
# ===================================================================
try
    global c05r_result = C05R.recover_fullbook(c05r_panel,c05r_markets,C05R_OUT;
        generation_seed=C05R_SEEDS["recovery"]["generation"],
        chain_seeds=C05R_SEEDS["recovery"]["chains"],truth=C05R_TRUTH)
finally
    # Canonical latest-run accounting plus the immutable run-local copy written
    # by fit_fullbook, including when any supported numerical failure propagates.
    C05R.write_newton_accounting(joinpath(@__DIR__,"results","C"); run="C1_synthetic")
end

# %%
# ===================================================================
# 7. Evidence, not production acceptance
# ===================================================================
show(stdout,MIME"text/plain"(),CSV.read(joinpath(C05R_OUT,"synthetic_recovery_c.csv"),DataFrame);
    allrows=true,allcols=true)
println()
println("R05_FULLBOOK_RECOVERY_DONE")
# R05_DONE/T05_DONE are reserved for complete production acceptance/reproduction.
