# Phase C C0/C1 core production STAGE, not complete C0/C1/H1/H2 acceptance.
# Question: does the pooled measurement change honest full-book forecasts?
# Control C0: Gaussian isolated targets. C1: exact book density, approximate
# Laplace collapse. Double Poisson, full books only, frozen priors/seeds/budgets.
# 10b theta fits only honest_train; FULL chronological pre-week filtering then
# scores honest_test. 10a is retrospective. No own-book forecast mean/Hessian.
# No DB writes, package/cache refresh, thin books or scientific threshold changes.
# A successfully converged synthetic recovery is a REQUIRED saved prerequisite.
# No production sampling starts on file existence alone: recheck draws/intervals.
# Output is a NEW directory; no resume/overwrite of immutable fit artifacts.
# Binary loading requires l05_pooled before deserializing prototype types.
# H1/H2, figures and two complete production reproductions are later stages;
# this runner must never print R05_DONE or PHASEC7_DONE by itself.
# Use a fresh OWNED persistent beast Julia REPL; 16 threads/core pinning/BLAS1.

# %%
# ===================================================================
# 1. Packages and implementations
# ===================================================================
using BayesianFootball, LinearAlgebra, CSV, DataFrames, SHA, TOML, ThreadPinning
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__,"l05_pooled.jl"))
end
if !isdefined(@__MODULE__, :FullBookWorkflow)
    include(joinpath(@__DIR__,"l05_fullbook_workflow.jl"))
end
const C05K = FullBookWorkflow
const C05KP = PooledMarket

# %%
# ===================================================================
# 2. Scientific configuration, protocol priority and frozen seeds
# ===================================================================
const C05K_CONFIG = C05KP.MM.scottish_lower_2425_2526()
const C05K_SEEDS = TOML.parsefile(joinpath(@__DIR__,"fullbook_seeds.toml"))
const C05K_RECOVERY = get(ENV,"C05_RECOVERY_INPUT",joinpath(@__DIR__,"results","C","v7_recovery"))
const C05K_OUT = get(ENV,"C05_CORE_OUT",joinpath(@__DIR__,"results","C","v7_core_1"))
const C05K_PROTOCOLS = ("10b","10a") # honour honest-fit priority on an interrupted run
const C05K_RUNGS = (:C0,:C1)
const C05K_RECOVERY_PANEL_SHA = "de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1"

# %%
# ===================================================================
# 3. Runtime and immutable output preparation
# ===================================================================
strip(read(`hostname`,String)) == "mcmc-beast" || error("posterior sampling requires mcmc-beast")
Threads.nthreads() == 16 || error("production requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
isdir(C05K_OUT) && !isempty(readdir(C05K_OUT)) && error("core output must be fresh: $C05K_OUT")
mkpath(C05K_OUT)

# %%
# ===================================================================
# 4. Pinned data snapshot, unchanged targets and honest filtration
# ===================================================================
c05k_cache = joinpath(pwd(),".cache","datastore_ScottishLower.jls")
open(io -> bytes2hex(SHA.sha256(io)),c05k_cache) ==
    "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" || error("wrong pinned cache")
c05k_ds = BayesianFootball.Data.load_datastore_cached(C05K_CONFIG.segment; max_age_hours=10^6)
c05k_panel = C05KP.CM.TB.phase_b_panel(c05k_ds; config=C05K_CONFIG).panel
c05k_markets = C05KP.fullbook_markets(c05k_ds,c05k_panel,C05K_CONFIG)
c05k_train,c05k_train_markets = C05KP.restrict_fullbook(c05k_panel,c05k_markets,C05K_CONFIG.honest_train)
c05k_train_ids = Set(c05k_train.obs_match)
c05k_test_ids = Set(c05k_panel.matches.match_id[in.(c05k_panel.matches.season,Ref(C05K_CONFIG.honest_test))])
isempty(intersect(c05k_train_ids,c05k_test_ids)) || error("honest train/test IDs overlap")
maximum(c05k_train.matches.match_date) < minimum(c05k_panel.matches.match_date[
    in.(c05k_panel.matches.match_id,Ref(c05k_test_ids))]) || error("honest chronological cut violated")
println("C0/C1 full population=$(length(c05k_markets)); honest train=$(length(c05k_train_markets)), test=$(length(c05k_test_ids)); all forecasts precede every update in their week")

# %%
# ===================================================================
# 5. Full Gate1 regeneration and state-engine gates
# ===================================================================
C05KP.reset_newton_accounting!()
try
    global c05k_gate = C05KP.laplace_gate(c05k_ds,C05K_CONFIG,C05K_OUT; kinds=("full",))
    all(c05k_gate.summary.gate_pass) || error("full-book Gate1 failed; no sampling")
    global c05k_engines = C05KP.fullbook_engine_gates()
    CSV.write(joinpath(C05K_OUT,"state_engine_gates_c.csv"),c05k_engines)
    all(c05k_engines.pass) || error("state engine gates failed; no sampling")
finally
    C05KP.write_newton_accounting(C05K_OUT; run="core_preflight")
end

# %%
# ===================================================================
# 6. Recovery prerequisite, rechecked from all retained chains
# ===================================================================
c05k_recovery_diag = C05K.require_recovery(C05K_RECOVERY;
    generation_seed=C05K_SEEDS["recovery"]["generation"],expected_panel_sha=C05K_RECOVERY_PANEL_SHA)
CSV.write(joinpath(C05K_OUT,"prerequisite_recovery_convergence.csv"),c05k_recovery_diag)

# %%
# ===================================================================
# 7. Frozen C0/C1 training and fail-loud convergence
# ===================================================================
c05k_fits = Dict{Tuple{Symbol,String},C05KP.MID.ArmFit}()
c05k_diagnostics = DataFrame[]
for protocol in C05K_PROTOCOLS
    for rung in C05K_RUNGS
        p,markets = protocol == "10a" ? (c05k_panel,c05k_markets) : (c05k_train,c05k_train_markets)
        a = C05KP.FullBookRung(rung)
        fit = C05K.train_rung(a,p,markets,C05K_OUT;
            protocol,seeds=C05K_SEEDS[String(rung)][protocol])
        c05k_fits[(rung,protocol)] = fit
        push!(c05k_diagnostics,CSV.read(joinpath(C05K_OUT,"convergence_$(rung)_$(protocol).csv"),DataFrame))
        CSV.write(joinpath(C05K_OUT,"convergence_c.csv"),vcat(c05k_diagnostics...))
    end

    # %%
    # ===============================================================
    # 8. Accepted inference, B2-compatible measures and paired C0 score
    # ===============================================================
    C05K.evaluate_fits(c05k_panel,c05k_markets,C05K_CONFIG,c05k_fits,C05K_OUT;
        prediction_seed=C05K_SEEDS["inference"]["prediction"])
    println("C0/C1 $protocol accepted measures saved; not H1/H2 or complete reproduction")
end

# %%
# ===================================================================
# 9. Stage report — no complete-production sentinel
# ===================================================================
println("R05_FULLBOOK_CORE_DONE")
