# Phase B2 only: fixture shocks versus persistent team-quality innovations.
# Control: refitted R2; candidates R5, R6, R7. No R3/R4, pooled fit or betting.
# Filtration: every fixture is predicted BEFORE any observation of its week.
# 10a fits theta retrospectively; 10b fits only configured honest_train seasons,
# then sequentially filters the full panel and scores only honest_test seasons.
# Persistence: seeded scientific CSVs replace results/B2; binary chains are ignored.
# Wall times and timestamps are confined to RUN_PROVENANCE.md. Default always REFITS.
# Usage: fresh own beast tmux REPL, --project -t 16 (8 during manager rerun).
# Include this file top to bottom; completion marker is R03_DONE.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, ThreadPinning, CSV, DataFrames
ENV["GKSwstype"] = "100"
using Plots
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l03_covariance.jl"))
const MC = CovarianceMarket

# %%
# ===================================================================
# 2. Configuration, runtime and output directories
# ===================================================================
const MC_CONFIG = MC.MM.scottish_lower_2425_2526()
const MC_SEED = 3922
const MC_OUT = joinpath(@__DIR__, "results", "B2")
const MC_FIG = joinpath(@__DIR__, "results", "figures")
mkpath(MC_OUT)
mkpath(MC_FIG)
mc_started = time()
println("Phase B2 SHA = ",strip(read(`git rev-parse HEAD`,String)))
println("threads = ",Threads.nthreads(),"; config = ",MC_CONFIG.name)

# %%
# ===================================================================
# 3. Pinned data and panel: no SQL, inversion changes or database writes
# ===================================================================
mc_ds = BayesianFootball.Data.load_datastore_cached(MC_CONFIG.segment; max_age_hours=10^6)
mc_panel = MC.TB.phase_b_panel(mc_ds; config=MC_CONFIG).panel
println("Fixtures = ",MC.MID.n_fixtures(mc_panel),"; season boundaries = ",MC.TB.season_break_weeks(mc_panel,MC_CONFIG))
println("10b train/test = ",MC_CONFIG.honest_train," / ",MC_CONFIG.honest_test)

# %%
# ===================================================================
# 4. Engine gates: reductions, independent batch Gaussian and FFBS moments
# ===================================================================
mc_gates = MC.engine_gates_b2()
CSV.write(joinpath(MC_OUT,"engine_gates_b2.csv"),mc_gates)
all(mc_gates.pass) || error("Phase B2 engine gates failed")

# %%
# ===================================================================
# 5. Training and convergence: 4 × (2000 + 3000), thin 1, both protocols
# ===================================================================
# R5: axis HalfNormal(.20) noise; independent weekly HalfNormal(.10) steps.
# R6: R2 noise; correlated attack/defence steps, Uniform(-1,1) rho on atanh.
# R7: both. Initial states, static league mu/gamma and summer evolution as R2.
mc_fits = MC.train_covariance(mc_panel,MC_CONFIG,MC_OUT; seed=MC_SEED)

# %%
# ===================================================================
# 6. Plug-in smoothed fit, pre-week evaluation and paired residual diagnostics
# ===================================================================
# Linear axes exact Gaussian. Nonlinear log-total: 4000 seeded draws / KDE.
# Team persistence CI: 2000 whole-team bootstrap replicates, seed 3924.
mc_evaluation = MC.evaluate_covariance(mc_panel,MC_CONFIG,mc_fits,MC_OUT)
# Human addendum: existing draws/RTS levels only; no hierarchy model or new fit.
mc_hierarchy = MC.hierarchy_postprocess(mc_panel,mc_fits,MC_OUT)

# %%
# ===================================================================
# 7. Figures and final provenance
# ===================================================================
MC.covariance_figures(Plots,mc_evaluation,MC_FIG)
open(joinpath(MC_OUT,"RUN_PROVENANCE.md"),"a") do io
    println(io,"- Full runner SHA `",strip(read(`git rev-parse HEAD`,String)),"`; wall time ",time()-mc_started," seconds; R03_DONE.")
end
println("R03_DONE")
