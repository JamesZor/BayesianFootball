# Phase B2 human addendum only: rotate already-converged posterior scales,
# compare smoothed attack/concessions levels, and test common weekly level errors.
# No new fits, hyperparameter changes, or group/tier hierarchy implementation.
# Existing ignored R2/R5/R6/R7 binary fits must be present beside convergence CSV.
# Include this file in a fresh owned beast tmux REPL; marker R03_HIERARCHY_DONE.

# %%
# ===================================================================
# 1. Packages and prototype definitions required for deserialization
# ===================================================================
using BayesianFootball, LinearAlgebra, ThreadPinning, CSV, DataFrames
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l03_covariance.jl"))
const MH = CovarianceMarket

# %%
# ===================================================================
# 2. Configuration and existing accepted inputs (no sampling)
# ===================================================================
const MH_CONFIG = MH.MM.scottish_lower_2425_2526()
const MH_OUT = joinpath(@__DIR__, "results", "B2")
mh_ds = BayesianFootball.Data.load_datastore_cached(MH_CONFIG.segment; max_age_hours=10^6)
mh_panel = MH.TB.phase_b_panel(mh_ds; config=MH_CONFIG).panel
mh_fits = MH.load_covariance_fits(MH_OUT)

# %%
# ===================================================================
# 3. Posterior rotation / season-average RTS levels / week-cluster ICC
# ===================================================================
# Innovation intervals: all retained theta draws. Levels: median-theta RTS means,
# averaged over each season's grid weeks, across its accepted-panel active teams.
# ICC: signed unbalanced ANOVA MOM; singleton/opening weeks retained;
# whole-week percentile 95% bootstrap, 2000 reps, seed 3928.
mh_result = MH.hierarchy_postprocess(mh_panel,mh_fits,MH_OUT)
println("R03_HIERARCHY_DONE")
