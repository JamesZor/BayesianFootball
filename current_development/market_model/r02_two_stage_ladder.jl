# ============================================================================== 
# Phase B only: decompose the close into team strength; compare rate likelihoods.
# This is descriptive market modelling, not Phase C pooled-book inference or betting.
#
# Filtration: R0–R3 exact Gaussian pre-week predictions. R4 prior-Gamma RBPF,
# one fresh shared weight per fixture, exact conditional Kalman ratings/HA per particle.
# 10a uses full-panel θ; 10b fits θ on 24/25 and scores only 25/26.
#
# Persistence: replaceable seeded scientific CSVs in results/B; PNGs in results/figures.
# Binary chains are git-ignored. RUN_PROVENANCE.md records non-deterministic wall times.
# Default runs REFIT, not checkpoint reuse; prototype loader required to read binaries.
# Usage: dedicated beast tmux REPL, julia --project -t 16, include this file.
# ============================================================================== 

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, Statistics, ThreadPinning
using CSV, DataFrames, Serialization
ENV["GKSwstype"] = "100"
using Plots
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l02_two_stage.jl"))
const MB = TwoStageMarket

# %%
# ===================================================================
# 2. Fixed configuration and outputs
# ===================================================================
const MB_CONFIG = MB.MM.scottish_lower_2425_2526()
const MB_SEED = 3902
const MB_PARTICLES = 20_000
const MB_RBPF_SEEDS = (3906,3907,3908)
const MB_OUT = joinpath(@__DIR__,"results","B")
const MB_FIG = joinpath(@__DIR__,"results","figures")
mkpath(MB_OUT)
mkpath(MB_FIG)
mb_started = time()
println("Phase B SHA = ",strip(read(`git rev-parse HEAD`,String)))
println("threads = ",Threads.nthreads(),"; R4 particles = ",MB_PARTICLES,"; prior Gamma proposal")

# %%
# ===================================================================
# 3. Pinned panel; exclusions and data-only EDA
# ===================================================================
mb_ds = BayesianFootball.Data.load_datastore_cached(MB_CONFIG.segment; max_age_hours=10^6)
mb_built = MB.phase_b_panel(mb_ds; config=MB_CONFIG)
mb_panel = mb_built.panel
CSV.write(joinpath(MB_OUT,"market_targets.csv"),mb_panel.matches)
CSV.write(joinpath(MB_OUT,"refusals.csv"),mb_panel.refusals)
mb_book,mb_market_refusals = MB.MM.gated_close(mb_ds, MB_CONFIG)
mb_distribution = MB.rate_eda(mb_panel,mb_built.exclusions,MB_OUT)
mb_goal_calibration = MB.rates_goals(mb_ds,mb_panel,mb_book,MB_OUT)
println(MB.MID.n_fixtures(mb_panel), " fixtures / ", MB.MID.n_obs(mb_panel),
    " observations; ", nrow(mb_built.exclusions), " thin books excluded")

# %%
# ===================================================================
# 4. Engine and AD gates (before any sampling)
# ===================================================================
mb_gates = vcat(MB.engine_gates(),MB.rbpf_gate())
CSV.write(joinpath(MB_OUT,"engine_gates.csv"),mb_gates)
all(mb_gates.pass) || error("Phase B engine gates failed")
mb_ad = MB.family_gradient_gates(MB.family_data(mb_panel))
CSV.write(joinpath(MB_OUT,"family_ad_gates.csv"),mb_ad)
all(mb_ad.pass) || error("Phase B family compiled AD gates failed")

# %%
# ===================================================================
# 5. Ladder training and convergence; all parameters must pass
# ===================================================================
# R0 no team states; R1 zero-step GRW1Break; R2 GRW1; R3 static home-only HA;
# R4 same + shared Student-t mixture. Gaussian: 4×(2000+3000), thin 1.
# R4: 4×(1000+2000×4). Same budgets for honest 24/25-only fits.
mb_fits = MB.train_ladder(mb_panel,MB_OUT; seed=MB_SEED, config=MB_CONFIG)

# %%
# ===================================================================
# 6. Smoothed fit; pre-week 10a/10b; residuals; R4 seed/ESS diagnostics
# ===================================================================
mb_smoothed = MB.ladder_evaluation(mb_panel,mb_fits,MB_OUT; particles=MB_PARTICLES,seeds=MB_RBPF_SEEDS, config=MB_CONFIG)
mb_variance = MB.mean_variance(mb_panel,mb_smoothed,MB_OUT)

# %%
# ===================================================================
# 7. Static Gamma question: full WAIC and 10-fold fixture CV
# ===================================================================
# Fixed fixture seed 3903; sides share a fold; masked likelihood on the full design.
# NUTS 4×(1000+1000), target acceptance .8, compiled ReverseDiff.
# All full/fold fits gated; λ-scale densities/Jacobians make scores comparable.
mb_family_summary = MB.family_comparison(mb_panel,MB_OUT)

# %%
# ===================================================================
# 8. Figures and final evidence
# ===================================================================
MB.phase_b_figures(Plots,mb_panel,mb_distribution,mb_goal_calibration,mb_variance,MB_OUT,MB_FIG)
open(joinpath(MB_OUT,"RUN_PROVENANCE.md"),"a") do io
    println(io,"- Full runner SHA `",strip(read(`git rev-parse HEAD`,String)),"`; wall time ",time()-mb_started," seconds; R02_DONE.")
end
show(stdout,MIME"text/plain"(),mb_family_summary; allrows=true,allcols=true)
println()
println("R02_DONE")
