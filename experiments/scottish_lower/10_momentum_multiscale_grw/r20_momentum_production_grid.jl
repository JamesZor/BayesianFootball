# ==============================================================================
# experiments/scottish_lower/10_momentum_multiscale_grw/r20_momentum_production_grid.jl
# ==============================================================================
#
# Stage 2: 40-Fold Walk-Forward Production Grid on mcmc-beast
# Samples all 40 folds (710 fixtures, 4 chains × 800 warmup + 800 samples)
# for m01 (Time Decay), m02 (1st-Order GRW), and m03 (Momentum GRW).
#
# Persists fits to PostgreSQL mcmc_experiments in namespace:
#   scottish_lower_momentum_grw
#
# USAGE:
#   julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r20_momentum_production_grid.jl
# ==============================================================================

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf

include(joinpath(@__DIR__, "l10_momentum_grw_loader.jl"))

println("=== [Stage 2] Momentum MultiScale GRW 40-Fold Production Grid ===")
# Pi (GPT-6 Astra): Implement full 40-fold walk-forward sampling and persistence.
