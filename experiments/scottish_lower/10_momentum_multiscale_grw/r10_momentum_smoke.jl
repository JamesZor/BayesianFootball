# ==============================================================================
# experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl
# ==============================================================================
#
# Stage 1: Smoke gate for Momentum MultiScale GRW vs Time Decay & 1st-Order GRW
# Runs on folds 1, 20, 40 to verify ReverseDiff compilation, 0 divergences,
# R̂ ≤ 1.05, ESS ≥ 200, score-grid coherency, and momentum parameter identification.
#
# USAGE:
#   julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl
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

println("=== [Stage 1] Momentum MultiScale GRW Smoke Gate ===")
# Pi (GPT-6 Astra): Implement full smoke verification across folds 1, 20, 40.
