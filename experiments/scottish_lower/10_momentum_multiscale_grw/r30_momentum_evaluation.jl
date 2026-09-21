# ==============================================================================
# experiments/scottish_lower/10_momentum_multiscale_grw/r30_momentum_evaluation.jl
# ==============================================================================
#
# Stage 3: Unified Evaluation and Portfolio Backtest
# Evaluates proper scores (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS),
# supremacy slope vs Betfair closing line, favourite-tail calibration (≥ 0.70),
# and portfolio backtest under BookSpec(1X2, OU2.5, BakerMcHale) and
# PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25)).
#
# USAGE:
#   julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r30_momentum_evaluation.jl
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

println("=== [Stage 3] Momentum MultiScale GRW Evaluation & Backtest ===")
# Pi (GPT-6 Astra): Implement evaluation vs Betfair closing line and portfolio simulation.
