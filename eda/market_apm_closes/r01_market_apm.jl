# Phase A: do starting XIs explain inverted closes beyond team strength?
# No player ridge, forecasts, reinversion, or database reads/writes.
# ===================================================================
# 1. Packages and loader
# ===================================================================
using BayesianFootball, LinearAlgebra
BLAS.set_num_threads(1)
if !isdefined(Main, :QualityStyleEDA)
    @eval module QualityStyleEDA
        import BayesianFootball
        struct MarketModelEnglish <: BayesianFootball.Data.DataTournemantSegment end
        BayesianFootball.Data.tournament_ids(::MarketModelEnglish) = [1,2,3,84]
    end
end
include("l01_market_apm.jl")
# ===================================================================
# 2. Configuration and pinned data
# ===================================================================
root = normpath(joinpath(@__DIR__, "../.."))
out = joinpath(@__DIR__, "results/A")
mkpath(out)
# ===================================================================
# 3. Coverage gate (must pass before statistical work)
# ===================================================================
blocked = MarketAPMPhaseA.coverage(root, out)
blocked && error("STOP: >15% join loss; ask user before downstream work. See coverage_gate.csv")
# ===================================================================
# 4. Team-season headroom, design diagnostics, cheap absence tests
# ===================================================================
MarketAPMPhaseA.analyze(root, out)
