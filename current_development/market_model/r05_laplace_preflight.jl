# Phase C necessary-condition preflight ONLY, not the pooled production runner.
# Question: can the brief's literal Gaussian likelihood/projections satisfy its
# exact-Dirichlet quadrature and total-direction requirements before sampling?
# Held fixed: Phase A's gated close, production double Poisson, configuration,
# pinned cache and thresholds. No posterior results or forecast claims.
# CSVs under results/C are replaceable deterministic diagnostic evidence.
# Run through an owned persistent laptop tmux Julia REPL; no packages/DB writes.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, CSV, DataFrames, SHA
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :LaplaceBookPreflight)
    include(joinpath(@__DIR__, "l05_laplace_preflight.jl"))
end
const C_PREFLIGHT = LaplaceBookPreflight

# %%
# ===================================================================
# 2. Configuration and pinned data (no cache refresh)
# ===================================================================
const C_PREFLIGHT_CONFIG = C_PREFLIGHT.MM.scottish_lower_2425_2526()
const C_PREFLIGHT_OUT = joinpath(@__DIR__, "results", "C")
mkpath(C_PREFLIGHT_OUT)
c_preflight_cache = joinpath(pwd(), ".cache", "datastore_ScottishLower.jls")
c_preflight_hash = open(io -> bytes2hex(SHA.sha256(io)), c_preflight_cache)
c_preflight_hash == "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" ||
    error("wrong pinned cache")
c_preflight_ds = BayesianFootball.Data.load_datastore_cached(C_PREFLIGHT_CONFIG.segment;
    max_age_hours=10^6)

# %%
# ===================================================================
# 3. Deterministic necessary conditions, before any sampler is implemented
# ===================================================================
# n = prior median times .25/1/4. These are NOT posterior medians: no posterior
# exists at preflight. Gaussian state predictions are centred at isolated rates,
# with explicit equal-side spreads. Both quadrature orders use exact integrands.
c_preflight = C_PREFLIGHT.preflight(c_preflight_ds, C_PREFLIGHT_CONFIG, C_PREFLIGHT_OUT)
show(stdout, MIME"text/plain"(), c_preflight.summary; allrows=true, allcols=true)
println()
println("Totals projection requirements: ", count(c_preflight.projections.projection_pass),
    "/", nrow(c_preflight.projections), " pass")

# %%
# ===================================================================
# 4. Promotion decision: no fitting after a failed fixed requirement
# ===================================================================
if all(c_preflight.summary.gate_pass) && all(c_preflight.projections.projection_pass)
    println("C_PREFLIGHT05_DONE")
else
    println("C_PREFLIGHT05_BLOCKED")
end
