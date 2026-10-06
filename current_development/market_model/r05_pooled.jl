# Phase C revision 3: can the skewness-corrected posterior mean pass all
# unchanged marginal AND posterior-moment gates before pooled sampling?
# The revision-2 log-marginal and covariance are unchanged; ALL available
# book types are checked before deciding promotion.
# Phase A's isolated KL rates remain the scoring targets, not expansion points.
# Double Poisson for C; later score-grid Phase C2 and Phase D are not authorised.
# Gate artifacts under results/C are deterministic and replaceable. Revision-1
# preflight artifacts remain untouched; revision-2 gate evidence is in v2_gate/. No package updates or database writes.
# At this stage no posterior fits are promoted; a failed gate stops the runner.
# Use include through a fresh owned persistent tmux Julia REPL, pane IDs only.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, CSV, DataFrames, SHA
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__, "l05_pooled.jl"))
end
const C05 = PooledMarket

# %%
# ===================================================================
# 2. Configuration and output
# ===================================================================
const C05_CONFIG = C05.MM.scottish_lower_2425_2526()
const C05_OUT = joinpath(@__DIR__, "results", "C")
mkpath(C05_OUT)

# %%
# ===================================================================
# 3. Pinned data snapshot (no refresh)
# ===================================================================
c05_cache = joinpath(pwd(), ".cache", "datastore_ScottishLower.jls")
c05_hash = open(io -> bytes2hex(SHA.sha256(io)), c05_cache)
c05_hash == "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" ||
    error("wrong pinned cache")
c05_ds = BayesianFootball.Data.load_datastore_cached(C05_CONFIG.segment;
    max_age_hours=10^6)

# %%
# ===================================================================
# 4. Revised book likelihood, exact quadrature and posterior moments
# ===================================================================
# n=250/1000/4000; equal-side prediction SD=.05/.20; first books in ID order.
# Phase A isolated centres where present; fixed-parameter C0 predictions for
# refused totals books. +.10 side-offset cases are retained, with no selection.
# No posterior median n is claimed at this prerequisite gate.
c05_gate = C05.laplace_gate(c05_ds, C05_CONFIG, C05_OUT)
show(stdout, MIME"text/plain"(), c05_gate.summary; allrows=true, allcols=true)
println()

# %%
# ===================================================================
# 5. Promotion gate: no sampling after a failed likelihood approximation
# ===================================================================
if !all(c05_gate.summary.gate_pass)
    println("C05_C3_LIKELIHOOD_BLOCKED")
else
    println("C05_C3_LIKELIHOOD_GATE_DONE")
end
# R05_DONE is reserved for the completed production pipeline, not this gate stage.
