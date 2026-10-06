# Phase C revision 5 full-book deterministic preflight, NOT a production pipeline.
# Does the authorised full-book update pass the unchanged likelihood/moment gates?
# Can C0/C1/H1/H2 filter the panel with independent Gaussian engine reductions?
# The revision-2 marginal/covariance and revision-3 corrected mean are unchanged.
# C2 thin-book diagnostics live in t05_c2_pending_tests.jl, excluded from acceptance.
# Phase A's isolated KL rates remain the scoring targets, not expansion points.
# Double Poisson for C; later score-grid Phase C2 and Phase D are not authorised.
# Gate artifacts under results/C/v4_preflight are deterministic and replaceable.
# Revision-1/2/3 evidence remains untouched. No packages or database writes.
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
const C05_OUT = get(ENV,"C05_PREFLIGHT_OUT",joinpath(@__DIR__, "results", "C", "v5_preflight"))
mkpath(C05_OUT)
C05.reset_newton_accounting!()
try

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
# Full-book Phase A isolated centres only. +.10 side-offset cases are retained.
# Thin-book centres and likelihoods are not constructed.
# No posterior median n is claimed at this prerequisite gate.
c05_gate = C05.laplace_gate(c05_ds, C05_CONFIG, C05_OUT; kinds=("full",))
show(stdout, MIME"text/plain"(), c05_gate.summary; allrows=true, allcols=true)
println()

# %%
# ===================================================================
# 5. Exact Gaussian and frozen-factor engine gates
# ===================================================================
all(c05_gate.summary.gate_pass) || error("full-book likelihood gate failed; no sampling")
c05_engines = C05.fullbook_engine_gates()
CSV.write(joinpath(C05_OUT,"state_engine_gates_c.csv"),c05_engines)
all(c05_engines.pass) || error("full-book state engine gate failed; no sampling")

# %%
# ===================================================================
# 6. Fixed-parameter full-panel engine preflight (NOT fitted parameters)
# ===================================================================
c05_panel = C05.CM.TB.phase_b_panel(c05_ds; config=C05_CONFIG).panel
c05_markets = C05.fullbook_markets(c05_ds,c05_panel,C05_CONFIG)
c05_fixed_rows = NamedTuple[]
for (name,theta) in ((:C0,log.([0.07,0.03,0.01])),
                     (:C1,log.([0.03,0.01,0.06,1000.0])),
                     (:H1,log.([0.07,0.03,0.01])),
                     (:H2,vcat(log.([0.07,0.03,0.01]),0.0)))
    a = C05.FullBookRung(name)
    elapsed = @elapsed c05_fixed = C05.fullbook_filter(a,c05_panel,theta;
        markets=name == :C1 ? c05_markets : nothing,store=true,predict=true)
    means,covs = C05.fullbook_smoothing(c05_fixed)
    min_eigenvalue = minimum(minimum(eigvals(Symmetric(covs[:,:,t]))) for t in axes(covs,3))
    push!(c05_fixed_rows,(; rung=String(name),fixtures=length(c05_markets),
        loglik=c05_fixed.loglik,min_smoothed_eigenvalue=min_eigenvalue,
        approximate=name == :C1,parameters="fixed preflight, not posterior"))
    CSV.write(joinpath(C05_OUT,"fixed_parameter_filters.csv"),DataFrame(c05_fixed_rows))
    println("C4 fixed $name: $(c05_fixed.loglik), min smoothed eigenvalue=$min_eigenvalue; $elapsed seconds")
    isfinite(c05_fixed.loglik) && min_eigenvalue >= -1e-10 ||
        error("$name fixed-parameter preflight failed")
end
println("C05_C5_ENGINE_PREFLIGHT_DONE")
finally
    C05.write_newton_accounting(C05_OUT; run="real_preflight")
    C05.write_newton_accounting(joinpath(@__DIR__,"results","C"); run="real_preflight")
end
# Sampling, recovery, convergence, evaluation and production reproduction still
# require implementation. Neither R05_DONE nor PHASEC4_DONE is claimed here.
