# Diagnose the failed toy likelihood assertion before changing any production/test code.
# Compare the returned-only submodel with the production accumulation seam, in both spaces.
# No fits, SQL writes, source fixes or acceptance-threshold changes.

# %% 1. Packages and diagnostic implementation
using LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l02_likelihood_audit.jl"))

# %% 2. Fixed sigma and original toy panel; separate density components
w2_audit_rows = Wave2LikelihoodAudit.audit("/root/BF_runs/qs_experiment_w2_out/phase1")
println("PHASE1_LIKELIHOOD_DIAGNOSIS_DONE")
