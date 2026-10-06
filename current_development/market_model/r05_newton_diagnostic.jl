# Revision6 recovery failure diagnostic, NOT a sampler/solver retry or a remedy.
# Trace the root initial coordinate from chain4961 on the unchanged generated panel.
# Inject observation into the same joint_mode; derivatives and decisions are unchanged.
# No data/seed/grid/prior/budget/tolerance changes, database writes or promotion.
# Include only through an owned persistent Julia REPL after recovery has stopped.

# %%
# ===================================================================
# 1. Implementation and prototype deserialization requirements
# ===================================================================
using LinearAlgebra, Serialization, CSV, DataFrames
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__, "l05_pooled.jl"))
end
if !isdefined(@__MODULE__, :FullBookNewtonDiagnostics)
    include(joinpath(@__DIR__, "l05_newton_diagnostics.jl"))
end

# %%
# ===================================================================
# 2. Frozen failure coordinate and binary, separate diagnostic output
# ===================================================================
const C06D_THETA = [-4.3667389598945885,-4.699137880221636,-3.0309405303003616,7.090705824646739]
const C06D_BINARY = get(ENV,"C06D_BINARY",joinpath(@__DIR__,"results","C","v6_recovery","synthetic_panel.jls"))
const C06D_OUT = get(ENV,"C06D_OUT",joinpath(@__DIR__,"results","C","v6_diagnostic"))
c06d_generated = deserialize(C06D_BINARY)

# %%
# ===================================================================
# 3. Observe production Newton iterations; persist failure, never a fit
# ===================================================================
c06d_summary = FullBookNewtonDiagnostics.trace_fullbook_mode(c06d_generated,C06D_THETA,C06D_OUT)
show(stdout,MIME"text/plain"(),c06d_summary; allrows=true,allcols=true)
println()
println("C06_NEWTON_DIAGNOSTIC_REPORTED")
