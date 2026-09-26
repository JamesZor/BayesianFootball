# Stage 0: real feature geometry and compiled AD, no sampling or DB writes.
# 1. Packages and runtime
using ThreadPinning, LinearAlgebra, DataFrames, CSV
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l10_momentum_grw_loader.jl"))
const M = MomentumGRW

# 2. Cohort and models (pure Poisson; macro priors unchanged)
const C = M.MomentumGRWConfig()
ds = M.gph_load_data()
splitter = M.gph_splitter(C.target_seasons)
rows = NamedTuple[]
references = Dict(M.models(; optimized=false))
mkpath(joinpath(C.save_root, "preflight"))

# 3. Filtration and AD: no-target, late first season, late second season
for (name, model) in M.models()
    all_inputs = M.gph_fold_inputs(ds, splitter, model)
    @assert length(all_inputs.boundaries) == C.expected_folds
    inputs = (; boundaries=all_inputs.boundaries[C.smoke_folds],
                feature_sets=all_inputs.feature_sets[C.smoke_folds],
                oos=all_inputs.oos[C.smoke_folds])
    filtration = M.gph_filtration_report(ds, inputs)
    @assert all(filtration.ordered)
    filtration.fold = C.smoke_folds
    CSV.write(joinpath(C.save_root, "preflight", name * "_filtration.csv"), filtration)
    for (fold, fs) in zip(C.smoke_folds, inputs.feature_sets)
        println("AUDITING ", name, " fold=", fold)
        audit = M.allocation_audit(model, references[name], first(fs); seed=22)
        push!(rows, (; model=name, fold, audit...))
        println(last(rows))
        CSV.write(joinpath(C.save_root, "preflight", "gradients.csv"), DataFrame(rows))
    end
end
println("PREFLIGHT PASS: zero replay allocations; linked-space density and gradient parity")
