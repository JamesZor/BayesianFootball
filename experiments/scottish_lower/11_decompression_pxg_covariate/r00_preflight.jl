# Stage 0: real feature geometry and compiled AD; no sampling or database writes.
# Question: is the direct proxy-xG form design point-in-time, exactly parameterised,
# and allocation-free under a compiled ReverseDiff tape?
# USAGE: julia --project -t 16 experiments/scottish_lower/11_decompression_pxg_covariate/r00_preflight.jl

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, DataFrames, CSV
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l11_decompression_loader.jl"))
const D = DecompressionPXG

# ===================================================================
# 2. Fixed cohort and model recipes
# ===================================================================
const CONFIG = D.DecompressionConfig()
const OUTPUT = joinpath(CONFIG.save_root, "preflight", D.source_fingerprint())
mkpath(OUTPUT)
ds = D.gph_load_data()
splitter = D.gph_splitter(CONFIG.target_seasons)
models = Dict(D.models())
reference = D.candidate_model(optimized = false)

# ===================================================================
# 3. Filtration, feature provenance, and all-arm tape compilation
# ===================================================================
gradients = NamedTuple[]
parity = NamedTuple[]
features = NamedTuple[]
for (name, model) in models
    inputs, filtration = D.selected_inputs(ds, splitter, model, CONFIG; smoke = true)
    CSV.write(joinpath(OUTPUT, name * "_filtration.csv"), filtration)
    for (fold, feature_sets) in zip(CONFIG.smoke_folds, inputs.feature_sets)
        feature_set = first(feature_sets)
        generic = D.gph_gradient_audit(model, feature_sets; replays = 100, seed = 24)
        push!(gradients, (; model = name, fold, engine = "sampling", generic...))
        if name == "m03_negbin_pxg_covariate"
            audit = D.allocation_audit(model, reference, feature_set; seed = 24)
            push!(parity, (; model = name, fold, audit...))
            push!(features, (; model = name, fold, D.feature_audit(feature_sets)...))
        end
        CSV.write(joinpath(OUTPUT, "gradients.csv"), DataFrame(gradients))
        isempty(parity) || CSV.write(joinpath(OUTPUT, "candidate_parity.csv"), DataFrame(parity))
        isempty(features) || CSV.write(joinpath(OUTPUT, "features.csv"), DataFrame(features))
        println("PREFLIGHT ", last(gradients))
    end
end

all(row.allocated_bytes == 0 for row in parity) ||
    error("candidate compiled replay was not allocation-free")
all(row.shot_count_observations == 0 && row.goal_observations == 0 for row in features) ||
    error("candidate feature used a forbidden fallback below live-text commentary")
println("STAGE 0 PASS: exact density/gradient parity and zero-allocation candidate replay")
