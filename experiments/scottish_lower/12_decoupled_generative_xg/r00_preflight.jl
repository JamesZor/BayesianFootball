# Stage 0: mathematical identity, real feature geometry, and compiled AD; no sampling.
# Question: are all four densities correct and allocation-free, and is the proposed
# shared-kappa "funnel" actually distinct from the canonical joint control?
# USAGE: julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r00_preflight.jl

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, DataFrames, CSV
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l12_loader.jl"))
const D = DecoupledGenerativeXG

# ===================================================================
# 2. Fixed cohort and explicit four-arm recipes
# ===================================================================
const CONFIG = D.FunnelConfig()
const SOURCE = D.source_fingerprint()
const OUTPUT = joinpath(CONFIG.save_root, "preflight", SOURCE)
mkpath(OUTPUT)
ds = D.gph_load_data()
splitter = D.gph_splitter(CONFIG.target_seasons)
models = Dict(D.models())
references = Dict(D.reference_models())

# ===================================================================
# 3. Filtration, feature provenance, density parity, and compiled AD
# ===================================================================
gradients = NamedTuple[]
parity = NamedTuple[]
features = NamedTuple[]
for (name, model) in models
    inputs, filtration = D.selected_inputs(ds, splitter, model, CONFIG; smoke = true)
    CSV.write(joinpath(OUTPUT, name * "_filtration.csv"), filtration)
    for (fold, feature_sets) in zip(CONFIG.smoke_folds, inputs.feature_sets)
        generic = D.gph_gradient_audit(model, feature_sets; replays = 100, seed = 25)
        push!(gradients, (; model = name, fold, generic...))
        exact = D.engine_audit(model, references[name], first(feature_sets); seed = 25)
        push!(parity, (; model = name, fold, exact...))
        if name != "m01_poisson_time_decay"
            push!(features, (; model = name, fold, D.proxy_feature_audit(feature_sets)...))
        end
        CSV.write(joinpath(OUTPUT, "gradients.csv"), DataFrame(gradients))
        CSV.write(joinpath(OUTPUT, "engine_parity.csv"), DataFrame(parity))
        isempty(features) || CSV.write(joinpath(OUTPUT, "proxy_features.csv"), DataFrame(features))
        println("PREFLIGHT ", last(parity))
    end
end

# ===================================================================
# 4. Decisive architecture check: m02 ≡ m03
# ===================================================================
identity_rows = NamedTuple[]
identity_inputs, _ = D.selected_inputs(
    ds, splitter, models["m02_joint_gamma_poisson"], CONFIG; smoke = true)
for (fold, feature_sets) in zip(CONFIG.smoke_folds, identity_inputs.feature_sets)
    identity = D.shared_identity_audit(
        models["m02_joint_gamma_poisson"],
        models["m03_funnel_shared_kappa"],
        first(feature_sets);
        seed = 25,
    )
    push!(identity_rows, (; fold, identity...))
end
CSV.write(joinpath(OUTPUT, "m02_m03_identity.csv"), DataFrame(identity_rows))

all(row.allocated_bytes == 0 for row in parity) ||
    error("at least one compiled optimized replay allocated")
all(row.worst_density == 0.0 && row.worst_gradient == 0.0 for row in identity_rows) ||
    error("m02 and m03 are not identical under the stated equations")
println("STAGE 0 PASS: all-arm density/gradient parity, zero-allocation replay, and m02 ≡ m03")
println("IMPORTANT: ordinary joint Bayes lets goals update μ; the stated m03 does not cut feedback")
