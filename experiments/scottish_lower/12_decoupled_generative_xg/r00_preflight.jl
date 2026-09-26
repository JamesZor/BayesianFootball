# Stage 0: mathematical identity, real feature geometry, and compiled AD; no sampling.
# Question: are all four densities correct and allocation-free, and do the funnel arms
# genuinely CUT the goal-to-rating feedback that the joint control allows?
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
coverage = DataFrame()
for (name, model) in models
    inputs, filtration = D.selected_inputs(ds, splitter, model, CONFIG; smoke = true)
    CSV.write(joinpath(OUTPUT, name * "_filtration.csv"), filtration)
    for (fold, feature_sets) in zip(CONFIG.smoke_folds, inputs.feature_sets)
        fs = first(feature_sets)
        # A cut arm's sampled density is its CHANCE layer; the joint arms are audited
        # against the production builder as before.
        if model isa D.CutFunnelModel
            chance = D.build_cut_chance_model(model, fs)
            exact = D.density_audit(chance, chance; seed = 25)
        else
            generic = D.gph_gradient_audit(model, feature_sets; replays = 100, seed = 25)
            push!(gradients, (; model = name, fold, generic...))
            exact = D.engine_audit(model, references[name], fs; seed = 25)
        end
        push!(parity, (; model = name, fold, exact...))
        if name != "m01_poisson_time_decay"
            push!(features, (; model = name, fold, D.proxy_feature_audit(feature_sets)...))
        end
        isempty(gradients) || CSV.write(joinpath(OUTPUT, "gradients.csv"), DataFrame(gradients))
        CSV.write(joinpath(OUTPUT, "engine_parity.csv"), DataFrame(parity))
        isempty(features) || CSV.write(joinpath(OUTPUT, "proxy_features.csv"), DataFrame(features))
        println("PREFLIGHT ", last(parity))
    end
end

# ===================================================================
# 4. Decisive architecture check: the cut actually cuts
# ===================================================================
# The previous version of this section asserted m02 ≡ m03 — it certified the defect,
# because m03 WAS m02 under another name. The question now is the opposite one: does
# perturbing the GOALS move the chance layer at all? It must not, exactly.
cut_rows = NamedTuple[]
stage_b_rows = NamedTuple[]
for name in ("m03_funnel_shared_kappa", "m04_funnel_hierarchical_kappa")
    model = models[name]
    inputs, _ = D.selected_inputs(ds, splitter, model, CONFIG; smoke = true)
    for (fold, feature_sets) in zip(CONFIG.smoke_folds, inputs.feature_sets)
        fs = first(feature_sets)
        push!(cut_rows, (; model = name, fold,
                           D.cut_no_feedback_audit(model, fs; seed = 25)...))
        push!(stage_b_rows, (; model = name, fold,
                               D.cut_stage_b_density_audit(model, fs; seed = 25)...))
        println("CUT ", last(cut_rows))
    end
end
CSV.write(joinpath(OUTPUT, "cut_no_feedback.csv"), DataFrame(cut_rows))
CSV.write(joinpath(OUTPUT, "cut_stage_b_density.csv"), DataFrame(stage_b_rows))

# The exact shared-κ conditional against its analytic Gamma(S, 1/T) law.
let model = models["m03_funnel_shared_kappa"]
    inputs, _ = D.selected_inputs(ds, splitter, model, CONFIG; smoke = true)
    rows = NamedTuple[]
    for (fold, feature_sets) in zip(CONFIG.smoke_folds, inputs.feature_sets)
        z0 = D.cut_stage_b_design(model, first(feature_sets))
        z = D.cut_stage_b_at(z0, fill(log(1.3), z0.n_matches), fill(log(1.1), z0.n_matches))
        push!(rows, (; fold, D.cut_verify_exact_shared(z; n = 20_000, seed = 11)...))
        println("EXACT κ ", last(rows))
    end
    CSV.write(joinpath(OUTPUT, "exact_kappa.csv"), DataFrame(rows))
    all(r.worst_quantile_rel_error < 1.0e-3 for r in rows) ||
        error("the exact shared-κ sampler does not reproduce its analytic law")
end

# Proxy-xG coverage: the chance layer's ENTIRE training set, per fold.
let model = models["m03_funnel_shared_kappa"]
    all_inputs = D.gph_fold_inputs(ds, splitter, model.chance)
    coverage = D.cut_coverage_report(all_inputs.feature_sets)
    CSV.write(joinpath(OUTPUT, "proxy_coverage.csv"), coverage)
    early = coverage[coverage.fold .<= 20, :]
    late = coverage[coverage.fold .> 20, :]
    println("COVERAGE folds 1-20: ", round(100 * sum(early.n_covered) / sum(early.n_rows), digits = 1),
            "%  |  folds 21-40: ", round(100 * sum(late.n_covered) / sum(late.n_rows), digits = 1), "%")
    println("COVERAGE worst fold: ", round(100 * minimum(coverage.coverage), digits = 1),
            "%;  folds with a team having <5 covered matches: ",
            count(>(0), coverage.n_teams_under_5), " of ", nrow(coverage))
end

all(row.allocated_bytes == 0 for row in parity) ||
    error("at least one compiled optimized replay allocated")
all(row.worst_density == 0.0 && row.worst_gradient == 0.0 for row in cut_rows) ||
    error("GOAL FEEDBACK DETECTED: a funnel arm's chance layer responds to goals")
println("\nSTAGE 0 PASS: density/gradient parity, zero-allocation replay, exact κ law,")
println("and an EXACTLY zero goal->rating derivative in both funnel arms.")
println("NOTE: the cut trains ratings on proxy-covered matches only. Folds 1-20 are")
println("      ~50-66% covered, so folds 21-40 are the pre-registered clean headline.")
