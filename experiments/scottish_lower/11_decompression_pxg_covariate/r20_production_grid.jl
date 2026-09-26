# Stage 2: matched 40-fold / 710-fixture benchmark, mcmc-beast only.
# Fixed 4 x (800 warmup + 800 retained), acceptance=0.90, max_depth=10.
# A passing source-matched Stage 1 certificate is mandatory. All retained draws
# are audited and persisted to PostgreSQL namespace scottish_lower_decompression.
# USAGE: julia --project -t 16 experiments/scottish_lower/11_decompression_pxg_covariate/r20_production_grid.jl
# PXG_PREPARE_ONLY=true validates every fold and recipe without sampling.

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Dates
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l11_decompression_loader.jl"))
const D = DecompressionPXG

# ===================================================================
# 2. Visible configuration and smoke-promotion gate
# ===================================================================
const CONFIG = D.DecompressionConfig()
const RUNTIME = D.runtime_config(CONFIG; smoke = false)
const PREPARE_ONLY = parse(Bool, get(ENV, "PXG_PREPARE_ONLY", "false"))
const SOURCE = D.source_fingerprint()
const OUTPUT = joinpath(CONFIG.save_root, "production", SOURCE)
const CERTIFICATE = joinpath(CONFIG.save_root, "smoke_gate.jls")
isfile(CERTIFICATE) || error("Stage 1 has not produced a smoke certificate")
certificate = D.Serialization.deserialize(CERTIFICATE)
certificate.passed || error("Stage 1 failed; production is forbidden")
certificate.source == SOURCE || error("Source changed since smoke; rerun Stage 1")
mkpath(OUTPUT)
smoke_db = D.PostgresStorage(CONFIG.smoke_experiment)
for (name, _) in D.models()
    haskey(certificate.runs, name) || error("Smoke certificate lacks $name")
    smoke = D.load_fit(smoke_db, D.UUID(certificate.runs[name]))
    D.convergence_pass(smoke, 3) || error("Stored smoke fit $name fails convergence")
    if name == "m03_negbin_pxg_covariate"
        D.pxg_identification_pass(D.pxg_posterior(smoke, CONFIG.smoke_folds)) ||
            error("Stored candidate smoke fit fails the w_pxg identification gate")
    end
end
println("PRODUCTION source=", SOURCE,
        " tasks per arm=40 folds x 4 chains; concurrency=16")

# ===================================================================
# 3. Cohort, models, and experiment database namespace
# ===================================================================
ds = D.gph_load_data()
splitter = D.gph_splitter(CONFIG.target_seasons)
models = D.models()
db = D.gph_database(CONFIG.experiment)
rows = NamedTuple[]
posterior = NamedTuple[]

for (name, model) in models
    # ===============================================================
    # 4. All-fold feature/filtration preflight and config truth
    # ===============================================================
    inputs, filtration = D.selected_inputs(ds, splitter, model, CONFIG; smoke = false)
    CSV.write(joinpath(OUTPUT, name * "_filtration.csv"), filtration)
    config = D.fit_recipe(CONFIG, name, model, splitter, RUNTIME; smoke = false)
    D.register_recipe!(db, name, config)
    existing = D.gph_completed_run(db, config)
    recipe_hash = D.gph_run_hash(db, config)
    println("RECIPE ", name, " hash=", recipe_hash, " existing=", existing)
    PREPARE_ONLY && continue

    # ===============================================================
    # 5. Native queued sampling with recipe-addressed checkpoints
    # ===============================================================
    checkpoint = joinpath(OUTPUT, "checkpoints", recipe_hash)
    fit = existing === nothing ?
        D.gph_sample(config, inputs, RUNTIME; checkpoint_dir = checkpoint) :
        D.load_fit(db, existing)
    D.gph_assert_coverage(name, fit; folds = CONFIG.expected_folds, oos = CONFIG.expected_oos)

    # ===============================================================
    # 6. Full convergence, latent/grid audit, and exact persistence
    # ===============================================================
    full_path = existing === nothing ?
        D.save_fit(fit, D.FileStorage(joinpath(OUTPUT, "full_fits"))) :
        "loaded existing database run"
    D.gph_latent_audit(fit)
    grid = D.score_grid_audit(fit)
    run_id = existing === nothing ? D.gph_save_and_verify(db, fit) : existing
    convergence_row = D.gph_convergence_row(name, fit, RUNTIME; run_id)
    passed = D.convergence_pass(fit, CONFIG.expected_folds)
    if name == "m03_negbin_pxg_covariate"
        arm_posterior = D.pxg_posterior(fit, collect(1:CONFIG.expected_folds))
        append!(posterior, [(; model = name, row...) for row in arm_posterior])
        CSV.write(joinpath(OUTPUT, "pxg_posterior.csv"), DataFrame(posterior))
    end
    push!(rows, (; convergence_row..., grid..., full_path, gate_pass = passed,
                  source = SOURCE, recipe_hash))
    CSV.write(joinpath(OUTPUT, "production_runs.csv"), DataFrame(rows))
    println("PRODUCTION ARM ", last(rows))
    passed || error("$name production convergence failed; evaluation promotion blocked")
end

# ===================================================================
# 7. Immutable run UUID manifest; Stage 3 never fits or regenerates latents
# ===================================================================
if PREPARE_ONLY
    println("PRODUCTION PREPARE_ONLY PASS — no sampling")
else
    length(rows) == 3 || error("production arm coverage incomplete")
    manifest = (;
        source = SOURCE,
        generated = now(),
        runs = Dict(row.model => row.run_id for row in rows),
    )
    D.Serialization.serialize(joinpath(CONFIG.save_root, "production_manifest.jls"), manifest)
    println("PRODUCTION PASS: ", manifest.runs)
end
