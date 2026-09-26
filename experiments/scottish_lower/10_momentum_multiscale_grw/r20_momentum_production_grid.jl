# Stage 2: matched 40-fold / 710-fixture Poisson benchmark, mcmc-beast only.
# Fixed 4 x (800 warmup + 800 retained), acceptance=0.90, max_depth=10.
# This script refuses a missing, failed or source-stale Stage 1 certificate.
# Definitions must be loaded before reading prototype checkpoint/fit artifacts.
# Registry and configs.config_hash are checked BEFORE expensive sampling.
# User-approved persistence: all retained draws audited and saved locally;
# every fourth draw persisted to PostgreSQL, with latents reconstructed from it.
# USAGE: julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r20_momentum_production_grid.jl
# MMG_PREPARE_ONLY=true validates all fold inputs and recipes without sampling.

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Dates
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l10_momentum_grw_loader.jl"))
const M = MomentumGRW

# ===================================================================
# 2. Visible configuration and smoke-promotion gate
# ===================================================================
const C = M.MomentumGRWConfig()
const R = M.runtime_config(C;smoke=false)
const PREPARE_ONLY = parse(Bool,get(ENV,"MMG_PREPARE_ONLY","false"))
const PERSIST_STRIDE = 4
const SOURCE = M.source_fingerprint()
const OUT = joinpath(C.save_root,"production",SOURCE)
const CERTIFICATE = joinpath(C.save_root,"smoke_gate.jls")
isfile(CERTIFICATE) || error("Stage 1 has not produced a smoke certificate")
certificate = M.Serialization.deserialize(CERTIFICATE)
certificate.passed || error("Stage 1 failed; production is forbidden")
certificate.source==SOURCE || error("Source changed since smoke; rerun Stage 1")
mkpath(OUT)
smoke_db = M.PostgresStorage(C.smoke_experiment)
for (name,_) in M.models()
    haskey(certificate.runs,name) || error("Smoke certificate lacks $name")
    smoke = M.load_fit(smoke_db,M.UUID(certificate.runs[name]))
    M.convergence_pass(smoke,3) || error("Stored smoke fit $name fails convergence")
end
println("PRODUCTION source=",SOURCE," tasks per arm=40 folds x 4 chains; concurrency=16")

# ===================================================================
# 3. Cohort, models and database namespace
# ===================================================================
ds = M.gph_load_data()
splitter = M.gph_splitter(C.target_seasons)
models = M.models()
db = M.gph_database(C.experiment)
rows = NamedTuple[]

for (name,model) in models
    # ===============================================================
    # 4. All-fold feature/filtration preflight and config truth
    # ===============================================================
    inputs,filtration = M.selected_inputs(ds,splitter,model,C;smoke=false)
    CSV.write(joinpath(OUT,name*"_filtration.csv"),filtration)
    base = M.fit_recipe(C,name,model,splitter,R;smoke=false)
    config = M.FitConfig(name=base.name,model=base.model,splitter=base.splitter,
        sampler=base.sampler,execution=base.execution,tags=base.tags,
        description=base.description*"; persistence_stride=4; diagnostics=all retained draws",
        save_dir=base.save_dir)
    M.register_recipe!(db,name,config)
    existing = M.gph_completed_run(db,config)
    recipe_hash = M.gph_run_hash(db,config)
    println("RECIPE ",name," hash=",recipe_hash," existing=",existing)
    PREPARE_ONLY && continue

    # ===============================================================
    # 5. Native queued sampling with recipe-addressed fold checkpoints
    # ===============================================================
    checkpoint = joinpath(OUT,"checkpoints",recipe_hash)
    fit = existing===nothing ? M.gph_sample(config,inputs,R;checkpoint_dir=checkpoint) : M.load_fit(db,existing)
    M.gph_assert_coverage(name,fit;folds=C.expected_folds,oos=C.expected_oos)

    # ===============================================================
    # 6. Full-draw convergence audit, latents, score mass and persistence
    # ===============================================================
    # Keep a full local Fit before attempting a potentially large DB transaction.
    full_path = existing===nothing ? M.save_fit(fit,M.FileStorage(joinpath(OUT,"full_fits"))) : "loaded existing DB run"
    M.gph_latent_audit(fit)
    full_grid = M.score_grid_audit(fit)
    stored = existing===nothing ? M.gph_thin_for_persistence(fit,inputs,PERSIST_STRIDE) : fit
    M.gph_latent_audit(stored)
    grid = M.score_grid_audit(stored)
    run_id = existing===nothing ? M.gph_save_and_verify(db,stored) : existing
    row = M.gph_convergence_row(name,stored,R;run_id)
    passed = M.convergence_pass(stored,C.expected_folds)
    push!(rows,(;row...,grid...,full_grid_worst_tail=existing===nothing ? full_grid.worst_tail : missing,
        full_path,persistence_stride=PERSIST_STRIDE,gate_pass=passed,source=SOURCE,recipe_hash))
    CSV.write(joinpath(OUT,"production_runs.csv"),DataFrame(rows))
    println("PRODUCTION ARM ",last(rows))
    passed || error("$name production convergence failed; evaluation promotion blocked")
end

# ===================================================================
# 7. Immutable run UUID manifest; Stage 3 never fits or regenerates latents
# ===================================================================
if !PREPARE_ONLY
    length(rows)==3 || error("production arm coverage incomplete")
    manifest = (;source=SOURCE,generated=now(),runs=Dict(r.model=>r.run_id for r in rows))
    M.Serialization.serialize(joinpath(C.save_root,"production_manifest.jls"),manifest)
    println("PRODUCTION PASS: ",manifest.runs)
else
    println("PRODUCTION PREPARE_ONLY PASS — no sampling")
end
