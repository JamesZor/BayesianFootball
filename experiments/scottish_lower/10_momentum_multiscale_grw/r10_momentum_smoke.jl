# Stage 1: mechanical/convergence gate, NOT a predictive-performance result.
# Fixed 4 x (400 warmup + 400 retained), folds 1/20/40. All arms must pass.
# Conditional-mean OOS forecasts; no future outcomes or future innovations.
# PostgreSQL namespace: smoke_scottish_lower_momentum_grw. Checkpoint paths are
# recipe-hashed; completed recipes are loaded rather than resampled. Include l10
# before deserializing prototype fits. Data are cache-backed; cohort is asserted.
# USAGE: julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl
# Set MMG_PREPARE_ONLY=true for features, AD and registry preflight without sampling.

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Dates
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l10_momentum_grw_loader.jl"))
const M = MomentumGRW

# ===================================================================
# 2. Configuration and immutable source identity
# ===================================================================
const C = M.MomentumGRWConfig()
const R = M.runtime_config(C;smoke=true)
const PREPARE_ONLY = parse(Bool,get(ENV,"MMG_PREPARE_ONLY","false"))
const SOURCE = M.source_fingerprint()
const OUT = joinpath(C.save_root,"smoke",SOURCE)
mkpath(OUT)
println("SMOKE source=",SOURCE," folds=",C.smoke_folds," prepare_only=",PREPARE_ONLY)

# ===================================================================
# 3. Data, split and explicit three-arm construction
# ===================================================================
ds = M.gph_load_data()
splitter = M.gph_splitter(C.target_seasons)
models = M.models() # TimeDecay(180), MultiScaleGRW, MomentumMultiScaleGRW; Poisson
references = Dict(M.models(;optimized=false))
db = M.gph_database(C.smoke_experiment)
odds = M.gph_betfair_closing_odds(ds)
rows = NamedTuple[]
gradients = NamedTuple[]
posteriors = NamedTuple[]

for (name,model) in models
    # ===============================================================
    # 4. Filtration, full linked AD parity and zero-allocation replay
    # ===============================================================
    inputs,filtration = M.selected_inputs(ds,splitter,model,C;smoke=true)
    CSV.write(joinpath(OUT,name*"_filtration.csv"),filtration)
    for (fold,fs) in zip(C.smoke_folds,inputs.feature_sets)
        audit = M.allocation_audit(model,references[name],first(fs))
        push!(gradients,(;model=name,fold,audit...))
        println("AD ",last(gradients))
    end
    CSV.write(joinpath(OUT,"gradients.csv"),DataFrame(gradients))

    # ===============================================================
    # 5. Register recipe, deduplicate, then native fold x chain queue
    # ===============================================================
    config = M.fit_recipe(C,name,model,splitter,R;smoke=true)
    M.register_recipe!(db,name,config)
    existing = M.gph_completed_run(db,config)
    println("RECIPE ",name," hash=",M.gph_run_hash(db,config)," existing=",existing)
    PREPARE_ONLY && continue
    checkpoint = joinpath(OUT,"checkpoints",M.gph_run_hash(db,config))
    fit = existing === nothing ? M.gph_sample(config,inputs,R;checkpoint_dir=checkpoint) : M.load_fit(db,existing)

    # ===============================================================
    # 6. Six-part convergence audit, coverage, extraction and score mass
    # ===============================================================
    n_oos = sum(nrow,inputs.oos)
    M.gph_assert_coverage(name,fit;folds=3,oos=n_oos)
    M.gph_latent_audit(fit)
    grid = M.score_grid_audit(fit)
    passed = M.convergence_pass(fit,3)
    append!(posteriors,M.momentum_posterior(fit,C.smoke_folds))

    # ===============================================================
    # 7. Fit round-trip; only converged fits may enter portfolio checks
    # ===============================================================
    run_id = existing === nothing ? M.gph_save_and_verify(db,fit) : existing
    portfolio_id = ""
    if passed
        pid,_ = M.portfolio_roundtrip(db,run_id,fit,ds,odds)
        portfolio_id = string(pid)
    end
    row = M.gph_convergence_row(name,fit,R;run_id)
    push!(rows,(;row...,grid...,gate_pass=passed,portfolio_id,source=SOURCE))
    CSV.write(joinpath(OUT,"smoke_gates.csv"),DataFrame(rows))
    isempty(posteriors) || CSV.write(joinpath(OUT,"momentum_posterior.csv"),DataFrame(posteriors))
    println("SMOKE ARM ",last(rows))
end

# ===================================================================
# 8. Promotion certificate: missing/failed/stale certificate blocks r20
# ===================================================================
if !PREPARE_ONLY
    verdict = length(rows)==3 && all(r.gate_pass for r in rows)
    certificate = (;source=SOURCE,passed=verdict,generated=now(),
        runs=Dict(r.model=>r.run_id for r in rows),report=joinpath(OUT,"smoke_gates.csv"))
    M.Serialization.serialize(joinpath(C.save_root,"smoke_gate.jls"),certificate)
    println("SMOKE_VERDICT ",verdict ? "PASS" : "FAIL"," report=",certificate.report)
    verdict || error("Smoke convergence gate failed; production forbidden")
else
    println("PREPARE_ONLY PASS — no sampling or promotion certificate")
end
