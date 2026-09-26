# Stage 2: matched 40-fold / 710-fixture benchmark, mcmc-beast only.
# Fixed 4 × (800 warmup + 800 retained), acceptance=0.90, max_depth=10.
# A source-matched passing Stage 1 certificate is mandatory. All retained draws
# are persisted under PostgreSQL namespace scottish_lower_decoupled_xg.
# USAGE: julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r20_production_grid.jl
# FUNNEL_PREPARE_ONLY=true validates every fold and recipe without sampling.

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Dates
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l12_loader.jl"))
const D = DecoupledGenerativeXG

# ===================================================================
# 2. Visible configuration and smoke-promotion gate
# ===================================================================
const CONFIG = D.FunnelConfig()
const RUNTIME = D.runtime_config(CONFIG)
const PREPARE_ONLY = parse(Bool, get(ENV, "FUNNEL_PREPARE_ONLY", "false"))
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
end
println("PRODUCTION source=", SOURCE,
        " tasks per arm=40 folds × 4 chains; concurrency=16")

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
    zero_sum_error = D.hierarchical_zero_sum_audit(fit)
    # Stage B is gated per fold: the fit-level audit only ever sees the spliced chain,
    # so an inner conditional that failed to mix would otherwise pass unnoticed.
    stage_b = if model isa D.CutFunnelModel
        gates = [D.cut_stage_b_gate(f.chain) for f in fit.folds]
        bad = findall(g -> !g.passed, gates)
        isempty(bad) || error("$name Stage B failed on folds $bad: $(gates[first(bad)])")
        (; stage_b_pass = true,
           stage_b_worst_rhat = maximum(g.max_rhat for g in gates),
           stage_b_worst_frac = maximum(g.frac_rhat_gt for g in gates),
           stage_b_min_ess = minimum(g.min_ess for g in gates),
           stage_b_divergences = sum(g.divergences for g in gates),
           stage_b_worst_div_frac = maximum(g.div_frac for g in gates),
           stage_b_runs = sum(g.runs for g in gates))
    else
        (; stage_b_pass = true, stage_b_worst_rhat = NaN, stage_b_worst_frac = 0.0,
           stage_b_min_ess = NaN, stage_b_divergences = 0,
           stage_b_worst_div_frac = 0.0, stage_b_runs = 0)
    end
    run_id = existing === nothing ? D.gph_save_and_verify(db, fit) : existing
    convergence_row = D.gph_convergence_row(name, fit, RUNTIME; run_id)
    passed = D.convergence_pass(fit, CONFIG.expected_folds) && stage_b.stage_b_pass
    arm_posterior = D.kappa_posterior(fit, collect(1:CONFIG.expected_folds))
    append!(posterior, [(; model = name, row...) for row in arm_posterior])
    isempty(posterior) || CSV.write(joinpath(OUTPUT, "kappa_posterior.csv"), DataFrame(posterior))
    push!(rows, (; convergence_row..., grid..., zero_sum_error, stage_b..., full_path,
                  gate_pass = passed, source = SOURCE, recipe_hash))
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
    length(rows) == 4 || error("production arm coverage incomplete")
    manifest = (;
        source = SOURCE,
        generated = now(),
        runs = Dict(row.model => row.run_id for row in rows),
    )
    D.Serialization.serialize(joinpath(CONFIG.save_root, "production_manifest.jls"), manifest)
    println("PRODUCTION PASS: ", manifest.runs)
end
