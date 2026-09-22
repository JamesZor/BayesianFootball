# Stage 1: mechanical/convergence gate, not a predictive-performance result.
# Fixed 4 x (400 warmup + 400 retained), folds 1/20/40, all three arms.
# PostgreSQL namespace: smoke_scottish_lower_decompression. Recipe-addressed
# checkpoints are resumable; completed recipes are loaded instead of resampled.
# USAGE: julia --project -t 16 experiments/scottish_lower/11_decompression_pxg_covariate/r10_smoke.jl
# Set PXG_PREPARE_ONLY=true for filtration, AD, and registry preflight only.

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Dates, Statistics
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l11_decompression_loader.jl"))
const D = DecompressionPXG

# ===================================================================
# 2. Visible configuration and immutable source identity
# ===================================================================
const CONFIG = D.DecompressionConfig()
const RUNTIME = D.runtime_config(CONFIG; smoke = true)
const PREPARE_ONLY = parse(Bool, get(ENV, "PXG_PREPARE_ONLY", "false"))
const SOURCE = D.source_fingerprint()
const OUTPUT = joinpath(CONFIG.save_root, "smoke", SOURCE)
mkpath(OUTPUT)
println("SMOKE source=", SOURCE, " folds=", CONFIG.smoke_folds,
        " prepare_only=", PREPARE_ONLY)

# ===================================================================
# 3. Data, split, closing book, and explicit three-arm construction
# ===================================================================
ds = D.gph_load_data()
splitter = D.gph_splitter(CONFIG.target_seasons)
models = D.models()
reference_candidate = D.candidate_model(optimized = false)
db = D.gph_database(CONFIG.smoke_experiment)
odds = D.gph_betfair_closing_odds(ds)
rows = NamedTuple[]
gradients = NamedTuple[]
parity = NamedTuple[]
posterior = NamedTuple[]
fits = Dict{String,Any}()

for (name, model) in models
    # ===============================================================
    # 4. Filtration, feature audit, and compiled AD gates
    # ===============================================================
    inputs, filtration = D.selected_inputs(ds, splitter, model, CONFIG; smoke = true)
    CSV.write(joinpath(OUTPUT, name * "_filtration.csv"), filtration)
    for (fold, feature_sets) in zip(CONFIG.smoke_folds, inputs.feature_sets)
        feature_set = first(feature_sets)
        audit = D.gph_gradient_audit(model, feature_sets; replays = 100, seed = 24)
        push!(gradients, (; model = name, fold, audit...))
        if name == "m03_negbin_pxg_covariate"
            candidate_audit = D.allocation_audit(model, reference_candidate, feature_set; seed = 24)
            push!(parity, (; model = name, fold, candidate_audit...))
        end
    end
    CSV.write(joinpath(OUTPUT, "gradients.csv"), DataFrame(gradients))
    CSV.write(joinpath(OUTPUT, "candidate_parity.csv"), DataFrame(parity))

    # ===============================================================
    # 5. Register recipe, deduplicate, then native fold x chain queue
    # ===============================================================
    config = D.fit_recipe(CONFIG, name, model, splitter, RUNTIME; smoke = true)
    D.register_recipe!(db, name, config)
    existing = D.gph_completed_run(db, config)
    recipe_hash = D.gph_run_hash(db, config)
    println("RECIPE ", name, " hash=", recipe_hash, " existing=", existing)
    PREPARE_ONLY && continue
    checkpoint = joinpath(OUTPUT, "checkpoints", recipe_hash)
    fit = existing === nothing ?
        D.gph_sample(config, inputs, RUNTIME; checkpoint_dir = checkpoint) :
        D.load_fit(db, existing)

    # ===============================================================
    # 6. Convergence, posterior identification, extraction, score mass
    # ===============================================================
    n_oos = sum(nrow, inputs.oos)
    D.gph_assert_coverage(name, fit; folds = 3, oos = n_oos)
    D.gph_latent_audit(fit)
    grid = D.score_grid_audit(fit)
    convergence = D.convergence_pass(fit, 3)
    pxg_rows = name == "m03_negbin_pxg_covariate" ?
        D.pxg_posterior(fit, CONFIG.smoke_folds) : NamedTuple[]
    append!(posterior, [(; model = name, row...) for row in pxg_rows])
    identified = name != "m03_negbin_pxg_covariate" || D.pxg_identification_pass(pxg_rows)

    # ===============================================================
    # 7. Exact fit round-trip; common portfolio follows after all arms
    # ===============================================================
    run_id = existing === nothing ? D.gph_save_and_verify(db, fit) : existing
    fits[name] = fit
    convergence_row = D.gph_convergence_row(name, fit, RUNTIME; run_id)
    weight_mean = isempty(pxg_rows) ? NaN : mean(row.mean for row in pxg_rows)
    weight_q05_min = isempty(pxg_rows) ? NaN : minimum(row.q05 for row in pxg_rows)
    gate_pass = convergence && identified
    push!(rows, (; convergence_row..., grid..., weight_mean, weight_q05_min,
                  weight_identified = identified, gate_pass,
                  portfolio_id = "", source = SOURCE, recipe_hash))
    CSV.write(joinpath(OUTPUT, "smoke_gates.csv"), DataFrame(rows))
    isempty(posterior) || CSV.write(joinpath(OUTPUT, "pxg_posterior.csv"), DataFrame(posterior))
    println("SMOKE ARM ", last(rows))
end

# ===================================================================
# 8. Promotion certificate and common-panel persistence round-trip
# ===================================================================
if PREPARE_ONLY
    println("PREPARE_ONLY PASS — no sampling or promotion certificate")
else
    verdict = length(rows) == 3 && all(row.gate_pass for row in rows)
    if verdict
        panel, refusals = D.tradeable_panel(fits, odds, ds)
        CSV.write(joinpath(OUTPUT, "portfolio_refusals.csv"), refusals)
        CSV.write(joinpath(OUTPUT, "portfolio_panel.csv"), DataFrame(match_id = panel))
        for (index, row) in enumerate(rows)
            portfolio_id, _ = D.portfolio_roundtrip(
                db, D.UUID(row.run_id), fits[row.model], ds, odds; panel)
            rows[index] = merge(row, (; portfolio_id = string(portfolio_id)))
        end
        CSV.write(joinpath(OUTPUT, "smoke_gates.csv"), DataFrame(rows))
    end
    certificate = (;
        source = SOURCE,
        passed = verdict,
        generated = now(),
        runs = Dict(row.model => row.run_id for row in rows),
        report = joinpath(OUTPUT, "smoke_gates.csv"),
    )
    D.Serialization.serialize(joinpath(CONFIG.save_root, "smoke_gate.jls"), certificate)
    println("SMOKE_VERDICT ", verdict ? "PASS" : "FAIL", " report=", certificate.report)
    verdict || error("Smoke convergence/identification gate failed; production forbidden")
end
