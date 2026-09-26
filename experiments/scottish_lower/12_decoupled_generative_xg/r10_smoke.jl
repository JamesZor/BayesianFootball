# Stage 1: mechanical/convergence gate, not a predictive-performance result.
# Fixed 4 × (400 warmup + 400 retained), folds 1/20/40, all four arms.
# PostgreSQL namespace: smoke_scottish_lower_decoupled_xg.
# USAGE: julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r10_smoke.jl
# Set FUNNEL_PREPARE_ONLY=true for filtration, AD, and registry preflight only.

# ===================================================================
# 1. Packages and runtime
# ===================================================================
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Dates, Statistics
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l12_loader.jl"))
const D = DecoupledGenerativeXG

# ===================================================================
# 2. Visible configuration and immutable source identity
# ===================================================================
const CONFIG = D.FunnelConfig()
const RUNTIME = D.runtime_config(CONFIG)
const PREPARE_ONLY = parse(Bool, get(ENV, "FUNNEL_PREPARE_ONLY", "false"))
const SOURCE = D.source_fingerprint()
const OUTPUT = joinpath(CONFIG.save_root, "smoke", SOURCE)
mkpath(OUTPUT)
println("SMOKE source=", SOURCE, " folds=", CONFIG.smoke_folds,
        " prepare_only=", PREPARE_ONLY)

# ===================================================================
# 3. Data, split, closing book, and explicit four-arm construction
# ===================================================================
ds = D.gph_load_data()
splitter = D.gph_splitter(CONFIG.target_seasons)
models = D.models()
references = Dict(D.reference_models())
db = D.gph_database(CONFIG.smoke_experiment)
odds = D.gph_betfair_closing_odds(ds)
rows = NamedTuple[]
gradients = NamedTuple[]
posterior = NamedTuple[]
fits = Dict{String,Any}()

for (name, model) in models
    # ===============================================================
    # 4. Filtration, feature audit, and compiled AD gates
    # ===============================================================
    inputs, filtration = D.selected_inputs(ds, splitter, model, CONFIG; smoke = true)
    CSV.write(joinpath(OUTPUT, name * "_filtration.csv"), filtration)
    for (fold, feature_sets) in zip(CONFIG.smoke_folds, inputs.feature_sets)
        fs = first(feature_sets)
        if model isa D.CutFunnelModel
            # A cut arm's sampled density is its chance layer, and the gate that
            # matters is that goals cannot move it AT ALL.
            chance = D.build_cut_chance_model(model, fs)
            audit = D.density_audit(chance, chance; seed = 25)
            nofb = D.cut_no_feedback_audit(model, fs; seed = 25)
            (nofb.worst_density == 0.0 && nofb.worst_gradient == 0.0) || error(
                "$name fold $fold: goals reach the chance layer " *
                "(density $(nofb.worst_density), gradient $(nofb.worst_gradient)) - not a cut")
        else
            audit = D.engine_audit(model, references[name], fs; seed = 25)
        end
        audit.allocated_bytes == 0 || error(
            "$name fold $fold replay allocates $(audit.allocated_bytes) bytes")
        push!(gradients, (; model = name, fold, audit...))
    end
    CSV.write(joinpath(OUTPUT, "gradients.csv"), DataFrame(gradients))

    # ===============================================================
    # 5. Register recipe, deduplicate, then native fold × chain queue
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
    # 6. Convergence, extraction, score mass, and kappa geometry
    # ===============================================================
    n_oos = sum(nrow, inputs.oos)
    D.gph_assert_coverage(name, fit; folds = 3, oos = n_oos)
    D.gph_latent_audit(fit)
    grid = D.score_grid_audit(fit)
    convergence = D.convergence_pass(fit, 3)
    # The fit-level audit sees the SPLICED chain, which cannot tell whether each
    # inner conditional run mixed. Gate Stage B separately, per fold.
    stage_b = if model isa D.CutFunnelModel
        gates = [D.cut_stage_b_gate(f.chain) for f in fit.folds]
        for (fold, g) in zip(CONFIG.smoke_folds, gates)
            println("  STAGE B fold $fold ", g)
            g.passed || error("$name fold $fold Stage B conditional did not mix: $g")
        end
        (; stage_b_pass = all(g.passed for g in gates),
           stage_b_max_rhat = maximum(g.max_rhat for g in gates),
           stage_b_frac_rhat_gt = maximum(g.frac_rhat_gt for g in gates),
           stage_b_divergences = sum(g.divergences for g in gates),
           stage_b_worst_div_frac = maximum(g.div_frac for g in gates),
           stage_b_runs = sum(g.runs for g in gates))
    else
        (; stage_b_pass = true, stage_b_max_rhat = NaN,
           stage_b_frac_rhat_gt = 0.0, stage_b_divergences = 0,
           stage_b_worst_div_frac = 0.0, stage_b_runs = 0)
    end
    zero_sum_error = D.hierarchical_zero_sum_audit(fit)
    arm_posterior = D.kappa_posterior(fit, CONFIG.smoke_folds)
    append!(posterior, [(; model = name, row...) for row in arm_posterior])

    # ===============================================================
    # 7. Exact fit round-trip; common portfolio follows after all arms
    # ===============================================================
    run_id = existing === nothing ? D.gph_save_and_verify(db, fit) : existing
    fits[name] = fit
    convergence_row = D.gph_convergence_row(name, fit, RUNTIME; run_id)
    gate_pass = convergence && stage_b.stage_b_pass
    push!(rows, (; convergence_row..., grid..., zero_sum_error, stage_b..., gate_pass,
                  portfolio_id = "", source = SOURCE, recipe_hash))
    CSV.write(joinpath(OUTPUT, "smoke_gates.csv"), DataFrame(rows))
    isempty(posterior) || CSV.write(joinpath(OUTPUT, "kappa_posterior.csv"), DataFrame(posterior))
    println("SMOKE ARM ", last(rows))
end

# ===================================================================
# 8. Promotion certificate and common-panel persistence round-trip
# ===================================================================
if PREPARE_ONLY
    println("PREPARE_ONLY PASS — no sampling or promotion certificate")
else
    verdict = length(rows) == 4 && all(row.gate_pass for row in rows)
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
    verdict || error("Smoke convergence gate failed; production forbidden")
end
