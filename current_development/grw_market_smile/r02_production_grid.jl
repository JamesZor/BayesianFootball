# ==============================================================================
# r02 — 43-fold walk-forward production grid, market-anchored MultiScaleGRW
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The inference half of Task 015: the four market-anchored rungs sampled over the
# canonical Scottish Lower walk-forward grid and persisted to `mcmc_experiments` under
# `scottish_lower_grw_market_smile`. Proper scores live in `r04_evaluate.jl` and the
# 2026-09-12 counterfactual in `r05_slate_repricing.jl`; nothing here reads a price for
# scoring.
#
#   m05_joint_grw_supremacy_w040         C1 @ 0.40
#   m05_joint_grw_smile_supremacy_w020   C1 + C2 @ 0.20
#   m05_joint_grw_smile_supremacy_w040   C1 + C2 @ 0.40
#   m05_joint_grw_smile_supremacy_w070   C1 + C2 @ 0.70
#
# The baseline rung is NOT sampled. Task 013 persisted the identical recipe
# (`m05_wealth_grw`, run b0961bc4, 4 × (500 + 1000), extended in place to 43 folds). §6
# asserts the model and sampler are string-identical before treating it as the control —
# the config-truth protocol's "load, don't resample".
#
# FILTRATION / COMPARABILITY CONTRACT
#
# * Split: `gph_splitter(["24/25", "25/26", "26/27"])` — 43 folds on a DataStore that ends
#   before the 2026-09-12 card. Folds 1–40 are asserted fixture-for-fixture identical to the
#   40-fold walk-forward split, so the 710-fixture panel is embedded unchanged; folds 41–43
#   are the ones the baseline control was extended with, and Fold 43 is the fold that priced
#   the live card.
# * Every fold: no fixture both trained on and held out; last training kickoff before first
#   held-out kickoff.
# * Both pillars read closing prices of TRAINING fixtures only.
# * Each run's held-out fixture set must EQUAL the baseline control's.
#
# GATES (per rung): R̂ ≤ 1.05 · bulk/tail ESS ≥ 400 · divergence rate < 0.1% · BFMI ≥ 0.30 ·
# tree-depth saturation < 5% · 43 folds · OOS set == control's · latents audited · smile
# O/U pricing equals cdf(Poisson(λ_tot·φ)) · persisted and reloaded identically.
#
# PERSISTENCE CAVEAT
#
# The audit runs on every retained draw; the artefact keeps every `persist_stride`-th with
# latents re-extracted from exactly those draws. Smile rungs are persisted with the latent
# panel detached (ticket T010) and a file copy under `results/latents/<run_id>/`. A rung that
# fails its gate is NOT persisted; its per-fold checkpoints stay under
# `results/<model>/checkpoints_<budget>/` and a rerun AT THE SAME BUDGET resumes from them.
#
# USAGE (mcmc-beast, from /root/BF_grw_market_smile, after r01 passes)
#
#   julia --project -t 16 current_development/grw_market_smile/r02_production_grid.jl
#   R02_MODELS=m05_joint_grw_smile_supremacy_w040 julia ... r02_production_grid.jl
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf

include(joinpath(@__DIR__, "l01_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R02_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = GMSConfig()
    GMSConfig(samples = env("R02_SAMPLES", base.samples),
              warmup = env("R02_WARMUP", base.warmup),
              chains = env("R02_CHAINS", base.chains),
              persist_stride = env("R02_STRIDE", base.persist_stride))
end
const R02_SELECTED = let raw = strip(get(ENV, "R02_MODELS", ""))
    isempty(raw) ? copy(GMS_GRID_MODEL_NAMES) : String.(strip.(split(raw, ",")))
end
all(in(GMS_GRID_MODEL_NAMES), R02_SELECTED) ||
    error("R02_MODELS names an unknown grid rung: $(setdiff(R02_SELECTED, GMS_GRID_MODEL_NAMES))")
# Stamped with the budget: `fit_model` resumes from whatever checkpoints it finds.
const R02_BUDGET_TAG = "$(R02_CONFIG.chains)x$(R02_CONFIG.warmup)w$(R02_CONFIG.samples)s"
const R02_OUT_DIR = R02_CONFIG.save_root
const R02_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end

println("\n" * "="^96)
println("  r02 PRODUCTION GRID — market-anchored MultiScaleGRW")
println("  rungs      : ", join(R02_SELECTED, ", "))
println("  split      : GroupedCVConfig 56/57, target ", join(R02_CONFIG.extension_seasons, " + "),
        ", 2 history seasons, match-biweek")
println("  sampler    : QueuedNUTS  ", R02_CONFIG.chains, " chains × ", R02_CONFIG.warmup,
        " warmup + ", R02_CONFIG.samples, " retained, δ = ", R02_CONFIG.accept_rate,
        ", max depth ", R02_CONFIG.max_depth)
println("  queue      : ", R02_CONFIG.max_concurrent_tasks, " concurrent fold×chain tasks on ",
        Threads.nthreads(), " threads (BLAS pinned to 1)")
println("  experiment : ", R02_CONFIG.experiment, "   persist stride: ", R02_CONFIG.persist_stride)
println("  git        : ", R02_GIT)
println("="^96)

# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================
mkpath(R02_OUT_DIR)
r02_db = gph_database(R02_CONFIG.experiment)

# %%
# ===================================================================
# 4. Data snapshot and temporal splits
# ===================================================================
r02_ds = gph_load_data()
r02_splitter = gph_splitter(R02_CONFIG.extension_seasons)
r02_n_folds = gms_split_prefix_check(r02_ds, R02_CONFIG)
println("\n  store: ", nrow(r02_ds.matches), " matches, latest ", maximum(r02_ds.matches.match_date),
        " | rows dated 2026-09-12: ", count(==(Date(2026, 9, 12)), Date.(r02_ds.matches.match_date)),
        " | folds: ", r02_n_folds, " (1–", R02_CONFIG.expected_folds, " == the 40-fold split)")

# %%
# ===================================================================
# 5. Engine / model construction and config registration
# ===================================================================
r02_all_models = gms_models()
r02_models = gms_select(r02_all_models, R02_SELECTED)
r02_baseline = last(first(r02_all_models))
r02_sampler = gms_production_sampler(R02_CONFIG)
r02_configs = gms_fit_configs(R02_CONFIG, r02_all_models, r02_splitter, r02_sampler)
r02_registry = gms_register!(r02_db, r02_all_models, r02_splitter, r02_sampler, r02_configs)
println("  registered models: ", r02_registry.model_ids,
        " | splitter #", r02_registry.splitter_id, " | sampler #", r02_registry.sampler_id)

# %%
# ===================================================================
# 6. The baseline rung: Task 013's persisted run of the same recipe
# ===================================================================
r02_control = load_fit(PostgresStorage(GMS_BASELINE_CONTROL.experiment), GMS_BASELINE_CONTROL.run_id)
string(r02_control.config.model) == string(r02_baseline) || error(
    "baseline control $(GMS_BASELINE_CONTROL.run_id) is not the recipe this loader builds; " *
    "it cannot stand in for m05_joint_grw_baseline")
# The SAMPLED budget, checked three ways. Task 013 sampled 4 × (500 + 1000), persisted every
# 2nd draw, and `extend_fit` then re-recorded `config.sampler` at the thinned 500 per chain it
# matched for folds 41–43 — so the recorded sampler legitimately reads 500 (first r02 attempt).
# (1) recorded sampler == production with n_samples = samples ÷ stride;
# (2) what was SAMPLED: Task 013's committed pre-extension report counted 40 × chains × samples
#     transitions; and fold 1's CURRENT audit counts chains × samples ÷ stride, because
#     `extend_fit` re-audited every fold from the thinned chains (second r02 attempt measured 2,000);
# (3) the persisted panel carries chains × samples ÷ stride draws, as the candidates will.
#
# Checked against the CANONICAL production budget, never R02_CONFIG: a rung re-run at a larger
# budget (R02_SAMPLES / R02_WARMUP / R02_STRIDE) does not change what the pinned control is, and
# reading the override here refused the control on the @0.70 re-run.
let C = GMSConfig(),
    thinned = QueuedNUTSConfig(n_samples = C.samples ÷ C.persist_stride,
                               n_warmup = C.warmup, n_chains = C.chains,
                               accept_rate = C.accept_rate, max_depth = C.max_depth,
                               show_progress = false)
    string(r02_control.config.sampler) == string(thinned) || error(
        "baseline control records sampler $(r02_control.config.sampler); expected $(thinned)")
    GMS_BASELINE_SAMPLED_TRANSITIONS_40 == C.expected_folds * C.chains * C.samples ||
        error("Task 013 recorded $(GMS_BASELINE_SAMPLED_TRANSITIONS_40) transitions over 40 folds; " *
              "the production budget implies $(C.expected_folds * C.chains * C.samples)")
    fold1 = only(filter(f -> f.fold == 1, r02_control.diagnostics.folds))
    fold1.n_transitions == C.chains * C.samples ÷ C.persist_stride || error(
        "baseline control fold 1 audit counts $(fold1.n_transitions) transitions; the thinned re-audit " *
        "`extend_fit` performs implies $(C.chains * C.samples ÷ C.persist_stride)")
    n_draws(r02_control.latents) == C.chains * C.samples ÷ C.persist_stride ||
        error("baseline control persists $(n_draws(r02_control.latents)) draws per fixture")
    # Every arm must persist the same number of draws per fixture, whatever it was sampled at.
    R02_CONFIG.chains * R02_CONFIG.samples ÷ R02_CONFIG.persist_stride == n_draws(r02_control.latents) ||
        error("this invocation would persist $(R02_CONFIG.chains * R02_CONFIG.samples ÷ R02_CONFIG.persist_stride) " *
              "draws per fixture; the baseline persists $(n_draws(r02_control.latents)) — adjust R02_STRIDE")
end
length(r02_control.folds) == R02_CONFIG.expected_extended_folds || error(
    "baseline control holds $(length(r02_control.folds)) folds; expected $(R02_CONFIG.expected_extended_folds)")
const R02_CONTROL_OOS = Set(Int.(r02_control.latents.match_ids))
@printf("  baseline control %s: %d folds, %d OOS fixtures, R̂ %.4f, %s — model and sampler identical\n",
        GMS_BASELINE_CONTROL.run_id, length(r02_control.folds), length(R02_CONTROL_OOS),
        r02_control.diagnostics.max_rhat, r02_control.diagnostics.passed ? "converged" : "NOT converged")
r02_control = nothing
GC.gc()

# %%
# ===================================================================
# 7–11. Per rung: config truth → features → training → audit → persist
# ===================================================================
r02_rows = NamedTuple[]

for (name, model) in r02_models
    fit_config = r02_configs[name]
    println("\n" * "-"^96)
    println("RUNG ", name, "   supremacy: ", nameof(typeof(model.supremacy)),
            "   smile: ", nameof(typeof(model.smile)), "   family: ",
            nameof(typeof(latent_family(model))))
    println("-"^96)

    # --- 7. config truth: never resample a persisted recipe --------------------
    existing = gph_completed_run(r02_db, fit_config)
    if existing !== nothing
        println("  recipe already persisted as run ", existing, " — not sampling")
        fit = load_fit(r02_db, existing)
        push!(r02_rows, (; gms_convergence_row(name, fit, R02_CONFIG; run_id = existing)...,
                           gate_pass = fit.diagnostics.passed, reused = true))
        println("R02_MODEL_DONE ", name, " reused ", existing)
        continue
    end

    # --- 8. features, filtration, market coverage -----------------------------
    inputs = gph_fold_inputs(r02_ds, r02_splitter, model)
    filtration = gph_filtration_report(r02_ds, inputs)
    CSV.write(joinpath(R02_OUT_DIR, "r02_filtration_$(name).csv"), filtration)
    all(filtration.ordered) || error("$name: training/OOS kickoff ordering violated")
    coverage = gms_market_coverage(model, inputs)
    CSV.write(joinpath(R02_OUT_DIR, "r02_market_coverage_$(name).csv"), coverage)
    all(coverage.supremacy_observed .> 0) || error("$name: a fold's supremacy pillar reads no match")
    @printf("  folds %d | train %d–%d | OOS %d | target steps 0–%d | supremacy share %.2f–%.2f | smile share %.2f–%.2f\n",
            nrow(filtration), minimum(filtration.n_train), maximum(filtration.n_train),
            sum(filtration.n_oos), maximum(filtration.n_target),
            minimum(coverage.supremacy_share), maximum(coverage.supremacy_share),
            minimum(coverage.smile_share), maximum(coverage.smile_share))

    # --- 9. training -------------------------------------------------------------
    # NUTS chains are single-threaded; QueuedExecution flattens 43 folds × 4 chains into
    # one 172-task queue over the 16 pinned threads.
    checkpoint_dir = joinpath(R02_OUT_DIR, name, "checkpoints_" * R02_BUDGET_TAG)
    fit = gph_sample(fit_config, inputs, gms_gph_config(R02_CONFIG); checkpoint_dir)

    # --- 10. convergence (six-part audit on every retained draw) ---------------
    d = fit.diagnostics
    pillars = gms_pillar_summary(fit)
    @printf("  audit: R̂ %.4f (fold %d) | ESS bulk %.0f tail %.0f | div %d/%d | depth %.2f%% | BFMI %.3f | %s\n",
            d.max_rhat, d.worst_rhat_fold, d.min_ess_bulk, d.min_ess_tail,
            d.n_divergent, d.n_transitions, 100 * d.treedepth_rate, d.min_bfmi,
            d.passed ? "PASS" : "FAIL: " * join(d.failures, "; "))
    @printf("  pillars: σ_sup %.3f [%.3f, %.3f] | σ_smile %.3f [%.3f, %.3f] | κ %.3f | φ %s\n",
            pillars.σ_sup_median, pillars.σ_sup_q05, pillars.σ_sup_q95,
            pillars.σ_smile_median, pillars.σ_smile_q05, pillars.σ_smile_q95,
            pillars.κ_median, pillars.φ_median)

    if !d.passed
        push!(r02_rows, (; gms_convergence_row(name, fit, R02_CONFIG)..., gate_pass = false, reused = false))
        println("R02_MODEL_FAIL ", name, " — not persisted; checkpoints kept in ", checkpoint_dir)
        continue
    end

    # --- 11. thin, audit latents, persist, verify round-trip -------------------
    persisted = gph_thin_for_persistence(fit, inputs, R02_CONFIG.persist_stride)
    length(persisted.folds) == R02_CONFIG.expected_extended_folds || error(
        "$name has $(length(persisted.folds)) folds; expected $(R02_CONFIG.expected_extended_folds)")
    Set(Int.(persisted.latents.match_ids)) == R02_CONTROL_OOS || error(
        "$name holds a different held-out fixture set from the baseline control — not comparable")
    latent = gms_latent_audit(persisted)
    @printf("  latents: %s | %d fixtures × %d draws | mean λ_h %.3f λ_a %.3f | φ mean %s\n",
            latent.family, latent.n_matches, latent.n_draws, latent.mean_lambda_h,
            latent.mean_lambda_a, latent.φ_mean)

    if persisted.latents isa SmileLatents
        pricing = gms_smile_pricing_gate(persisted; n_fixtures = 12)
        CSV.write(joinpath(R02_OUT_DIR, "r02_smile_pricing_$(name).csv"), pricing)
        maximum(abs.(pricing.p_under_typed .- pricing.p_under_ref)) <= 1e-12 &&
            maximum(abs.(pricing.p_under_legacy .- pricing.p_under_ref)) <= 1e-12 ||
            error("$name: smile O/U pricing disagrees with cdf(Poisson(λ_tot·φ))")
        @printf("  smile O/U 2.5 vs plain grid at the same λ: mean shift %+.4f\n",
                mean(pricing.smile_shift))
    end

    run_id = gms_save_and_verify(r02_db, persisted, inputs;
                                 latent_dir = joinpath(R02_OUT_DIR, "latents"))
    println("  persisted and reloaded identically: run ", run_id)

    push!(r02_rows, (; gms_convergence_row(name, persisted, R02_CONFIG; run_id)...,
                       gate_pass = true, reused = false))
    println("R02_MODEL_DONE ", name, " ", run_id)

    fit = nothing
    persisted = nothing
    inputs = nothing
    GC.gc()
end

# %%
# ===================================================================
# 12. Final report
# ===================================================================
r02_summary = DataFrame(r02_rows)
const R02_STAMP = R02_BUDGET_TAG * "_" * join(sort(R02_SELECTED), "+")
CSV.write(joinpath(R02_OUT_DIR, "r02_production_runs_$(R02_STAMP).csv"), r02_summary)

open(joinpath(R02_OUT_DIR, "r02_production_report_$(R02_STAMP).md"), "w") do io
    println(io, "# r02 production grid — Task 015 (market-anchored MultiScaleGRW)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R02_GIT,
            "` on ", gethostname(), ". Namespace `", R02_CONFIG.experiment, "`. Store latest kickoff ",
            maximum(r02_ds.matches.match_date), "; ", r02_n_folds, " folds.\n")
    println(io, "Sampler: QueuedNUTS ", R02_CONFIG.chains, " × (", R02_CONFIG.warmup, " warmup + ",
            R02_CONFIG.samples, " retained), δ = ", R02_CONFIG.accept_rate,
            ". Audit on all retained draws; artefact keeps every ", R02_CONFIG.persist_stride,
            "nd draw. Baseline rung = Task 013 run `", GMS_BASELINE_CONTROL.run_id, "`.\n")
    println(io, "## Convergence\n")
    print(io, gph_markdown_table(select(r02_summary,
        :model, :folds, :oos, :draws, :max_rhat, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :divergence_rate, :treedepth_rate, :min_bfmi, :strict_rhat_pass,
        :gate_pass, :wall_min, :run_id);
        formats = Dict(:divergence_rate => v -> @sprintf("%.5f", v),
                       :treedepth_rate => v -> @sprintf("%.4f", v),
                       :wall_min => v -> gph_num(v; digits = 1),
                       :min_ess_bulk => v -> gph_num(v; digits = 0),
                       :min_ess_tail => v -> gph_num(v; digits = 0))))
    println(io, "\n## Pillar posteriors (pooled over folds)\n")
    print(io, gph_markdown_table(select(r02_summary,
        :model, :σ_sup_median, :σ_sup_q05, :σ_sup_q95,
        :σ_smile_median, :σ_smile_q05, :σ_smile_q95, :κ_median, :φ_median);
        formats = Dict(c => (v -> gph_num(v; digits = 3)) for c in
                       (:σ_sup_median, :σ_sup_q05, :σ_sup_q95, :σ_smile_median,
                        :σ_smile_q05, :σ_smile_q95, :κ_median))))
end

println("\nR02_VERDICT ", all(r02_summary.gate_pass) ? "PASS" : "FAIL", "  (",
        count(r02_summary.gate_pass), "/", nrow(r02_summary), " rungs)")
println("R02_DONE report=", joinpath(R02_OUT_DIR, "r02_production_report_$(R02_STAMP).md"))
