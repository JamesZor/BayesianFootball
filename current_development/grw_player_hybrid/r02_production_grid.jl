# ==============================================================================
# r02 — 40-fold walk-forward production grid, MultiScaleGRW × PlayerLineupPillar
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The inference half of Task 013: four ladder models sampled over the canonical
# Scottish Lower grid (pooled tournaments 56/57, target seasons 24/25 + 25/26,
# 40 match-biweek folds, 710 held-out fixtures) and persisted to
# `mcmc_experiments` under `scottish_lower_grw_player_hybrid`.
#
# It is NOT the comparison. Proper scores, the bootstrap and the portfolio live in
# `r04_evaluate.jl` and `r05_portfolio_attribution.jl`, which load what this
# runner persists by UUID. Nothing here reads a price.
#
# FILTRATION / COMPARABILITY CONTRACT
#
# * The split is `gph_splitter(["24/25", "25/26"])` — the same `GroupedCVConfig` the
#   Exp 06 TimeDecay controls and the Task 007 GRW runs were fitted under, so the
#   710 held-out fixtures are the identical set and the GRW micro step is the
#   identical match-biweek.
# * RAPM is ridge-fitted on each fold's history block only (`fit_on = :history`).
# * Every fold is asserted, before sampling, to hold no fixture in both its training
#   block and its held-out block, and to have its last training kickoff precede its
#   first held-out kickoff.
#
# GATES (per model, all six parts of the audit plus coverage)
#
#   R̂ ≤ 1.05 · bulk ESS ≥ 400 · tail ESS ≥ 400 · divergence rate < 0.1% ·
#   BFMI ≥ 0.30 · tree-depth saturation < 5% · 40 folds · 710 unique OOS fixtures
#
#   Task 007's strict R̂ ≤ 1.01 is reported beside the gate as advisory.
#
# PERSISTENCE CAVEAT
#
# The audit runs on all 4 × 1,000 retained draws. The artefact persists every 2nd
# draw (2,000 per fold) with latents re-extracted from exactly those draws — see
# `GPHConfig.persist_stride` for why 4,000 would not fit the bytea path. A model
# that fails its gate is NOT persisted; its per-fold checkpoints stay on disk under
# `results/<model>/checkpoints/` and a rerun resumes from them.
#
# A recipe already persisted in this namespace is loaded, not resampled.
#
# USAGE (mcmc-beast, from /root/BF_grw_player_hybrid)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/grw_player_hybrid/r02_production_grid.jl
#   R02_MODELS=m10_lineup_grw,m12_joint_hybrid_synergy_grw  julia ... r02_production_grid.jl
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
const R02_CONFIG = GPHConfig()
const R02_SELECTED = let raw = strip(get(ENV, "R02_MODELS", ""))
    isempty(raw) ? copy(GPH_MODEL_NAMES) : String.(strip.(split(raw, ",")))
end
all(in(GPH_MODEL_NAMES), R02_SELECTED) ||
    error("R02_MODELS names an unknown model: $(setdiff(R02_SELECTED, GPH_MODEL_NAMES))")
const R02_OUT_DIR = R02_CONFIG.save_root
const R02_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end

println("\n" * "="^96)
println("  r02 PRODUCTION GRID — MultiScaleGRW × PlayerLineupPillar")
println("  models     : ", join(R02_SELECTED, ", "))
println("  split      : GroupedCVConfig 56/57, target ", join(R02_CONFIG.target_seasons, " + "),
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
r02_splitter = gph_splitter(R02_CONFIG.target_seasons)
r02_boundaries = Data.create_id_boundaries(r02_ds, r02_splitter)
length(r02_boundaries) == R02_CONFIG.expected_folds || error(
    "splitter produced $(length(r02_boundaries)) folds; expected $(R02_CONFIG.expected_folds)")
println("\n  store: ", nrow(r02_ds.matches), " matches, latest ", maximum(r02_ds.matches.match_date),
        " | folds: ", length(r02_boundaries))

# %%
# ===================================================================
# 5. Engine / model construction and config registration
# ===================================================================
r02_models = gph_models()
r02_sampler = gph_production_sampler(R02_CONFIG)
r02_configs = gph_fit_configs(R02_CONFIG, r02_models, r02_splitter, r02_sampler)
r02_registry = gph_register!(r02_db, r02_models, r02_splitter, r02_sampler, r02_configs)
println("  registered models: ", r02_registry.model_ids,
        " | splitter #", r02_registry.splitter_id, " | sampler #", r02_registry.sampler_id)

# %%
# ===================================================================
# 6–10. Per model: config truth → features → training → audit → persist
# ===================================================================
r02_rows = NamedTuple[]

for (name, model) in r02_models
    name in R02_SELECTED || continue
    fit_config = r02_configs[name]
    println("\n" * "-"^96)
    println("MODEL ", name)
    println("-"^96)

    # --- 6. config truth: never resample a persisted recipe --------------------
    existing = gph_completed_run(r02_db, fit_config)
    if existing !== nothing
        println("  recipe already persisted as run ", existing, " — loading, not sampling")
        fit = load_fit(r02_db, existing)
        push!(r02_rows, (; gph_convergence_row(name, fit, R02_CONFIG; run_id = existing)...,
                           gate_pass = fit.diagnostics.passed, reused = true))
        println("R02_MODEL_DONE ", name, " reused ", existing)
        continue
    end

    # --- 7. features and filtration -------------------------------------------
    inputs = gph_fold_inputs(r02_ds, r02_splitter, model)
    filtration = gph_filtration_report(r02_ds, inputs)
    CSV.write(joinpath(R02_OUT_DIR, "r02_filtration_$(name).csv"), filtration)
    all(filtration.ordered) || error("$name: training/OOS kickoff ordering violated")
    @printf("  folds %d | train %d–%d | OOS %d fixtures | target steps 0–%d\n",
            nrow(filtration), minimum(filtration.n_train), maximum(filtration.n_train),
            sum(filtration.n_oos), maximum(filtration.n_target))
    sum(filtration.n_oos) == R02_CONFIG.expected_oos || error(
        "$name: $(sum(filtration.n_oos)) OOS fixtures; expected $(R02_CONFIG.expected_oos)")

    # --- 8. training ------------------------------------------------------------
    # NUTS chains are single-threaded; QueuedExecution flattens 40 folds × 4 chains
    # into one 160-task queue and keeps all 16 pinned threads busy until it drains.
    checkpoint_dir = joinpath(R02_OUT_DIR, name, "checkpoints")
    fit = gph_sample(fit_config, inputs, R02_CONFIG; checkpoint_dir)

    # --- 9. convergence (six-part audit on every retained draw) ---------------
    d = fit.diagnostics
    @printf("  audit: R̂ %.4f (fold %d) | ESS bulk %.0f tail %.0f | div %d/%d | depth %.2f%% | BFMI %.3f | %s\n",
            d.max_rhat, d.worst_rhat_fold, d.min_ess_bulk, d.min_ess_tail,
            d.n_divergent, d.n_transitions, 100 * d.treedepth_rate, d.min_bfmi,
            d.passed ? "PASS" : "FAIL: " * join(d.failures, "; "))

    if !d.passed
        push!(r02_rows, (; gph_convergence_row(name, fit, R02_CONFIG)..., gate_pass = false,
                           reused = false))
        println("R02_MODEL_FAIL ", name, " — not persisted; checkpoints kept in ", checkpoint_dir)
        continue
    end

    # --- 10. thin for persistence, audit latents, persist, verify round-trip ---
    persisted = gph_thin_for_persistence(fit, inputs, R02_CONFIG.persist_stride)
    gph_assert_coverage(name, persisted; folds = R02_CONFIG.expected_folds,
                        oos = R02_CONFIG.expected_oos)
    latent = gph_latent_audit(persisted)
    @printf("  latents: %d fixtures × %d draws | mean λ_h %.3f λ_a %.3f | sd %.3f–%.3f\n",
            latent.n_matches, latent.n_draws, latent.mean_lambda_h, latent.mean_lambda_a,
            latent.min_sd, latent.max_sd)
    run_id = gph_save_and_verify(r02_db, persisted)
    println("  persisted and reloaded identically: run ", run_id)

    push!(r02_rows, (; gph_convergence_row(name, persisted, R02_CONFIG; run_id)...,
                       gate_pass = true, reused = false))
    println("R02_MODEL_DONE ", name, " ", run_id)

    fit = nothing
    persisted = nothing
    inputs = nothing
    GC.gc()
end

# %%
# ===================================================================
# 11. Final report
# ===================================================================
r02_summary = DataFrame(r02_rows)
CSV.write(joinpath(R02_OUT_DIR, "r02_production_runs.csv"), r02_summary)

open(joinpath(R02_OUT_DIR, "r02_production_report.md"), "w") do io
    println(io, "# r02 production grid — Task 013\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R02_GIT,
            "` on ", gethostname(), ". Namespace `", R02_CONFIG.experiment, "`.\n")
    println(io, "Sampler: QueuedNUTS ", R02_CONFIG.chains, " × (", R02_CONFIG.warmup, " warmup + ",
            R02_CONFIG.samples, " retained), δ = ", R02_CONFIG.accept_rate,
            ". Audit on all retained draws; artefact keeps every ", R02_CONFIG.persist_stride,
            "nd draw.\n")
    print(io, gph_markdown_table(select(r02_summary,
        :model, :folds, :oos, :draws, :max_rhat, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :divergence_rate, :treedepth_rate, :min_bfmi, :strict_rhat_pass,
        :gate_pass, :wall_min, :run_id);
        formats = Dict(:divergence_rate => v -> @sprintf("%.5f", v),
                       :treedepth_rate => v -> @sprintf("%.4f", v),
                       :wall_min => v -> gph_num(v; digits = 1),
                       :min_ess_bulk => v -> gph_num(v; digits = 0),
                       :min_ess_tail => v -> gph_num(v; digits = 0))))
end

println("\nR02_VERDICT ", all(r02_summary.gate_pass) ? "PASS" : "FAIL", "  (",
        count(r02_summary.gate_pass), "/", nrow(r02_summary), " models)")
println("R02_DONE report=", joinpath(R02_OUT_DIR, "r02_production_report.md"))
