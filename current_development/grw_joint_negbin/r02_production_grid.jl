# ==============================================================================
# r02 — 40-fold walk-forward production grid, JointGammaNegBinObservation ladder
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The inference half of Task 014: four NegBin ladder models sampled over the canonical
# Scottish Lower grid (pooled tournaments 56/57, target seasons 24/25 + 25/26, 40
# match-biweek folds, 710 held-out fixtures) and persisted to `mcmc_experiments` under
# `scottish_lower_grw_joint_negbin`.
#
# It is NOT the comparison. Proper scores, the bootstrap and the portfolio live in
# `r04_evaluate.jl` and `r05_portfolio.jl`, which load what this runner persists by
# UUID. Nothing here reads a price.
#
# FILTRATION / COMPARABILITY CONTRACT
#
# * The split is `gjn_splitter(["24/25", "25/26"])` — the same `GroupedCVConfig` the
#   Task 013 Poisson controls were fitted under, so the 710 held-out fixtures are the
#   identical set and the GRW micro step is the identical match-biweek.
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
# THE DISPERSION IS REPORTED, NOT GATED. `r̂` is the study's own subject: a posterior
# that piles up at large `r` is a negative binomial saying it is a Poisson, which is a
# RESULT (the latent state already absorbed the overdispersion), not a failure. It is
# printed per model and carried into the report so the evaluation can be read against it.
#
# PERSISTENCE CAVEAT
#
# The audit runs on every retained draw. The artefact persists every `persist_stride`-th,
# with latents re-extracted from exactly those draws. A model that fails its gate is NOT
# persisted; its per-fold checkpoints stay on disk under
# `results/<model>/checkpoints_<chains>x<warmup>w<samples>s/` and a rerun AT THE SAME BUDGET
# resumes from them. A recipe already persisted in this namespace is loaded, not resampled.
#
# USAGE (mcmc-beast, from /root/BF_grw_joint_negbin)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/grw_joint_negbin/r02_production_grid.jl
#   R02_MODELS=m05_wealth_grw_negbin  julia ... r02_production_grid.jl
#
#   # a model that failed the audit on ESS alone, re-run at a larger budget:
#   R02_MODELS=m12_joint_hybrid_synergy_negbin R02_WARMUP=1000 R02_SAMPLES=2500 R02_STRIDE=5 \
#     julia ... r02_production_grid.jl
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
# The production budget, overridable per invocation.
#
# WHY THE OVERRIDE EXISTS. A model can fail the six-part audit on ESS alone — enough draws
# to settle R̂ and produce no divergences, but not enough to resolve the tails of every
# parameter at every fold. That is a BUDGET shortfall, not a geometry pathology, and the
# fix is more draws rather than a different model. Keeping the knob here, stamped into the
# checkpoint path and reported in the run row, means a re-run at a larger budget is a
# recorded fact rather than an edit to the file.
#
# `R02_STRIDE` exists because the two move together. The audit runs on every retained draw;
# the artefact keeps every `persist_stride`-th. Task 013 measured a 40-fold GRW fit at 4,000
# draws per fold as ~725 MB, and its hex text form has to stay under PostgreSQL's 1 GB field
# limit — so raising `samples` without raising `persist_stride` in step will eventually push
# `save_fit` over that edge.
const R02_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = GJNConfig()
    GJNConfig(samples = env("R02_SAMPLES", base.samples),
              warmup = env("R02_WARMUP", base.warmup),
              chains = env("R02_CHAINS", base.chains),
              persist_stride = env("R02_STRIDE", base.persist_stride))
end
const R02_SELECTED = let raw = strip(get(ENV, "R02_MODELS", ""))
    isempty(raw) ? copy(GJN_MODEL_NAMES) : String.(strip.(split(raw, ",")))
end
# Stamped with the budget. `fit_model` RESUMES from whatever per-fold checkpoints it finds,
# so re-running a failed model at a larger budget against the old directory would hand back
# the old draws and reproduce the failure exactly — with a new config hash to make it look
# like a fresh result.
const R02_BUDGET_TAG = "$(R02_CONFIG.chains)x$(R02_CONFIG.warmup)w$(R02_CONFIG.samples)s"
all(in(GJN_MODEL_NAMES), R02_SELECTED) ||
    error("R02_MODELS names an unknown model: $(setdiff(R02_SELECTED, GJN_MODEL_NAMES))")
const R02_OUT_DIR = R02_CONFIG.save_root
const R02_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end

println("\n" * "="^96)
println("  r02 PRODUCTION GRID — JointGammaNegBinObservation ladder")
println("  models     : ", join(R02_SELECTED, ", "))
println("  split      : GroupedCVConfig 56/57, target ", join(R02_CONFIG.target_seasons, " + "),
        ", 2 history seasons, match-biweek")
println("  sampler    : QueuedNUTS  ", R02_CONFIG.chains, " chains × ", R02_CONFIG.warmup,
        " warmup + ", R02_CONFIG.samples, " retained, δ = ", R02_CONFIG.accept_rate,
        ", max depth ", R02_CONFIG.max_depth)
println("  queue      : ", R02_CONFIG.max_concurrent_tasks, " concurrent fold×chain tasks on ",
        Threads.nthreads(), " threads (BLAS pinned to 1)")
println("  experiment : ", R02_CONFIG.experiment, "   persist stride: ", R02_CONFIG.persist_stride)
println("  checkpoints: <model>/checkpoints_", R02_BUDGET_TAG)
println("  git        : ", R02_GIT)
println("="^96)

# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================
mkpath(R02_OUT_DIR)
r02_db = gjn_database(R02_CONFIG.experiment)

# %%
# ===================================================================
# 4. Data snapshot and temporal splits
# ===================================================================
r02_ds = gjn_load_data()
r02_splitter = gjn_splitter(R02_CONFIG.target_seasons)
r02_boundaries = Data.create_id_boundaries(r02_ds, r02_splitter)
length(r02_boundaries) == R02_CONFIG.expected_folds || error(
    "splitter produced $(length(r02_boundaries)) folds; expected $(R02_CONFIG.expected_folds)")
println("\n  store: ", nrow(r02_ds.matches), " matches, latest ", maximum(r02_ds.matches.match_date),
        " | folds: ", length(r02_boundaries))

# %%
# ===================================================================
# 5. Engine / model construction and config registration
# ===================================================================
r02_models = gjn_models()
r02_sampler = gjn_production_sampler(R02_CONFIG)
r02_configs = gjn_fit_configs(R02_CONFIG, r02_models, r02_splitter, r02_sampler)
r02_registry = gjn_register!(r02_db, r02_models, r02_splitter, r02_sampler, r02_configs)
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
    println("MODEL ", name, "   obs: ", nameof(typeof(model.observation)),
            "   family: ", latent_family(model))
    println("-"^96)

    # --- 6. config truth: never resample a persisted recipe --------------------
    existing = gjn_completed_run(r02_db, fit_config)
    if existing !== nothing
        println("  recipe already persisted as run ", existing, " — loading, not sampling")
        fit = load_fit(r02_db, existing)
        disp = gjn_dispersion_summary(fit)
        push!(r02_rows, (; gjn_convergence_row(name, fit, R02_CONFIG; run_id = existing)...,
                           mean_r = disp.mean_r, median_r = disp.median_r,
                           r_q05 = disp.q05, r_q95 = disp.q95,
                           gate_pass = fit.diagnostics.passed, reused = true))
        println("R02_MODEL_DONE ", name, " reused ", existing)
        continue
    end

    # --- 7. features and filtration -------------------------------------------
    inputs = gjn_fold_inputs(r02_ds, r02_splitter, model)
    filtration = gjn_filtration_report(r02_ds, inputs)
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
    checkpoint_dir = joinpath(R02_OUT_DIR, name, "checkpoints_" * R02_BUDGET_TAG)
    fit = gjn_sample(fit_config, inputs, R02_CONFIG; checkpoint_dir)

    # --- 9. convergence (six-part audit on every retained draw) ---------------
    d = fit.diagnostics
    disp = gjn_dispersion_summary(fit)
    @printf("  audit: R̂ %.4f (fold %d) | ESS bulk %.0f tail %.0f | div %d/%d | depth %.2f%% | BFMI %.3f | %s\n",
            d.max_rhat, d.worst_rhat_fold, d.min_ess_bulk, d.min_ess_tail,
            d.n_divergent, d.n_transitions, 100 * d.treedepth_rate, d.min_bfmi,
            d.passed ? "PASS" : "FAIL: " * join(d.failures, "; "))
    @printf("  dispersion: r̂ median %.2f  mean %.2f  90%% [%.2f, %.2f]  (reported, not gated)\n",
            disp.median_r, disp.mean_r, disp.q05, disp.q95)

    if !d.passed
        push!(r02_rows, (; gjn_convergence_row(name, fit, R02_CONFIG)...,
                           mean_r = disp.mean_r, median_r = disp.median_r,
                           r_q05 = disp.q05, r_q95 = disp.q95,
                           gate_pass = false, reused = false))
        println("R02_MODEL_FAIL ", name, " — not persisted; checkpoints kept in ", checkpoint_dir)
        continue
    end

    # --- 10. thin for persistence, audit latents, persist, verify round-trip ---
    persisted = gjn_thin_for_persistence(fit, inputs, R02_CONFIG.persist_stride)
    gjn_assert_coverage(name, persisted; folds = R02_CONFIG.expected_folds,
                        oos = R02_CONFIG.expected_oos)
    latent = gjn_latent_audit(persisted)
    @printf("  latents: %d fixtures × %d draws | mean λ_h %.3f λ_a %.3f | sd %.3f–%.3f | r %.2f–%.2f\n",
            latent.n_matches, latent.n_draws, latent.mean_lambda_h, latent.mean_lambda_a,
            latent.min_sd, latent.max_sd, latent.min_r, latent.max_r)

    grid = gjn_grid_gate(persisted)
    @printf("  grid vs double-Poisson at the same λ: |Δ O/U 3.5| %.5f  |Δ BTTS| %.5f  |Δ 1X2| %.5f\n",
            mean(abs.(grid.d_over35)), mean(abs.(grid.d_btts)), mean(abs.(grid.d_home)))
    CSV.write(joinpath(R02_OUT_DIR, "r02_grid_gate_$(name)_$(R02_BUDGET_TAG).csv"), grid)

    run_id = gjn_save_and_verify(r02_db, persisted)
    println("  persisted and reloaded identically: run ", run_id)

    push!(r02_rows, (; gjn_convergence_row(name, persisted, R02_CONFIG; run_id)...,
                       mean_r = disp.mean_r, median_r = disp.median_r,
                       r_q05 = disp.q05, r_q95 = disp.q95,
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

# STAMPED, NOT OVERWRITTEN. A re-run of one model at a different budget used to clobber the
# report and CSV describing the others — silently destroying the record of every model this
# invocation did not touch. The stamp makes each invocation its own artefact; the merged
# four-model table lives in the README, assembled from these.
const R02_STAMP = R02_BUDGET_TAG * "_" * join(sort(R02_SELECTED), "+")
CSV.write(joinpath(R02_OUT_DIR, "r02_production_runs_$(R02_STAMP).csv"), r02_summary)

open(joinpath(R02_OUT_DIR, "r02_production_report_$(R02_STAMP).md"), "w") do io
    println(io, "# r02 production grid — Task 014 (JointGammaNegBinObservation)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R02_GIT,
            "` on ", gethostname(), ". Namespace `", R02_CONFIG.experiment, "`.\n")
    println(io, "Sampler: QueuedNUTS ", R02_CONFIG.chains, " × (", R02_CONFIG.warmup, " warmup + ",
            R02_CONFIG.samples, " retained), δ = ", R02_CONFIG.accept_rate,
            ". Audit on all retained draws; artefact keeps every ", R02_CONFIG.persist_stride,
            "nd draw.\n")
    println(io, "## Convergence\n")
    print(io, gjn_markdown_table(select(r02_summary,
        :model, :folds, :oos, :draws, :max_rhat, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :divergence_rate, :treedepth_rate, :min_bfmi, :strict_rhat_pass,
        :gate_pass, :wall_min, :run_id);
        formats = Dict(:divergence_rate => v -> @sprintf("%.5f", v),
                       :treedepth_rate => v -> @sprintf("%.4f", v),
                       :wall_min => v -> gjn_num(v; digits = 1),
                       :min_ess_bulk => v -> gjn_num(v; digits = 0),
                       :min_ess_tail => v -> gjn_num(v; digits = 0))))
    println(io, "\n## Posterior dispersion `r` (reported, not gated)\n")
    println(io, "Pooled over folds. Experiment 02 measured `r̂ ≈ 26.0–26.5` on this league ",
            "with a single-arm NegBin and TimeDecay state.\n")
    print(io, gjn_markdown_table(select(r02_summary,
        :model, :median_r, :mean_r, :r_q05, :r_q95);
        formats = Dict(:median_r => v -> gjn_num(v; digits = 2),
                       :mean_r => v -> gjn_num(v; digits = 2),
                       :r_q05 => v -> gjn_num(v; digits = 2),
                       :r_q95 => v -> gjn_num(v; digits = 2))))
end

println("\nR02_VERDICT ", all(r02_summary.gate_pass) ? "PASS" : "FAIL", "  (",
        count(r02_summary.gate_pass), "/", nrow(r02_summary), " models)")
println("R02_DONE report=", joinpath(R02_OUT_DIR, "r02_production_report_$(R02_STAMP).md"))
