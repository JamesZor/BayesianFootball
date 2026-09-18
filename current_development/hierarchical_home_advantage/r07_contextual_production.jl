# ==============================================================================
# r07 — 40-fold walk-forward production grid, contextual home advantage ladder
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The inference half of Task 008 Phase 2: the three m05 rungs sampled over the canonical
# Scottish Lower grid (pooled 56/57, target seasons 24/25 + 25/26, 40 match-biweek folds,
# 710 held-out fixtures) and persisted to `mcmc_experiments` under
# `scottish_lower_contextual_ha`. Proper scores live in r08, which loads these by UUID.
#
# COMPARABILITY. Same splitter as every Exp 06 control, so the 710 held-out fixtures are
# the identical set. Each rung differs from the persisted flat control `m05_joint_td_raw`
# in the HA slot and its contextual terms only (the HA priors are the work package's, not
# the control's N(0.2, 0.2) — disclosed in the README).
#
# GATES (per rung): R̂ ≤ 1.05 · ESS bulk/tail ≥ 400 · divergence rate < 0.1% · BFMI ≥ 0.30 ·
# tree-depth saturation < 5% · 40 folds · 710 OOS. A failing rung is not persisted; its
# checkpoints stay under `results/phase2/<model>/checkpoints/` and a rerun resumes.
#
# SIDE ARTEFACTS (written while fold features exist), under results/phase2/:
#   r07_coefficients_<model>.csv   every contextual weight, γ_base, σ_γ, per fold, vs prior
#   r07_sites_<model>.csv          worst R̂ / ESS per ha.* and contextual site
#   r07_design_<model>.csv         per-fold count of training fixtures switching each term on
#   r07_oos_design.csv             contextual design of every held-out fixture (for r08)
#   r07_unmapped_home_<model>.csv  T003 fixtures
#
# USAGE (mcmc-beast, from /root/BF_hier_ha)
#
#   julia --project -t 16 current_development/hierarchical_home_advantage/r07_contextual_production.jl
#   R07_MODELS=m05_joint_td_turf_asym  julia ... r07_contextual_production.jl
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

include(joinpath(@__DIR__, "l04_contextual_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R07_CONFIG = CtxConfig()
const R07_SELECTED = let raw = strip(get(ENV, "R07_MODELS", ""))
    isempty(raw) ? copy(CTX_MODEL_NAMES) : String.(strip.(split(raw, ",")))
end
all(in(CTX_MODEL_NAMES), R07_SELECTED) ||
    error("R07_MODELS names an unknown model: $(setdiff(R07_SELECTED, CTX_MODEL_NAMES))")
const R07_OUT_DIR = R07_CONFIG.save_root
const R07_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unsynced-rsync" end

println("\n" * "="^96)
println("  r07 PRODUCTION GRID — contextual home advantage ladder")
println("  models     : ", join(R07_SELECTED, ", "))
println("  split      : GroupedCVConfig 56/57, target ", join(R07_CONFIG.target_seasons, " + "),
        ", 2 history seasons, match-biweek")
println("  sampler    : QueuedNUTS  ", R07_CONFIG.chains, " chains × ", R07_CONFIG.warmup,
        " warmup + ", R07_CONFIG.samples, " retained, δ = ", R07_CONFIG.accept_rate,
        ", max depth ", R07_CONFIG.max_depth)
println("  experiment : ", R07_CONFIG.experiment, "   persist stride: ", R07_CONFIG.persist_stride)
println("  git        : ", R07_GIT, "   threads: ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Runtime, data, splits
# ===================================================================
mkpath(R07_OUT_DIR)
r07_db = gph_database(R07_CONFIG.experiment)
r07_ds = gph_load_data()
r07_splitter = gph_splitter(R07_CONFIG.target_seasons)
r07_boundaries = Data.create_id_boundaries(r07_ds, r07_splitter)
length(r07_boundaries) == R07_CONFIG.expected_folds || error(
    "splitter produced $(length(r07_boundaries)) folds; expected $(R07_CONFIG.expected_folds)")
println("\n  store: ", nrow(r07_ds.matches), " matches, latest ", maximum(r07_ds.matches.match_date),
        " | folds: ", length(r07_boundaries))

# %%
# ===================================================================
# 4. Models and config registration
# ===================================================================
r07_models = ctx_models()
r07_sampler = ctx_production_sampler(R07_CONFIG)
r07_configs = ctx_fit_configs(R07_CONFIG, r07_models, r07_splitter, r07_sampler)
r07_registry = ctx_register!(r07_db, r07_models, r07_splitter, r07_sampler, r07_configs)
println("  registered models: ", r07_registry.model_ids,
        " | splitter #", r07_registry.splitter_id, " | sampler #", r07_registry.sampler_id)

# %%
# ===================================================================
# 5. Per rung: config truth → features → training → audit → persist
# ===================================================================
r07_rows = NamedTuple[]
r07_oos_written = false

for (name, model) in r07_models
    name in R07_SELECTED || continue
    fit_config = r07_configs[name]
    println("\n" * "-"^96)
    println("MODEL ", name)
    println("-"^96)

    existing = gph_completed_run(r07_db, fit_config)
    if existing !== nothing
        println("  recipe already persisted as run ", existing, " — loading, not sampling")
        fit = load_fit(r07_db, existing)
        push!(r07_rows, (; hha_convergence_row(name, fit, ctx_as_hha(R07_CONFIG); run_id = existing)...,
                           max_site_rhat = NaN, n_unmapped_home = -1,
                           gate_pass = fit.diagnostics.passed, reused = true))
        println("R07_MODEL_DONE ", name, " reused ", existing)
        continue
    end

    inputs = gph_fold_inputs(r07_ds, r07_splitter, model)
    filtration = gph_filtration_report(r07_ds, inputs)
    CSV.write(joinpath(R07_OUT_DIR, "r07_filtration_$(name).csv"), filtration)
    all(filtration.ordered) || error("$name: training/OOS kickoff ordering violated")
    sum(filtration.n_oos) == R07_CONFIG.expected_oos || error(
        "$name: $(sum(filtration.n_oos)) OOS fixtures; expected $(R07_CONFIG.expected_oos)")

    CSV.write(joinpath(R07_OUT_DIR, "r07_design_$(name).csv"), ctx_design_summary(inputs))
    if !r07_oos_written
        oos_design = ctx_oos_design(inputs)
        CSV.write(joinpath(R07_OUT_DIR, "r07_oos_design.csv"), oos_design)
        @printf("  OOS design: %d fixtures | turf home %d | turf asym %d | midweek %d | rest≠0 %d\n",
                nrow(oos_design), sum(oos_design.turf_home), sum(oos_design.turf_asym),
                sum(oos_design.midweek), count(!iszero, oos_design.rest_diff))
        global r07_oos_written = true
    end
    unmapped = hha_unmapped_home_report(inputs)
    CSV.write(joinpath(R07_OUT_DIR, "r07_unmapped_home_$(name).csv"), unmapped)
    println("  held-out fixtures with an unmapped home club (T003): ", nrow(unmapped))

    checkpoint_dir = joinpath(R07_OUT_DIR, name, "checkpoints")
    fit = ctx_sample(fit_config, inputs, R07_CONFIG; checkpoint_dir)

    d = fit.diagnostics
    sites = ctx_site_report(fit)
    CSV.write(joinpath(R07_OUT_DIR, "r07_sites_$(name).csv"), sites)
    worst = sites[argmax(sites.max_rhat), :]
    @printf("  audit: R̂ %.4f (fold %d) | site R̂ %.4f at %s | ESS bulk %.0f tail %.0f | div %d/%d | depth %.2f%% | BFMI %.3f | %s\n",
            d.max_rhat, d.worst_rhat_fold, worst.max_rhat, worst.site,
            d.min_ess_bulk, d.min_ess_tail, d.n_divergent, d.n_transitions,
            100 * d.treedepth_rate, d.min_bfmi,
            d.passed ? "PASS" : "FAIL: " * join(d.failures, "; "))

    coefs = vcat([ctx_coefficients(fit, f) for f in eachindex(fit.folds)]...)
    CSV.write(joinpath(R07_OUT_DIR, "r07_coefficients_$(name).csv"), coefs)
    println("  final-fold coefficients:")
    show(stdout, MIME"text/plain"(),
         select(filter(:fold => ==(length(fit.folds)), coefs),
                :site, :mean, :sd, :q05, :q95, :p_positive, :prior_p_positive, :contraction);
         allrows = true)
    println()

    if !d.passed
        push!(r07_rows, (; hha_convergence_row(name, fit, ctx_as_hha(R07_CONFIG))...,
                           max_site_rhat = worst.max_rhat, n_unmapped_home = nrow(unmapped),
                           gate_pass = false, reused = false))
        println("R07_MODEL_FAIL ", name, " — not persisted; checkpoints kept in ", checkpoint_dir)
        continue
    end

    persisted = gph_thin_for_persistence(fit, inputs, R07_CONFIG.persist_stride)
    gph_assert_coverage(name, persisted; folds = R07_CONFIG.expected_folds,
                        oos = R07_CONFIG.expected_oos)
    latent = gph_latent_audit(persisted)
    @printf("  latents: %d fixtures × %d draws | mean λ_h %.3f λ_a %.3f\n",
            latent.n_matches, latent.n_draws, latent.mean_lambda_h, latent.mean_lambda_a)
    run_id = gph_save_and_verify(r07_db, persisted)
    println("  persisted and reloaded identically: run ", run_id)

    push!(r07_rows, (; hha_convergence_row(name, persisted, ctx_as_hha(R07_CONFIG); run_id)...,
                       max_site_rhat = worst.max_rhat, n_unmapped_home = nrow(unmapped),
                       gate_pass = true, reused = false))
    println("R07_MODEL_DONE ", name, " ", run_id)

    fit = nothing
    persisted = nothing
    inputs = nothing
    GC.gc()
end

# %%
# ===================================================================
# 6. Report
# ===================================================================
r07_summary = DataFrame(r07_rows)
CSV.write(joinpath(R07_OUT_DIR, "r07_production_runs.csv"), r07_summary)

open(joinpath(R07_OUT_DIR, "r07_production_report.md"), "w") do io
    println(io, "# r07 production grid — Task 008 Phase 2\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R07_GIT,
            "` on ", gethostname(), ". Namespace `", R07_CONFIG.experiment, "`.\n")
    println(io, "Sampler: QueuedNUTS ", R07_CONFIG.chains, " × (", R07_CONFIG.warmup, " warmup + ",
            R07_CONFIG.samples, " retained), δ = ", R07_CONFIG.accept_rate,
            ". Audit on all retained draws; artefact keeps every ", R07_CONFIG.persist_stride,
            "nd draw.\n")
    print(io, gph_markdown_table(select(r07_summary,
        :model, :folds, :oos, :draws, :max_rhat, :max_site_rhat, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :divergence_rate, :treedepth_rate, :min_bfmi, :strict_rhat_pass,
        :n_unmapped_home, :gate_pass, :wall_min, :run_id);
        formats = Dict(:divergence_rate => v -> @sprintf("%.5f", v),
                       :treedepth_rate => v -> @sprintf("%.4f", v),
                       :wall_min => v -> gph_num(v; digits = 1),
                       :min_ess_bulk => v -> gph_num(v; digits = 0),
                       :min_ess_tail => v -> gph_num(v; digits = 0))))
end

println("\nR07_VERDICT ", all(r07_summary.gate_pass) ? "PASS" : "FAIL", "  (",
        count(r07_summary.gate_pass), "/", nrow(r07_summary), " models)")
println("R07_DONE report=", joinpath(R07_OUT_DIR, "r07_production_report.md"))
