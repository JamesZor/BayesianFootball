# ==============================================================================
# r02 — 43-fold walk-forward production grid, 1-parameter smile spine
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The inference half of Task 016: the two spine rungs sampled over the canonical Scottish Lower
# walk-forward grid and persisted to `mcmc_experiments` under `scottish_lower_grw_smile_spine`,
# plus the H1 benchmark against Task 015's five-strike smile. Proper scores are r04's, portfolios
# r06–r08's; nothing here reads a price for scoring.
#
#   m05_joint_grw_smile_spine_w020   C1 @ 0.20 + C2 spine @ 0.20     SAMPLED
#   m05_joint_grw_smile_spine_w040   C1 @ 0.40 + C2 spine @ 0.40     SAMPLED
#
# The other four rungs are NOT sampled. §6 loads each pinned run and asserts it is the recipe
# `gss_models()` builds, at the production budget, converged, over 43 folds — the config-truth
# protocol's "load, don't resample" — and records the Julia version, thread count and commit each
# was fitted with.
#
# QUESTIONS THIS RUNNER ANSWERS (and the ones it does not)
#
# H1 — wall time per rung and min bulk / tail ESS against Task 015's five-strike rungs at the same
#      weight, run-level and FOLD BY FOLD (both audited on all 4 × 1000 draws). The work package's
#      targets (≤ 90 min, min bulk ESS ≥ 600) are REPORTED against, not gated: the gate is the
#      six-part audit. Task 015's wall times were measured on the same host at -t 16; a comparison
#      is only fair if this report's header says the same.
# H2 — β_spine per fold, 43 folds.
# It does NOT answer H3–H5.
#
# FILTRATION / COMPARABILITY CONTRACT
#
# * Split: `gph_splitter(["24/25", "25/26", "26/27"])` — 43 folds. Folds 1–40 asserted
#   fixture-for-fixture identical to the 40-fold split, so the 710-fixture panel is embedded
#   unchanged. The DataStore must end before the 2026-09-12 card (Task 015's cache of 2026-09-12).
# * Every fold: no fixture both trained on and held out; last training kickoff before first
#   held-out kickoff. Both pillars read closing prices of TRAINING fixtures only.
# * Each run's held-out fixture set must EQUAL the pinned baseline's.
#
# GATES (per rung): R̂ ≤ 1.05 · bulk/tail ESS ≥ 400 · divergence rate < 0.1% · BFMI ≥ 0.30 ·
# tree-depth saturation < 5% · 43 folds · OOS set == baseline's · latents audited · persisted and
# reloaded identically · smile O/U = cdf(Poisson(λ_tot·φ)) ≤ 1e-12 · anti-diagonal reweighting
# (T011) on the persisted container: totals = smile CDF ≤ 1e-9, Σ = 1, uniform diagonals, φ ≡ 1
# shortcut bit-identical and un-shortcut path within the truncation-mass bound.
#
# ORDER CHANGE FROM TASK 015. A converged rung is PERSISTED BEFORE the pricing and reweighting
# gates run, and those gates record failures in the row rather than throwing. They check the
# pricer, not the posterior; an error there must not discard hours of converged sampling.
#
# PERSISTENCE CAVEAT
#
# The audit runs on every retained draw; the artefact keeps every `persist_stride`-th draw with
# latents re-extracted from exactly those draws. Spine rungs are persisted with the latent panel
# detached (T010) and a file copy under `results/latents/<run_id>/`. A rung that fails its
# convergence gate is NOT persisted; its per-fold checkpoints stay under
# `results/<model>/checkpoints_<budget>/` and a rerun AT THE SAME BUDGET resumes from them.
# Deserialising any artefact here requires `include("l01_loader.jl")`.
#
# USAGE (mcmc-beast, from /root/BF_grw_smile_spine, after r01 passes)
#
#   julia --project -t 16 current_development/grw_smile_spine/r02_production_grid.jl
#   R02_MODELS=m05_joint_grw_smile_spine_w040 julia ... r02_production_grid.jl
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
using Statistics

include(joinpath(@__DIR__, "l01_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R02_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = gss_config()
    gss_config(samples = env("R02_SAMPLES", base.samples),
               warmup = env("R02_WARMUP", base.warmup),
               chains = env("R02_CHAINS", base.chains),
               persist_stride = env("R02_STRIDE", base.persist_stride))
end
# The pinned rungs are checked against the CANONICAL production budget, never an override: a
# rung re-run at a larger budget does not change what the pinned runs are (Task 015 r02 §6).
const R02_CANONICAL = gss_config()
const R02_SELECTED = let raw = strip(get(ENV, "R02_MODELS", ""))
    isempty(raw) ? copy(GSS_GRID_MODEL_NAMES) : String.(strip.(split(raw, ",")))
end
all(in(GSS_GRID_MODEL_NAMES), R02_SELECTED) ||
    error("R02_MODELS names an unknown grid rung: $(setdiff(R02_SELECTED, GSS_GRID_MODEL_NAMES))")

const R02_H1_WALL_TARGET_MIN = 90.0
const R02_H1_ESS_TARGET = 600.0
const R02_PRICING_TOL = 1.0e-12
const R02_G4_TOL = 1.0e-9
const R02_SPREAD_TOL = 1.0e-12

# Stamped with the budget: `fit_model` resumes from whatever checkpoints it finds.
const R02_BUDGET_TAG = "$(R02_CONFIG.chains)x$(R02_CONFIG.warmup)w$(R02_CONFIG.samples)s"
const R02_OUT_DIR = R02_CONFIG.save_root
const R02_STAMP = R02_BUDGET_TAG * "_" * join(sort(R02_SELECTED), "+")
const R02_GIT = try
    readchomp(`git rev-parse --short HEAD`)
catch
    "unknown"
end

println("\n" * "="^96)
println("  r02 PRODUCTION GRID — 1-parameter smile spine on market-anchored MultiScaleGRW")
println("  rungs      : ", join(R02_SELECTED, ", "))
println("  split      : GroupedCVConfig 56/57, target ", join(R02_CONFIG.extension_seasons, " + "),
        ", 2 history seasons, match-biweek")
println("  sampler    : QueuedNUTS  ", R02_CONFIG.chains, " chains × ", R02_CONFIG.warmup,
        " warmup + ", R02_CONFIG.samples, " retained, δ = ", R02_CONFIG.accept_rate,
        ", max depth ", R02_CONFIG.max_depth)
println("  queue      : ", R02_CONFIG.max_concurrent_tasks, " concurrent fold×chain tasks on ",
        Threads.nthreads(), " threads (BLAS pinned to 1)")
println("  experiment : ", R02_CONFIG.experiment, "   persist stride: ", R02_CONFIG.persist_stride)
println("  git        : ", R02_GIT, "   julia ", VERSION, "   host ", gethostname())
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
r02_ladder = gss_models()
r02_models = gms_select(r02_ladder, R02_SELECTED)
r02_by_name = Dict{String,Any}(name => model for (name, model) in r02_ladder)
r02_sampler = gms_production_sampler(R02_CONFIG)
r02_configs = gss_fit_configs(R02_CONFIG, r02_ladder, r02_splitter, r02_sampler)
r02_registry = gss_register!(r02_db, r02_ladder, r02_splitter, r02_sampler, r02_configs)
println("  registered models: ", r02_registry.model_ids,
        " | splitter #", r02_registry.splitter_id, " | sampler #", r02_registry.sampler_id)

# %%
# ===================================================================
# 6. The pinned rungs: load, prove identical, record provenance
# ===================================================================
# Before any sampling: a pinned run that is not the recipe, does not deserialise, or did not
# converge is found here rather than in r04 after hours of compute.
r02_pinned_rows = NamedTuple[]
r02_pinned_folds = Dict{String,DataFrame}()
r02_baseline_oos = nothing
for pinned in GSS_PINNED_RUNS
    check = gss_check_pinned_run(pinned, r02_by_name[pinned.rung], R02_CANONICAL)
    push!(r02_pinned_rows, check.summary)
    r02_pinned_folds[pinned.rung] = check.folds
    if pinned.rung == "m05_joint_grw_baseline"
        global r02_baseline_oos = check.oos
    elseif check.oos !== nothing
        check.oos == r02_baseline_oos || error(
            "pinned $(pinned.rung) holds a different held-out fixture set from the baseline")
    end
    s = check.summary
    @printf("  pinned %-36s %s  folds %d  R̂ %.4f  ESS %.0f/%.0f  wall %.0f min  julia %s  threads %d  latents %s\n",
            s.rung, s.run_id[1:8], s.folds, s.max_rhat, s.min_ess_bulk, s.min_ess_tail, s.wall_min,
            s.julia_version, s.n_threads, s.latents)
    GC.gc()
end
r02_baseline_oos === nothing && error("the pinned baseline reloaded without latents; no OOS set to compare against")
r02_pinned_frame = DataFrame(r02_pinned_rows)
CSV.write(joinpath(R02_OUT_DIR, "r02_pinned_runs.csv"), r02_pinned_frame)
println("  every pinned rung is the rebuilt recipe, converged, 43 folds; baseline OOS fixtures: ",
        length(r02_baseline_oos))

# Every arm must persist the same number of draws per fixture, whatever it was sampled at.
R02_CONFIG.chains * R02_CONFIG.samples ÷ R02_CONFIG.persist_stride ==
    R02_CANONICAL.chains * R02_CANONICAL.samples ÷ R02_CANONICAL.persist_stride ||
    error("this invocation would persist a different draw count from the pinned rungs — adjust R02_STRIDE")

# %%
# ===================================================================
# 7–12. Per rung: config truth → features → training → audit → persist → pricer gates
# ===================================================================
r02_rows = NamedTuple[]
r02_spine_folds = Dict{String,DataFrame}()
r02_beta = DataFrame[]

for (name, model) in r02_models
    fit_config = r02_configs[name]
    println("\n" * "-"^96)
    println("RUNG ", name, "   supremacy: ", nameof(typeof(model.supremacy)),
            @sprintf("   spine: w=%.2f β~%s pivot=%d", model.smile.weight, model.smile.β_prior, model.smile.pivot),
            "   family: ", nameof(typeof(latent_family(model))))
    println("-"^96)

    # --- 7. config truth: never resample a persisted recipe --------------------
    existing = gph_completed_run(r02_db, fit_config)
    if existing !== nothing
        println("  recipe already persisted as run ", existing, " — not sampling")
        fit = load_fit(r02_db, existing)
        r02_spine_folds[name] = gss_fold_convergence_frame(name, fit)
        push!(r02_rows, (; gss_convergence_row(name, fit, R02_CONFIG; run_id = existing)...,
                           gate_pass = fit.diagnostics.passed, pricer_failures = "not re-run (reused)",
                           reused = true))
        println("R02_MODEL_DONE ", name, " reused ", existing)
        continue
    end

    # --- 8. features, filtration, market coverage -----------------------------
    inputs = gph_fold_inputs(r02_ds, r02_splitter, model)
    filtration = gph_filtration_report(r02_ds, inputs)
    CSV.write(joinpath(R02_OUT_DIR, "r02_filtration_$(name).csv"), filtration)
    all(filtration.ordered) || error("$name: training/OOS kickoff ordering violated")
    coverage = gss_market_coverage(model, inputs)
    CSV.write(joinpath(R02_OUT_DIR, "r02_market_coverage_$(name).csv"), coverage)
    all(coverage.supremacy_observed .> 0) || error("$name: a fold's supremacy pillar reads no match")
    all(coverage.smile_matches .> 0) || error("$name: a fold's spine pillar reads no match")
    @printf("  folds %d | train %d–%d | OOS %d | target steps 0–%d | supremacy share %.2f–%.2f | smile share %.2f–%.2f\n",
            nrow(filtration), minimum(filtration.n_train), maximum(filtration.n_train),
            sum(filtration.n_oos), maximum(filtration.n_target),
            minimum(coverage.supremacy_share), maximum(coverage.supremacy_share),
            minimum(coverage.smile_share), maximum(coverage.smile_share))

    # --- 9. training -------------------------------------------------------------
    # NUTS chains are single-threaded; QueuedExecution flattens 43 folds × 4 chains into one
    # 172-task queue over the 16 pinned threads.
    checkpoint_dir = joinpath(R02_OUT_DIR, name, "checkpoints_" * R02_BUDGET_TAG)
    fit = gph_sample(fit_config, inputs, gms_gph_config(R02_CONFIG); checkpoint_dir)

    # --- 10. convergence (six-part audit on every retained draw) and β per fold ---
    d = fit.diagnostics
    pillars = gss_pillar_summary(fit)
    r02_spine_folds[name] = gss_fold_convergence_frame(name, fit)
    beta = gss_beta_by_fold(fit)
    CSV.write(joinpath(R02_OUT_DIR, "r02_beta_by_fold_$(name).csv"), beta)
    push!(r02_beta, insertcols(beta, 1, :model => name))
    @printf("  audit: R̂ %.4f (fold %d) | ESS bulk %.0f tail %.0f | div %d/%d | depth %.2f%% | BFMI %.3f | wall %.1f min | %s\n",
            d.max_rhat, d.worst_rhat_fold, d.min_ess_bulk, d.min_ess_tail,
            d.n_divergent, d.n_transitions, 100 * d.treedepth_rate, d.min_bfmi,
            fit.metadata.elapsed_seconds / 60, d.passed ? "PASS" : "FAIL: " * join(d.failures, "; "))
    @printf("  pillars: σ_sup %.3f | σ_smile %.3f [%.3f, %.3f] | κ %.3f | β %.4f [%.4f, %.4f] | fold β medians %.4f–%.4f | φ %s\n",
            pillars.σ_sup_median, pillars.σ_smile_median, pillars.σ_smile_q05, pillars.σ_smile_q95,
            pillars.κ_median, pillars.β_median, pillars.β_q05, pillars.β_q95,
            minimum(beta.β_median), maximum(beta.β_median), pillars.φ_median)

    if !d.passed
        push!(r02_rows, (; gss_convergence_row(name, fit, R02_CONFIG)..., gate_pass = false,
                           pricer_failures = "not run (convergence gate failed)", reused = false))
        println("R02_MODEL_FAIL ", name, " — not persisted; checkpoints kept in ", checkpoint_dir)
        continue
    end

    # --- 11. thin, audit latents, persist, verify round-trip -------------------
    persisted = gph_thin_for_persistence(fit, inputs, R02_CONFIG.persist_stride)
    length(persisted.folds) == R02_CONFIG.expected_extended_folds || error(
        "$name has $(length(persisted.folds)) folds; expected $(R02_CONFIG.expected_extended_folds)")
    Set(Int.(persisted.latents.match_ids)) == r02_baseline_oos || error(
        "$name holds a different held-out fixture set from the baseline control — not comparable")
    latent = gms_latent_audit(persisted)
    @printf("  latents: %s | %d fixtures × %d draws | mean λ_h %.3f λ_a %.3f | φ mean %s\n",
            latent.family, latent.n_matches, latent.n_draws, latent.mean_lambda_h,
            latent.mean_lambda_a, latent.φ_mean)

    run_id = gms_save_and_verify(r02_db, persisted, inputs;
                                 latent_dir = joinpath(R02_OUT_DIR, "latents"))
    println("  persisted and reloaded identically: run ", run_id)

    # --- 12. pricer gates on the persisted container (recorded, not thrown) -----
    pricer_failures = String[]
    try
        pricing = gms_smile_pricing_gate(persisted; n_fixtures = 12)
        CSV.write(joinpath(R02_OUT_DIR, "r02_smile_pricing_$(name).csv"), pricing)
        maximum(abs.(pricing.p_under_typed .- pricing.p_under_ref)) <= R02_PRICING_TOL ||
            push!(pricer_failures, "typed smile O/U ≠ cdf(Poisson(λ_tot·φ))")
        maximum(abs.(pricing.p_under_legacy .- pricing.p_under_ref)) <= R02_PRICING_TOL ||
            push!(pricer_failures, "legacy row route smile O/U ≠ cdf(Poisson(λ_tot·φ))")

        reweight = gss_reweight_gate(persisted.latents)
        CSV.write(joinpath(R02_OUT_DIR, "r02_reweight_$(name).csv"), reweight)
        append!(pricer_failures, gss_reweight_failures(reweight; tol = R02_G4_TOL, spread_tol = R02_SPREAD_TOL))

        identity = gss_identity_path_gate(persisted.latents)
        identity.shortcut_bit_identical || push!(pricer_failures, "φ ≡ 1 shortcut changed the grid")
        identity.forced_within_truncation || push!(pricer_failures,
            @sprintf("un-shortcut φ ≡ 1 path exceeds the truncation-mass bound by %.2e",
                     identity.worst_excess_over_truncation))
        @printf("  pricer: %d fixtures | totals gap %.2e | mass gap %.2e | Δp_home %+.5f Δp_draw %+.5f Δp_away %+.5f | forced φ≡1 %.2e (trunc %.2e) | %s\n",
                nrow(reweight), maximum(reweight.max_draw_cdf_gap), maximum(reweight.max_mass_gap),
                mean(reweight.Δp_home), mean(reweight.Δp_draw), mean(reweight.Δp_away),
                identity.forced_max_abs_gap, identity.max_truncation_mass,
                isempty(pricer_failures) ? "PASS" : "FAIL: " * join(pricer_failures, "; "))
    catch err
        push!(pricer_failures, "pricer gate errored: " * sprint(showerror, err))
        println("  pricer gate ERRORED (run is persisted): ", first(pricer_failures))
    end

    push!(r02_rows, (; gss_convergence_row(name, persisted, R02_CONFIG; run_id)...,
                       gate_pass = isempty(pricer_failures), pricer_failures = join(pricer_failures, "; "),
                       reused = false))
    println(isempty(pricer_failures) ? "R02_MODEL_DONE " : "R02_MODEL_PRICER_FAIL ", name, " ", run_id)

    fit = nothing
    persisted = nothing
    inputs = nothing
    GC.gc()
end

# %%
# ===================================================================
# 13. Benchmark against Task 015 (H1) and β recovery (H2)
# ===================================================================
r02_summary = DataFrame(r02_rows)

# Run level: pinned rungs as recorded in their artefacts, beside the spine rungs.
r02_benchmark = vcat(
    select(r02_pinned_frame, :rung => :model, :source, :max_rhat, :min_ess_bulk, :min_ess_tail,
           :n_divergent, :wall_min),
    DataFrame([(; model = r.model, source = "Task 016 (this run)", max_rhat = r.max_rhat,
                  min_ess_bulk = r.min_ess_bulk, min_ess_tail = r.min_ess_tail,
                  n_divergent = r.n_divergent, wall_min = r.wall_min) for r in eachrow(r02_summary)]);
    cols = :union)
r02_readme_wall = Dict(r.rung => r.wall_min for r in GSS_TASK015_PRODUCTION)
r02_benchmark.readme_wall_min = [get(r02_readme_wall, m, missing) for m in r02_benchmark.model]
CSV.write(joinpath(R02_OUT_DIR, "r02_benchmark_$(R02_STAMP).csv"), r02_benchmark)

# Fold level: each spine rung against the five-strike rung at the same weight.
r02_fold_pairs = DataFrame[]
r02_pair_rows = NamedTuple[]
for (spine_name, five_name) in GSS_LINE_PAIRS
    haskey(r02_spine_folds, spine_name) || continue
    pair = gss_fold_benchmark(r02_spine_folds[spine_name], r02_pinned_folds[five_name])
    push!(r02_fold_pairs, insertcols(pair, 1, :spine => spine_name, :reference => five_name))
    push!(r02_pair_rows, (; spine = spine_name, reference = five_name, folds = nrow(pair),
                            median_bulk_ratio = median(pair.bulk_ratio),
                            min_bulk_ratio = minimum(pair.bulk_ratio),
                            median_tail_ratio = median(pair.tail_ratio),
                            folds_spine_bulk_below_reference = count(pair.bulk_ratio .< 1.0),
                            spine_min_bulk = minimum(pair.spine_ess_bulk),
                            reference_min_bulk = minimum(pair.reference_ess_bulk)))
end
r02_fold_frame = isempty(r02_fold_pairs) ? DataFrame() : vcat(r02_fold_pairs...)
r02_pair_frame = DataFrame(r02_pair_rows)
nrow(r02_fold_frame) > 0 && CSV.write(joinpath(R02_OUT_DIR, "r02_fold_benchmark_$(R02_STAMP).csv"), r02_fold_frame)
r02_beta_frame = isempty(r02_beta) ? DataFrame() : vcat(r02_beta...)
r02_line = gss_task015_line_fit()

for r in eachrow(r02_summary)
    @printf("  H1 %-34s wall %.1f min (target ≤ %.0f) | min bulk ESS %.0f (target ≥ %.0f) | tail %.0f\n",
            r.model, r.wall_min, R02_H1_WALL_TARGET_MIN, r.min_ess_bulk, R02_H1_ESS_TARGET, r.min_ess_tail)
end
for r in eachrow(r02_pair_frame)
    @printf("  H1 per fold %s vs %s: median bulk ratio %.2f (min %.2f), spine below reference on %d/%d folds\n",
            r.spine, r.reference, r.median_bulk_ratio, r.min_bulk_ratio,
            r.folds_spine_bulk_below_reference, r.folds)
end

# %%
# ===================================================================
# 14. Final report
# ===================================================================
CSV.write(joinpath(R02_OUT_DIR, "r02_production_runs_$(R02_STAMP).csv"), r02_summary)

open(joinpath(R02_OUT_DIR, "r02_production_report_$(R02_STAMP).md"), "w") do io
    println(io, "# r02 production grid — Task 016 (1-parameter smile spine)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R02_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads, Julia ", VERSION,
            ". Namespace `", R02_CONFIG.experiment, "`. Store latest kickoff ",
            maximum(r02_ds.matches.match_date), "; ", r02_n_folds, " folds; ",
            length(r02_baseline_oos), " held-out fixtures.\n")
    println(io, "Sampler: QueuedNUTS ", R02_CONFIG.chains, " × (", R02_CONFIG.warmup, " warmup + ",
            R02_CONFIG.samples, " retained), δ = ", R02_CONFIG.accept_rate,
            ". Audit on all retained draws; artefact keeps every ", R02_CONFIG.persist_stride,
            "nd draw.\n")

    println(io, "## Pinned rungs (loaded, not sampled)\n")
    println(io, "Each is asserted to be the recipe `gss_models()` builds, converged, 43 folds, at the production budget. ",
            "The baseline's per-fold audit was re-run on thinned chains by `extend_fit`; its ESS is not comparable to the others'.\n")
    print(io, gph_markdown_table(r02_pinned_frame;
        formats = Dict(:max_rhat => v -> gph_num(v; digits = 4),
                       :min_ess_bulk => v -> gph_num(v; digits = 0),
                       :min_ess_tail => v -> gph_num(v; digits = 0),
                       :min_bfmi => v -> gph_num(v; digits = 3),
                       :wall_min => v -> gph_num(v; digits = 1))))

    println(io, "\n## Convergence and persistence — spine rungs\n")
    print(io, gph_markdown_table(select(r02_summary,
        :model, :folds, :oos, :draws, :max_rhat, :worst_rhat_fold, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :divergence_rate, :treedepth_rate, :min_bfmi, :strict_rhat_pass,
        :passed, :gate_pass, :wall_min, :run_id, :reused);
        formats = Dict(:divergence_rate => v -> @sprintf("%.5f", v),
                       :treedepth_rate => v -> @sprintf("%.4f", v),
                       :wall_min => v -> gph_num(v; digits = 1),
                       :min_ess_bulk => v -> gph_num(v; digits = 0),
                       :min_ess_tail => v -> gph_num(v; digits = 0))))
    failed = filter(r -> !r.gate_pass, r02_summary)
    for r in eachrow(failed)
        println(io, "\n* `", r.model, "` — ", r.failures, isempty(r.failures) ? "" : "; ", r.pricer_failures)
    end

    println(io, "\n## H1 — benchmark against Task 015\n")
    println(io, @sprintf("Work-package targets, reported not gated: wall ≤ %.0f min per rung, min bulk ESS ≥ %.0f. ",
                         R02_H1_WALL_TARGET_MIN, R02_H1_ESS_TARGET),
            "`wall_min` is read from each artefact's metadata; `readme_wall_min` is Task 015's README figure.\n")
    print(io, gph_markdown_table(r02_benchmark;
        formats = Dict(:max_rhat => v -> gph_num(v; digits = 4),
                       :min_ess_bulk => v -> gph_num(v; digits = 0),
                       :min_ess_tail => v -> gph_num(v; digits = 0),
                       :wall_min => v -> gph_num(v; digits = 1),
                       :readme_wall_min => v -> ismissing(v) ? "—" : gph_num(v; digits = 0))))
    if nrow(r02_pair_frame) > 0
        println(io, "\n### Fold by fold, against the five-strike rung at the same weight\n")
        print(io, gph_markdown_table(r02_pair_frame;
            formats = Dict(:median_bulk_ratio => v -> gph_num(v; digits = 2),
                           :min_bulk_ratio => v -> gph_num(v; digits = 2),
                           :median_tail_ratio => v -> gph_num(v; digits = 2),
                           :spine_min_bulk => v -> gph_num(v; digits = 0),
                           :reference_min_bulk => v -> gph_num(v; digits = 0))))
        println(io, "\nPer-fold rows: `r02_fold_benchmark_", R02_STAMP, ".csv`.")
    end

    println(io, "\n## H2 — pillar posteriors and β_spine\n")
    println(io, @sprintf("Task 015's five-strike medians imply β_LS = %.4f, with line residuals %s at K = 0…4.\n",
                         r02_line.β_least_squares, join(round.(r02_line.residuals; digits = 3), " / ")))
    print(io, gph_markdown_table(select(r02_summary,
        :model, :σ_sup_median, :σ_smile_median, :σ_smile_q05, :σ_smile_q95, :κ_median,
        :β_median, :β_q05, :β_q95, :β_sd, :φ_median);
        formats = Dict(:σ_sup_median => v -> gph_num(v; digits = 3),
                       :σ_smile_median => v -> gph_num(v; digits = 3),
                       :σ_smile_q05 => v -> gph_num(v; digits = 3),
                       :σ_smile_q95 => v -> gph_num(v; digits = 3),
                       :κ_median => v -> gph_num(v; digits = 3),
                       :β_median => v -> gph_num(v; digits = 4), :β_q05 => v -> gph_num(v; digits = 4),
                       :β_q95 => v -> gph_num(v; digits = 4), :β_sd => v -> gph_num(v; digits = 4))))
    if nrow(r02_beta_frame) > 0
        beta_summary = combine(groupby(r02_beta_frame, :model),
                               :β_median => minimum => :fold_β_median_min,
                               :β_median => maximum => :fold_β_median_max,
                               :β_sd => maximum => :max_fold_β_sd,
                               :rhat => maximum => :max_β_rhat,
                               :ess_bulk => minimum => :min_β_ess_bulk)
        println(io)
        print(io, gph_markdown_table(beta_summary;
            formats = Dict(:fold_β_median_min => v -> gph_num(v; digits = 4),
                           :fold_β_median_max => v -> gph_num(v; digits = 4),
                           :max_fold_β_sd => v -> gph_num(v; digits = 4),
                           :max_β_rhat => v -> gph_num(v; digits = 4),
                           :min_β_ess_bulk => v -> gph_num(v; digits = 0))))
        println(io, "\nPer-fold rows: `r02_beta_by_fold_<model>.csv`.")
    end
end

println("\nR02_VERDICT ", all(r02_summary.gate_pass) ? "PASS" : "FAIL", "  (",
        count(r02_summary.gate_pass), "/", nrow(r02_summary), " rungs)")
println("R02_DONE report=", joinpath(R02_OUT_DIR, "r02_production_report_$(R02_STAMP).md"))
