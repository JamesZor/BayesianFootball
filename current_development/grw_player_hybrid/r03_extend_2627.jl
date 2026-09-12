# ==============================================================================
# r03 — Extend the four ladder runs into 2026/27 (folds 41–43)
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# Rolls each persisted Task 013 run forward over the 2026/27 opening match-biweeks
# so the MatchDay T−25 backtest (`r06_t25_backtest.jl`) has a conditioning chain
# for every opening Saturday. Only the fold positions absent from `fold_results`
# are sampled; the 40 walk-forward folds are reloaded, never resampled. The run
# UUID is unchanged — `extend_fit` updates diagnostics, latents, the artefact and
# telemetry in one transaction.
#
# It does not score anything. The 24/25 + 25/26 proper scores in `r04` restrict
# every extended run back to its 710 walk-forward fixtures before scoring.
#
# FILTRATION CONTRACT
#
# The splitter is the r02 splitter with "26/27" appended; fold k of the extended
# run conditions on everything before match-biweek k of 26/27 and holds out
# biweek k. Folds 1–40 are byte-identical to r02 (asserted by fold count and by
# the untouched first 40 `fold_results` rows).
#
# SAMPLER
#
# `extend_fit` matches the persisted per-chain draw count: r02 persisted every
# 2nd of 1,000 retained draws, so new folds are sampled at 500 warmup + 500
# retained × 4 chains and every fold of the extended container holds 2,000 draws.
#
# USAGE (mcmc-beast, from /root/BF_grw_player_hybrid)
#
#   R03_PREVIEW=1 julia --project -t 16 current_development/grw_player_hybrid/r03_extend_2627.jl
#   julia --project -t 16 current_development/grw_player_hybrid/r03_extend_2627.jl
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
const R03_CONFIG = GPHConfig()
const R03_PREVIEW = get(ENV, "R03_PREVIEW", "0") == "1"
const R03_OUT_DIR = R03_CONFIG.save_root

println("\n" * "="^96)
println("  r03 EXTEND TO 2026/27 — ", R03_PREVIEW ? "PREVIEW ONLY" : "EXTEND (samples new folds)")
println("  experiment : ", R03_CONFIG.experiment)
println("  seasons    : ", join(R03_CONFIG.extension_seasons, " + "))
println("="^96)

# %%
# ===================================================================
# 3. Runtime, data and runs
# ===================================================================
r03_db = gph_database(R03_CONFIG.experiment)
r03_ds = gph_load_data()
r03_splitter = gph_splitter(R03_CONFIG.extension_seasons)

r03_runs = Dict{String,UUID}()
for name in GPH_MODEL_NAMES
    run_id = gph_run_by_name(r03_db, name)
    run_id === nothing && error("no completed r02 run named $name")
    r03_runs[name] = run_id
end

# %%
# ===================================================================
# 4. Extension plan
# ===================================================================
r03_plans = Dict{String,Any}()
for name in GPH_MODEL_NAMES
    plan = preview_extension(r03_db, string(r03_runs[name]), r03_ds; splitter = r03_splitter)
    r03_plans[name] = plan
    println("  ", rpad(name, 32), " run ", r03_runs[name], " → ", plan.new_count, " new fold(s)")
end
if R03_PREVIEW
    println("\nR03_DONE preview")
    exit(0)
end

# %%
# ===================================================================
# 5. Training — only the missing folds
# ===================================================================
r03_rows = NamedTuple[]
for name in GPH_MODEL_NAMES
    run_id = r03_runs[name]
    before = load_fit(r03_db, run_id)
    length(before.folds) in (R03_CONFIG.expected_folds, R03_CONFIG.expected_extended_folds) ||
        error("$name holds $(length(before.folds)) folds before extension")
    println("\n--- extending ", name)
    t0 = time()
    extended = gph_extend!(r03_db, run_id, r03_ds, R03_CONFIG)
    minutes = (time() - t0) / 60

    # --- 6. audit: coverage, latents, the delta folds' own convergence ------
    reloaded = load_fit(r03_db, run_id)
    length(reloaded.folds) == R03_CONFIG.expected_extended_folds || error(
        "$name has $(length(reloaded.folds)) folds after extension")
    kept = restrict_latents(reloaded.latents, Int.(before.latents.match_ids))
    order_kept = sortperm(kept.match_ids)
    order_before = sortperm(before.latents.match_ids)
    kept.λ_home[order_kept, :] == before.latents.λ_home[order_before, :] ||
        error("$name: extension altered the walk-forward latents")
    latent = gph_latent_audit(reloaded)
    new_folds = filter(f -> f.fold > R03_CONFIG.expected_folds, reloaded.diagnostics.folds)
    new_rhat = maximum(f.max_rhat for f in new_folds)
    new_ess = minimum(min(f.min_ess_bulk, f.min_ess_tail) for f in new_folds)
    new_div = sum(f.n_divergent for f in new_folds)
    @printf("  %-32s folds %d | OOS %d × %d draws | new folds: R̂ %.4f ESS %.0f div %d | %.1f min\n",
            name, length(reloaded.folds), latent.n_matches, latent.n_draws,
            new_rhat, new_ess, new_div, minutes)
    push!(r03_rows, (; model = name, run_id = string(run_id),
                       folds = length(reloaded.folds), oos = latent.n_matches,
                       oos_2627 = latent.n_matches - n_matches(before.latents),
                       draws = latent.n_draws,
                       new_fold_max_rhat = new_rhat, new_fold_min_ess = new_ess,
                       new_fold_divergences = new_div,
                       run_max_rhat = reloaded.diagnostics.max_rhat,
                       run_passed = reloaded.diagnostics.passed,
                       wall_min = minutes))
    println("R03_MODEL_DONE ", name)
end

# %%
# ===================================================================
# 7. Final report
# ===================================================================
r03_summary = DataFrame(r03_rows)
CSV.write(joinpath(R03_OUT_DIR, "r03_extension.csv"), r03_summary)
open(joinpath(R03_OUT_DIR, "r03_extension_report.md"), "w") do io
    println(io, "# r03 extension to 2026/27 — Task 013\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". Runs updated in place.\n")
    print(io, gph_markdown_table(r03_summary;
        formats = Dict(:wall_min => v -> gph_num(v; digits = 1),
                       :new_fold_min_ess => v -> gph_num(v; digits = 0))))
end
println("\nR03_DONE report=", joinpath(R03_OUT_DIR, "r03_extension_report.md"))
