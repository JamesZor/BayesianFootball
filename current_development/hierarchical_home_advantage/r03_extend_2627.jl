# ==============================================================================
# r03 — Extend the TimeDecay hierarchical-HA hybrid into 2026/27 (folds 41–43)
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# Rolls `m12_joint_hybrid_synergy_hier_ha` forward over the 2026/27 opening
# match-biweeks so `r05_slate_repricing.jl` has a Fold-43 posterior for the
# 2026-09-12 card — the same fold Run 67 (the flat production twin) priced that card
# from. Only fold positions absent from `fold_results` are sampled; folds 1–40 are
# reloaded, never resampled, and the run UUID is unchanged.
#
# Only this candidate is extended by default: it is the one the work package names
# for re-pricing and the one with a live flat twin. `R03_MODELS` overrides.
#
# It does not score anything. `r04` restricts every run back to its 710 walk-forward
# fixtures before scoring.
#
# FILTRATION CONTRACT
#
# The splitter is r02's with "26/27" appended; fold k of the extension conditions on
# everything before match-biweek k of 26/27 and holds out biweek k. The runner
# asserts the 710 walk-forward latents are bit-identical before and after.
#
# DATA SNAPSHOT CAVEAT
#
# The DataStore holds PLAYED matches, so the unplayed 2026-09-12 card is not in it and
# does not need to be: 43 folds exist on data through 2026-09-05, and `select_split`
# serves an unplayed card from the last fold through its `exclude` path — exactly how
# Run 67 priced it live. A first version of this runner refused to extend unless the
# card's rows were in the store; that guard was wrong and is now a printed count.
#
# SAMPLER
#
# `extend_fit` matches the persisted per-chain draw count: r02 persisted every 2nd of
# 1,000 retained draws, so new folds are sampled at 500 warmup + 500 retained × 4.
#
# USAGE (mcmc-beast, from /root/BF_hier_ha_slate, after r02)
#
#   R03_PREVIEW=1 julia --project -t 16 current_development/hierarchical_home_advantage/r03_extend_2627.jl
#   julia --project -t 16 current_development/hierarchical_home_advantage/r03_extend_2627.jl
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
const R03_CONFIG = HHAConfig()
const R03_PREVIEW = get(ENV, "R03_PREVIEW", "0") == "1"
const R03_SELECTED = let raw = strip(get(ENV, "R03_MODELS", "m12_joint_hybrid_synergy_hier_ha"))
    String.(strip.(split(raw, ",")))
end
all(in(HHA_MODEL_NAMES), R03_SELECTED) ||
    error("R03_MODELS names an unknown model: $(setdiff(R03_SELECTED, HHA_MODEL_NAMES))")
const R03_EXPECTED_EXTENDED_FOLDS = 43
const R03_REQUIRED_FIXTURE_DATE = Date(2026, 9, 12)
const R03_OUT_DIR = R03_CONFIG.save_root

println("\n" * "="^96)
println("  r03 EXTEND TO 2026/27 — ", R03_PREVIEW ? "PREVIEW ONLY" : "EXTEND (samples new folds)")
println("  experiment : ", R03_CONFIG.experiment)
println("  models     : ", join(R03_SELECTED, ", "))
println("  seasons    : ", join(R03_CONFIG.extension_seasons, " + "))
println("="^96)

# %%
# ===================================================================
# 3. Runtime, data and runs
# ===================================================================
mkpath(R03_OUT_DIR)
r03_db = gph_database(R03_CONFIG.experiment)
r03_ds = gph_load_data()
r03_splitter = gph_splitter(R03_CONFIG.extension_seasons)

r03_slate_rows = count(==(R03_REQUIRED_FIXTURE_DATE), Date.(r03_ds.matches.match_date))
# Informational only. The DataStore holds played matches; an unplayed card is served by
# `select_split`'s `exclude` path from the last fold, exactly as Run 67 served it live.
# What r05 needs is 43 folds, which section 6 asserts.
println("  store: ", nrow(r03_ds.matches), " matches, latest ", maximum(r03_ds.matches.match_date),
        " | rows dated ", R03_REQUIRED_FIXTURE_DATE, ": ", r03_slate_rows)

r03_runs = Dict{String,UUID}()
for name in R03_SELECTED
    run_id = gph_run_by_name(r03_db, name)
    run_id === nothing && error("no completed r02 run named $name")
    r03_runs[name] = run_id
end

# %%
# ===================================================================
# 4. Extension plan
# ===================================================================
for name in R03_SELECTED
    plan = preview_extension(r03_db, string(r03_runs[name]), r03_ds; splitter = r03_splitter)
    println("  ", rpad(name, 38), " run ", r03_runs[name], " → ", plan.new_count, " new fold(s)")
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
for name in R03_SELECTED
    run_id = r03_runs[name]
    before = load_fit(r03_db, run_id)
    length(before.folds) in (R03_CONFIG.expected_folds, R03_EXPECTED_EXTENDED_FOLDS) ||
        error("$name holds $(length(before.folds)) folds before extension")
    println("\n--- extending ", name)
    t0 = time()
    extend_fit(r03_db, string(run_id), r03_ds;
               splitter = r03_splitter,
               execution = hha_execution(R03_CONFIG))
    minutes = (time() - t0) / 60

    # --- 6. audit: coverage, walk-forward latents unchanged, delta folds converged
    reloaded = load_fit(r03_db, run_id)
    length(reloaded.folds) == R03_EXPECTED_EXTENDED_FOLDS || error(
        "$name has $(length(reloaded.folds)) folds after extension; expected $R03_EXPECTED_EXTENDED_FOLDS")
    kept = restrict_latents(reloaded.latents, Int.(before.latents.match_ids))
    order_kept = sortperm(kept.match_ids)
    order_before = sortperm(before.latents.match_ids)
    kept.λ_home[order_kept, :] == before.latents.λ_home[order_before, :] ||
        error("$name: extension altered the walk-forward latents")
    kept.λ_away[order_kept, :] == before.latents.λ_away[order_before, :] ||
        error("$name: extension altered the walk-forward latents")
    latent = gph_latent_audit(reloaded)
    new_folds = filter(f -> f.fold > R03_CONFIG.expected_folds, reloaded.diagnostics.folds)
    new_rhat = maximum(f.max_rhat for f in new_folds)
    new_ess = minimum(min(f.min_ess_bulk, f.min_ess_tail) for f in new_folds)
    new_div = sum(f.n_divergent for f in new_folds)
    @printf("  %-38s folds %d | OOS %d × %d draws | new folds: R̂ %.4f ESS %.0f div %d | %.1f min\n",
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
    println(io, "# r03 extension to 2026/27 — Task 008 Phase 1\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". Runs updated in place.\n")
    print(io, gph_markdown_table(r03_summary;
        formats = Dict(:wall_min => v -> gph_num(v; digits = 1),
                       :new_fold_min_ess => v -> gph_num(v; digits = 0))))
end
println("\nR03_DONE report=", joinpath(R03_OUT_DIR, "r03_extension_report.md"))
