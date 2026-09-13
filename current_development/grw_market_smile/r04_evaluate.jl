# ==============================================================================
# r04 — Out-of-sample proper scores and paired bootstrap, 24/25 + 25/26
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A predictive-fit comparison of the market-anchored rungs against the pure GRW baseline and
# the Betfair closing line. Not a betting study.
#
# HYPOTHESES, fixed before scoring, written so they can fail visibly:
#
#   H1  The supremacy pillar improves 1X2 LogLoss over the baseline
#       (refuted if the paired Δ interval for supremacy_w040 − baseline on 1X2 includes 0
#       or is positive).
#   H2  The smile adds to supremacy on totals
#       (refuted if smile_w040 − supremacy_w040 on OU2.5 includes 0 or is positive).
#   H3  The home-favourite compression shrinks: in the market p_home ≥ 0.50 bins the
#       anchored rungs' mean p_model − p_market is closer to 0 than the baseline's.
#   H4  There is a best weight: one of 0.20 / 0.70 beats 0.40 with an interval excluding 0.
#
# The prior that must be argued against: Experiment 06 found the Betfair close BEATS every
# football model on this panel (0.64182 vs 0.64315 for this baseline). A model pulled toward
# the close can therefore gain LogLoss merely by becoming more like the benchmark — so every
# arm is also reported against the close itself, and "beats the baseline" is not read as
# "beats the market".
#
# COMPARABILITY CONTRACT
#
# * One panel: fixtures of 24/25 + 25/26 every arm holds a latent for; must be exactly 710.
#   The 43-fold runs are restricted to it BEFORE scoring.
# * One book: de-vigged Betfair TWA(−20, 0] close.
# * Reproduction gate: the baseline control must reproduce Task 013's published
#   LogLoss 0.64315 / ECE 0.0123 on 1X2 + O/U 2.5 + BTTS (2,899 rows) before any contrast.
# * 10,000 fixture-clustered paired resamples; LogLoss AND Brier.
#
# USAGE (mcmc-beast, from /root/BF_grw_market_smile, after r02)
#
#   julia --project -t 16 current_development/grw_market_smile/r04_evaluate.jl
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
include(joinpath(@__DIR__, "l02_evaluation.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R04_CONFIG = GMSConfig()
const R04_BOOTSTRAP_B = 10_000
const R04_SEED = 20260915
const R04_OUT_DIR = joinpath(R04_CONFIG.save_root, "evaluation")
const R04_LATENT_DIR = joinpath(R04_CONFIG.save_root, "latents")

mkpath(R04_OUT_DIR)
println("\n" * "="^96)
println("  r04 PROPER SCORES — market-anchored GRW vs pure GRW baseline and the Betfair close")
println("  bootstrap  : ", R04_BOOTSTRAP_B, " fixture-clustered paired resamples, seed ", R04_SEED)
println("  scopes     : all (primary), ", join(GMS_SCOPES, ", "))
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot, the book and the arms
# ===================================================================
r04_ds = gph_load_data()
r04_splitter = gph_splitter(R04_CONFIG.extension_seasons)
r04_odds = gms_betfair_closing_odds(r04_ds)
r04_families = gms_family_selections(r04_odds)
println("  Betfair close: ", nrow(r04_odds), " rows over ", length(unique(r04_odds.match_id)), " fixtures")

r04_arms = gms_arms(R04_CONFIG)
r04_raw = Dict{String,Any}()
for arm in r04_arms
    t0 = time()
    r04_raw[arm.label] = gms_load_arm(arm, r04_ds; splitter = r04_splitter, latent_dir = R04_LATENT_DIR)
    @printf("  loaded %-36s run %s  %s  %.0f s\n", arm.label, arm.run_id,
            nameof(typeof(r04_raw[arm.label].latents)), time() - t0)
end

r04_panel_ids = gms_common_panel(r04_ds, r04_raw, R04_CONFIG.target_seasons)
length(r04_panel_ids) == R04_CONFIG.expected_oos || error(
    "common panel is $(length(r04_panel_ids)) fixtures; expected $(R04_CONFIG.expected_oos)")
r04_fits = Dict(label => gms_restrict(fit, r04_panel_ids) for (label, fit) in r04_raw)
r04_raw = nothing
GC.gc()

# %%
# ===================================================================
# 4. Reproduction gate, then proper scores per arm
# ===================================================================
r04_ctx = Dict{String,Any}()
r04_ctx2 = Dict{String,Any}()
for arm in r04_arms
    r04_ctx[arm.label] = gms_context(r04_fits[arm.label], r04_odds, r04_ds; markets = GMS_PRIMARY_MARKETS)
    r04_ctx2[arm.label] = gms_context(r04_fits[arm.label], r04_odds, r04_ds; markets = GMS_SECONDARY_MARKETS)
end

r04_repro = evaluate_predictions(r04_ctx[GMS_BASELINE_CONTROL.label]; n_bins = 10)
@printf("\n  reproduction: baseline LogLoss %.5f (published %.5f)  ECE %.4f (published %.4f)  rows %d (published %d)\n",
        r04_repro.model.logloss, GMS_BASELINE_PUBLISHED.logloss, r04_repro.model.ece,
        GMS_BASELINE_PUBLISHED.ece, r04_repro.model.n_obs, GMS_BASELINE_PUBLISHED.n_obs)
r04_repro.model.n_obs == GMS_BASELINE_PUBLISHED.n_obs || error("reproduction basis has the wrong row count")
abs(r04_repro.model.logloss - GMS_BASELINE_PUBLISHED.logloss) < 5e-5 ||
    error("baseline control does not reproduce its published LogLoss")
abs(r04_repro.model.ece - GMS_BASELINE_PUBLISHED.ece) < 5e-4 ||
    error("baseline control does not reproduce its published ECE")

r04_score_rows = NamedTuple[]
r04_obs = Dict{String,DataFrame}()
r04_obs2 = Dict{String,DataFrame}()
for arm in r04_arms
    append!(r04_score_rows, gms_scores(arm.label, r04_ctx[arm.label], r04_ctx2[arm.label], r04_families))
    r04_obs[arm.label] = gms_observation_frame(arm.label, r04_ctx[arm.label], r04_odds)
    r04_obs2[arm.label] = gms_observation_frame(arm.label, r04_ctx2[arm.label], r04_odds)
end
r04_scores = DataFrame(r04_score_rows)
# One frame per arm holding primary and secondary rows; the `all` scope uses primary only.
r04_frames = Dict(k => vcat(r04_obs[k], r04_obs2[k]) for k in keys(r04_obs))

for scope in vcat(["all"], GMS_SCOPES)
    sub = sort(filter(:scope => ==(scope), r04_scores), :logloss)
    nrow(sub) == 0 && continue
    println("\n=== PROPER SCORES · scope = ", scope, " ===")
    show(stdout, MIME"text/plain"(),
         select(sub, :model, :n_obs, :logloss, :market_logloss, :brier, :ece, :market_ece, :rps);
         allrows = true, allcols = true)
    println()
end

# %%
# ===================================================================
# 5. Paired, fixture-clustered bootstrap
# ===================================================================
# The `all` scope must pool the primary markets only, so it is computed on `r04_obs`; the
# per-family scopes come from the combined frames.
function r04_contrasts(column::Symbol)
    rows = NamedTuple[]
    for (cand, ref) in GMS_PAIRS
        r = gms_paired_bootstrap(r04_obs[cand], r04_obs[ref]; B = R04_BOOTSTRAP_B, seed = R04_SEED, column)
        push!(rows, (; contrast = "$cand − $ref", scope = "all", rule = String(column),
                       n_obs = r.n_obs, n_fixtures = r.n_fixtures, delta = r.delta,
                       lo = r.lo, hi = r.hi, p_better = r.p_negative))
        for fam in GMS_SCOPES
            r = gms_paired_bootstrap(r04_frames[cand], r04_frames[ref]; B = R04_BOOTSTRAP_B,
                                     seed = R04_SEED, family = fam, column)
            r.n_obs == 0 && continue
            push!(rows, (; contrast = "$cand − $ref", scope = fam, rule = String(column),
                           n_obs = r.n_obs, n_fixtures = r.n_fixtures, delta = r.delta,
                           lo = r.lo, hi = r.hi, p_better = r.p_negative))
        end
    end
    return DataFrame(rows)
end
r04_boot = vcat(r04_contrasts(:ll), r04_contrasts(:brier))
r04_boot.significant = (r04_boot.hi .< 0) .| (r04_boot.lo .> 0)

r04_market_rows = NamedTuple[]
for arm in r04_arms
    for (scope, frame, fam) in vcat([("all", r04_obs[arm.label], nothing)],
                                    [(f, r04_frames[arm.label], f) for f in GMS_SCOPES])
        b = gms_paired_bootstrap(frame, :market; B = R04_BOOTSTRAP_B, seed = R04_SEED, family = fam)
        b.n_obs == 0 && continue
        push!(r04_market_rows, (; model = arm.label, scope, b...))
    end
end
r04_market = DataFrame(r04_market_rows)
r04_market.significant = (r04_market.hi .< 0) .| (r04_market.lo .> 0)

println("\n=== PAIRED ΔLogLoss ===")
show(stdout, MIME"text/plain"(),
     select(filter(r -> r.rule == "ll", r04_boot), :contrast, :scope, :n_obs, :delta, :lo, :hi,
            :p_better, :significant); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 6. Mechanism: home-favourite compression (H3)
# ===================================================================
r04_compression = gms_compression_table(r04_obs)
println("\n=== 1X2 HOME: mean p_model − p_market by Betfair-close p_home bin ===")
show(stdout, MIME"text/plain"(), r04_compression; allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 7. Final report
# ===================================================================
CSV.write(joinpath(R04_OUT_DIR, "r04_proper_scores.csv"), r04_scores)
CSV.write(joinpath(R04_OUT_DIR, "r04_paired_bootstrap.csv"), r04_boot)
CSV.write(joinpath(R04_OUT_DIR, "r04_vs_market.csv"), r04_market)
CSV.write(joinpath(R04_OUT_DIR, "r04_compression.csv"), r04_compression)

r04_f5 = v -> gph_num(v; digits = 5)
r04_s5 = v -> gph_signed(v; digits = 5)
open(joinpath(R04_OUT_DIR, "r04_evaluation_report.md"), "w") do io
    println(io, "# r04 proper scores — Task 015 (market-anchored MultiScaleGRW)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". Panel ", length(r04_panel_ids),
            " fixtures (24/25 + 25/26 walk-forward). Book: de-vigged Betfair TWA(−20, 0] close. ",
            @sprintf("Baseline reproduction: LogLoss %.5f / ECE %.4f on %d rows (published %.5f / %.4f).",
                     r04_repro.model.logloss, r04_repro.model.ece, r04_repro.model.n_obs,
                     GMS_BASELINE_PUBLISHED.logloss, GMS_BASELINE_PUBLISHED.ece), "\n")
    println(io, "Runs: ", join(["`$(a.label)` `$(a.run_id)`" for a in r04_arms], "; "), ".\n")
    println(io, "`all` pools 1X2 + O/U 2.5 + BTTS. OU1.5 / OU3.5 are secondary and never pooled.\n")
    for scope in vcat(["all"], GMS_SCOPES)
        sub = sort(filter(:scope => ==(scope), r04_scores), :logloss)
        nrow(sub) == 0 && continue
        println(io, "## Scope: ", scope, "\n")
        cols = scope in ("all", "1X2") ?
            [:model, :n_obs, :logloss, :market_logloss, :brier, :market_brier, :rps, :market_rps, :ece, :market_ece] :
            [:model, :n_obs, :logloss, :market_logloss, :brier, :market_brier, :ece, :market_ece]
        print(io, gph_markdown_table(select(sub, cols);
            formats = Dict(c => (c in (:ece, :market_ece) ? (v -> gph_num(v; digits = 4)) : r04_f5)
                           for c in cols if c ∉ (:model, :n_obs))))
        println(io)
    end
    println(io, "## Paired Δ (candidate − reference), fixture-clustered bootstrap B = ", R04_BOOTSTRAP_B, "\n")
    println(io, "Negative Δ favours the candidate. 95% percentile interval.\n")
    for rule in ("ll", "brier")
        println(io, "### ", rule == "ll" ? "LogLoss" : "Brier", "\n")
        print(io, gph_markdown_table(select(filter(r -> r.rule == rule, r04_boot),
            :contrast, :scope, :n_obs, :n_fixtures, :delta, :lo, :hi, :p_better, :significant);
            formats = Dict(:delta => r04_s5, :lo => r04_s5, :hi => r04_s5,
                           :p_better => v -> gph_num(v; digits = 4))))
        println(io)
    end
    println(io, "## Every arm against the Betfair close (ΔLogLoss)\n")
    print(io, gph_markdown_table(select(r04_market, :model, :scope, :n_obs, :delta, :lo, :hi,
                                        :p_negative, :significant);
        formats = Dict(:delta => r04_s5, :lo => r04_s5, :hi => r04_s5,
                       :p_negative => v -> gph_num(v; digits = 4))))
    println(io, "\n## Home-favourite compression (1X2 home selection)\n")
    print(io, gph_markdown_table(r04_compression;
        formats = Dict(:mean_p_market => v -> gph_num(v; digits = 4),
                       :mean_p_model => v -> gph_num(v; digits = 4),
                       :mean_gap => v -> gph_signed(v; digits = 4),
                       :home_win_rate => v -> gph_num(v; digits = 4))))
end

println("\nR04_SIGNIFICANT ", count(r04_boot.significant), " of ", nrow(r04_boot), " contrasts")
println("R04_DONE report=", joinpath(R04_OUT_DIR, "r04_evaluation_report.md"))
