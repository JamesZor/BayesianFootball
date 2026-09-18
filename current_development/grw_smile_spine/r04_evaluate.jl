# ==============================================================================
# r04 — Out-of-sample proper scores, six arms, 24/25 + 25/26
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A predictive-fit comparison of the 1-parameter spine against the 5-parameter smile, the
# supremacy-only and pure-GRW controls, and the Betfair closing line. Not a betting study:
# r06–r08 stake these posteriors.
#
# THE HYPOTHESIS, and how it can fail visibly
#
#   H3 (predictive parity): the spine's out-of-sample proper scores are statistically
#      indistinguishable from the five-strike smile's at the same weight — ΔLogLoss ≈ 0 with a
#      paired interval containing 0.
#
# H3 is a NULL hypothesis, which makes it the easy kind to "confirm" by accident: a wide
# interval containing 0 is not parity, it is ignorance. So two things are reported beside every
# Δ — the interval width, and whether the same test can resolve a contrast that is known to be
# real (spine − baseline). A Δ of 0 ± 0.02 says nothing; a Δ of 0 ± 0.001 on a panel where the
# spine − baseline interval excludes 0 is evidence.
#
# WHERE THE SPINE SHOULD LOSE IF IT LOSES ANYWHERE. r02 measured φ = 0.900 / 0.949 / 1.000 /
# 1.054 / 1.111 against the five-strike 0.843 / 0.976 / 1.001 / 1.026 / 1.069. The line cannot
# bend: it over-prices Under 0.5 (φ₀ too high) and over-prices Over 4.5 (φ₄ too high). §6's
# per-strike table is therefore not decoration — it is the only place a pooled totals score
# would hide the one-parameter restriction's cost. O/U 0.5 and 4.5 are scored and NEVER pooled
# into `all`.
#
# The prior to argue against: Experiment 06 found the Betfair close BEATS every model on this
# panel (0.64182 vs 0.64315 for this baseline). A model anchored to the close can gain LogLoss
# merely by becoming more like the benchmark, so every arm is also reported against the close
# itself and "beats the baseline" is never read as "beats the market".
#
# COMPARABILITY CONTRACT
#
# * One panel: fixtures of 24/25 + 25/26 every arm holds a latent for; must be exactly 710.
#   The 43-fold runs are restricted to it BEFORE scoring.
# * One book: de-vigged Betfair TWA(−20, 0] close, for every arm.
# * Reproduction gate: the pinned baseline must reproduce Task 013's published LogLoss 0.64315 /
#   ECE 0.0123 on 1X2 + O/U 2.5 + BTTS (2,899 rows) before any contrast is printed.
# * 10,000 fixture-clustered paired resamples; LogLoss AND Brier.
#
# USAGE (mcmc-beast, from /root/BF_grw_smile_spine, after r02)
#
#   julia --project -t 16 current_development/grw_smile_spine/r04_evaluate.jl
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
const R04_CONFIG = gss_config()
const R04_BOOTSTRAP_B = 10_000
const R04_SEED = 20260915
const R04_OUT_DIR = joinpath(R04_CONFIG.save_root, "evaluation")
const R04_LATENT_DIRS = [joinpath(R04_CONFIG.save_root, "latents"),
                         joinpath(@__DIR__, "..", "grw_market_smile", "results", "latents")]
const R04_GIT = try
    readchomp(`git rev-parse --short HEAD`)
catch
    "unknown"
end

mkpath(R04_OUT_DIR)
println("\n" * "="^96)
println("  r04 PROPER SCORES — 1-parameter spine vs 5-parameter smile, controls, Betfair close")
println("  bootstrap  : ", R04_BOOTSTRAP_B, " fixture-clustered paired resamples, seed ", R04_SEED)
println("  scopes     : all (1X2 + OU2.5 + BTTS), then ", join(GSS_SCOPES, ", "))
println("  git        : ", R04_GIT, "   host ", gethostname(), "   threads ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot, the book and the arms
# ===================================================================
r04_ds = gph_load_data()
r04_splitter = gph_splitter(R04_CONFIG.extension_seasons)
r04_odds = gms_betfair_closing_odds(r04_ds)
r04_families = gss_family_selections(r04_odds)
println("  Betfair close: ", nrow(r04_odds), " rows over ", length(unique(r04_odds.match_id)),
        " fixtures | scopes quoted: ", join(sort(collect(keys(r04_families))), ", "))

r04_arms = gss_arms(R04_CONFIG)
r04_raw = gss_load_arms(r04_arms, r04_ds; splitter = r04_splitter, latent_dirs = R04_LATENT_DIRS)

r04_panel_ids = gms_common_panel(r04_ds, r04_raw, R04_CONFIG.target_seasons)
length(r04_panel_ids) == R04_CONFIG.expected_oos || error(
    "common panel is $(length(r04_panel_ids)) fixtures; expected $(R04_CONFIG.expected_oos)")
r04_fits = Dict(label => gms_restrict(fit, r04_panel_ids) for (label, fit) in r04_raw)
r04_raw = nothing
GC.gc()
println("  panel  : ", length(r04_panel_ids), " fixtures (24/25 + 25/26), every arm restricted to it")

# %%
# ===================================================================
# 4. Reproduction gate
# ===================================================================
# Before any contrast: if the pinned baseline does not reproduce its published score on its
# published basis, the panel or the book is not Task 013's and no Δ below means what it says.
r04_ctx = Dict{String,Any}()
r04_ctx2 = Dict{String,Any}()
for arm in r04_arms
    r04_ctx[arm.label] = gms_context(r04_fits[arm.label], r04_odds, r04_ds; markets = GSS_PRIMARY_MARKETS)
    r04_ctx2[arm.label] = gms_context(r04_fits[arm.label], r04_odds, r04_ds; markets = GSS_SECONDARY_MARKETS)
end

r04_repro = evaluate_predictions(r04_ctx["m05_joint_grw_baseline"]; n_bins = 10)
@printf("\n  reproduction: baseline LogLoss %.5f (published %.5f)  ECE %.4f (published %.4f)  rows %d (published %d)\n",
        r04_repro.model.logloss, GMS_BASELINE_PUBLISHED.logloss, r04_repro.model.ece,
        GMS_BASELINE_PUBLISHED.ece, r04_repro.model.n_obs, GMS_BASELINE_PUBLISHED.n_obs)
r04_repro.model.n_obs == GMS_BASELINE_PUBLISHED.n_obs ||
    error("reproduction basis has the wrong row count")
abs(r04_repro.model.logloss - GMS_BASELINE_PUBLISHED.logloss) < 5e-5 ||
    error("baseline control does not reproduce its published LogLoss")
abs(r04_repro.model.ece - GMS_BASELINE_PUBLISHED.ece) < 5e-4 ||
    error("baseline control does not reproduce its published ECE")

# %%
# ===================================================================
# 5. Proper scores per arm and scope
# ===================================================================
r04_score_rows = NamedTuple[]
r04_obs = Dict{String,DataFrame}()
r04_obs2 = Dict{String,DataFrame}()
for arm in r04_arms
    append!(r04_score_rows, gss_scores(arm.label, r04_ctx[arm.label], r04_ctx2[arm.label], r04_families))
    r04_obs[arm.label] = gms_observation_frame(arm.label, r04_ctx[arm.label], r04_odds)
    r04_obs2[arm.label] = gms_observation_frame(arm.label, r04_ctx2[arm.label], r04_odds)
end
r04_scores = DataFrame(r04_score_rows)
# One frame per arm holding primary and secondary rows; `all` uses the primary frame only.
r04_frames = Dict(k => vcat(r04_obs[k], r04_obs2[k]) for k in keys(r04_obs))

for scope in vcat(["all"], GSS_SCOPES)
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
# 6. The strike ladder — where the one-parameter restriction must show
# ===================================================================
r04_strikes = gss_strike_table(r04_frames, [a.label for a in r04_arms])
println("\n=== PER-STRIKE (UNDER selection): LogLoss and mean p_model − p_market ===")
show(stdout, MIME"text/plain"(), r04_strikes; allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 7. Paired, fixture-clustered bootstrap
# ===================================================================
# `all` pools the primary markets only, so it is computed on `r04_obs`; the per-scope rows come
# from the combined frames.
function r04_contrasts(column::Symbol)
    rows = NamedTuple[]
    for (cand, ref) in GSS_PAIRS
        haskey(r04_obs, cand) && haskey(r04_obs, ref) || continue
        r = gms_paired_bootstrap(r04_obs[cand], r04_obs[ref]; B = R04_BOOTSTRAP_B, seed = R04_SEED, column)
        push!(rows, (; contrast = "$cand − $ref", scope = "all", rule = String(column),
                       n_obs = r.n_obs, n_fixtures = r.n_fixtures, delta = r.delta,
                       lo = r.lo, hi = r.hi, width = r.hi - r.lo, p_better = r.p_negative))
        for fam in GSS_SCOPES
            r = gms_paired_bootstrap(r04_frames[cand], r04_frames[ref]; B = R04_BOOTSTRAP_B,
                                     seed = R04_SEED, family = fam, column)
            r.n_obs == 0 && continue
            push!(rows, (; contrast = "$cand − $ref", scope = fam, rule = String(column),
                           n_obs = r.n_obs, n_fixtures = r.n_fixtures, delta = r.delta,
                           lo = r.lo, hi = r.hi, width = r.hi - r.lo, p_better = r.p_negative))
        end
    end
    return DataFrame(rows)
end
r04_boot = vcat(r04_contrasts(:ll), r04_contrasts(:brier))
r04_boot.significant = (r04_boot.hi .< 0) .| (r04_boot.lo .> 0)

r04_market_rows = NamedTuple[]
for arm in r04_arms
    for (scope, frame, fam) in vcat([("all", r04_obs[arm.label], nothing)],
                                    [(f, r04_frames[arm.label], f) for f in GSS_SCOPES])
        b = gms_paired_bootstrap(frame, :market; B = R04_BOOTSTRAP_B, seed = R04_SEED, family = fam)
        b.n_obs == 0 && continue
        push!(r04_market_rows, (; model = arm.label, scope, b...))
    end
end
r04_market = DataFrame(r04_market_rows)
r04_market.significant = (r04_market.hi .< 0) .| (r04_market.lo .> 0)

println("\n=== PAIRED ΔLogLoss (negative favours the candidate) ===")
show(stdout, MIME"text/plain"(),
     select(filter(r -> r.rule == "ll", r04_boot), :contrast, :scope, :n_obs, :delta, :lo, :hi,
            :width, :p_better, :significant); allrows = true, allcols = true)
println()

# H3 read out explicitly: parity claims need a resolving test beside them.
println("\n=== H3 — spine vs five-strike smile at the same weight ===")
for w in ("w020", "w040")
    cand = "m05_joint_grw_smile_spine_$w"
    ref = "m05_joint_grw_smile_supremacy_$w"
    control = only(filter(r -> r.rule == "ll" && r.scope == "all" &&
                               r.contrast == "$cand − m05_joint_grw_baseline", eachrow(r04_boot)))
    for scope in ("all", "OU2.5", "OU0.5", "OU4.5")
        hits = filter(r -> r.rule == "ll" && r.scope == scope && r.contrast == "$cand − $ref",
                      eachrow(r04_boot))
        isempty(hits) && continue
        r = only(hits)
        @printf("  %-4s %-6s Δ %+0.5f [%+0.5f, %+0.5f] width %.5f  %s   (same-panel control: spine − baseline Δ %+0.5f, %s)\n",
                w, scope, r.delta, r.lo, r.hi, r.width,
                r.significant ? "RESOLVED" : "indistinguishable",
                control.delta, control.significant ? "resolved" : "unresolved")
    end
end

# %%
# ===================================================================
# 8. Mechanism: home-favourite compression
# ===================================================================
r04_compression = gms_compression_table(r04_obs)
println("\n=== 1X2 HOME: mean p_model − p_market by Betfair-close p_home bin ===")
show(stdout, MIME"text/plain"(), r04_compression; allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 9. Final report
# ===================================================================
CSV.write(joinpath(R04_OUT_DIR, "r04_proper_scores.csv"), r04_scores)
CSV.write(joinpath(R04_OUT_DIR, "r04_strike_ladder.csv"), r04_strikes)
CSV.write(joinpath(R04_OUT_DIR, "r04_paired_bootstrap.csv"), r04_boot)
CSV.write(joinpath(R04_OUT_DIR, "r04_vs_market.csv"), r04_market)
CSV.write(joinpath(R04_OUT_DIR, "r04_compression.csv"), r04_compression)

r04_f5 = v -> gph_num(v; digits = 5)
r04_s5 = v -> gph_signed(v; digits = 5)
open(joinpath(R04_OUT_DIR, "r04_evaluation_report.md"), "w") do io
    println(io, "# r04 proper scores — Task 016 (1-parameter smile spine)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R04_GIT, "` on ",
            gethostname(), ". Panel ", length(r04_panel_ids),
            " fixtures (24/25 + 25/26 walk-forward). Book: de-vigged Betfair TWA(−20, 0] close. ",
            @sprintf("Baseline reproduction: LogLoss %.5f / ECE %.4f on %d rows (published %.5f / %.4f).",
                     r04_repro.model.logloss, r04_repro.model.ece, r04_repro.model.n_obs,
                     GMS_BASELINE_PUBLISHED.logloss, GMS_BASELINE_PUBLISHED.ece), "\n")
    println(io, "Runs: ", join(["`$(a.label)` `$(a.run_id)`" for a in r04_arms], "; "), ".\n")
    println(io, "`all` pools 1X2 + O/U 2.5 + BTTS (Task 013's published basis). O/U 0.5, 1.5, 3.5 ",
            "and 4.5 are secondary and never pooled — the spine's line forces φ₀ = 0.900 and ",
            "φ₄ = 1.111 against the five-strike 0.843 and 1.069, so the ends are where the ",
            "one-parameter restriction can cost something.\n")

    for scope in vcat(["all"], GSS_SCOPES)
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

    println(io, "## Strike ladder — UNDER selection, per O/U line\n")
    println(io, "`mean_gap` is mean `p_model − p_market`; `realised_under_rate` is the outcome ",
            "frequency on the same rows.\n")
    print(io, gph_markdown_table(r04_strikes;
        formats = Dict(:logloss => r04_f5, :market_logloss => r04_f5,
                       :delta_logloss => r04_s5,
                       :mean_p_model => v -> gph_num(v; digits = 4),
                       :mean_p_market => v -> gph_num(v; digits = 4),
                       :mean_gap => v -> gph_signed(v; digits = 4),
                       :realised_under_rate => v -> gph_num(v; digits = 4))))

    println(io, "\n## Paired Δ (candidate − reference), fixture-clustered bootstrap B = ",
            R04_BOOTSTRAP_B, "\n")
    println(io, "Negative Δ favours the candidate; 95% percentile interval. `width` is hi − lo — ",
            "a Δ near 0 with a wide interval is an unresolved test, not parity.\n")
    for rule in ("ll", "brier")
        println(io, "### ", rule == "ll" ? "LogLoss" : "Brier", "\n")
        print(io, gph_markdown_table(select(filter(r -> r.rule == rule, r04_boot),
            :contrast, :scope, :n_obs, :n_fixtures, :delta, :lo, :hi, :width, :p_better, :significant);
            formats = Dict(:delta => r04_s5, :lo => r04_s5, :hi => r04_s5, :width => r04_f5,
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
