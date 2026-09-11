# ==============================================================================
# r04 — Out-of-sample proper scores and paired bootstrap, 24/25 + 25/26
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A predictive-fit comparison: the four NegBin ladder models against their Task 013
# Poisson counterparts and against the Betfair closing line. It is not a betting study
# — `r05_portfolio.jl` stakes.
#
# HYPOTHESIS UNDER TEST. Replacing the conditional Poisson goals density with a
# negative binomial moves probability mass toward 0 and toward 4+ at a fixed mean. That
# is invisible to 1X2, which sums scoreline diagonals, and visible on the markets that
# partition the tail. So:
#
#   H1  Totals (O/U 1.5, 2.5, 3.5, 4.5): NegBin improves LogLoss or ECE.
#   H2  BTTS: NegBin improves LogLoss or ECE.
#   H0  1X2: no material change — the CONTROL that says the mean did not move.
#
# A null on H1/H2 is a live possibility and the honest prior. Experiment 02 measured
# `r̂ ≈ 26` on this league — about 5% excess variance — and ΔLogLoss = +0.0001 on 1X2.
# `MultiScaleGRW` absorbs rate variation into latent state, so the residual conditional
# overdispersion this component prices may simply be small. The study is designed so
# that outcome is reportable rather than embarrassing: every contrast is a matched pair
# differing in exactly one component, and the interval is what gets read, not the sign.
#
# COMPARABILITY CONTRACT
#
# * One panel: the fixtures of seasons 24/25 + 25/26 that EVERY arm holds a latent for.
#   The Task 013 controls were extended in place to 43 folds / 769 fixtures, so they are
#   restricted to the intersection before scoring. The intersection must be the expected
#   710; any other size is an error, not a caveat.
# * One book: de-vigged Betfair TWA(−20, 0] close — Task 013 `r04`'s frame exactly.
#   `m12_poisson` must reproduce its published LogLoss 0.64437 / ECE 0.0086 on this
#   panel; the runner asserts it before scoring anything new.
# * Inference clusters on the FIXTURE: 10,000 paired bootstrap resamples of the panel
#   fixtures, keeping every fixture's rows together. Reported for LogLoss AND Brier,
#   because a redistribution of tail mass can move a quadratic rule and a logarithmic
#   rule differently, and reporting only the one that moved would be a choice made after
#   seeing the data.
#
# USAGE (mcmc-beast, from /root/BF_grw_joint_negbin)
#
#   julia --project -t 16 current_development/grw_joint_negbin/r04_evaluate.jl
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
const R04_CONFIG = GJNConfig()
const R04_BOOTSTRAP_B = 10_000
const R04_SEED = 20260914
const R04_OUT_DIR = joinpath(R04_CONFIG.save_root, "evaluation")

# The published Task 013 figures the pinned control must reproduce (README §4).
const R04_M12_POISSON_LOGLOSS = 0.64437
const R04_M12_POISSON_ECE = 0.0086

mkpath(R04_OUT_DIR)
println("\n" * "="^96)
println("  r04 PROPER SCORES — NegBin ladder vs Task 013 Poisson controls and the Betfair close")
println("  bootstrap  : ", R04_BOOTSTRAP_B, " fixture-clustered paired resamples, seed ", R04_SEED)
println("  scopes     : all, ", join(GJN_SCOPES, ", "))
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot, the book and the arms
# ===================================================================
r04_ds = gjn_load_data()
r04_odds = gjn_betfair_closing_odds(r04_ds)
r04_families = gjn_family_selections(r04_odds)
println("  Betfair close: ", nrow(r04_odds), " rows over ", length(unique(r04_odds.match_id)),
        " fixtures")
for f in GJN_SCOPES
    haskey(r04_families, f) &&
        println("    ", rpad(f, 7), " selections: ", join(string.(r04_families[f]), ", "))
end

r04_arms = gjn_arms(R04_CONFIG)
r04_raw = Dict{String,Any}()
for arm in r04_arms
    r04_raw[arm.label] = gjn_load_arm(arm)
end

# The panel is the INTERSECTION, computed once, then imposed on every arm.
r04_panel_ids = gjn_common_panel(r04_ds, r04_raw, R04_CONFIG.target_seasons)
length(r04_panel_ids) == R04_CONFIG.expected_oos || error(
    "common panel is $(length(r04_panel_ids)) fixtures; expected $(R04_CONFIG.expected_oos)")

r04_fits = Dict{String,Any}()
for arm in r04_arms
    fit = r04_raw[arm.label]
    r04_fits[arm.label] = gjn_restrict(fit, r04_panel_ids)
    @printf("  %-34s %-20s %2d folds  %4d → %d fixtures  run %s\n",
            arm.label, arm.likelihood, length(fit.folds), n_matches(fit.latents),
            length(r04_panel_ids), arm.run_id)
end
r04_panel = Set(r04_panel_ids)
all(Set(f.latents.match_ids) == r04_panel for f in values(r04_fits)) ||
    error("arms do not cover the identical panel after restriction")

# %%
# ===================================================================
# 4. Proper scores per arm and market family
# ===================================================================
r04_score_rows = NamedTuple[]
r04_obs = Dict{String,DataFrame}()
for arm in r04_arms
    ctx = gjn_context(r04_fits[arm.label], r04_odds, r04_ds)
    append!(r04_score_rows, [(; row..., likelihood = arm.likelihood, role = arm.role)
                             for row in gjn_scores(arm.label, ctx, r04_families)])
    r04_obs[arm.label] = gjn_observation_frame(arm.label, ctx, r04_odds)
end
r04_scores = DataFrame(r04_score_rows)

# Reproduction gate: the pinned control, scored here, must be the published number. If it
# is not, every contrast below is against the wrong thing.
#
# Scored on GJN_LEGACY_MARKETS, NOT on this study's wider set. Task 013's 0.64437 is a
# pooled figure over 1X2 + O/U 2.5 + BTTS and its 2,899 rows; pooling six markets and
# comparing against a three-market number would fail for a reason unrelated to whether
# the control loaded correctly. The gate reproduces the published basis exactly.
r04_legacy_ctx = gjn_context(r04_fits["m12_poisson"], r04_odds, r04_ds;
                             markets = GJN_LEGACY_MARKETS)
r04_legacy = evaluate_predictions(r04_legacy_ctx; n_bins = 10)
@printf("\n  reproduction (1X2 + O/U 2.5 + BTTS, %d rows): m12_poisson LogLoss %.5f (published %.5f)  ECE %.4f (published %.4f)\n",
        r04_legacy.model.n_obs, r04_legacy.model.logloss, R04_M12_POISSON_LOGLOSS,
        r04_legacy.model.ece, R04_M12_POISSON_ECE)
r04_legacy.model.n_obs == 2899 ||
    error("reproduction basis is $(r04_legacy.model.n_obs) rows; Task 013 published 2899")
abs(r04_legacy.model.logloss - R04_M12_POISSON_LOGLOSS) < 5e-5 ||
    error("m12_poisson control does not reproduce its published LogLoss")
abs(r04_legacy.model.ece - R04_M12_POISSON_ECE) < 5e-4 ||
    error("m12_poisson control does not reproduce its published ECE")
r04_m12_ctl = only(filter(r -> r.model == "m12_poisson" && r.scope == "all", r04_scores))

for scope in vcat(["all"], GJN_SCOPES)
    sub = sort(filter(:scope => ==(scope), r04_scores), :logloss)
    nrow(sub) == 0 && continue
    println("\n=== PROPER SCORES · scope = ", scope, " ===")
    show(stdout, MIME"text/plain"(),
         select(sub, :model, :likelihood, :n_obs, :logloss, :market_logloss,
                :brier, :ece, :market_ece);
         allrows = true, allcols = true)
    println()
end

# %%
# ===================================================================
# 5. Paired, fixture-clustered bootstrap — the four matched contrasts
# ===================================================================
# LogLoss AND Brier, every scope. `p_better` is the share of resamples in which the
# NegBin arm scored lower (better) than its Poisson control.
r04_boot = vcat(
    gjn_pair_table(r04_obs; B = R04_BOOTSTRAP_B, seed = R04_SEED, column = :ll),
    gjn_pair_table(r04_obs; B = R04_BOOTSTRAP_B, seed = R04_SEED, column = :brier),
)
r04_boot.significant = (r04_boot.hi .< 0) .| (r04_boot.lo .> 0)

# Every arm against the closing line, for context on where all of them sit.
r04_market_rows = NamedTuple[]
for arm in r04_arms
    frame = r04_obs[arm.label]
    quoted = [f for f in GJN_SCOPES if any(frame.family .== f)]
    for scope in vcat([nothing], quoted)
        b = gjn_paired_bootstrap(frame, :market; B = R04_BOOTSTRAP_B,
                                 seed = R04_SEED, family = scope, column = :ll)
        b.n_obs == 0 && continue
        push!(r04_market_rows, (; model = arm.label,
                                  scope = scope === nothing ? "all" : scope, b...))
    end
end
r04_market = DataFrame(r04_market_rows)
r04_market.significant = (r04_market.hi .< 0) .| (r04_market.lo .> 0)

println("\n=== PAIRED ΔLogLoss · NegBin − its Poisson control ===")
show(stdout, MIME"text/plain"(),
     select(filter(r -> r.rule == "ll", r04_boot),
            :contrast, :scope, :n_obs, :n_fixtures, :delta, :lo, :hi, :p_better, :significant);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 6. Final report
# ===================================================================
CSV.write(joinpath(R04_OUT_DIR, "r04_proper_scores.csv"), r04_scores)
CSV.write(joinpath(R04_OUT_DIR, "r04_paired_bootstrap.csv"), r04_boot)
CSV.write(joinpath(R04_OUT_DIR, "r04_vs_market.csv"), r04_market)

r04_fmt5 = v -> gjn_num(v; digits = 5)
r04_fmt4 = v -> gjn_num(v; digits = 4)
open(joinpath(R04_OUT_DIR, "r04_evaluation_report.md"), "w") do io
    println(io, "# r04 proper scores — Task 014 (JointGammaNegBinObservation)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Panel: ", length(r04_panel), " fixtures (24/25 + 25/26 walk-forward). ",
            "Book: de-vigged Betfair TWA(−20, 0] close. Control reproduction: `m12_poisson` ",
            @sprintf("LogLoss %.5f / ECE %.4f", r04_legacy.model.logloss, r04_legacy.model.ece),
            " vs published ", R04_M12_POISSON_LOGLOSS, " / ", R04_M12_POISSON_ECE,
            " on Task 013's own 1X2 + O/U 2.5 + BTTS basis (", r04_legacy.model.n_obs, " rows).\n")
    println(io, "Scored markets here are wider than Task 013's: 1X2, BTTS and O/U 1.5 / 2.5 / ",
            "3.5 / 4.5. The `all` scope therefore pools more rows than the 2,899 that ",
            "reproduction figure is computed on, and is not comparable with it.\n")

    println(io, "Read the totals and BTTS scopes as the test and the 1X2 scope as the control: ",
            "a negative binomial redistributes mass within a fixed mean, so a change on ",
            "O/U 3.5 with none on 1X2 is the mechanism behaving as stated.\n")

    for scope in vcat(["all"], GJN_SCOPES)
        sub = sort(filter(:scope => ==(scope), r04_scores), :logloss)
        nrow(sub) == 0 && continue
        println(io, "## Scope: ", scope, "\n")
        cols = scope in ("all", "1X2") ?
            [:model, :likelihood, :n_obs, :logloss, :market_logloss, :brier, :market_brier,
             :rps, :market_rps, :ece, :market_ece] :
            [:model, :likelihood, :n_obs, :logloss, :market_logloss, :brier, :market_brier,
             :ece, :market_ece]
        print(io, gjn_markdown_table(select(sub, cols);
            formats = Dict(:logloss => r04_fmt5, :market_logloss => r04_fmt5,
                           :brier => r04_fmt5, :market_brier => r04_fmt5,
                           :rps => r04_fmt5, :market_rps => r04_fmt5,
                           :ece => r04_fmt4, :market_ece => r04_fmt4)))
        println(io)
    end

    println(io, "## Paired Δ, fixture-clustered bootstrap (B = ", R04_BOOTSTRAP_B, ")\n")
    println(io, "Each row is a NegBin rung minus its Task 013 Poisson counterpart, the two ",
            "differing in exactly one component. Negative Δ favours the NegBin arm; ",
            "`p_better` is the share of resamples in which it scored lower. ",
            "95% percentile interval.\n")
    for rule in ("ll", "brier")
        sub = filter(r -> r.rule == rule, r04_boot)
        nrow(sub) == 0 && continue
        println(io, "### ", rule == "ll" ? "LogLoss" : "Brier", "\n")
        print(io, gjn_markdown_table(select(sub, :contrast, :scope, :n_obs, :n_fixtures,
                                            :delta, :lo, :hi, :p_better, :significant);
            formats = Dict(:delta => v -> gjn_signed(v; digits = 5),
                           :lo => v -> gjn_signed(v; digits = 5),
                           :hi => v -> gjn_signed(v; digits = 5),
                           :p_better => r04_fmt4)))
        println(io)
    end

    println(io, "## Every arm against the Betfair close (ΔLogLoss)\n")
    print(io, gjn_markdown_table(select(r04_market, :model, :scope, :n_obs, :n_fixtures,
                                        :delta, :lo, :hi, :p_negative, :significant);
        formats = Dict(:delta => v -> gjn_signed(v; digits = 5),
                       :lo => v -> gjn_signed(v; digits = 5),
                       :hi => v -> gjn_signed(v; digits = 5),
                       :p_negative => r04_fmt4)))
end

r04_sig = count(r04_boot.significant)
println("\nR04_SIGNIFICANT ", r04_sig, " of ", nrow(r04_boot), " matched contrasts")
println("R04_DONE report=", joinpath(R04_OUT_DIR, "r04_evaluation_report.md"))
