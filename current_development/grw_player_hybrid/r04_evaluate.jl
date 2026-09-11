# ==============================================================================
# r04 — Out-of-sample proper scores and paired bootstrap, 24/25 + 25/26
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A predictive-fit comparison: the four GRW ladder models against the Betfair
# closing line and against the TimeDecay controls (`m05_joint_td_raw`,
# `m12_hybrid_td_raw`) plus Task 007's team-level `m05_joint_grw_raw`. It is not a
# betting study — `r05` and `r06` stake.
#
# Hypothesis under test: fusing the two-speed GRW team state with the announced-XI
# lineup pillar improves on BOTH parents — lower LogLoss than `m12_hybrid_td_raw`
# (TimeDecay + lineup), and better calibration (ECE) than `m05_wealth_grw` (GRW,
# no lineup). A null here is a live possibility: Task 007 found the GRW's gain
# nearly vanishes once the proxy-xG arm is present (ΔLL −0.0003), and Exp 06
# found the lineup pillar buys calibration, not LogLoss.
#
# COMPARABILITY CONTRACT
#
# * One panel: the 710 fixtures of seasons 24/25 + 25/26 that every arm holds a
#   latent for. Runs extended into 2026/27 are RESTRICTED to it before scoring.
#   A panel of any other size is an error, not a caveat.
# * One book: de-vigged Betfair TWA(−20, 0] close — Exp 06 `r62`'s frame exactly.
#   `m12_hybrid_td_raw` must reproduce its published LogLoss 0.64337 / ECE 0.0100
#   on this panel; the runner asserts it before scoring anything new.
# * Inference clusters on the FIXTURE: 10,000 paired bootstrap resamples of the
#   710 fixtures, keeping every fixture's rows together.
#
# USAGE (mcmc-beast, from /root/BF_grw_player_hybrid)
#
#   julia --project -t 16 current_development/grw_player_hybrid/r04_evaluate.jl
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
const R04_CONFIG = GPHConfig()
const R04_BOOTSTRAP_B = 10_000
const R04_SEED = 20260911
const R04_OUT_DIR = joinpath(R04_CONFIG.save_root, "evaluation")

# The published Exp 06 figures the pinned control must reproduce (README §results).
const R04_M12_TD_LOGLOSS = 0.64337
const R04_M12_TD_ECE = 0.0100

# Paired contrasts, each read as "left − right" (negative favours the left).
const R04_PAIRS = [
    ("m12_joint_hybrid_synergy_grw", "m12_hybrid_td_raw"),   # GRW vs TimeDecay, full hybrid
    ("m05_wealth_grw", "m05_joint_td_raw"),                  # GRW vs TimeDecay, team level
    ("m12_joint_hybrid_synergy_grw", "m05_wealth_grw"),      # lineup pillar under joint + GRW
    ("m10_lineup_grw", "m00_baseline_grw"),                  # lineup pillar under Poisson + GRW
    ("m05_wealth_grw", "m05_joint_grw_raw"),                 # reproduction of Task 007
    ("m12_joint_hybrid_synergy_grw", "m05_joint_grw_raw"),   # hybrid vs the T−25 GRW benchmark
]

mkpath(R04_OUT_DIR)
println("\n" * "="^96)
println("  r04 PROPER SCORES — GRW × lineup ladder vs TimeDecay controls and Betfair close")
println("  bootstrap  : ", R04_BOOTSTRAP_B, " fixture-clustered paired resamples, seed ", R04_SEED)
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot, the book and the arms
# ===================================================================
r04_ds = gph_load_data()
r04_odds = gph_betfair_closing_odds(r04_ds)
r04_families = gph_family_selections(r04_odds)
println("  Betfair close: ", nrow(r04_odds), " rows over ", length(unique(r04_odds.match_id)),
        " fixtures | families: ", join(["$k=$(v)" for (k, v) in sort(collect(r04_families))], "  "))

r04_arms = gph_arms(R04_CONFIG)
r04_fits = Dict{String,Any}()
for arm in r04_arms
    fit = gph_load_arm(arm)
    panel = gph_season_panel(r04_ds, fit, R04_CONFIG.target_seasons)
    length(panel) == R04_CONFIG.expected_oos || error(
        "$(arm.label) holds $(length(panel)) 24/25+25/26 fixtures; expected $(R04_CONFIG.expected_oos)")
    r04_fits[arm.label] = gph_restrict(fit, panel)
    @printf("  %-30s %-15s %2d folds  %4d → %d fixtures  run %s\n",
            arm.label, arm.dynamics, length(fit.folds), n_matches(fit.latents),
            length(panel), arm.run_id)
end
r04_panel = Set(r04_fits[first(GPH_MODEL_NAMES)].latents.match_ids)
all(Set(f.latents.match_ids) == r04_panel for f in values(r04_fits)) ||
    error("arms do not cover the identical 710-fixture panel")

# %%
# ===================================================================
# 4. Proper scores per arm and market family
# ===================================================================
r04_score_rows = NamedTuple[]
r04_obs = Dict{String,DataFrame}()
for arm in r04_arms
    ctx = gph_context(r04_fits[arm.label], r04_odds, r04_ds)
    append!(r04_score_rows, [(; row..., dynamics = arm.dynamics, role = arm.role)
                             for row in gph_scores(arm.label, ctx, r04_families)])
    r04_obs[arm.label] = gph_observation_frame(arm.label, ctx, r04_odds)
end
r04_scores = DataFrame(r04_score_rows)

# Reproduction gate: the pinned production control, scored here, must be the
# published number. If it is not, every contrast below is against the wrong thing.
r04_m12_td = only(filter(r -> r.model == "m12_hybrid_td_raw" && r.scope == "all", r04_scores))
@printf("\n  reproduction: m12_hybrid_td_raw LogLoss %.5f (published %.5f)  ECE %.4f (published %.4f)\n",
        r04_m12_td.logloss, R04_M12_TD_LOGLOSS, r04_m12_td.ece, R04_M12_TD_ECE)
abs(r04_m12_td.logloss - R04_M12_TD_LOGLOSS) < 5e-5 || error("m12 TD control does not reproduce its published LogLoss")
abs(r04_m12_td.ece - R04_M12_TD_ECE) < 5e-4 || error("m12 TD control does not reproduce its published ECE")

println("\n=== PROPER SCORES (scope = all) ===")
show(stdout, MIME"text/plain"(),
     sort(select(filter(:scope => ==("all"), r04_scores),
                 :model, :dynamics, :n_obs, :logloss, :market_logloss, :brier, :rps, :ece, :market_ece),
          :logloss); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 5. Paired, fixture-clustered bootstrap
# ===================================================================
r04_boot_rows = NamedTuple[]
for scope in ("all", "1X2", "OU2.5", "BTTS")
    fam = scope == "all" ? nothing : scope
    for arm in r04_arms
        b = gph_paired_bootstrap(r04_obs[arm.label], :market; B = R04_BOOTSTRAP_B,
                                 seed = R04_SEED, family = fam)
        push!(r04_boot_rows, (; left = arm.label, right = "betfair_close", scope, b...))
    end
    for (left, right) in R04_PAIRS
        b = gph_paired_bootstrap(r04_obs[left], r04_obs[right]; B = R04_BOOTSTRAP_B,
                                 seed = R04_SEED, family = fam)
        push!(r04_boot_rows, (; left, right, scope, b...))
    end
end
r04_boot = DataFrame(r04_boot_rows)
r04_boot.significant = (r04_boot.hi .< 0) .| (r04_boot.lo .> 0)

println("\n=== PAIRED ΔLogLoss (left − right), scope = all ===")
show(stdout, MIME"text/plain"(),
     select(filter(:scope => ==("all"), r04_boot),
            :left, :right, :n_obs, :n_fixtures, :delta, :lo, :hi, :p_negative, :significant);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 6. Final report
# ===================================================================
CSV.write(joinpath(R04_OUT_DIR, "r04_proper_scores.csv"), r04_scores)
CSV.write(joinpath(R04_OUT_DIR, "r04_paired_bootstrap.csv"), r04_boot)

r04_fmt5 = v -> gph_num(v; digits = 5)
r04_fmt4 = v -> gph_num(v; digits = 4)
open(joinpath(R04_OUT_DIR, "r04_evaluation_report.md"), "w") do io
    println(io, "# r04 proper scores — Task 013\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Panel: ", length(r04_panel), " fixtures (24/25 + 25/26 walk-forward). ",
            "Book: de-vigged Betfair TWA(−20, 0] close. Control reproduction: `m12_hybrid_td_raw` ",
            @sprintf("LogLoss %.5f / ECE %.4f", r04_m12_td.logloss, r04_m12_td.ece),
            " vs published ", R04_M12_TD_LOGLOSS, " / ", R04_M12_TD_ECE, ".\n")
    for scope in ("all", "1X2", "OU2.5", "BTTS")
        sub = sort(filter(:scope => ==(scope), r04_scores), :logloss)
        nrow(sub) == 0 && continue
        println(io, "## Scope: ", scope, "\n")
        cols = scope in ("all", "1X2") ?
            [:model, :dynamics, :n_obs, :logloss, :market_logloss, :brier, :market_brier, :rps, :market_rps, :ece, :market_ece] :
            [:model, :dynamics, :n_obs, :logloss, :market_logloss, :brier, :market_brier, :ece, :market_ece]
        print(io, gph_markdown_table(select(sub, cols);
            formats = Dict(:logloss => r04_fmt5, :market_logloss => r04_fmt5,
                           :brier => r04_fmt5, :market_brier => r04_fmt5,
                           :rps => r04_fmt5, :market_rps => r04_fmt5,
                           :ece => r04_fmt4, :market_ece => r04_fmt4)))
        println(io)
    end
    println(io, "## Paired ΔLogLoss, fixture-clustered bootstrap (B = ", R04_BOOTSTRAP_B, ")\n")
    println(io, "Negative Δ favours the left arm. 95% percentile interval.\n")
    print(io, gph_markdown_table(select(r04_boot, :scope, :left, :right, :n_obs, :n_fixtures,
                                        :delta, :lo, :hi, :p_negative, :significant);
        formats = Dict(:delta => v -> gph_signed(v; digits = 5),
                       :lo => v -> gph_signed(v; digits = 5),
                       :hi => v -> gph_signed(v; digits = 5),
                       :p_negative => v -> gph_num(v; digits = 3))))
end
println("\nR04_DONE report=", joinpath(R04_OUT_DIR, "r04_evaluation_report.md"))
