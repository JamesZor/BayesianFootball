# ==============================================================================
# r06 — Raw vs Calibrated portfolio, at the close and at the tradeable T−25 book
# ==============================================================================
#
# WHAT THIS ANSWERS
#
# r05 staked all eight arms RAW against the de-vigged close and found the NegBin ladder
# a little behind its Poisson controls: 92–94% bet overlap, and a drag concentrated on
# Over 1.5, which the negative binomial declines more often because it moves mass onto
# 0 goals at a fixed mean. That was a raw, closing-line simulation.
#
# The trader's question is whether the production Layer 2 seam changes it. So each arm is
# staked FOUR more ways:
#
#   Environment A — de-vigged Betfair TWA(−20, 0] close, the r05 book
#     raw                       the r05 row, reproduced as a gate
#     close_std                 GenerativeRateCalibrator("scot_lower_close_std"),
#                               StandardGaussianLaw(w_base = 0.85, sigma = 0.15), T+0
#     close_std_t007            the SAME name as canonicalised by Task 007 — the same law
#                               at (w_base = 0.30, sigma = 0.40). See §2.
#
#   Environment B — tradeable T−25 point-in-time order book
#     raw                       no calibration, T−25 prices
#     t25_inv                   MatchDay.option_b_calibrator(), i.e.
#                               InverseGaussianLaw(w_base = 0.25, sigma = 0.35) at T−25
#
# WHY TWO CLOSING CALIBRATORS. The work package asks for
# `canonical_calibrator(:scot_lower_close_std)`. There is no such function in this
# repository, and the parenthetical it expands to — `StandardGaussianLaw(w_base = 0.85,
# sigma = 0.15)` — does not match the only place that name is pinned:
# `current_development/multiscale_grw/l02_portfolio.jl` and `todos/007` both define
# `scot_lower_close_std` as `StandardGaussianLaw(w_base = 0.30, sigma = 0.40)`. The two
# are not close: w_base = 0.85 keeps 85% of the model's log-rate, w_base = 0.30 hands 70%
# of it to the market. Rather than pick one and have the answer hinge on the pick, BOTH
# are run and both are reported. The T−25 calibrator has no such ambiguity — the work
# package's spec is `MatchDay.option_b_calibrator()` verbatim, and that is what is used.
#
# WHAT THE CALIBRATOR DOES TO A NEGATIVE BINOMIAL. `calibrate_latents` pools the LOG
# RATES toward the market's inverted (λ_h, λ_a) and passes `observation_params` — the
# per-fixture dispersion draws r_h, r_a — through untouched. So a calibrated NegBin arm
# is the market's location with the model's shape. That is exactly the contrast the
# trader is asking about: if the Over 1.5 drag is a LOCATION problem, pooling λ toward
# the market removes it; if it is a SHAPE problem, r survives the pool and so does the
# drag.
#
# WHAT IS NOT HERE. No MCMC. Every posterior is loaded by UUID from `mcmc_experiments`.
# No derivative-coherence audit either: it re-prices the whole 12×12 tensor, it would
# roughly double the runtime, and it is structural on `PoolDispersion` — Task 007
# measured worst family spread 6.66e-16 through this same code path.
#
# ONE PANEL PER ENVIRONMENT. The close quotes more fixtures than the T−25 book does, so
# the two environments have different buildable panels and their bankroll numbers are NOT
# comparable across environments. Within an environment every arm and every variant
# stakes the identical panel, which is what the raw/calibrated and NegBin/Poisson
# contrasts need.
#
# USAGE (mcmc-beast, from /root/BF_grw_joint_negbin)
#
#   julia --project -t 16 current_development/grw_joint_negbin/r06_calibrated_portfolio.jl
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

const R06_CAL = BayesianFootball.Calibration

# %%
# ===================================================================
# 2. Configuration — environments, calibrators, variants
# ===================================================================
const R06_CONFIG = GJNConfig()
const R06_BOOTSTRAP_B = 4000
const R06_SEED = 1
const R06_OUT_DIR = joinpath(R06_CONFIG.save_root, "calibrated_portfolio")
const R06_SUMMARY_PATH = joinpath(R06_CONFIG.save_root, "r06_calibrated_portfolio_summary.csv")

"The work package's closing calibrator, as its parenthetical literally specifies it."
r06_cal_close_prompt() = GenerativeRateCalibrator(
    name = "scot_lower_close_std",
    law = StandardGaussianLaw(w_base = 0.85, sigma = 0.15),
    dispersion = PoolDispersion(),
    anchor = :pool_mean,
    fallback = :identity,
    book_as_of_minutes = 0.0)

"The same registry name as Task 007 pinned it — the repository's own `scot_lower_close_std`."
r06_cal_close_task007() = GenerativeRateCalibrator(
    name = "scot_lower_close_std_t007",
    law = StandardGaussianLaw(w_base = 0.30, sigma = 0.40),
    dispersion = PoolDispersion(),
    anchor = :pool_mean,
    fallback = :identity,
    book_as_of_minutes = 0.0)

mkpath(R06_OUT_DIR)
println("\n" * "="^96)
println("  r06 RAW vs CALIBRATED PORTFOLIO — Option B, close and T−25")
println("="^96)

# %%
# ===================================================================
# 3. Helpers
# ===================================================================

"""
    r06_simulate(book, policy, source, odds, ds, panel; label)

Simulate one ALREADY-RESTRICTED source (a `Fit` or a `CalibratedFit`) over `panel`.

`gjn_simulate` restricts internally and therefore only accepts a `Fit`; a calibrated
container must not be re-restricted, because restriction happens BEFORE calibration so
that the inversion is only paid for the fixtures that are staked. Everything else — the
zero-skip refusal, the metric set, the bootstrap — is `gjn_simulate`'s, verbatim.
"""
function r06_simulate(book, policy, source, odds, ds, panel::Vector{Int};
                      label::AbstractString, B::Int = R06_BOOTSTRAP_B, seed::Int = R06_SEED)
    books, report = GJN_PORTFOLIO.build_books_reported(
        book, source, odds, ds; require_converged = false, quiet = true)
    GJN_PORTFOLIO.n_skipped(report) == 0 || error(
        "$label skipped $(GJN_PORTFOLIO.n_skipped(report)) panel fixtures")
    length(books) == length(panel) || error(
        "$label built $(length(books)) books for $(length(panel)) fixtures")
    return GJN_PORTFOLIO.simulate_portfolio(
        policy, books, report;
        bootstrap = true, B = B, seed = seed,
        metrics = GJN_BACKTESTING.AbstractWealthMetric[
            GJN_BACKTESTING.CalmarRatio(), GJN_BACKTESTING.SharpeRatio()])
end

"Every family row of one ledger, tagged with the environment and variant it came from."
function r06_family_rows(env, variant, label, likelihood, bets)
    out = NamedTuple[]
    nrow(bets) == 0 && return out
    for sub in groupby(sort(DataFrame(bets), :family), :family)
        s = gjn_edge_summary(sub)
        push!(out, (; environment = env, variant = variant, model = label,
                      likelihood = likelihood, selection_family = first(sub.family),
                      n_bets = s.n_bets, win_rate_pct = 100 * s.win_rate,
                      roi_pct = s.roi, stake_sum = s.stake_sum, pnl_sum = s.pnl_sum,
                      edge_mean_pp = s.edge_mean, p_model_mean = s.p_model_mean,
                      p_market_mean = s.p_market_mean, capture_ratio = s.capture_ratio))
    end
    return out
end

"`true` for the Over 1.5 leg of the Option B basket, whatever the ledger spells it."
r06_is_over15(family) = occursin("1.5", String(family))

"""
    r06_declined_rows(pair, env, variant, name_a, name_b, bets_a, bets_b; family_filter)

The bets model B struck and model A did not, and what they were worth to B.

`family_filter = r06_is_over15` is the Over 1.5 cut the r05 finding lives in: "NegBin declined
30–40% of Over 1.5 bets that were profitable" is a statement about `n_only_b` and about
the ROI of that exclusive set, and the question here is whether calibration shrinks it.
"""
function r06_declined_rows(pair, env, variant, name_a, name_b,
                           bets_a::AbstractDataFrame, bets_b::AbstractDataFrame;
                           family_filter = nothing)
    ff = family_filter
    a = ff === nothing ? DataFrame(bets_a) : filter(r -> ff(r.family), DataFrame(bets_a))
    b = ff === nothing ? DataFrame(bets_b) : filter(r -> ff(r.family), DataFrame(bets_b))
    (nrow(a) == 0 && nrow(b) == 0) && return NamedTuple[]
    both_a, both_b, only_a, only_b = gjn_partition_bets(a, b)
    sa, sb = gjn_edge_summary(only_a), gjn_edge_summary(only_b)
    sh_a, sh_b = gjn_edge_summary(both_a), gjn_edge_summary(both_b)
    total_b = nrow(both_b) + nrow(only_b)
    return NamedTuple[(; environment = env, variant = variant, pair = pair,
        model_a = name_a, model_b = name_b,
        n_a = nrow(a), n_b = nrow(b), n_shared = nrow(both_a),
        n_only_a = nrow(only_a), n_only_b = nrow(only_b),
        declined_by_a_pct = total_b == 0 ? NaN : 100 * nrow(only_b) / total_b,
        only_b_roi_pct = sb.roi, only_b_win_rate_pct = 100 * sb.win_rate,
        only_b_pnl = sb.pnl_sum,
        only_a_roi_pct = sa.roi, only_a_win_rate_pct = 100 * sa.win_rate,
        only_a_pnl = sa.pnl_sum,
        shared_p_model_a = sh_a.p_model_mean, shared_p_model_b = sh_b.p_model_mean,
        shared_roi_a_pct = sh_a.roi, shared_roi_b_pct = sh_b.roi)]
end

# %%
# ===================================================================
# 4. Data, arms and the 710-fixture walk-forward panel
# ===================================================================
r06_ds = gjn_load_data()
r06_book, r06_policy = gjn_option_b()
println("  book   : ", r06_book)
println("  policy : ", r06_policy)

r06_arms = gjn_arms(R06_CONFIG)
r06_raw_fits = Dict{String,Any}()
for arm in r06_arms
    r06_raw_fits[arm.label] = gjn_load_arm(arm)
end
r06_panel_ids = gjn_common_panel(r06_ds, r06_raw_fits, R06_CONFIG.target_seasons)
length(r06_panel_ids) == R06_CONFIG.expected_oos || error(
    "common panel is $(length(r06_panel_ids)) fixtures; expected $(R06_CONFIG.expected_oos)")

r06_fits = Dict{String,Any}()
for arm in r06_arms
    r06_fits[arm.label] = gjn_restrict(r06_raw_fits[arm.label], r06_panel_ids)
end
println("  arms   : ", length(r06_arms), " over a common ", length(r06_panel_ids),
        "-fixture walk-forward panel")

# %%
# ===================================================================
# 5. The two market environments
# ===================================================================
# Environment A stakes the r05 book — the de-vigged TWA(−20, 0] close. Its INVERSION,
# though, is read off a `point_in_time_book` at T+0: that is the frame that carries
# `:as_of_minutes`, so `calibrate_fit`'s instant assertion can actually fire, and it is
# the frame Task 007 inverted for this same calibrator name. The two describe the same
# closing market by two estimators (window average vs last visible tick).
r06_close_odds = gjn_betfair_closing_odds(r06_ds)
r06_close_pit, r06_close_refusals = R06_CAL.point_in_time_book(
    r06_ds; config = R06_CAL.PointInTimeBookConfig(as_of_minutes = 0.0))
r06_t25_book, r06_t25_refusals = R06_CAL.point_in_time_book(
    r06_ds; config = R06_CAL.PointInTimeBookConfig(as_of_minutes = -25.0))

for (key, bk, rf) in (("close_pit T+0", r06_close_pit, r06_close_refusals),
                      ("t25 T−25", r06_t25_book, r06_t25_refusals))
    cov = R06_CAL.book_coverage(bk, rf)
    @printf("  book %-14s %6d rows | %4d fixtures | staleness med %.0f p90 %.0f | overround %.4f | %d refused markets\n",
            key, cov.n_rows, cov.n_fixtures, cov.median_staleness, cov.p90_staleness,
            cov.median_overround, cov.n_refused_markets)
end

# `environments`: each is a staking book, an inversion book, and the variants to run.
const R06_ENVIRONMENTS = [
    (; key = "close", as_of = 0.0, odds = r06_close_odds, inversion_book = r06_close_pit,
       label = "de-vigged Betfair TWA(−20, 0] close",
       variants = [(; key = "raw", cal = nothing),
                   (; key = "close_std", cal = r06_cal_close_prompt()),
                   (; key = "close_std_t007", cal = r06_cal_close_task007())]),
    (; key = "t25", as_of = -25.0, odds = r06_t25_book, inversion_book = r06_t25_book,
       label = "tradeable T−25 point-in-time order book",
       variants = [(; key = "raw", cal = nothing),
                   (; key = "t25_inv", cal = GJN_MD.option_b_calibrator())]),
]

# %%
# ===================================================================
# 6. Panels, inversions and the simulation grid
# ===================================================================
r06_rows = NamedTuple[]
r06_family_out = NamedTuple[]
r06_weight_rows = NamedTuple[]
r06_panel_rows = NamedTuple[]
r06_inversion_rows = NamedTuple[]
r06_results = Dict{Tuple{String,String,String},Any}()   # (env, variant, model)
r06_panels = Dict{String,Vector{Int}}()

for env in R06_ENVIRONMENTS
    println("\n" * "-"^96)
    println("  ENVIRONMENT ", uppercase(env.key), " — ", env.label)
    println("-"^96)

    quoted = sort!(collect(intersect(Set(r06_panel_ids), Set(Int.(env.odds.match_id)))))
    env_odds = filter(:match_id => in(Set(quoted)), DataFrame(env.odds))
    panel, dropped = gjn_buildable_panel(r06_book, r06_fits, env_odds, r06_ds, quoted)
    r06_panels[env.key] = panel
    CSV.write(joinpath(R06_OUT_DIR, "r06_dropped_$(env.key).csv"), dropped)
    @printf("  panel  : %d walk-forward → %d quoted → %d buildable by every arm (%d dropped)\n",
            length(r06_panel_ids), length(quoted), length(panel), nrow(dropped))
    push!(r06_panel_rows, (; environment = env.key, as_of_minutes = env.as_of,
                             book = env.label, n_walk_forward = length(r06_panel_ids),
                             n_quoted = length(quoted), n_buildable = length(panel),
                             n_dropped = nrow(dropped)))

    # The Nelder-Mead inversion back to (λ_mkt_h, λ_mkt_a) depends on the BOOK ALONE —
    # not on the model, not on the law. One pass per environment, reused by every arm and
    # every calibrator in it.
    rates = R06_CAL.invert_market_rates(first(v for v in env.variants if v.cal !== nothing).cal,
                                        env.inversion_book; match_ids = panel)
    icov = R06_CAL.inversion_coverage(rates, panel)
    @printf("  invert : %d/%d panel fixtures accepted (%.1f%% of all, %.1f%% of quoted)\n",
            icov.n_accepted, icov.n_fixtures, 100 * icov.coverage, 100 * icov.coverage_quoted)
    for (reason, n) in R06_CAL.inversion_refusals(rates)
        @printf("           refused %4d  %s\n", n, reason)
    end
    push!(r06_inversion_rows, (; environment = env.key, n_fixtures = icov.n_fixtures,
                                 n_accepted = icov.n_accepted, n_refused = icov.n_refused,
                                 n_absent = icov.n_absent,
                                 coverage_pct = 100 * icov.coverage,
                                 coverage_quoted_pct = 100 * icov.coverage_quoted))
    CSV.write(joinpath(R06_OUT_DIR, "r06_inversion_$(env.key).csv"),
              R06_CAL.inversion_frame(rates))

    for arm in r06_arms
        fit = gjn_restrict(r06_fits[arm.label], panel)
        for v in env.variants
            source = if v.cal === nothing
                fit
            else
                cf = calibrate_fit(v.cal, fit, env.inversion_book; rates = rates, quiet = true)
                ws = R06_CAL.weight_summary(cf.rate_diagnostics)
                push!(r06_weight_rows, (; environment = env.key, variant = v.key,
                                          model = arm.label, calibrator = v.cal.name,
                                          law = R06_CAL.law_label(v.cal.law),
                                          n_shifted = ws.n_shifted, w_median = ws.w_median,
                                          w_p10 = ws.w_p10, w_p90 = ws.w_p90,
                                          var_retention_median = ws.var_retention_median,
                                          market_share_median = ws.market_share_median))
                cf
            end
            label = string(arm.label, "/", env.key, "/", v.key)
            result = r06_simulate(r06_book, r06_policy, source, env_odds, r06_ds, panel;
                                  label = label)
            r06_results[(env.key, v.key, arm.label)] = result
            row = gjn_portfolio_row(arm.label, arm.likelihood, result; n_panel = length(panel))
            edge = gjn_edge_summary(result.trajectory.bets)
            push!(r06_rows, (; environment = env.key, variant = v.key,
                               calibrator = v.cal === nothing ? "none" : v.cal.name,
                               law = v.cal === nothing ? "none" : R06_CAL.law_label(v.cal.law),
                               role = arm.role, row...,
                               capture_ratio = edge.capture_ratio,
                               cap_weighted_win_rate_pct = 100 * edge.cap_weighted_win_rate,
                               run_id = string(arm.run_id)))
            append!(r06_family_out, r06_family_rows(env.key, v.key, arm.label,
                                                    arm.likelihood, result.trajectory.bets))
            @printf("  %-34s %-15s return %+9.2f%%  ROI %+6.2f%%  Sharpe %6.3f  MDD %7.2f%%  bets %4d  capture %.3f\n",
                    arm.label, v.key, row.total_return_pct, row.roi_pct, row.sharpe_ann,
                    row.max_drawdown_pct, row.n_bets, edge.capture_ratio)
        end
    end
end

r06_summary = DataFrame(r06_rows)
r06_families = DataFrame(r06_family_out)
r06_weights = DataFrame(r06_weight_rows)

# %%
# ===================================================================
# 7. Gate — the close/raw rows must reproduce r05
# ===================================================================
# Environment A's raw variant IS r05: the same book, the same contract, the same panel
# construction, the same seed. If it does not reproduce r05's returns to the last decimal,
# something in this runner changed the simulation and every calibrated row beside it is
# describing a different experiment.
r06_gate_rows = NamedTuple[]
let r05_path = joinpath(R06_CONFIG.save_root, "portfolio", "r05_portfolio_summary.csv")
    if isfile(r05_path)
        r05 = CSV.read(r05_path, DataFrame)
        raw_close = filter(r -> r.environment == "close" && r.variant == "raw", r06_summary)
        for r in eachrow(raw_close)
            j = findfirst(==(r.model), r05.model)
            j === nothing && continue
            d_ret = r.total_return_pct - r05.total_return_pct[j]
            d_bets = r.n_bets - r05.n_bets[j]
            push!(r06_gate_rows, (; model = r.model, r06_return_pct = r.total_return_pct,
                                    r05_return_pct = r05.total_return_pct[j],
                                    delta_return_pct = d_ret,
                                    r06_n_bets = r.n_bets, r05_n_bets = r05.n_bets[j],
                                    delta_n_bets = d_bets))
        end
        worst = isempty(r06_gate_rows) ? 0.0 : maximum(abs(g.delta_return_pct) for g in r06_gate_rows)
        bad = count(g -> g.delta_n_bets != 0, r06_gate_rows)
        @printf("\n  r05 REPRODUCTION GATE: %d arms compared, worst |Δ return| %.6f pp, %d bet-count mismatches\n",
                length(r06_gate_rows), worst, bad)
        (worst < 1e-6 && bad == 0) || error(
            "the close/raw rows do not reproduce r05 (worst |Δ return| $(worst) pp, " *
            "$(bad) bet-count mismatches); the calibrated rows beside them cannot be read " *
            "as a change to r05's experiment")
    else
        println("\n  r05 REPRODUCTION GATE: skipped — $(r05_path) not present")
    end
end

# %%
# ===================================================================
# 8. Raw vs calibrated, arm by arm — what the pool actually moved
# ===================================================================
r06_rawcal_rows = NamedTuple[]
r06_rawcal_attr = NamedTuple[]
for env in R06_ENVIRONMENTS, arm in r06_arms
    raw = r06_results[(env.key, "raw", arm.label)].trajectory.bets
    for v in env.variants
        v.key == "raw" && continue
        cal = r06_results[(env.key, v.key, arm.label)].trajectory.bets
        rows, sizing = gjn_pair_attribution("$(arm.label) cal[$(v.key)] vs raw", "raw", cal, raw)
        for r in rows
            push!(r06_rawcal_attr, (; environment = env.key, variant = v.key,
                                      model = arm.label, likelihood = arm.likelihood, r...))
        end
        push!(r06_rawcal_rows, (; environment = env.key, variant = v.key, model = arm.label,
                                  likelihood = arm.likelihood, role = arm.role, sizing...))
    end
end
r06_rawcal = DataFrame(r06_rawcal_rows)
r06_rawcal_attribution = DataFrame(r06_rawcal_attr)

# %%
# ===================================================================
# 9. NegBin vs Poisson, within each environment and variant
# ===================================================================
# The r05 contrast, repeated inside every calibration state. If calibration collapses the
# difference between the likelihoods, `overlap_pct` rises toward 100 and
# `sizing_delta_pnl` toward zero; if the dispersion still drives the book, they do not.
r06_pair_rows = NamedTuple[]
r06_pair_attr = NamedTuple[]
r06_over15_rows = NamedTuple[]
r06_all_declined = NamedTuple[]
for env in R06_ENVIRONMENTS, v in env.variants, (nb, po) in GJN_PAIRS
    a = r06_results[(env.key, v.key, nb)].trajectory.bets
    b = r06_results[(env.key, v.key, po)].trajectory.bets
    rows, sizing = gjn_pair_attribution(nb, po, a, b)
    for r in rows
        push!(r06_pair_attr, (; environment = env.key, variant = v.key, r...))
    end
    push!(r06_pair_rows, (; environment = env.key, variant = v.key, sizing...))
    append!(r06_over15_rows, r06_declined_rows("$(nb) vs $(po)", env.key, v.key, nb, po,
                                               a, b; family_filter = r06_is_over15))
    append!(r06_all_declined, r06_declined_rows("$(nb) vs $(po)", env.key, v.key, nb, po,
                                                a, b))
    @printf("  %-8s %-15s %-34s shared %4d (%.0f%%)  sizing ΔPnL %+.4f  shared ROI %+.2f%% vs %+.2f%%\n",
            env.key, v.key, nb, sizing.n_shared, sizing.overlap_pct,
            sizing.sizing_delta_pnl, sizing.shared_roi_a_pct, sizing.shared_roi_b_pct)
end
r06_pairs = DataFrame(r06_pair_rows)
r06_pair_attribution = DataFrame(r06_pair_attr)
r06_over15 = DataFrame(r06_over15_rows)
r06_declined = DataFrame(r06_all_declined)

# %%
# ===================================================================
# 10. Write everything
# ===================================================================
sort!(r06_summary, [:environment, :variant, order(:total_return_pct, rev = true)])
CSV.write(R06_SUMMARY_PATH, r06_summary)
CSV.write(joinpath(R06_OUT_DIR, "r06_portfolio_summary.csv"), r06_summary)
CSV.write(joinpath(R06_OUT_DIR, "r06_family_returns.csv"), r06_families)
CSV.write(joinpath(R06_OUT_DIR, "r06_calibration_weights.csv"), r06_weights)
CSV.write(joinpath(R06_OUT_DIR, "r06_raw_vs_calibrated.csv"), r06_rawcal)
CSV.write(joinpath(R06_OUT_DIR, "r06_raw_vs_calibrated_attribution.csv"), r06_rawcal_attribution)
CSV.write(joinpath(R06_OUT_DIR, "r06_negbin_vs_poisson.csv"), r06_pairs)
CSV.write(joinpath(R06_OUT_DIR, "r06_negbin_vs_poisson_attribution.csv"), r06_pair_attribution)
CSV.write(joinpath(R06_OUT_DIR, "r06_over15_declines.csv"), r06_over15)
CSV.write(joinpath(R06_OUT_DIR, "r06_all_declines.csv"), r06_declined)
CSV.write(joinpath(R06_OUT_DIR, "r06_panels.csv"), DataFrame(r06_panel_rows))
CSV.write(joinpath(R06_OUT_DIR, "r06_inversion_coverage.csv"), DataFrame(r06_inversion_rows))
isempty(r06_gate_rows) || CSV.write(joinpath(R06_OUT_DIR, "r06_r05_gate.csv"),
                                    DataFrame(r06_gate_rows))

r06_f2 = v -> gjn_num(v; digits = 2)
r06_f3 = v -> gjn_num(v; digits = 3)
open(joinpath(R06_OUT_DIR, "r06_calibrated_portfolio_report.md"), "w") do io
    println(io, "# r06 raw vs calibrated portfolio — Task 014\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Contract: `MatchDay.option_b_system()`, identical for every row. ",
            "No MCMC: all eight posteriors loaded by UUID.\n")
    println(io, "Two market environments, each with its own buildable panel. Bankroll ",
            "figures are comparable WITHIN an environment and not across them.\n")
    print(io, gjn_markdown_table(DataFrame(r06_panel_rows)))

    println(io, "\n## Headline — every arm, every variant\n")
    print(io, gjn_markdown_table(select(r06_summary,
        :environment, :variant, :model, :likelihood, :n_bets, :total_return_pct,
        :roi_pct, :sharpe_ann, :max_drawdown_pct, :win_rate_pct, :capture_ratio,
        :mean_edge_pp, :p_roi_positive);
        formats = Dict(:total_return_pct => r06_f2, :roi_pct => r06_f2,
                       :sharpe_ann => r06_f3, :max_drawdown_pct => r06_f2,
                       :win_rate_pct => r06_f2, :capture_ratio => r06_f3,
                       :mean_edge_pp => r06_f2, :p_roi_positive => r06_f3)))

    println(io, "\n## What the calibrator did to the posterior\n")
    println(io, "`w_median` is the weight kept on the MODEL's log-rate; `1 - w` is the ",
            "market's share of the pooled location. `var_retention_median` is the ",
            "retained posterior log-variance.\n")
    print(io, gjn_markdown_table(select(r06_weights,
        :environment, :variant, :model, :law, :n_shifted, :w_median, :w_p10, :w_p90,
        :var_retention_median, :market_share_median);
        formats = Dict(:w_median => r06_f3, :w_p10 => r06_f3, :w_p90 => r06_f3,
                       :var_retention_median => r06_f3, :market_share_median => r06_f3)))

    println(io, "\n## Raw vs calibrated, arm by arm\n")
    println(io, "`n_only_a` are bets only the CALIBRATED ledger struck, `n_only_b` only ",
            "the raw one. On the shared set price and settlement are identical, so ",
            "`sizing_delta_pnl` is stake size alone.\n")
    print(io, gjn_markdown_table(select(r06_rawcal,
        :environment, :variant, :model, :likelihood, :n_shared, :n_only_a, :n_only_b,
        :overlap_pct, :sizing_delta_pnl, :shared_roi_a_pct, :shared_roi_b_pct);
        formats = Dict(:overlap_pct => r06_f2,
                       :sizing_delta_pnl => v -> gjn_signed(v; digits = 4),
                       :shared_roi_a_pct => r06_f2, :shared_roi_b_pct => r06_f2)))

    println(io, "\n## NegBin vs Poisson, inside each calibration state\n")
    print(io, gjn_markdown_table(select(r06_pairs,
        :environment, :variant, :pair, :n_shared, :n_only_a, :n_only_b, :overlap_pct,
        :sizing_delta_pnl, :shared_roi_a_pct, :shared_roi_b_pct,
        :capture_ratio_a, :capture_ratio_b);
        formats = Dict(:overlap_pct => r06_f2,
                       :sizing_delta_pnl => v -> gjn_signed(v; digits = 4),
                       :shared_roi_a_pct => r06_f2, :shared_roi_b_pct => r06_f2,
                       :capture_ratio_a => r06_f3, :capture_ratio_b => r06_f3)))

    println(io, "\n## The Over 1.5 drag\n")
    println(io, "`n_only_b` is the count of Over 1.5 bets the POISSON control struck and ",
            "the NegBin rung declined; `declined_by_a_pct` is that as a share of the ",
            "control's whole Over 1.5 book, and `only_b_roi_pct` is what those declined ",
            "bets returned the control. The question is whether calibration shrinks the ",
            "first two.\n")
    print(io, gjn_markdown_table(select(r06_over15,
        :environment, :variant, :pair, :n_a, :n_b, :n_shared, :n_only_a, :n_only_b,
        :declined_by_a_pct, :only_b_roi_pct, :only_b_win_rate_pct,
        :shared_p_model_a, :shared_p_model_b);
        formats = Dict(:declined_by_a_pct => r06_f2, :only_b_roi_pct => r06_f2,
                       :only_b_win_rate_pct => r06_f2,
                       :shared_p_model_a => v -> gjn_num(v; digits = 4),
                       :shared_p_model_b => v -> gjn_num(v; digits = 4))))

    println(io, "\n## Return by selection family\n")
    print(io, gjn_markdown_table(select(sort(r06_families,
            [:environment, :variant, :selection_family, :model]),
        :environment, :variant, :selection_family, :model, :likelihood, :n_bets,
        :win_rate_pct, :roi_pct, :edge_mean_pp, :capture_ratio);
        formats = Dict(:win_rate_pct => r06_f2, :roi_pct => r06_f2,
                       :edge_mean_pp => r06_f2, :capture_ratio => r06_f3)))

    if !isempty(r06_gate_rows)
        println(io, "\n## r05 reproduction gate\n")
        print(io, gjn_markdown_table(DataFrame(r06_gate_rows);
            formats = Dict(:r06_return_pct => r06_f2, :r05_return_pct => r06_f2,
                           :delta_return_pct => v -> gjn_signed(v; digits = 8))))
    end
end

println("\nR06_DONE report=", joinpath(R06_OUT_DIR, "r06_calibrated_portfolio_report.md"))
println("R06_SUMMARY=", R06_SUMMARY_PATH)
