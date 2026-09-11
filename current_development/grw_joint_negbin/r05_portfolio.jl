# ==============================================================================
# r05 — Closing-line portfolio under Option B, with Task 012 attribution
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# Every arm staked through ONE contract — `MatchDay.option_b_system()`, the audited
# production book and policy — against the de-vigged Betfair TWA(−20, 0] close, over
# the identical buildable subset of the walk-forward panel. Only the posterior differs
# between rows.
#
# Then the Task 012 decomposition, pair by pair, for each NegBin rung against its
# Task 013 Poisson counterpart:
#
#   capture ratio      E[edge | won] / E[edge | lost] over each whole ledger
#   shared-bet sizing  on bets both models took (same fixture, selection, price,
#                      outcome): Σ(s_a − s_b)·settle is pure sizing alpha
#   disjoint bets      turnover, strike rate and ROI of each model's exclusive bets
#   cap-weighted WR    stake-weighted strike rate vs the plain one
#
# WHY THE ATTRIBUTION IS THE INTERESTING PART HERE. Option B's TieredTrust stakes FIVE
# selections — 1X2 home/draw/away, Under 2.5 at full trust, and Over 1.5 at 1/1.4. (The
# work package describes the basket as "1X2 + O/U 2.5"; Over 1.5 is in it too, and it
# matters here because it is a second totals line this component moves.)
#
# A negative binomial changes the two totals prices and barely touches 1X2, so the two
# ledgers should overlap heavily on the result market and diverge on the totals. The
# shared/exclusive split is what separates "it priced the same bets differently" from
# "it took different bets", and at this sample size that decomposition is more
# informative than the bankroll ranking it sits beside.
#
# It is a closing-line simulation: prices are the close, not a tradeable T−25 book. At
# n ≈ 600 fixtures the bootstrap growth intervals of neighbouring arms overlap, so a
# ranking here is descriptive, not a result.
#
# PERSISTENCE. The four ladder portfolios are written to `portfolio_runs` /
# `portfolio_bets` / `portfolio_artifacts`, linked to their model run UUIDs, and each is
# reloaded and its bet ledger compared for identity.
#
# USAGE (mcmc-beast, from /root/BF_grw_joint_negbin)
#
#   julia --project -t 16 current_development/grw_joint_negbin/r05_portfolio.jl
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
const R05_CONFIG = GJNConfig()
const R05_BOOTSTRAP_B = 4000
const R05_SEED = 1
const R05_OUT_DIR = joinpath(R05_CONFIG.save_root, "portfolio")

# The four matched contrasts, plus the two that place the NegBin hybrid against the
# ladder rungs it is built from.
const R05_PAIRS = vcat(
    [(negbin, poisson) for (negbin, poisson) in GJN_PAIRS],
    [("m12_joint_hybrid_synergy_negbin", "m05_wealth_grw_negbin"),
     ("m10_lineup_grw_negbin", "m00_baseline_grw_negbin")],
)

mkpath(R05_OUT_DIR)
println("\n" * "="^96)
println("  r05 CLOSING-LINE PORTFOLIO — Option B, Task 012 attribution")
println("="^96)

# %%
# ===================================================================
# 3. Data, book and arms
# ===================================================================
r05_ds = gjn_load_data()
r05_odds = gjn_betfair_closing_odds(r05_ds)
r05_book, r05_policy = gjn_option_b()
println("  book   : ", r05_book)
println("  policy : ", r05_policy)

r05_arms = gjn_arms(R05_CONFIG)
r05_raw = Dict{String,Any}()
for arm in r05_arms
    r05_raw[arm.label] = gjn_load_arm(arm)
end
r05_panel_ids = gjn_common_panel(r05_ds, r05_raw, R05_CONFIG.target_seasons)
length(r05_panel_ids) == R05_CONFIG.expected_oos || error(
    "common panel is $(length(r05_panel_ids)) fixtures; expected $(R05_CONFIG.expected_oos)")

r05_fits = Dict{String,Any}()
for arm in r05_arms
    r05_fits[arm.label] = gjn_restrict(r05_raw[arm.label], r05_panel_ids)
end
r05_quoted = sort!(collect(intersect(Set(r05_panel_ids), Set(r05_odds.match_id))))
r05_panel, r05_dropped = gjn_buildable_panel(r05_book, r05_fits, r05_odds, r05_ds, r05_quoted)
CSV.write(joinpath(R05_OUT_DIR, "r05_dropped_fixtures.csv"), r05_dropped)
println("  panel  : ", length(r05_panel_ids), " walk-forward → ", length(r05_quoted),
        " quoted → ", length(r05_panel), " buildable by every arm (",
        nrow(r05_dropped), " dropped)")

# %%
# ===================================================================
# 4. Simulation — identical contract, identical panel
# ===================================================================
r05_results = Dict{String,Any}()
r05_rows = NamedTuple[]
for arm in r05_arms
    result = gjn_simulate(r05_book, r05_policy, r05_fits[arm.label], r05_odds, r05_ds, r05_panel;
                          label = arm.label, B = R05_BOOTSTRAP_B, seed = R05_SEED)
    r05_results[arm.label] = result
    row = gjn_portfolio_row(arm.label, arm.likelihood, result; n_panel = length(r05_panel))
    edge = gjn_edge_summary(result.trajectory.bets)
    push!(r05_rows, (; row..., capture_ratio = edge.capture_ratio,
                       cap_weighted_win_rate_pct = 100 * edge.cap_weighted_win_rate,
                       run_id = string(arm.run_id)))
    @printf("  %-34s return %+8.2f%%  ROI %+6.2f%%  Sharpe %6.3f  MDD %6.2f%%  bets %4d  capture %.3f\n",
            arm.label, row.total_return_pct, row.roi_pct, row.sharpe_ann, row.max_drawdown_pct,
            row.n_bets, edge.capture_ratio)
end
r05_summary = sort(DataFrame(r05_rows), :total_return_pct; rev = true)

# %%
# ===================================================================
# 5. Task 012 attribution, pair by pair
# ===================================================================
r05_attr_rows = NamedTuple[]
r05_sizing_rows = NamedTuple[]
for (a, b) in R05_PAIRS
    haskey(r05_results, a) && haskey(r05_results, b) || continue
    rows, sizing = gjn_pair_attribution(a, b, r05_results[a].trajectory.bets,
                                        r05_results[b].trajectory.bets)
    append!(r05_attr_rows, rows)
    push!(r05_sizing_rows, sizing)
    @printf("  %-62s shared %4d (%.0f%%)  sizing ΔPnL %+.4f  shared ROI %+.2f%% vs %+.2f%%\n",
            sizing.pair, sizing.n_shared, sizing.overlap_pct, sizing.sizing_delta_pnl,
            sizing.shared_roi_a_pct, sizing.shared_roi_b_pct)
end
r05_attribution = DataFrame(r05_attr_rows)
r05_sizing = DataFrame(r05_sizing_rows)

# Per selection family, per arm: where each ledger's return came from. This is the cut
# that should separate the two likelihoods — the totals rows, not the 1X2 rows.
r05_family_rows = NamedTuple[]
for arm in r05_arms
    bets = r05_results[arm.label].trajectory.bets
    nrow(bets) == 0 && continue
    for sub in groupby(sort(DataFrame(bets), :family), :family)
        s = gjn_edge_summary(sub)
        push!(r05_family_rows, (; model = arm.label, likelihood = arm.likelihood,
                                  selection_family = first(sub.family),
                                  n_bets = s.n_bets, win_rate_pct = 100 * s.win_rate,
                                  roi_pct = s.roi, stake_sum = s.stake_sum,
                                  pnl_sum = s.pnl_sum, edge_mean_pp = s.edge_mean,
                                  capture_ratio = s.capture_ratio))
    end
end
r05_families = DataFrame(r05_family_rows)

# %%
# ===================================================================
# 6. Persistence — the four ladder portfolios, with ledger round-trip
# ===================================================================
r05_persisted = NamedTuple[]
for arm in r05_arms
    arm.role == "ladder" || continue
    db = PostgresStorage(arm.experiment)
    result = r05_results[arm.label]
    portfolio_id = save_portfolio_db(result, arm.run_id, db;
        book_spec = r05_book, policy_spec = r05_policy,
        metadata = (; runner = "r05_portfolio", contract = "option_b_system",
                      odds = "betfair_twa_-20_0", n_panel = length(r05_panel), task = "014"))
    reloaded = load_portfolio_db(portfolio_id, db)
    reloaded.trajectory.bets == result.trajectory.bets || error(
        "$(arm.label): reloaded portfolio ledger differs from the simulated one")
    push!(r05_persisted, (; model = arm.label, model_run_id = string(arm.run_id),
                            portfolio_run_id = string(portfolio_id)))
    println("  persisted ", arm.label, " → portfolio ", portfolio_id, " (ledger reloads identically)")
end

# %%
# ===================================================================
# 7. Final report
# ===================================================================
CSV.write(joinpath(R05_OUT_DIR, "r05_portfolio_summary.csv"), r05_summary)
CSV.write(joinpath(R05_OUT_DIR, "r05_attribution.csv"), r05_attribution)
CSV.write(joinpath(R05_OUT_DIR, "r05_sizing.csv"), r05_sizing)
CSV.write(joinpath(R05_OUT_DIR, "r05_family_returns.csv"), r05_families)
CSV.write(joinpath(R05_OUT_DIR, "r05_persisted.csv"), DataFrame(r05_persisted))

r05_f2 = v -> gjn_num(v; digits = 2)
r05_f3 = v -> gjn_num(v; digits = 3)
open(joinpath(R05_OUT_DIR, "r05_portfolio_report.md"), "w") do io
    println(io, "# r05 closing-line portfolio — Task 014 (JointGammaNegBinObservation)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Contract: `MatchDay.option_b_system()`. Book: de-vigged Betfair TWA(−20, 0] close. ",
            "Panel: ", length(r05_panel), " fixtures buildable by every arm.\n")
    println(io, "Closing-line simulation, not a tradeable-price test. Growth intervals of ",
            "neighbouring arms overlap at this sample size; read the attribution, not the ranking.\n")

    println(io, "## Headline\n")
    print(io, gjn_markdown_table(select(r05_summary,
        :model, :likelihood, :n_bets, :total_return_pct, :roi_pct, :sharpe_ann,
        :calmar, :max_drawdown_pct, :win_rate_pct, :capture_ratio, :mean_edge_pp);
        formats = Dict(:total_return_pct => r05_f2, :roi_pct => r05_f2,
                       :sharpe_ann => r05_f3, :calmar => r05_f3,
                       :max_drawdown_pct => r05_f2, :win_rate_pct => r05_f2,
                       :capture_ratio => r05_f3, :mean_edge_pp => r05_f2)))

    println(io, "\n## Shared-bet sizing and overlap (Task 012)\n")
    println(io, "On the shared set the fixture, selection, price and outcome are identical, so ",
            "`sizing_delta_pnl` is attributable to stake size alone.\n")
    print(io, gjn_markdown_table(select(r05_sizing,
        :pair, :n_shared, :n_only_a, :n_only_b, :overlap_pct, :stake_mean_a, :stake_mean_b,
        :sizing_delta_pnl, :shared_roi_a_pct, :shared_roi_b_pct,
        :capture_ratio_a, :capture_ratio_b);
        formats = Dict(:overlap_pct => r05_f2, :sizing_delta_pnl => v -> gjn_signed(v; digits = 4),
                       :shared_roi_a_pct => r05_f2, :shared_roi_b_pct => r05_f2,
                       :capture_ratio_a => r05_f3, :capture_ratio_b => r05_f3,
                       :stake_mean_a => v -> gjn_num(v; digits = 5),
                       :stake_mean_b => v -> gjn_num(v; digits = 5))))

    println(io, "\n## Shared and exclusive bet sets\n")
    print(io, gjn_markdown_table(select(r05_attribution,
        :pair, :bet_set, :owner, :n_bets, :win_rate_pct, :cap_weighted_win_rate_pct,
        :roi_pct, :edge_mean_pp, :capture_ratio);
        formats = Dict(:win_rate_pct => r05_f2, :cap_weighted_win_rate_pct => r05_f2,
                       :roi_pct => r05_f2, :edge_mean_pp => r05_f2,
                       :capture_ratio => r05_f3)))

    println(io, "\n## Return by selection family\n")
    println(io, "Option B stakes 1X2 home/draw/away, Under 2.5 and Over 1.5. The two totals ",
            "rows are where the likelihoods are expected to differ; the 1X2 rows are the ",
            "control.\n")
    print(io, gjn_markdown_table(select(sort(r05_families, [:selection_family, :model]),
        :model, :likelihood, :selection_family, :n_bets, :win_rate_pct, :roi_pct,
        :edge_mean_pp, :capture_ratio);
        formats = Dict(:win_rate_pct => r05_f2, :roi_pct => r05_f2,
                       :edge_mean_pp => r05_f2, :capture_ratio => r05_f3)))
end

println("\nR05_DONE report=", joinpath(R05_OUT_DIR, "r05_portfolio_report.md"))
