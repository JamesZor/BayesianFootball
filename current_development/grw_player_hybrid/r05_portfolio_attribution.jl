# ==============================================================================
# r05 — Closing-line portfolio under Option B, with Task 012 attribution
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# Every arm staked through ONE contract — `MatchDay.option_b_system()`, the audited
# production book and policy — against the de-vigged Betfair TWA(−20, 0] close,
# over the identical buildable subset of the 710-fixture walk-forward panel. Only
# the posterior differs between rows.
#
# Then the Task 012 decomposition, pair by pair:
#
#   capture ratio      E[edge | won] / E[edge | lost] over each whole ledger
#   shared-bet sizing  on bets both models took (same fixture, selection, price,
#                      outcome): Σ(s_a − s_b)·settle is pure sizing alpha
#   disjoint bets      turnover, strike rate and ROI of each model's exclusive bets
#   cap-weighted WR    stake-weighted strike rate vs the plain one
#
# It is a closing-line simulation: prices are the close, not a tradeable T−25
# book. `r06` is the executable-price test. At n ≈ 600 fixtures the bootstrap
# growth intervals of neighbouring arms overlap; a ranking here is descriptive.
#
# PERSISTENCE. The four ladder portfolios are written to `portfolio_runs` /
# `portfolio_bets` / `portfolio_artifacts`, linked to their model run UUIDs, and
# each is reloaded and its bet ledger compared for identity.
#
# USAGE (mcmc-beast, from /root/BF_grw_player_hybrid)
#
#   julia --project -t 16 current_development/grw_player_hybrid/r05_portfolio_attribution.jl
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
const R05_CONFIG = GPHConfig()
const R05_BOOTSTRAP_B = 4000
const R05_SEED = 1
const R05_OUT_DIR = joinpath(R05_CONFIG.save_root, "portfolio")

# "a vs b": each pair answers one question about what changed between the two.
const R05_PAIRS = [
    ("m12_joint_hybrid_synergy_grw", "m12_hybrid_td_raw"),   # GRW vs TimeDecay in the full hybrid
    ("m05_wealth_grw", "m05_joint_td_raw"),                  # GRW vs TimeDecay at team level
    ("m12_joint_hybrid_synergy_grw", "m05_wealth_grw"),      # what the lineup adds on GRW + joint
    ("m10_lineup_grw", "m00_baseline_grw"),                  # what the lineup adds on GRW + Poisson
    ("m12_joint_hybrid_synergy_grw", "m05_joint_grw_raw"),   # hybrid vs the T−25 GRW benchmark
]

mkpath(R05_OUT_DIR)
println("\n" * "="^96)
println("  r05 CLOSING-LINE PORTFOLIO — Option B, Task 012 attribution")
println("="^96)

# %%
# ===================================================================
# 3. Data, book and arms
# ===================================================================
r05_ds = gph_load_data()
r05_odds = gph_betfair_closing_odds(r05_ds)
r05_book, r05_policy = gph_option_b()
println("  book   : ", r05_book)
println("  policy : ", r05_policy)

r05_arms = gph_arms(R05_CONFIG)
r05_fits = Dict{String,Any}()
for arm in r05_arms
    fit = gph_load_arm(arm)
    panel = gph_season_panel(r05_ds, fit, R05_CONFIG.target_seasons)
    length(panel) == R05_CONFIG.expected_oos || error("$(arm.label): $(length(panel)) panel fixtures")
    r05_fits[arm.label] = gph_restrict(fit, panel)
end
r05_quoted = sort!(collect(intersect(Set(r05_fits[first(GPH_MODEL_NAMES)].latents.match_ids),
                                     Set(r05_odds.match_id))))
r05_panel, r05_dropped = gph_buildable_panel(r05_book, r05_fits, r05_odds, r05_ds, r05_quoted)
CSV.write(joinpath(R05_OUT_DIR, "r05_dropped_fixtures.csv"), r05_dropped)
println("  panel  : 710 walk-forward → ", length(r05_quoted), " quoted → ",
        length(r05_panel), " buildable by every arm (", nrow(r05_dropped), " dropped)")

# %%
# ===================================================================
# 4. Simulation — identical contract, identical panel
# ===================================================================
r05_results = Dict{String,Any}()
r05_rows = NamedTuple[]
for arm in r05_arms
    result = gph_simulate(r05_book, r05_policy, r05_fits[arm.label], r05_odds, r05_ds, r05_panel;
                          label = arm.label, B = R05_BOOTSTRAP_B, seed = R05_SEED)
    r05_results[arm.label] = result
    row = gph_portfolio_row(arm.label, arm.dynamics, result; n_panel = length(r05_panel))
    edge = gph_edge_summary(result.trajectory.bets)
    push!(r05_rows, (; row..., capture_ratio = edge.capture_ratio,
                       cap_weighted_win_rate_pct = 100 * edge.cap_weighted_win_rate,
                       run_id = string(arm.run_id)))
    @printf("  %-30s return %+8.2f%%  ROI %+6.2f%%  Sharpe %6.3f  MDD %6.2f%%  bets %4d  capture %.3f\n",
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
    rows, sizing = gph_pair_attribution(a, b, r05_results[a].trajectory.bets,
                                        r05_results[b].trajectory.bets)
    append!(r05_attr_rows, rows)
    push!(r05_sizing_rows, sizing)
    @printf("  %-50s shared %4d (%.0f%%)  sizing ΔPnL %+.4f  shared ROI %+.2f%% vs %+.2f%%  capture %.3f vs %.3f\n",
            sizing.pair, sizing.n_shared, sizing.overlap_pct, sizing.sizing_delta_pnl,
            sizing.shared_roi_a_pct, sizing.shared_roi_b_pct,
            sizing.capture_ratio_a, sizing.capture_ratio_b)
end
r05_attribution = DataFrame(r05_attr_rows)
r05_sizing = DataFrame(r05_sizing_rows)

# Per selection family, per arm: where each ledger's return came from.
r05_family_rows = NamedTuple[]
for arm in r05_arms
    bets = r05_results[arm.label].trajectory.bets
    nrow(bets) == 0 && continue
    for sub in groupby(sort(DataFrame(bets), :family), :family)
        s = gph_edge_summary(sub)
        push!(r05_family_rows, (; model = arm.label, selection_family = first(sub.family),
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
        metadata = (; runner = "r05_portfolio_attribution", contract = "option_b_system",
                      odds = "betfair_twa_-20_0", n_panel = length(r05_panel), task = "013"))
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
CSV.write(joinpath(R05_OUT_DIR, "r05_shared_bet_sizing.csv"), r05_sizing)
CSV.write(joinpath(R05_OUT_DIR, "r05_selection_families.csv"), r05_families)
CSV.write(joinpath(R05_OUT_DIR, "r05_persisted_portfolios.csv"), DataFrame(r05_persisted))

r05_p2 = v -> gph_num(v; digits = 2)
r05_p3 = v -> gph_num(v; digits = 3)
open(joinpath(R05_OUT_DIR, "r05_portfolio_report.md"), "w") do io
    println(io, "# r05 closing-line portfolio and attribution — Task 013\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Contract: `MatchDay.option_b_system()` (canonical markets, DeArb, FractionalKelly 0.30, ",
            "2% commission; TieredTrust Home/U2.5 = 1, Draw/Away/O1.5 = 1/1.4; SlateDrawdown 8; ",
            "FixedCap 0.25; DailySlate). Prices: de-vigged Betfair TWA(−20, 0] close. Panel: ",
            length(r05_panel), " fixtures buildable by every arm (of ", length(r05_quoted),
            " quoted). Bootstrap B = ", R05_BOOTSTRAP_B, ".\n")
    println(io, "## Headline\n")
    print(io, gph_markdown_table(select(r05_summary, :model, :dynamics, :n_bets, :total_return_pct,
                                        :roi_pct, :growth_lo, :growth_hi, :sharpe_ann, :calmar,
                                        :max_drawdown_pct, :win_rate_pct, :cap_weighted_win_rate_pct,
                                        :capture_ratio);
        formats = Dict(:total_return_pct => r05_p2, :roi_pct => r05_p2, :sharpe_ann => r05_p3,
                       :calmar => r05_p2, :max_drawdown_pct => r05_p2, :win_rate_pct => r05_p2,
                       :cap_weighted_win_rate_pct => r05_p2, :capture_ratio => r05_p3,
                       :growth_lo => v -> gph_signed(v; digits = 5),
                       :growth_hi => v -> gph_signed(v; digits = 5))))
    println(io, "\n## Shared-bet sizing and disjoint sets\n")
    println(io, "`sizing_delta_pnl = Σ (s_a − s_b) · settle` over shared bets, in bankroll fractions; ",
            "`roi_when_a_larger` is the return on the extra stake where model A sized up.\n")
    print(io, gph_markdown_table(r05_sizing;
        formats = Dict(:overlap_pct => r05_p2, :stake_mean_a => v -> gph_num(v; digits = 5),
                       :stake_mean_b => v -> gph_num(v; digits = 5),
                       :roi_when_a_larger_pct => r05_p2, :roi_when_b_larger_pct => r05_p2,
                       :sizing_delta_pnl => v -> gph_signed(v; digits = 4),
                       :shared_roi_a_pct => r05_p2, :shared_roi_b_pct => r05_p2,
                       :capture_ratio_a => r05_p3, :capture_ratio_b => r05_p3)))
    println(io, "\n## Partition detail\n")
    print(io, gph_markdown_table(select(r05_attribution, :pair, :bet_set, :owner, :n_bets,
                                        :win_rate_pct, :cap_weighted_win_rate_pct, :roi_pct,
                                        :stake_mean, :odds_mean, :edge_win_pp, :edge_loss_pp,
                                        :capture_ratio);
        formats = Dict(:win_rate_pct => r05_p2, :cap_weighted_win_rate_pct => r05_p2,
                       :roi_pct => r05_p2, :stake_mean => v -> gph_num(v; digits = 5),
                       :odds_mean => r05_p2, :edge_win_pp => r05_p2, :edge_loss_pp => r05_p2,
                       :capture_ratio => r05_p3)))
    println(io, "\n## Persisted portfolios\n")
    print(io, gph_markdown_table(DataFrame(r05_persisted)))
end
println("\nR05_DONE report=", joinpath(R05_OUT_DIR, "r05_portfolio_report.md"))
