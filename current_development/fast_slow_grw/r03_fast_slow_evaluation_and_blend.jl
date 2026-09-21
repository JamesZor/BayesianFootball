# ==============================================================================
# r03 — Draw-concatenation mixtures: headline benchmark on the 40-fold panel
# ==============================================================================
#
# WHAT THIS IS
#
# Loads the four `r02` arms by name from `fast_slow_grw_scottish_lower`, forms the
# tight ⊕ loose posterior mixtures at ρ ∈ {0, 0.25, 0.5, 0.75, 1} (ρ = share of
# draws from the loose arm; `mixture_latents`), and scores every mixture on:
#
#   1. supremacy slope   OLS of E[log λ_h − log λ_a] on the inverted Betfair close
#   2. capital ≥ 4.0     stake-weighted share of the ledger at odds ≥ 4.0
#   3. capital ≤ 1.8     stake-weighted share at odds ≤ 1.8
#   4. max drawdown      of the compounding bankroll
#   5. Sharpe            annualised, from `simulate_portfolio`
#   6. flat ROI          one unit on every bet the policy placed
#
# plus the favourite-tail table and derivative-market (1X2 / O/U 2.5 / BTTS)
# log loss and mean-probability drift against the close.
#
# STAKING. The work package's system: BookSpec(1X2, O/U 2.5, BakerMcHale), PolicySpec(
# FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25)), priced off the de-vigged
# Betfair (−20, 0] TWA close. Every mixture stakes the SAME panel — fixtures any arm
# cannot build a book for are dropped once, for all.
#
# A mixture is a linear probability pool at the fixture level (each market price is
# an average over draws), so no separate linear-pool comparison is run.
#
# USAGE (mcmc-beast, from /root/BF_fast_slow_grw)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/fast_slow_grw/r03_fast_slow_evaluation_and_blend.jl
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

include(joinpath(@__DIR__, "l02_fast_slow_evaluation.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R03_CONFIG = FSGConfig()
const R03_BOOTSTRAP_B = 1000
const R03_SEED = 20260921
const R03_OUT_DIR = joinpath(R03_CONFIG.save_root, "evaluation")
const R03_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
mkpath(R03_OUT_DIR)

println("\n" * "="^96)
println("  r03 DRAW-MIXTURE BENCHMARK — TODO 021, 40 folds / 710 fixtures")
println("  ρ grid : ", R03_CONFIG.rhos, "   git: ", R03_GIT, "   threads: ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Data, book, arms, panel
# ===================================================================
r03_ds = gph_load_data()
r03_odds = gph_betfair_closing_odds(r03_ds)
r03_book, r03_policy = fsg_system()
println("  book   : ", r03_book)
println("  policy : ", r03_policy)

r03_fits, r03_run_ids = fsg_load_arms(r03_ds, R03_CONFIG)
for name in FSG_MODEL_NAMES
    println("  arm ", rpad(name, 36), r03_run_ids[name], "  ",
            n_matches(r03_fits[name].latents), " × ", n_draws(r03_fits[name].latents))
end

r03_quoted = sort!(collect(intersect(Set(r03_fits[FSG_TIGHT].latents.match_ids),
                                     Set(r03_odds.match_id))))
r03_panel, r03_dropped = gph_buildable_panel(r03_book, r03_fits, r03_odds, r03_ds, r03_quoted)
CSV.write(joinpath(R03_OUT_DIR, "r03_dropped_fixtures.csv"), r03_dropped)
println("  panel  : 710 walk-forward → ", length(r03_quoted), " quoted → ",
        length(r03_panel), " buildable by every arm")

r03_market = fsg_market_frame(r03_odds, r03_fits[FSG_TIGHT].latents.match_ids)
println("  market : ", nrow(r03_market), " fixtures with an accepted rate inversion")

# %%
# ===================================================================
# 4. Mixtures → six headline metrics
# ===================================================================
r03_model = r03_fits[FSG_TIGHT].config.model
r03_grid = fsg_mixture_grid(r03_fits, R03_CONFIG)
r03_rows = NamedTuple[]
r03_tails = DataFrame()
r03_families = DataFrame()

for entry in r03_grid
    t0 = time()
    lat = entry.fit.latents
    sup = fsg_supremacy_report(lat, r03_model, r03_market)
    result = gph_simulate(r03_book, r03_policy, entry.fit, r03_odds, r03_ds, r03_panel;
                          label = entry.label, B = R03_BOOTSTRAP_B, seed = R03_SEED)
    row = gph_portfolio_row(entry.label, "MultiScaleGRW", result; n_panel = length(r03_panel))
    ledger = fsg_ledger_metrics(result.trajectory.bets)
    push!(r03_rows, (; label = entry.label, loose = entry.loose, rho = entry.rho,
                       sup_slope = sup.slope, sup_r2 = sup.r2, sup_n = sup.n, sup_sd = sup.sup_sd,
                       ledger.capital_ge4_pct, ledger.capital_le18_pct,
                       max_drawdown_pct = row.max_drawdown_pct, sharpe_ann = row.sharpe_ann,
                       ledger.flat_roi_pct,
                       total_return_pct = row.total_return_pct, roi_pct = row.roi_pct,
                       p_roi_positive = row.p_roi_positive, n_bets = row.n_bets,
                       median_odds = ledger.median_odds, mean_edge_pp = row.mean_edge_pp,
                       max_win_prob = sup.max_win_prob))
    tail = fsg_favourite_tail(lat, r03_model, r03_market)
    tail.label .= entry.label
    tail.rho .= entry.rho
    append!(r03_tails, tail)
    fam = fsg_family_scores(lat, r03_model, r03_odds)
    fam.label .= entry.label
    fam.rho .= entry.rho
    append!(r03_families, fam)
    @printf("  %-44s slope %.3f  ≥4.0 %5.1f%%  ≤1.8 %5.1f%%  MDD %5.1f%%  Sharpe %6.3f  flatROI %+6.2f%%  (%.0fs)\n",
            entry.label, sup.slope, ledger.capital_ge4_pct, ledger.capital_le18_pct,
            row.max_drawdown_pct, row.sharpe_ann, ledger.flat_roi_pct, time() - t0)
end
r03_summary = DataFrame(r03_rows)

# %%
# ===================================================================
# 5. Report
# ===================================================================
CSV.write(joinpath(R03_OUT_DIR, "r03_headline_metrics.csv"), r03_summary)
CSV.write(joinpath(R03_OUT_DIR, "r03_favourite_tail.csv"), r03_tails)
CSV.write(joinpath(R03_OUT_DIR, "r03_family_scores.csv"), r03_families)

open(joinpath(R03_OUT_DIR, "r03_evaluation_report.md"), "w") do io
    println(io, "# r03 draw-mixture benchmark — TODO 021\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R03_GIT, "` on ",
            gethostname(), ". Panel: ", length(r03_panel), " buildable of ", length(r03_quoted),
            " quoted of 710 walk-forward fixtures; ", nrow(r03_market),
            " with an accepted market-rate inversion (supremacy slope). Bootstrap B = ",
            R03_BOOTSTRAP_B, ".\n")
    println(io, "Runs: ", join(["`$(n)` $(r03_run_ids[n])" for n in FSG_MODEL_NAMES], ", "), ".\n")
    println(io, "## Headline metrics\n")
    print(io, gph_markdown_table(select(r03_summary, :label, :rho, :sup_slope, :sup_r2,
        :capital_ge4_pct, :capital_le18_pct, :max_drawdown_pct, :sharpe_ann, :flat_roi_pct,
        :total_return_pct, :p_roi_positive, :n_bets);
        formats = Dict(:rho => v -> gph_num(v; digits = 2),
                       :capital_ge4_pct => v -> gph_num(v; digits = 1),
                       :capital_le18_pct => v -> gph_num(v; digits = 1),
                       :max_drawdown_pct => v -> gph_num(v; digits = 1),
                       :flat_roi_pct => v -> gph_num(v; digits = 2),
                       :total_return_pct => v -> gph_num(v; digits = 1))))
    println(io, "\n## Favourite tail (1X2, market favourite side)\n")
    print(io, gph_markdown_table(select(r03_tails, :label, :band, :n, :p_market, :p_model)))
    println(io, "\n## Derivative markets vs the close\n")
    print(io, gph_markdown_table(select(r03_families, :label, :family, :n_fixtures,
        :logloss_model, :logloss_close, :headline, :mean_p_model, :mean_p_close);
        formats = Dict(:logloss_model => v -> gph_num(v; digits = 5),
                       :logloss_close => v -> gph_num(v; digits = 5))))
end

println("\nR03_DONE report=", joinpath(R03_OUT_DIR, "r03_evaluation_report.md"))
