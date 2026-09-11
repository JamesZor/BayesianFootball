# ==============================================================================
# r02 — MultiScaleGRW portfolio backtest and Layer 2 calibration benchmark
# ==============================================================================
#
# WHAT THIS IS. A no-MCMC repricing of seven persisted posteriors — three
# MultiScaleGRW candidates, their three matched TimeDecay controls, and the
# production `m12_joint_hybrid_synergy` — on one frozen Betfair panel, under one
# staking contract, plus a Layer 2 generative rate calibration of each at two
# price instants.
#
# WHAT THIS IS NOT. A prospective return claim, and not a promotion decision.
# It is a historical simulation on an exchange archive carrying the portfolio's
# stated fill assumptions.
#
# Run:  julia --project -t 16 current_development/multiscale_grw/r02_portfolio_calibration.jl
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l02_portfolio.jl"))
using .MultiScaleGRWPortfolio

const R02_BOOTSTRAP_B = 4000
const R02_SEED = 1

banner(text) = println("\n", "="^78, "\n ", text, "\n", "="^78)

# %%
# ==============================================================================
# 2. Fixed portfolio contract
# ==============================================================================
# De-vigged Betfair TWA[-20,0], 1X2 + O/U 2.5, Baker-McHale, 2% commission.
# 30% FlatTrust, SlateDrawdown(23), FixedCap(20%), DailySlate.
#
# Every arm — raw and calibrated — is priced through this one contract. The
# calibrated variants change only the posterior; see the loader header.
book = l02_book_spec()
policy = l02_policy_spec()

banner("1. CONTRACT")
println("  book   : ", book.markets, " | Baker-McHale | 2% commission")
println("  policy : FlatTrust(0.30), SlateDrawdown(23.0), FixedCap(0.20), DailySlate")

# %%
# ==============================================================================
# 3. Data, posteriors and the frozen fixture panel
# ==============================================================================
l02_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)

banner("2. LOADING SEVEN PERSISTED POSTERIORS")
dbs, fits = l02_load_all_fits()

odds = l02_betfair_closing_odds(ds)
quoted_panel = l02_common_match_ids(fits, odds)

# Quoted is not buildable: a fixture can carry prices and still be refused as
# unplayed or for an incomplete market. Drop that union ONCE, up front, so every
# arm stakes an identical panel and the zero-skip assertion can actually pass.
panel, dropped = l02_buildable_panel(book, fits, filter(:match_id => in(Set(quoted_panel)), odds),
                                     ds, quoted_panel)
common_odds = filter(:match_id => in(Set(panel)), odds)

@printf("\n  Betfair book : %d graded selections over %d fixtures\n",
        nrow(odds), length(unique(odds.match_id)))
@printf("  quoted panel : %d fixtures common to all seven arms AND quoted\n", length(quoted_panel))
@printf("  dropped      : %d not buildable by every arm\n", nrow(dropped))
if nrow(dropped) > 0
    for sub in groupby(dropped, :reason)
        @printf("                 %-24s %d\n", first(sub.reason), nrow(sub))
    end
end
@printf("  FROZEN PANEL : %d fixtures\n", length(panel))
@printf("  panel book   : %d selections\n", nrow(common_odds))

# %%
# ==============================================================================
# 4. Layer 2 calibration — invert each instant's book once
# ==============================================================================
# The Nelder-Mead inversion back to (lambda_mkt_h, lambda_mkt_a) depends on the
# BOOK ONLY, not on the model and not on the law. Inverting once per instant and
# reusing it across the seven arms is the difference between one inversion pass
# and fourteen identical ones.
banner("3. LAYER 2 — POINT-IN-TIME BOOKS AND MARKET INVERSION")

calibrators = l02_calibrators()
pit_books = Dict{String,DataFrame}()
pit_rates = Dict{String,Any}()

for c in calibrators
    pit, refusals = BayesianFootball.point_in_time_book(
        ds; config = BayesianFootball.PointInTimeBookConfig(as_of_minutes = c.as_of))
    pit_books[c.key] = pit
    @printf("  %-10s T%+.0f : %d selections | %d fixtures | %d refusals\n",
            c.key, c.as_of, nrow(pit), length(unique(pit.match_id)), nrow(refusals))
    pit_rates[c.key] = BayesianFootball.invert_market_rates(c.cal, pit; match_ids = panel)
    # `MarketRateFit.accepted`, not `.inverted` — the latter is the diagnostics
    # frame's column name for the same fact.
    accepted = count(f -> f.accepted, values(pit_rates[c.key]))
    @printf("             inverted %d / %d panel fixtures\n", accepted, length(panel))
end

# %%
# ==============================================================================
# 5. Reprice, simulate and collect every arm
# ==============================================================================
banner("4. PORTFOLIO SIMULATION — RAW AND CALIBRATED")

summary_rows = NamedTuple[]
tearsheet_rows = NamedTuple[]
coherence_rows = NamedTuple[]
weight_rows = NamedTuple[]
results = Dict{Tuple{String,String},Any}()

for arm in L02_ARMS
    # Restrict to the frozen panel BEFORE building. Each arm holds 710 posterior
    # fixtures but the book quotes 635; restricting first is what makes the
    # zero-skip assertion in `l02_simulate_arm` meaningful.
    fit = l02_restrict_fit(fits[arm.label], panel)

    # ---- raw ----------------------------------------------------------------
    result, _ = l02_simulate_arm(book, policy, fit, common_odds, ds, panel;
                                 label = arm.label * "/raw",
                                 B = R02_BOOTSTRAP_B, seed = R02_SEED)
    results[(arm.label, "raw")] = result
    push!(summary_rows, l02_arm_row(arm, "raw", result; n_panel = length(panel)))
    append!(tearsheet_rows, l02_tearsheet_rows(arm, "raw", result))

    s = result.summary
    @printf("  %-34s raw       bets %5d | return %+8.2f%% | g %+.5f | Sharpe %6.3f | MDD %6.1f%%\n",
            arm.label, s.n_bets, s.total_return_pct, s.growth_per_slate, s.sharpe_ann, s.mdd)

    # ---- calibrated, one variant per instant --------------------------------
    for c in calibrators
        cf = try
            l02_calibrate_checked(c.cal, fit, pit_books[c.key]; rates = pit_rates[c.key])
        catch err
            @warn("calibration failed; skipping this arm/instant",
                  arm = arm.label, instant = c.key,
                  exception = (err, catch_backtrace()))
            continue
        end

        # Derivative coherence: 1X2, O/U and BTTS are three partitions of one
        # 12x12 tensor, so the spread between families must be zero to rounding.
        coh = l02_coherence_check(cf, book.markets.markets)
        push!(coherence_rows, (; model = arm.label, variant = c.key,
                                 max_family_spread = coh.max_family_spread,
                                 max_deviation_from_one = coh.max_deviation_from_one,
                                 mean_grid_mass = coh.mean_grid_mass))

        cresult, _ = l02_simulate_arm(book, policy, cf, common_odds, ds, panel;
                                      label = arm.label * "/" * c.key,
                                      B = R02_BOOTSTRAP_B, seed = R02_SEED)
        results[(arm.label, c.key)] = cresult
        push!(summary_rows, l02_arm_row(arm, c.key, cresult; n_panel = length(panel)))
        append!(tearsheet_rows, l02_tearsheet_rows(arm, c.key, cresult))

        # The weight summary belongs beside every calibration result: a median w of
        # 0.41 means the market supplied most of the location and most of the
        # posterior log-variance was destroyed, and no headline score says that.
        ws = BayesianFootball.weight_summary(cf.rate_diagnostics)
        push!(weight_rows, (; model = arm.label, variant = c.key,
                              n_shifted = ws.n_shifted, w_median = ws.w_median,
                              w_p10 = ws.w_p10, w_p90 = ws.w_p90,
                              var_retention_median = ws.var_retention_median,
                              market_share_median = ws.market_share_median))

        cs = cresult.summary
        @printf("  %-34s %-9s bets %5d | return %+8.2f%% | g %+.5f | Sharpe %6.3f | MDD %6.1f%% | w̃ %.2f\n",
                "", c.key, cs.n_bets, cs.total_return_pct, cs.growth_per_slate,
                cs.sharpe_ann, cs.mdd, ws.w_median)
    end
end

summary = DataFrame(summary_rows)
tearsheet = DataFrame(tearsheet_rows)
coherence = DataFrame(coherence_rows)
weights = DataFrame(weight_rows)

# %%
# ==============================================================================
# 6. Derivative coherence audit
# ==============================================================================
banner("5. DERIVATIVE COHERENCE (calibrated containers)")

coherence_note = if nrow(coherence) == 0
    println("  no calibrated containers were produced")
    "_No calibrated containers were produced._"
else
    worst = maximum(coherence.max_family_spread)
    @printf("  worst max_family_spread across %d calibrated containers : %.3e\n",
            nrow(coherence), worst)
    println("  (1X2 / O/U / BTTS are partitions of one score tensor, so this is ~0 by construction)")
    worst < 1e-8 || @warn "family spread exceeds rounding tolerance; the tensor partition is not holding" worst
    # `@sprintf` requires a LITERAL format string, so the prose is concatenated
    # around the formatted number rather than spliced into a built-up format.
    string("Worst `max_family_spread` across ", nrow(coherence),
           " calibrated containers: **", @sprintf("%.3e", worst), "**. ",
           "1X2, O/U and BTTS are three partitions of one 12x12 score tensor, so this ",
           "is zero to rounding by construction; measuring it verifies the construction ",
           "rather than assuming it.")
end

# %%
# ==============================================================================
# 7. Results
# ==============================================================================
banner("6. HEADLINE — RAW POSTERIORS")
show(stdout, MIME("text/plain"),
     sort(select(filter(:variant => ==("raw"), summary),
                 :model, :family, :n_bets, :total_return_pct, :cagr_pct,
                 :growth_per_slate, :sharpe_ann, :calmar, :max_drawdown_pct,
                 :win_rate_pct, :expected_edge_pct),
          :total_return_pct, rev = true);
     allrows = true, allcols = true, truncate = 0)
println()

banner("7. RAW vs CALIBRATED")
show(stdout, MIME("text/plain"),
     sort(select(summary, :model, :variant, :n_bets, :total_return_pct,
                 :growth_per_slate, :sharpe_ann, :max_drawdown_pct, :expected_edge_pct),
          [:model, :variant]);
     allrows = true, allcols = true, truncate = 0)
println()

# %%
# ==============================================================================
# 8. Persist
# ==============================================================================
banner("8. WRITING RESULTS")

summary_path, tearsheet_path = l02_write_outputs!(summary, tearsheet)
context = (; n_panel = length(panel),
             span_days = maximum(summary.span_days),
             B = R02_BOOTSTRAP_B,
             weights = weights,
             weight_table = l02_weight_table(weights),
             coherence_note = coherence_note)
report_path = joinpath(L02_RESULTS_DIR, "r02_portfolio_calibration_report.md")
write(report_path, l02_report(summary, tearsheet, context))
coherence_path = joinpath(L02_RESULTS_DIR, "r02_coherence.csv")
CSV.write(coherence_path, coherence)
dropped_path = joinpath(L02_RESULTS_DIR, "r02_dropped_fixtures.csv")
CSV.write(dropped_path, dropped)
weights_path = joinpath(L02_RESULTS_DIR, "r02_calibration_weights.csv")
CSV.write(weights_path, weights)

println("  summary   : ", summary_path)
println("  tearsheet : ", tearsheet_path)
println("  coherence : ", coherence_path)
println("  weights   : ", weights_path)
println("  dropped   : ", dropped_path)
println("  report    : ", report_path)
println("\nDone. ", nrow(summary), " arm/variant rows over ", length(panel), " fixtures.")
