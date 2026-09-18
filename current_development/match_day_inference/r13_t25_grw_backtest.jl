# r13_t25_grw_backtest.jl
#
# ===================================================================
# WHAT THIS IS AND IS NOT
# ===================================================================
# QUESTION. Over the opening Saturdays of the 2026/27 Scottish League One and League Two
# season, how would MultiScaleGRW models (`m05_joint_production_wealth_grw` and
# `m05_production_wealth_grw`) have performed, starting with £500 initial wealth, at the prices
# resting in `betfair_live.order_book_1m` 25 minutes before kick-off (T-25), under the
# audited Option B staking policy -- compared directly against the TimeDecay control (`m05`)
# and the current production Hybrid model (`m12`)?
#
# HELD FIXED:
# - Opening bankroll: £500.00 compounding per arm per fill model
# - Exact instant: kick-off (14:00 UTC) minus 25 min = 13:35 UTC
# - Order book: betfair_live.order_book_1m (latest snapshot at or before T-25)
# - Fill models: TouchOnly (realistic resting size) and LadderSweep (3-level sweep, max 2% slip)
# - Commission: 2.0% on net winnings per winning leg
# - Slates: 2026/27 opening Saturdays (2026-08-01, 2026-08-08, 2026-08-15, 2026-08-22, 2026-09-05)
# - Policy: Option B (TieredTrust, FractionalKelly 0.3, SlateDrawdown 8.0, FixedCap 0.25)
#
# NO DATABASE WRITE. Runs purely as an in-memory evaluation and outputs CSVs + Markdown report.
# ===================================================================

using ThreadPinning, LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using DataFrames, Dates, Printf
using CSV
import LibPQ

include(joinpath(@__DIR__, "l10_t25_backtest.jl"))

# ===================================================================
# 1. Configuration
# ===================================================================
const R13_TOURNAMENTS = [56, 57]                     # Scottish League One, League Two
const R13_SEGMENT     = DD.ScottishLower()
const R13_GRW_EXP     = "scottish_lower_multiscale_grw_2426"
const R13_PLAYER_EXP  = "scottish_lower_joint_player_2426"

const R13_BANKROLL    = 500.00
const R13_T_MINUS     = Minute(25)
const R13_KICKOFF     = Time(14, 0)                  # 15:00 BST == 14:00 UTC
const R13_DAYS        = [Date(2026, 8, 1), Date(2026, 8, 8), Date(2026, 8, 15),
                         Date(2026, 8, 22), Date(2026, 9, 5)]
const R13_MAX_SLIPPAGE = 0.02
const R13_OUT_DIR      = joinpath(@__DIR__, "results")

println("\n" * "="^96)
println("  T-25 ORDER BOOK BACKTEST -- MultiScaleGRW vs TimeDecay vs Hybrid (2026/27 Season)")
println("  segment    : Scottish League One [56] + League Two [57]")
println("  slates     : ", join(string.(R13_DAYS), ", "))
println("  instant    : kick-off ", R13_KICKOFF, " UTC minus ", R13_T_MINUS, " (13:35 UTC)")
println("  bankroll   : ", @sprintf("£%.2f", R13_BANKROLL), " initial compounding per arm per fill model")
println("  writes     : CSV + Markdown report only. No database writes.")
println("="^96 * "\n")

mkpath(R13_OUT_DIR)
const R13_GIT_COMMIT = try readchomp(`git rev-parse HEAD`) catch; "" end
const R13_GIT_BRANCH = try readchomp(`git rev-parse --abbrev-ref HEAD`) catch; "" end
println("  git        : ", R13_GIT_BRANCH, " @ ", first(R13_GIT_COMMIT, 12))
println("  results    : ", R13_OUT_DIR, "\n")

# ===================================================================
# 2. Data snapshot: cards and T-25 archived books
# ===================================================================
@info "loading ScottishLower DataStore"
ds = DD.load_datastore_cached(R13_SEGMENT)

const R13_BOOK_SOURCE = MD.ArchivedOrderBook(max_age = Hour(2))

cards    = SlateCard[]
refusals = Tuple{Date,String}[]
conn = MD.paper_connection()
try
    for day in R13_DAYS
        loaded = load_slate_card(conn, day, R13_TOURNAMENTS, R13_BOOK_SOURCE;
                                 kickoff = DateTime(day, R13_KICKOFF), t_minus = R13_T_MINUS)
        card = loaded.card
        @printf("  %s  events=%2d  finished=%2d  dropped=%d  closing quotes=%4d  %s\n",
                card.day, loaded.n_events, length(card.fixtures), length(loaded.dropped),
                length(card.close), card.priceable ? "PRICEABLE" : "REFUSED")
        for d in loaded.dropped
            println("      dropped: ", d)
        end
        card.priceable || println("      refusal: ", card.refusal)
        push!(cards, card)
        card.priceable || push!(refusals, (card.day, card.refusal))
    end
finally
    close(conn)
end

const R13_LIVE_CARDS = SlateCard[c for c in cards if c.priceable]
isempty(R13_LIVE_CARDS) && error("r13: no slate has an archived order book at T-25.")

println("\n  priceable slates: ", length(R13_LIVE_CARDS), " of ", length(cards))

# ===================================================================
# 3. Canonical posteriors -- loaded, never sampled
# ===================================================================
storage_grw    = TT.PostgresStorage(R13_GRW_EXP)
storage_player = TT.PostgresStorage(R13_PLAYER_EXP)

@info "loading canonical fits"
fit_grw_joint  = MD.canonical_fit(storage_grw, "m05_joint_production_wealth_grw"; require_converged = false)
fit_grw_wealth = MD.canonical_fit(storage_grw, "m05_production_wealth_grw"; require_converged = false)
fit_m05_td     = MD.canonical_fit(storage_player, UUID("ed541a7c-01e2-447e-a771-783517728d47"); require_converged = false) # Run 63
fit_m12_hybrid = MD.canonical_fit(storage_player, UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef"); require_converged = false) # Run 67

models_to_audit = [
    ("m05_joint_production_wealth_grw", fit_grw_joint),
    ("m05_production_wealth_grw", fit_grw_wealth),
    ("m05_joint_production_wealth (TD control)", fit_m05_td),
    ("m12_joint_hybrid_synergy (TD prod)", fit_m12_hybrid),
]

for (name, cf) in models_to_audit
    @printf("  %-42s folds=%2d converged=%s\n", name, cf.n_folds, cf.converged)
    cf.n_folds >= 43 || error("r13: $name has $(cf.n_folds) folds; 43 are required.")
end

# ===================================================================
# 4. Option B System & Staking Setup
# ===================================================================
const R13_CALIBRATOR = MD.option_b_calibrator()
const R13_SYSTEM     = MD.option_b_system()

r13_spec(fixtures::Vector{MD.Fixture}) = MD.MatchDaySpec(
    fixtures   = MD.ExplicitFixtures(fixtures),
    identity   = MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
    lineups    = MD.SourceChain(MD.ProvisionalDB(), PriorMatchdayXI(ds)),
    book       = R13_BOOK_SOURCE,
    instrument = MD.BestOfBackLay(),
    rounding   = MD.FloorOrDrop(minimum = 1.0),
    gate       = MD.GateChain(MD.IdentityResolved(),
                              MD.MaxBookAge(Minute(10)),
                              MD.MaxSpread(0.08),
                              MD.MinMatched(minimum = 20.0)),
    markets    = MD.canonical_markets(),
)

# ===================================================================
# 5. Model Arms & Compounding Tracks
# ===================================================================
const R13_ARMS = ModelArm[
    ModelArm("m05_joint_grw_raw",       "m05 joint GRW (raw)",          fit_grw_joint,  nothing),
    ModelArm("m05_joint_grw_cal_optB",  "m05 joint GRW (Option B cal)", fit_grw_joint,  R13_CALIBRATOR),
    ModelArm("m05_wealth_grw_raw",      "m05 wealth GRW (raw)",         fit_grw_wealth, nothing),
    ModelArm("m05_wealth_grw_cal_optB", "m05 wealth GRW (Option B cal)",fit_grw_wealth, R13_CALIBRATOR),
    ModelArm("m05_joint_td_raw",        "m05 joint TimeDecay (raw)",    fit_m05_td,     nothing),
    ModelArm("m12_hybrid_td_raw",       "m12 hybrid TimeDecay (raw)",   fit_m12_hybrid, nothing),
    ModelArm("m12_hybrid_td_cal_optB",  "m12 hybrid TimeDecay (cal)",   fit_m12_hybrid, R13_CALIBRATOR),
]

const R13_FILL_MODELS = MD.AbstractFillModel[
    MD.TouchOnly(),
    MD.LadderSweep(max_slippage = R13_MAX_SLIPPAGE),
]

tracks = Track[Track(arm, fm, R13_BANKROLL) for arm in R13_ARMS for fm in R13_FILL_MODELS]
println("\n  tracks: ", length(tracks), " -> ",
        join([track_id(t) for t in tracks], ", "), "\n")

# ===================================================================
# 6. Execution -- slate by slate, compounding
# ===================================================================
for card in R13_LIVE_CARDS
    println("\n" * "-"^96)
    @printf("SLATE %s  |  T-25 = %s UTC  |  %d finished fixtures\n",
            card.day, card.as_of, length(card.fixtures))
    println("-"^96)
    for track in tracks
        t0 = time()
        out = run_slate!(track, card, r13_spec, R13_SYSTEM, R13_SEGMENT, ds)
        @printf("  %-32s fold %2d  legs %3d  filled %3d  pnl %+9.2f  bankroll %9.2f  (%.1fs)\n",
                track_id(track), out.fold_idx, out.n_legs, out.n_filled,
                out.net, out.bankroll, time() - t0)
        for (fx, why) in out.uncovered
            println("      NOT COVERED  ", fx.home, " v ", fx.away, ": ", why)
        end
        for b in out.blocked
            println("      blocked      ", b.fixture.home, " v ", b.fixture.away, ": ",
                    join([string(k, "=", v) for (k, v) in b.readiness.reasons], "; "))
        end
    end
end

# ===================================================================
# 7. Write Ledgers
# ===================================================================
const R13_TOUCH_CSV  = joinpath(R13_OUT_DIR, "r13_grw_trades_touch_only.csv")
const R13_SWEEP_CSV  = joinpath(R13_OUT_DIR, "r13_grw_trades_ladder_sweep.csv")
const R13_SLATES_CSV = joinpath(R13_OUT_DIR, "r13_grw_slate_trajectory.csv")

touch_ledger = write_trade_ledger(R13_TOUCH_CSV, tracks, :touch_only)
sweep_ledger = write_trade_ledger(R13_SWEEP_CSV, tracks, :ladder_sweep_v1)
slate_frame  = vcat([t.slates for t in tracks]...)
CSV.write(R13_SLATES_CSV, slate_frame)

println("\n=== LEDGERS WRITTEN ===")
@printf("  %-52s %5d rows\n", R13_TOUCH_CSV, nrow(touch_ledger))
@printf("  %-52s %5d rows\n", R13_SWEEP_CSV, nrow(sweep_ledger))
@printf("  %-52s %5d rows\n", R13_SLATES_CSV, nrow(slate_frame))

# ===================================================================
# 8. Executive Summary Table
# ===================================================================
summaries = summary_frame(tracks)
println("\n=== EXECUTIVE SUMMARY ===")
show(stdout, MIME"text/plain"(),
     summaries[:, [:arm, :fill_model, :final, :net_pnl, :total_staked, :roi_pct,
                   :win_rate_pct, :sharpe_ann, :max_drawdown_pct, :fill_rate_pct,
                   :beat_close_pct, :mean_clv]];
     allrows = true, allcols = true)
println("\n")

# ===================================================================
# 9. Custom Markdown Report
# ===================================================================
function write_grw_report(path::AbstractString, tracks::Vector{Track}, summaries::DataFrame,
                          cards::Vector{SlateCard}, refusals::Vector{Tuple{Date,String}};
                          touch_ledger::DataFrame, sweep_ledger::DataFrame,
                          slate_frame::DataFrame)
    live = SlateCard[c for c in cards if c.priceable]
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# T−25 Order Book Backtest — MultiScaleGRW vs TimeDecay & Hybrid")
        println(io)
        println(io, "Scottish League One [56] and League Two [57], **2026/27 Season Opening Slates**.")
        println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " from `", R13_GIT_BRANCH, "` @ ", first(R13_GIT_COMMIT, 12), ".")
        println(io)
        println(io, "Evaluated on `betfair_live.order_book_1m` 25 minutes before kick-off with **£500 opening bankroll**, compounding slate by slate.")
        println(io)
        println(io, "## 0. Model Arms Evaluated")
        println(io)
        println(io, "| Arm | Architecture | State Dynamics | Calibration |")
        println(io, "|---|---|---|---|")
        println(io, "| `m05_joint_grw_raw` | Two-arm Joint (proxy xG + goals) | **MultiScaleGRW** | Raw posterior |")
        println(io, "| `m05_joint_grw_cal_optB` | Two-arm Joint (proxy xG + goals) | **MultiScaleGRW** | Option B (`scot_lower_t25_inv`) |")
        println(io, "| `m05_wealth_grw_raw` | Production Wealth | **MultiScaleGRW** | Raw posterior |")
        println(io, "| `m05_wealth_grw_cal_optB` | Production Wealth | **MultiScaleGRW** | Option B (`scot_lower_t25_inv`) |")
        println(io, "| `m05_joint_td_raw` | Two-arm Joint | Time Decay | Raw posterior |")
        println(io, "| `m12_hybrid_td_raw` | Joint + Player RAPM Lineup | Time Decay | Raw posterior (Current Production) |")
        println(io, "| `m12_hybrid_td_cal_optB` | Joint + Player RAPM Lineup | Time Decay | Option B (`scot_lower_t25_inv`) |")
        println(io)

        println(io, "## 1. Slate Inventory")
        println(io)
        println(io, "| Slate | Finished fixtures | T−25 book | Closing quotes | Status |")
        println(io, "|---|---:|---|---:|---|")
        for c in cards
            @printf(io, "| %s | %d | %s | %d | %s |\n", c.day, length(c.fixtures),
                    c.priceable ? "present" : "**absent**", length(c.close),
                    c.priceable ? "priced" : "**REFUSED**")
        end
        println(io)
        if !isempty(refusals)
            println(io, "### Refused slates")
            println(io)
            for (day, why) in refusals
                println(io, "* **", day, "** — ", why)
            end
            println(io)
        end

        println(io, "## 2. Executive Summary")
        println(io)
        print(io, executive_table(summaries))
        println(io)
        println(io, "Initial bankroll is £500.00. Returns compound per slate. ROI is measured on filled risk.")
        println(io)

        println(io, "## 3. Bankroll Trajectory (£500 Initial)")
        println(io)
        print(io, trajectory_table(tracks))
        println(io)

        println(io, "## 4. Liquidity and Capacity Audit")
        println(io)
        print(io, liquidity_table(tracks))
        println(io)

        println(io, "### Slippage: LadderSweep vs TouchOnly")
        println(io)
        print(io, slippage_table(tracks))
        println(io)

        println(io, "## 5. Closing Line Value (CLV)")
        println(io)
        print(io, clv_table(tracks))
        println(io)

        println(io, "## 6. Where the Money Came From (`TouchOnly`)")
        println(io)
        print(io, market_breakdown(touch_ledger))
        println(io)

        println(io, "## 7. Artifacts")
        println(io)
        println(io, "| File | Rows |")
        println(io, "|---|---:|")
        println(io, "| `r13_grw_trades_touch_only.csv` | ", nrow(touch_ledger), " |")
        println(io, "| `r13_grw_trades_ladder_sweep.csv` | ", nrow(sweep_ledger), " |")
        println(io, "| `r13_grw_slate_trajectory.csv` | ", nrow(slate_frame), " |")
    end
    return path
end

const R13_REPORT = joinpath(R13_OUT_DIR, "REPORT_T25_GRW_2627.md")
write_grw_report(R13_REPORT, tracks, summaries, cards, refusals;
                 touch_ledger = touch_ledger, sweep_ledger = sweep_ledger,
                 slate_frame = slate_frame)

println("=== REPORT WRITTEN ===")
println("  ", R13_REPORT)
println("\nr13 complete. No database row was written.\n")
