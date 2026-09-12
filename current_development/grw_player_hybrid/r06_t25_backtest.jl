# ==============================================================================
# r06 — 2026/27 T−25 order-book backtest: GRW × lineup hybrid vs its benchmarks
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# QUESTION. Over the 2026/27 Scottish League One/Two opening Saturdays, starting from
# £500 and compounding slate by slate, how does `m12_joint_hybrid_synergy_grw` fare
# at the prices actually resting in `betfair_live.order_book_1m` at T−25 (13:35 UTC)
# against (a) `m05_joint_grw_raw`, the Task 007 team-level GRW that led the opening-
# slate report (£590.05 TouchOnly), and (b) `m12_hybrid_td_raw`, the Gen 4 production
# hybrid (£551.97)?
#
# It re-runs those two benchmarks and `m05_joint_td_raw` rather than quoting them, so
# every track shares one code path, one book read and one lineup source — and so the
# published figures are REPRODUCED here (asserted) before any new track is read.
#
# It is NOT a significance test. Four slates and ~60 legs per track cannot separate
# arms whose closing-line portfolios differ by a few percent; this is the executable-
# price sanity check that sits beside `r05`, not a verdict on its own.
#
# HELD FIXED (identical to `match_day_inference/r13_t25_grw_backtest.jl`)
#
# - £500 opening bankroll, compounding per arm per fill model
# - T−25 at a 14:00 UTC kick-off; archived book read at or before 13:35
# - TouchOnly (resting size at the touch) and LadderSweep (≤ 3 levels, ≤ 2% slip)
# - 2% commission on net winnings; FloorOrDrop(£1) stake rounding
# - gates: identity resolved, book ≤ 10 min old, spread ≤ 0.08, matched ≥ £20
# - Option B staking (`MatchDay.option_b_system()`); calibrated arm uses
#   `MatchDay.option_b_calibrator()` (scot_lower_t25_inv)
# - lineups: provisional XI scrape, else `PriorMatchdayXI` — the strictly-earlier-day
#   source; `MD.LastHistorical` would hand a finished fixture its played XI
#
# NO DATABASE WRITE. CSV ledgers and a Markdown report only.
#
# USAGE (mcmc-beast, from /root/BF_grw_player_hybrid; needs BF_DB_URL in .env)
#
#   julia --project -t 16 current_development/grw_player_hybrid/r06_t25_backtest.jl
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
using UUIDs
import DotEnv
import LibPQ

let env_file = joinpath(pkgdir(BayesianFootball), ".env")
    isfile(env_file) && DotEnv.load!(ENV, env_file)
end

include(joinpath(@__DIR__, "l01_loader.jl"))
include(joinpath(@__DIR__, "..", "match_day_inference", "l10_t25_backtest.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R06_CONFIG = GPHConfig()
const R06_TOURNAMENTS = [56, 57]
const R06_SEGMENT = DD.ScottishLower()
const R06_BANKROLL = 500.00
const R06_T_MINUS = Minute(25)
const R06_KICKOFF = Time(14, 0)
const R06_DAYS = [Date(2026, 8, 1), Date(2026, 8, 8), Date(2026, 8, 15),
                  Date(2026, 8, 22), Date(2026, 9, 5)]
const R06_MAX_SLIPPAGE = 0.02
const R06_OUT_DIR = joinpath(R06_CONFIG.save_root, "t25")

# REPORT_T25_GRW_2627.md, 2026-09-11 — the figures this runner must reproduce.
const R06_PUBLISHED = Dict(
    ("m05_joint_grw_raw", :touch_only) => 590.05,
    ("m05_joint_grw_raw", :ladder_sweep_v1) => 600.22,
    ("m12_hybrid_td_raw", :touch_only) => 551.97,
    ("m12_hybrid_td_raw", :ladder_sweep_v1) => 575.94,
    ("m05_joint_td_raw", :touch_only) => 569.45,
    ("m05_joint_td_raw", :ladder_sweep_v1) => 587.76,
)

const R06_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
mkpath(R06_OUT_DIR)

println("\n" * "="^96)
println("  r06 T−25 ORDER BOOK BACKTEST — GRW × lineup hybrid, 2026/27 opening slates")
println("  slates   : ", join(string.(R06_DAYS), ", "))
println("  instant  : ", R06_KICKOFF, " UTC − ", R06_T_MINUS, "   bankroll £", R06_BANKROLL)
println("  git      : ", R06_GIT)
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot: cards and T−25 archived books
# ===================================================================
r06_ds = DD.load_datastore_cached(R06_SEGMENT; max_age_hours = 10_000)
const R06_BOOK_SOURCE = MD.ArchivedOrderBook(max_age = Hour(2))

r06_cards = SlateCard[]
r06_refusals = Tuple{Date,String}[]
let conn = MD.paper_connection()
    try
        for day in R06_DAYS
            loaded = load_slate_card(conn, day, R06_TOURNAMENTS, R06_BOOK_SOURCE;
                                     kickoff = DateTime(day, R06_KICKOFF), t_minus = R06_T_MINUS)
            card = loaded.card
            @printf("  %s  finished=%2d  dropped=%d  closing quotes=%4d  %s\n",
                    card.day, length(card.fixtures), length(loaded.dropped), length(card.close),
                    card.priceable ? "PRICEABLE" : "REFUSED: " * card.refusal)
            push!(r06_cards, card)
            card.priceable || push!(r06_refusals, (card.day, card.refusal))
        end
    finally
        close(conn)
    end
end
r06_live = SlateCard[c for c in r06_cards if c.priceable]
isempty(r06_live) && error("r06: no slate has an archived order book at T−25")

# %%
# ===================================================================
# 4. Canonical posteriors — loaded, never sampled
# ===================================================================
r06_db = PostgresStorage(R06_CONFIG.experiment)
r06_fit(experiment, key) = MD.canonical_fit(PostgresStorage(experiment), key; require_converged = false)

r06_ladder = Dict(name => r06_fit(R06_CONFIG.experiment, string(gph_run_by_name(r06_db, name)))
                  for name in GPH_MODEL_NAMES)
r06_controls = Dict(c.label => r06_fit(c.experiment, c.run_id) for c in GPH_CONTROLS)

for (name, cf) in vcat(collect(r06_ladder), collect(r06_controls))
    @printf("  %-32s folds=%2d converged=%s\n", name, cf.n_folds, cf.converged)
    cf.n_folds >= R06_CONFIG.expected_extended_folds || error(
        "r06: $name has $(cf.n_folds) folds; run r03_extend_2627.jl first")
end

# %%
# ===================================================================
# 5. Option B system, spec and arms
# ===================================================================
const R06_CALIBRATOR = MD.option_b_calibrator()
const R06_SYSTEM = MD.option_b_system()

r06_spec(fixtures::Vector{MD.Fixture}) = MD.MatchDaySpec(
    fixtures = MD.ExplicitFixtures(fixtures),
    identity = MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
    lineups = MD.SourceChain(MD.ProvisionalDB(), PriorMatchdayXI(r06_ds)),
    book = R06_BOOK_SOURCE,
    instrument = MD.BestOfBackLay(),
    rounding = MD.FloorOrDrop(minimum = 1.0),
    gate = MD.GateChain(MD.IdentityResolved(),
                        MD.MaxBookAge(Minute(10)),
                        MD.MaxSpread(0.08),
                        MD.MinMatched(minimum = 20.0)),
    markets = MD.canonical_markets(),
)

const R06_ARMS = ModelArm[
    ModelArm("m12_hybrid_grw_raw", "m12 hybrid GRW (raw)",
             r06_ladder["m12_joint_hybrid_synergy_grw"], nothing),
    ModelArm("m12_hybrid_grw_cal_optB", "m12 hybrid GRW (Option B cal)",
             r06_ladder["m12_joint_hybrid_synergy_grw"], R06_CALIBRATOR),
    ModelArm("m05_wealth_grw_raw", "m05 wealth GRW joint (raw)",
             r06_ladder["m05_wealth_grw"], nothing),
    ModelArm("m10_lineup_grw_raw", "m10 lineup GRW Poisson (raw)",
             r06_ladder["m10_lineup_grw"], nothing),
    ModelArm("m00_baseline_grw_raw", "m00 baseline GRW (raw)",
             r06_ladder["m00_baseline_grw"], nothing),
    ModelArm("m05_joint_grw_raw", "Task 007 m05 joint GRW (raw)",
             r06_controls["m05_joint_grw_raw"], nothing),
    ModelArm("m12_hybrid_td_raw", "m12 hybrid TimeDecay (raw, production)",
             r06_controls["m12_hybrid_td_raw"], nothing),
    ModelArm("m05_joint_td_raw", "m05 joint TimeDecay (raw)",
             r06_controls["m05_joint_td_raw"], nothing),
]
const R06_FILL_MODELS = MD.AbstractFillModel[MD.TouchOnly(),
                                             MD.LadderSweep(max_slippage = R06_MAX_SLIPPAGE)]
r06_tracks = Track[Track(arm, fm, R06_BANKROLL) for arm in R06_ARMS for fm in R06_FILL_MODELS]

# %%
# ===================================================================
# 6. Execution — slate by slate, compounding
# ===================================================================
for card in r06_live
    println("\n" * "-"^96)
    @printf("SLATE %s  |  T−25 = %s UTC  |  %d finished fixtures\n",
            card.day, card.as_of, length(card.fixtures))
    println("-"^96)
    for track in r06_tracks
        t0 = time()
        out = run_slate!(track, card, r06_spec, R06_SYSTEM, R06_SEGMENT, r06_ds)
        @printf("  %-40s fold %2d  legs %3d  filled %3d  pnl %+9.2f  bankroll %9.2f  (%.1fs)\n",
                track_id(track), out.fold_idx, out.n_legs, out.n_filled,
                out.net, out.bankroll, time() - t0)
        for (fx, why) in out.uncovered
            println("      NOT COVERED  ", fx.home, " v ", fx.away, ": ", why)
        end
    end
end

# %%
# ===================================================================
# 7. Reproduction gate, ledgers, and summary
# ===================================================================
r06_summaries = summary_frame(r06_tracks)
for ((arm, fill), published) in R06_PUBLISHED
    row = only(filter(r -> r.arm == arm && r.fill_model == String(fill), r06_summaries))
    @printf("  reproduction %-20s %-16s £%.2f (published £%.2f)\n", arm, fill, row.final, published)
    abs(row.final - published) < 0.01 || @warn(
        "r06 does not reproduce the published T−25 figure", arm, fill, got = row.final, published)
end
r06_summaries.reproduces_published = [
    haskey(R06_PUBLISHED, (r.arm, Symbol(r.fill_model))) ?
        string(abs(r.final - R06_PUBLISHED[(r.arm, Symbol(r.fill_model))]) < 0.01) : "—"
    for r in eachrow(r06_summaries)]

# Capture ratio on the executed legs: E[edge | won] / E[edge | lost].
function r06_capture(trades::DataFrame)
    filled = filter(r -> r.risk_filled > 1e-9, trades)
    won = filled.outcome .== "WIN"
    (any(won) && any(.!won)) || return NaN
    e_loss = mean(filled.edge[.!won])
    return e_loss > 0 ? mean(filled.edge[won]) / e_loss : NaN
end
r06_summaries.capture_ratio = [r06_capture(t.trades) for t in r06_tracks]

r06_touch = write_trade_ledger(joinpath(R06_OUT_DIR, "r06_trades_touch_only.csv"), r06_tracks, :touch_only)
r06_sweep = write_trade_ledger(joinpath(R06_OUT_DIR, "r06_trades_ladder_sweep.csv"), r06_tracks, :ladder_sweep_v1)
r06_slates = vcat([t.slates for t in r06_tracks]...)
CSV.write(joinpath(R06_OUT_DIR, "r06_slate_trajectory.csv"), r06_slates)
CSV.write(joinpath(R06_OUT_DIR, "r06_summary.csv"), r06_summaries)

println("\n=== EXECUTIVE SUMMARY ===")
show(stdout, MIME"text/plain"(),
     r06_summaries[:, [:arm, :fill_model, :final, :net_pnl, :roi_pct, :win_rate_pct,
                       :sharpe_ann, :max_drawdown_pct, :fill_rate_pct, :mean_clv,
                       :capture_ratio, :reproduces_published]];
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 8. Final report
# ===================================================================
open(joinpath(R06_OUT_DIR, "r06_t25_report.md"), "w") do io
    println(io, "# r06 T−25 order-book backtest — Task 013\n")
    println(io, "Scottish League One [56] + League Two [57], 2026/27 opening slates. Generated ",
            Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R06_GIT, "`. £500 opening bankroll, ",
            "compounding per slate; `betfair_live.order_book_1m` at T−25; Option B staking.\n")
    println(io, "## Slates\n")
    println(io, "| Slate | Finished fixtures | T−25 book | Status |")
    println(io, "|---|---:|---|---|")
    for c in r06_cards
        println(io, "| ", c.day, " | ", length(c.fixtures), " | ", c.priceable ? "present" : "**absent**",
                " | ", c.priceable ? "priced" : "**REFUSED** — " * c.refusal, " |")
    end
    println(io, "\n## Executive summary\n")
    print(io, executive_table(r06_summaries))
    println(io, "\n| Arm | Fill model | Capture ratio | Reproduces published |")
    println(io, "|---|---|---:|---|")
    for r in eachrow(r06_summaries)
        println(io, "| `", r.arm, "` | `", r.fill_model, "` | ", _num(r.capture_ratio), " | ",
                r.reproduces_published, " |")
    end
    println(io, "\n## Bankroll trajectory\n")
    print(io, trajectory_table(r06_tracks))
    println(io, "\n## Liquidity\n")
    print(io, liquidity_table(r06_tracks))
    println(io, "\n### LadderSweep vs TouchOnly\n")
    print(io, slippage_table(r06_tracks))
    println(io, "\n## Closing-line value\n")
    print(io, clv_table(r06_tracks))
    println(io, "\n## Where the money came from (TouchOnly)\n")
    print(io, market_breakdown(r06_touch))
end
println("\nR06_DONE report=", joinpath(R06_OUT_DIR, "r06_t25_report.md"))
