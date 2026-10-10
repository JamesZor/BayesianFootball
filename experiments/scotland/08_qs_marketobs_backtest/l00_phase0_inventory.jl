module QSBacktestPhase0

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import LibPQ
import Statistics
import UUIDs

const PF = BF.Portfolio
const MD = BF.MatchDay
const CAL = BF.Calibration
const M12_RUN = UUIDs.UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef")
const M12_NAMESPACE = "scottish_lower_joint_player_2426"
const EXPECTED_WEALTH = 2.5152
const WEALTH_TOLERANCE = 0.00005
const EXPECTED_BETS = 1127

"Verify libpq's session default is read-only before loading a saved fit."
function assert_readonly(db)
    conn = BF.Training.Inference._db_connect(db)
    try
        status = DF.DataFrame(LibPQ.execute(conn, "SHOW default_transaction_read_only"))
        @assert String(status[1, 1]) == "on" "experiment DB session must be read-only"
    finally
        close(conn)
    end
    return nothing
end

"Load m12 by immutable UUID and restrict saved predictions to the two gate seasons."
function load_context()
    @assert occursin("default_transaction_read_only=on", get(ENV, "PGOPTIONS", ""))
    ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower(); max_age_hours = 100_000)
    db = BF.Training.PostgresStorage(M12_NAMESPACE)
    assert_readonly(db)
    raw = BF.Training.load_fit(db, M12_RUN)
    season_ids = Set(Int.(ds.matches.match_id[in.(ds.matches.season, Ref(["24/25", "25/26"]))]))
    latents = CAL.restrict_latents(raw.latents, season_ids)
    ids = Set(Int.(BF.Models.latent_match_ids(latents)))
    @assert length(ids) == 710 "Phase 0 requires exactly 710 held-out fixtures"
    fit = BF.Training.Fit(raw.config, raw.folds, latents, raw.diagnostics,
                          raw.metadata, raw.save_path)
    return (; ds, fit, ids)
end

"Build the requested snapshot only on the fixed 710-fixture OOS universe."
function snapshot(ctx, minutes; staleness = 90.0)
    config = CAL.PointInTimeBookConfig(as_of_minutes = Float64(minutes),
                                       max_staleness_minutes = Float64(staleness))
    book, refusals = CAL.point_in_time_book(ctx.ds; config)
    book = filter(r -> Int(r.match_id) in ctx.ids, book)
    if !isempty(refusals)
        refusals = filter(r -> Int(r.match_id) in ctx.ids, refusals)
    end
    isempty(book) || CAL.assert_book_as_of(book, minutes)
    @assert all(book.tick_minutes .<= minutes)
    @assert all(0.0 .<= book.staleness_minutes .<= staleness)
    return book, refusals
end

"Run one fixed reproduction arm; record all output before evaluating its parity gate."
function reproduce(ctx, book, spec, policy, label, out, rows)
    println("REPRODUCTION_START ", label)
    flush(stdout)
    result, books, report = PF.run_portfolio_simulation(spec, policy, ctx.fit, book, ctx.ds;
        initial_bankroll = 1.0, bootstrap = false, quiet = true)
    summary = result.summary
    passed = abs(summary.final_bankroll - EXPECTED_WEALTH) <= WEALTH_TOLERANCE &&
             summary.n_bets == EXPECTED_BETS
    push!(rows, (; arm = label, run_id = string(M12_RUN), as_of_minutes = -25.0,
                 max_staleness_minutes = 90.0, summary = nothing,
                 final_wealth = summary.final_bankroll, return_pct = summary.total_return_pct,
                 n_bets = summary.n_bets, n_slates = summary.n_slates,
                 n_books = length(books), n_build_errors = length(report.errored),
                 converged = report.converged, failed_gates = join(report.failed_gates, ";"),
                 expected_wealth = EXPECTED_WEALTH, expected_bets = EXPECTED_BETS,
                 wealth_tolerance = WEALTH_TOLERANCE, parity_pass = passed))
    frame = DF.DataFrame(rows)
    DF.select!(frame, DF.Not(:summary))
    CSV.write(joinpath(out, "phase0_reproduction.csv"), frame)
    CSV.write(joinpath(out, "phase0_" * label * "_daily.csv"), PF.states_frame(result))
    println("REPRODUCTION_RESULT arm=", label, " wealth=", summary.final_bankroll,
            " bets=", summary.n_bets, " slates=", summary.n_slates,
            " books=", length(books), " parity=", passed,
            " converged=", report.converged, " build_errors=", length(report.errored))
    flush(stdout)
    passed || error("PHASE0_REPRODUCTION_FAILED: $label; see phase0_reproduction.csv. Stop; no later phases.")
    return result
end

"Coverage includes every canonical market even if it has zero surviving selections."
function coverage_rows(book, minutes, bound, total)
    markets = [("1X2", 0.0, 3), ("BTTS", 0.0, 2),
               [("OverUnder", line, 2) for line in (0.5, 1.5, 2.5, 3.5)]...]
    rows = NamedTuple[]
    for (market, line, arity) in markets
        b = filter(r -> r.market_name == market && r.market_line == line, book)
        stale = Float64.(b.staleness_minutes)
        q(p) = isempty(stale) ? missing : Statistics.quantile(stale, p)
        fixtures = length(unique(b.match_id))
        push!(rows, (; as_of_minutes = Float64(minutes), max_staleness_minutes = bound,
                     market_name = market, market_line = line, universe_fixtures = total,
                     fixtures, fixture_coverage = fixtures / total, selections = DF.nrow(b),
                     expected_selections = total * arity,
                     selection_coverage = DF.nrow(b) / (total * arity),
                     staleness_min = q(0.0), staleness_p25 = q(0.25),
                     staleness_p50 = q(0.5), staleness_p75 = q(0.75),
                     staleness_p90 = q(0.9), staleness_p95 = q(0.95), staleness_max = q(1.0)))
    end
    return rows
end

"Phase 0 only: reproduce first; all later snapshot work is unreachable after failure."
function run(out)
    mkpath(out)
    ctx = load_context()
    CSV.write(joinpath(out, "phase0_fixture_universe.csv"),
              DF.DataFrame(match_id = sort!(collect(ctx.ids))))
    book, refusals = snapshot(ctx, -25.0)
    CSV.write(joinpath(out, "phase0_t25_book.csv"), book)
    isempty(refusals) || CSV.write(joinpath(out, "phase0_t25_refusals.csv"), refusals)
    rows = NamedTuple[]
    # Fixed historical control runs first, not a candidate selected after seeing results.
    historical_spec = PF.BookSpec(
        markets = BF.Data.MarketConfig(CAL.l2_tradeable_markets()),
        price = PF.DeArb(), allocator = PF.KellyLogUtility(), shrink = PF.FractionalKelly(0.30),
        exec = PF.ExecutionConfig(commission = PF.PerBetCommission(0.02), budget = 0.99,
                                  min_selection_stake = 0.001))
    policy = MD.canonical_scottish_lower_policy()
    reproduce(ctx, book, historical_spec, policy, "experiment07_historical_control", out, rows)
    # Exact constructor from r07_serve_console; no overrides to force historical parity.
    production_spec = PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb())
    reproduce(ctx, book, production_spec, policy, "canonical_matchday_default", out, rows)

    coverage = NamedTuple[]
    for minutes in (-25.0, -360.0, -1440.0)
        b, r = minutes == -25.0 ? (book, refusals) : snapshot(ctx, minutes)
        stem = "phase0_t" * string(Int(abs(minutes)))
        CSV.write(joinpath(out, stem * "_book.csv"), b)
        isempty(r) || CSV.write(joinpath(out, stem * "_refusals.csv"), r)
        append!(coverage, coverage_rows(b, minutes, 90.0, length(ctx.ids)))
        if minutes == -1440.0 && length(unique(b.match_id)) / length(ctx.ids) < 0.50
            bs, rs = snapshot(ctx, minutes; staleness = 360.0)
            CSV.write(joinpath(out, stem * "_sensitivity360_book.csv"), bs)
            isempty(rs) || CSV.write(joinpath(out, stem * "_sensitivity360_refusals.csv"), rs)
            append!(coverage, coverage_rows(bs, minutes, 360.0, length(ctx.ids)))
        end
    end
    CSV.write(joinpath(out, "phase0_snapshot_coverage.csv"), DF.DataFrame(coverage))
    println("PHASE0_PASS — stop at phase boundary; no allocator audit has run.")
    return nothing
end

end # module
