module QSBacktestPhase0

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import LibPQ
import Statistics
import Serialization
import UUIDs

const PF = BF.Portfolio
const MD = BF.MatchDay
const CAL = BF.Calibration
const M12_RUN = UUIDs.UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef")
const M12_NAMESPACE = "scottish_lower_joint_player_2426"
const EXPECTED_WEALTH = 2.5152
const WEALTH_TOLERANCE = 0.00005
const EXPECTED_BETS = 1127

"Execute only a single SELECT or SHOW statement; never accept an arbitrary SQL command."
function read_query(conn, sql, params = ())
    body = replace(strip(sql), r";$" => "")
    @assert occursin(r"^(SELECT|SHOW)\b"i, body) "database query must be a read"
    @assert !occursin(';', body) "database query must be a single statement"
    result = LibPQ.execute(conn, sql, params)
    try
        return DF.DataFrame(result)
    finally
        close(result)
    end
end

const AUDIT_TABLES = ["runs", "configs", "config_registry", "fold_results", "match_latents",
                      "fit_artifacts", "fit_fold_artifacts", "harness_scores",
                      "harness_experiments", "harness_checks"]

"Table-wide counts and timestamp maxima; absent tables/columns are explicit audit rows."
function audit_snapshot(conn)
    metadata = read_query(conn, """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND (table_name IN ('runs', 'configs', 'config_registry', 'fold_results',
                              'match_latents', 'fit_artifacts', 'fit_fold_artifacts')
               OR table_name LIKE 'harness_%')
        ORDER BY table_name, ordinal_position
    """)
    tables = sort!(unique(vcat(AUDIT_TABLES, String.(metadata.table_name))))
    rows = NamedTuple[]
    for table in tables
        @assert occursin(r"^[a-z_][a-z0-9_]*$", table)
        columns = filter(r -> r.table_name == table, metadata)
        if isempty(columns)
            push!(rows, (; table_name = table, table_present = false, column_name = "",
                         column_present = false, n_rows = -1, max_timestamp = missing))
            continue
        end
        names = Set(String.(columns.column_name))
        timestamps = String.(columns.column_name[startswith.(columns.data_type, "timestamp")])
        for column in timestamps
            @assert occursin(r"^[a-z_][a-z0-9_]*$", column)
        end
        expressions = vcat(["count(*)::bigint AS n_rows"],
                           ["max(\"$column\")::text AS \"$column\"" for column in timestamps])
        values = read_query(conn, "SELECT " * join(expressions, ", ") * " FROM public.\"$table\"")
        for column in sort!(unique(vcat(["created_at", "updated_at"], timestamps)))
            push!(rows, (; table_name = table, table_present = true, column_name = column,
                         column_present = column in names, n_rows = Int(values.n_rows[1]),
                         max_timestamp = column in timestamps ? values[1, Symbol(column)] : missing))
        end
    end
    return DF.DataFrame(rows)
end

"Always compare the post-run database snapshot, including exits from a failed scientific gate."
function run(out)
    mkpath(out)
    Base.run(`python3 $(joinpath(@__DIR__, "check_read_paths.py"))`)
    db = BF.Training.PostgresStorage(M12_NAMESPACE)
    conn = BF.Training.Inference._db_connect(db)
    try
        status = read_query(conn, "SHOW default_transaction_read_only")
        value = String(status[1, 1])
        println("SESSION_DEFAULT_TRANSACTION_READ_ONLY=", value)
        CSV.write(joinpath(out, "phase0_session_default.csv"),
                  DF.DataFrame(default_transaction_read_only = [value]))
        before = audit_snapshot(conn)
        CSV.write(joinpath(out, "phase0_db_before.csv"), before)
        println("DB_AUDIT_BEFORE tables=", length(unique(before.table_name)))
        flush(stdout)
        try
            run_backtest(out, db)
        finally
            after = audit_snapshot(conn)
            CSV.write(joinpath(out, "phase0_db_after.csv"), after)
            tables = sort!(unique(vcat(String.(before.table_name), String.(after.table_name))))
            checks = DF.DataFrame([(; table_name = table,
                unchanged = isequal(filter(r -> r.table_name == table, before),
                                     filter(r -> r.table_name == table, after))) for table in tables])
            CSV.write(joinpath(out, "phase0_db_audit.csv"), checks)
            unchanged = all(checks.unchanged)
            println("DB_AUDIT_AFTER unchanged=", unchanged, " tables=", length(tables))
            flush(stdout)
            @assert unchanged "before/after database audit changed; stop"
        end
    finally
        close(conn)
    end
    println("PHASE0_PASS — stop at phase boundary; no allocator audit has run.")
    return nothing
end

"Load m12 by immutable UUID and restrict saved predictions to the two gate seasons."
function load_context(db)
    cache = joinpath(pkgdir(BF), ".cache", "datastore_ScottishLower.jls")
    @assert isfile(cache) "Phase 0 requires the saved datastore cache; no SQL fallback"
    ds = Serialization.deserialize(cache)
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
function reproduce(ctx, book, spec, policy, label, out, rows;
                   parity_gate::Bool = false, initial_bankroll::Real = 1.0)
    println("REPRODUCTION_START ", label)
    flush(stdout)
    result, books, report = PF.run_portfolio_simulation(spec, policy, ctx.fit, book, ctx.ds;
        initial_bankroll = initial_bankroll, bootstrap = false, quiet = true)
    summary = result.summary
    wealth = summary.final_bankroll / summary.initial_bankroll
    passed = abs(wealth - EXPECTED_WEALTH) <= WEALTH_TOLERANCE &&
             summary.n_bets == EXPECTED_BETS
    push!(rows, (; arm = label, run_id = string(M12_RUN), as_of_minutes = -25.0,
                 max_staleness_minutes = 90.0,
                 initial_bankroll = summary.initial_bankroll, final_bankroll = summary.final_bankroll,
                 final_wealth = wealth, return_pct = summary.total_return_pct,
                 n_bets = summary.n_bets, n_slates = summary.n_slates,
                 n_books = length(books), n_build_errors = length(report.errored),
                 converged = report.converged, failed_gates = join(report.failed_gates, ";"),
                 expected_wealth = parity_gate ? EXPECTED_WEALTH : missing,
                 expected_bets = parity_gate ? EXPECTED_BETS : missing,
                 wealth_tolerance = parity_gate ? WEALTH_TOLERANCE : missing,
                 parity_status = parity_gate ? (passed ? "PASS" : "FAIL") : "NOT_APPLICABLE"))
    CSV.write(joinpath(out, "phase0_reproduction.csv"), DF.DataFrame(rows))
    daily = PF.states_frame(result)
    daily.unit_bankroll_open = daily.bankroll_open ./ summary.initial_bankroll
    daily.unit_bankroll_close = daily.bankroll_close ./ summary.initial_bankroll
    CSV.write(joinpath(out, "phase0_" * label * "_daily.csv"), daily)
    println("REPRODUCTION_RESULT arm=", label, " wealth=", wealth,
            " bets=", summary.n_bets, " slates=", summary.n_slates,
            " books=", length(books), " parity=", rows[end].parity_status,
            " converged=", report.converged, " build_errors=", length(report.errored))
    flush(stdout)
    !parity_gate || passed || error("PHASE0_REPRODUCTION_FAILED: $label; see phase0_reproduction.csv. Stop; no later phases.")
    return result
end

const CANONICAL_MARKET_KEYS = Set(vcat([("1X2", 0.0), ("BTTS", 0.0)],
                                     [("OverUnder", line) for line in (0.5, 1.5, 2.5, 3.5)]))

"All archived markets plus required canonical markets, including zero-coverage lines."
function market_catalog(ctx)
    archive = filter(r -> Int(r.match_id) in ctx.ids, ctx.ds.betfair_odds)
    pairs = Set((String(r.market_name), Float64(r.market_line)) for r in
                eachrow(DF.unique(DF.select(archive, [:market_name, :market_line]))))
    union!(pairs, CANONICAL_MARKET_KEYS)
    return [(market, line, CAL.expected_selection_count(market, line))
            for (market, line) in sort!(collect(pairs))]
end

"Coverage includes all archive lines and zero rows for every required canonical market."
function coverage_rows(book, minutes, bound, total, markets)
    rows = NamedTuple[]
    for (market, line, arity) in markets
        b = filter(r -> r.market_name == market && r.market_line == line, book)
        stale = Float64.(b.staleness_minutes)
        q(p) = isempty(stale) ? missing : Statistics.quantile(stale, p)
        fixtures = length(unique(b.match_id))
        push!(rows, (; as_of_minutes = Float64(minutes), max_staleness_minutes = bound,
                     market_name = market, market_line = line, universe_fixtures = total,
                     in_canonical_markets = (market, line) in CANONICAL_MARKET_KEYS,
                     fixtures, fixture_coverage = fixtures / total, selections = DF.nrow(b),
                     expected_selections = arity > 0 ? total * arity : missing,
                     selection_coverage = arity > 0 ? DF.nrow(b) / (total * arity) : missing,
                     staleness_min = q(0.0), staleness_p25 = q(0.25),
                     staleness_p50 = q(0.5), staleness_p75 = q(0.75),
                     staleness_p90 = q(0.9), staleness_p95 = q(0.95), staleness_max = q(1.0)))
    end
    return rows
end

"Exact experiment 07 m12/raw/B0_canonical/prod recipe, guarded against its committed CSV."
function historical_system(out)
    source = joinpath(pkgdir(BF), "experiments", "scottish_lower",
                      "07_calibrated_portfolio_and_trust_vector", "results",
                      "optimal_portfolio_comparison.csv")
    target = filter(r -> r.model == "m12" && r.container == "raw" &&
                        r.basket == "B0_canonical" && r.setting == "prod",
                    CSV.read(source, DF.DataFrame))
    @assert DF.nrow(target) == 1 "exact historical reference cell must be unique"
    r = target[1, :]
    @assert (r.t1, r.ratio, r.lambda, r.cap) == (0.35, 1.4, 23.0, 0.25)
    @assert r.n_bets == EXPECTED_BETS && r.n_slates == 99
    @assert abs(1.0 + r.return_pct / 100.0 - EXPECTED_WEALTH) <= WEALTH_TOLERANCE
    CSV.write(joinpath(out, "phase0_historical_target.csv"), target)
    # Runner:184,220,268,312–326; loader:131–134,194–200,209–211,273–287.
    spec = PF.BookSpec(
        markets = BF.Data.MarketConfig(CAL.l2_tradeable_markets()),
        price = PF.DeArb(), allocator = PF.KellyLogUtility(), shrink = PF.FractionalKelly(0.30),
        exec = PF.ExecutionConfig(commission = PF.PerBetCommission(0.02), budget = 0.99,
                                  min_selection_stake = 0.001))
    trust = PF.TieredTrust(Dict(
        ("1x2", 0.0, :home) => r.t1,
        ("over_under", 2.5, :under) => r.t1,
        ("1x2", 0.0, :draw) => r.t1 / r.ratio,
        ("1x2", 0.0, :away) => r.t1 / r.ratio,
    ); default = 0.0)
    policy = PF.PolicySpec(trust = trust, risk = PF.SlateDrawdown(r.lambda),
                           cap = PF.FixedCap(r.cap), grouping = PF.DailySlate())
    return PF.PortfolioSystem(spec, policy)
end

"Snapshot-wide coverage; separate archive, canonical-market and complete 1X2 denominators."
function snapshot_overview(book, minutes, bound, total)
    canonical = filter(r -> (String(r.market_name), Float64(r.market_line)) in
                            CANONICAL_MARKET_KEYS, book)
    x1x2 = filter(r -> r.market_name == "1X2", book)
    q(p) = isempty(book) ? missing : Statistics.quantile(book.staleness_minutes, p)
    return (; as_of_minutes = minutes, max_staleness_minutes = bound, universe_fixtures = total,
            archive_fixtures = length(unique(book.match_id)), archive_selections = DF.nrow(book),
            archive_fixture_coverage = length(unique(book.match_id)) / total,
            canonical_fixtures = length(unique(canonical.match_id)),
            canonical_selections = DF.nrow(canonical),
            canonical_fixture_coverage = length(unique(canonical.match_id)) / total,
            canonical_selection_coverage = DF.nrow(canonical) / (13 * total),
            x1x2_fixtures = length(unique(x1x2.match_id)),
            x1x2_fixture_coverage = length(unique(x1x2.match_id)) / total,
            staleness_p50 = q(0.5), staleness_p90 = q(0.9), staleness_p95 = q(0.95),
            staleness_max = q(1.0))
end

"Phase 0 only: historical reproduction first; snapshots remain unreachable on failure."
function run_backtest(out, db)
    ctx = load_context(db)
    CSV.write(joinpath(out, "phase0_fixture_universe.csv"),
              DF.DataFrame(match_id = sort!(collect(ctx.ids))))
    book, refusals = snapshot(ctx, -25.0)
    CSV.write(joinpath(out, "phase0_t25_book.csv"), book)
    isempty(refusals) || CSV.write(joinpath(out, "phase0_t25_refusals.csv"), refusals)
    previous = dirname(out)
    reference_ids = Set(Int.(CSV.read(joinpath(previous, "phase0_fixture_universe.csv"),
                                      DF.DataFrame).match_id))
    reference_book = CSV.read(joinpath(previous, "phase0_t25_book.csv"), DF.DataFrame)
    reference_book.selection = Symbol.(reference_book.selection)
    @assert reference_ids == ctx.ids "held-out universe changed from canonical inputs"
    @assert isequal(reference_book, book) "T−25 book changed from canonical inputs"
    println("INPUT_PARITY_PASS fixtures=710 T25_book_unchanged=true")
    rows = NamedTuple[]
    historical = historical_system(out)
    # Ruling 3: historical runner defaults to £1000; the gate uses normalised unit wealth.
    reproduce(ctx, book, historical.book, historical.policy, "experiment07_historical", out, rows;
              parity_gate = true, initial_bankroll = 1000.0)
    # Today's canonical arm already ran at source 8018dacf; do not rerun completed work.
    # This is the live Option B policy on RAW probabilities, never a fitted calibration container.
    option_b = MD.option_b_system()
    reproduce(ctx, book, option_b.book, option_b.policy, "option_b_live", out, rows)

    coverage = NamedTuple[]
    overviews = NamedTuple[]
    markets = market_catalog(ctx)
    for minutes in (-25.0, -360.0, -1440.0)
        b, r = minutes == -25.0 ? (book, refusals) : snapshot(ctx, minutes)
        stem = "phase0_t" * string(Int(abs(minutes)))
        CSV.write(joinpath(out, stem * "_book.csv"), b)
        isempty(r) || CSV.write(joinpath(out, stem * "_refusals.csv"), r)
        append!(coverage, coverage_rows(b, minutes, 90.0, length(ctx.ids), markets))
        overview = snapshot_overview(b, minutes, 90.0, length(ctx.ids))
        push!(overviews, overview)
        println("SNAPSHOT_RESULT minutes=", minutes, " bound=90 fixtures=", overview.archive_fixtures,
                " selections=", overview.archive_selections, " x1x2=", overview.x1x2_fixtures)
        flush(stdout)
        if minutes == -1440.0 && min(overview.archive_fixture_coverage,
                                    overview.canonical_fixture_coverage,
                                    overview.x1x2_fixture_coverage) < 0.50
            bs, rs = snapshot(ctx, minutes; staleness = 360.0)
            CSV.write(joinpath(out, stem * "_sensitivity360_book.csv"), bs)
            isempty(rs) || CSV.write(joinpath(out, stem * "_sensitivity360_refusals.csv"), rs)
            append!(coverage, coverage_rows(bs, minutes, 360.0, length(ctx.ids), markets))
            sensitivity = snapshot_overview(bs, minutes, 360.0, length(ctx.ids))
            push!(overviews, sensitivity)
            println("SNAPSHOT_RESULT minutes=", minutes, " bound=360 fixtures=", sensitivity.archive_fixtures,
                    " selections=", sensitivity.archive_selections, " x1x2=", sensitivity.x1x2_fixtures)
            flush(stdout)
        end
    end
    CSV.write(joinpath(out, "phase0_snapshot_coverage.csv"), DF.DataFrame(coverage))
    CSV.write(joinpath(out, "phase0_snapshot_overview.csv"), DF.DataFrame(overviews))
    return nothing
end

end # module
