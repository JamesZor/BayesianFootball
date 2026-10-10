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
function reproduce(ctx, book, spec, policy, label, out, rows; parity_gate::Bool = false)
    println("REPRODUCTION_START ", label)
    flush(stdout)
    result, books, report = PF.run_portfolio_simulation(spec, policy, ctx.fit, book, ctx.ds;
        initial_bankroll = 1.0, bootstrap = false, quiet = true)
    summary = result.summary
    passed = abs(summary.final_bankroll - EXPECTED_WEALTH) <= WEALTH_TOLERANCE &&
             summary.n_bets == EXPECTED_BETS
    push!(rows, (; arm = label, run_id = string(M12_RUN), as_of_minutes = -25.0,
                 max_staleness_minutes = 90.0,
                 final_wealth = summary.final_bankroll, return_pct = summary.total_return_pct,
                 n_bets = summary.n_bets, n_slates = summary.n_slates,
                 n_books = length(books), n_build_errors = length(report.errored),
                 converged = report.converged, failed_gates = join(report.failed_gates, ";"),
                 expected_wealth = parity_gate ? EXPECTED_WEALTH : missing,
                 expected_bets = parity_gate ? EXPECTED_BETS : missing,
                 wealth_tolerance = parity_gate ? WEALTH_TOLERANCE : missing,
                 parity_status = parity_gate ? (passed ? "PASS" : "FAIL") : "NOT_APPLICABLE"))
    CSV.write(joinpath(out, "phase0_reproduction.csv"), DF.DataFrame(rows))
    CSV.write(joinpath(out, "phase0_" * label * "_daily.csv"), PF.states_frame(result))
    println("REPRODUCTION_RESULT arm=", label, " wealth=", summary.final_bankroll,
            " bets=", summary.n_bets, " slates=", summary.n_slates,
            " books=", length(books), " parity=", rows[end].parity_status,
            " converged=", report.converged, " build_errors=", length(report.errored))
    flush(stdout)
    !parity_gate || passed || error("PHASE0_REPRODUCTION_FAILED: $label; see phase0_reproduction.csv. Stop; no later phases.")
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
function run_backtest(out, db)
    ctx = load_context(db)
    CSV.write(joinpath(out, "phase0_fixture_universe.csv"),
              DF.DataFrame(match_id = sort!(collect(ctx.ids))))
    book, refusals = snapshot(ctx, -25.0)
    CSV.write(joinpath(out, "phase0_t25_book.csv"), book)
    isempty(refusals) || CSV.write(joinpath(out, "phase0_t25_refusals.csv"), refusals)
    rows = NamedTuple[]
    # Manager ruling: canonical is the sole reproduction target and must run first.
    canonical_spec = PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb())
    reproduce(ctx, book, canonical_spec, MD.canonical_scottish_lower_policy(),
              "canonical", out, rows; parity_gate = true)
    # This is the live Option B policy on RAW probabilities, never a fitted calibration container.
    option_b = MD.option_b_system()
    reproduce(ctx, book, option_b.book, option_b.policy, "option_b_live", out, rows)

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
    return nothing
end

end # module
