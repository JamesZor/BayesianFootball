# r12_live_slate_grw_20260912.jl
#
# ===================================================================
# WHAT THIS IS
# ===================================================================
# The live Scottish Lower MultiScaleGRW Option B card for Saturday 2026-09-12.
# Prices the simultaneous 15:00 BST (14:00 UTC) settlement window at
# 14:35 BST (13:35 UTC, T-25) from the completed Fold-43 GRW posterior:
#   `m05_joint_production_wealth_grw` (UUID f870dbb7-9df0-4dae-a84a-cf570cf8113e)
# Applies the validated T-25 generative-rate calibrator, and sizes the Option B basket.
#
# Card: 9 active fixtures across Scottish League One (56) & League Two (57).
# (1 fixture postponed: Ross County vs Hamilton Academical).
#
# USAGE
#   julia --project -t 8 current_development/match_day_inference/r12_live_slate_grw_20260912.jl --dry-run
#   julia --project -t 8 current_development/match_day_inference/r12_live_slate_grw_20260912.jl --commit
#
# Runs on Port 8087 (shadowing live production on 8085).

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning, LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using DataFrames, Dates, Printf, UUIDs
import LibPQ
import Sockets

const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const FF = BayesianFootball.Features
const TT = BayesianFootball.Training

# %%
# ===================================================================
# 2. Frozen operational configuration
# ===================================================================
const R12_DAY           = Date(2026, 9, 12)
const R12_KICKOFF_UTC   = DateTime(2026, 9, 12, 14, 0)   # 15:00 BST
const R12_AS_OF_UTC     = DateTime(2026, 9, 12, 13, 35)  # 14:35 BST, exactly T-25
const R12_COMMIT_BY_UTC = DateTime(2026, 9, 12, 13, 58)  # commit allowed before 15:00 BST kickoff
const R12_TOURNAMENTS   = [56, 57]
const R12_BANKROLL      = 500.0
const R12_ACCOUNT       = "live_scottish_grw_500"
const R12_SCHEMA        = "paper_runbook"
const R12_EXPERIMENT    = "scottish_lower_multiscale_grw_2426"
const R12_RUN_UUID      = UUID("f870dbb7-9df0-4dae-a84a-cf570cf8113e") # m05_joint_production_wealth_grw (Fold 43)
const R12_HOST          = "0.0.0.0"
const R12_PORT          = 8087

function assert_console_port_free(port::Integer)
    socket = try
        Sockets.connect("127.0.0.1", Int(port))
    catch
        nothing
    end
    if socket !== nothing
        close(socket)
        error("commit refused: port $port is already serving another console. Stop the old " *
              "live console before writing a new slate.")
    end
    return nothing
end

function r12_mode(args)
    flags = Set(args)
    unknown = setdiff(flags, Set(["--dry-run", "--commit"]))
    isempty(unknown) || error("unknown argument(s): $(join(sort!(collect(unknown)), ", ")). " *
                              "Use exactly one of --dry-run or --commit.")
    xor("--dry-run" in flags, "--commit" in flags) || error(
        "choose exactly one mode: --dry-run (no writes) or --commit (paper_runbook write).")
    return "--commit" in flags ? :commit : :dry_run
end

const R12_MODE = r12_mode(ARGS)

println("\n" * "="^82)
println("  Scottish Lower MultiScaleGRW Option B — LIVE SHADOW slate (MatchDay 2026-09-12)")
println("  decision : 2026-09-12 14:35 BST / ", R12_AS_OF_UTC, " UTC (T-25)")
println("  kickoff  : 2026-09-12 15:00 BST / ", R12_KICKOFF_UTC, " UTC")
println("  model    : ", R12_EXPERIMENT, " / m05_joint_production_wealth_grw (Fold 43)")
println("  mode     : ", R12_MODE)
println("  ledger   : ", R12_SCHEMA, " (account: ", R12_ACCOUNT, ")")
println("="^82 * "\n")

# %%
# ===================================================================
# 3. Read-only operational pre-flight
# ===================================================================
function r12_load_fixtures(conn)
    lo = Int(round(datetime2unix(DateTime(R12_DAY))))
    hi = Int(round(datetime2unix(DateTime(R12_DAY) + Day(1))))
    frame = DataFrame(LibPQ.execute(conn, """
        SELECT match_id, home_team, away_team, start_timestamp, tournament_id, status_type
        FROM sofascore.events
        WHERE tournament_id = ANY(\$1) AND start_timestamp >= \$2 AND start_timestamp < \$3
        ORDER BY start_timestamp, tournament_id, match_id;
    """, (R12_TOURNAMENTS, lo, hi)))

    active_frame = filter(r -> r.status_type != "postponed", frame)
    fixtures = MD.Fixture[
        MD.Fixture(Int(row.match_id), String(row.home_team), String(row.away_team),
                   unix2datetime(row.start_timestamp), Int(row.tournament_id))
        for row in eachrow(active_frame)
    ]
    return frame, fixtures
end

function r12_book_health(conn, match_ids)
    isempty(match_ids) && return DataFrame()
    return DataFrame(LibPQ.execute(conn, """
        SELECT mm.match_id,
               count(DISTINCT md.market_id) AS markets,
               count(o.market_id) AS book_rows,
               min(o.ts) AS first_tick,
               max(o.ts) AS latest_tick,
               max(o.market_matched) / 10000.0 AS max_matched
        FROM betfair.match_meta mm
        LEFT JOIN betfair_live.market_metadata md ON md.event_id = mm.betfair_event_id
        LEFT JOIN betfair_live.order_book_1m o ON o.market_id = md.market_id
        WHERE mm.match_id = ANY(\$1)
        GROUP BY mm.match_id
        ORDER BY mm.match_id;
    """, (match_ids,)))
end

function r12_lineup_health(conn, match_ids)
    isempty(match_ids) && return DataFrame()
    return DataFrame(LibPQ.execute(conn, """
        SELECT e.match_id, e.home_team, e.away_team,
               count(l.*) AS player_rows,
               count(DISTINCT l.scraped_at) AS scrapes,
               min(l.scraped_at) AS first_scrape,
               max(l.scraped_at) AS latest_scrape,
               coalesce(bool_or(l.confirmed), false) AS any_confirmed
        FROM sofascore.events e
        LEFT JOIN sofascore.lineup_provisional l ON l.match_id = e.match_id
        WHERE e.match_id = ANY(\$1)
        GROUP BY e.match_id, e.home_team, e.away_team
        ORDER BY e.match_id;
    """, (match_ids,)))
end

function verify_fold_team_coverage(expr, ds, fixtures)
    boundaries = DD.create_id_boundaries(ds, expr.config.splitter)
    ids = Int[f.m_id for f in fixtures]
    selected = MD.select_split(expr, boundaries; exclude = ids, ds = ds,
                               config = expr.config.splitter, fixture_ids = ids)
    features = FF.create_features(boundaries, ds, expr.config.model, expr.config.splitter)
    team_map = features[selected.idx][1].data[:team_map]
    teams = sort(unique(vcat([f.home for f in fixtures], [f.away for f in fixtures])))
    report = DataFrame(team = teams, in_team_map = [haskey(team_map, t) for t in teams])
    all(report.in_team_map) || error(
        "fold $(selected.idx) team-map refusal: " *
        join(report.team[.!report.in_team_map], ", "))
    return selected, report
end

function r12_preflight()
    conn = MD.paper_connection()
    try
        all_events, fixtures = r12_load_fixtures(conn)
        println("=== G-A FIXTURE INVENTORY ===")
        show(stdout, MIME"text/plain"(), all_events; allrows = true, allcols = true)
        println("\n")
        length(fixtures) >= 9 || error(
            "pre-flight: expected at least 9 active Scottish Lower fixtures on $R12_DAY, " *
            "found $(length(fixtures)).")
        all(f -> f.kickoff == R12_KICKOFF_UTC, fixtures) || error(
            "pre-flight: the card does not form one 14:00 UTC settlement window: " *
            join(sort!(unique(string(f.kickoff) for f in fixtures)), ", "))

        println("=== G-B BETFAIR TICK HEALTH ===")
        book_health = r12_book_health(conn, [f.m_id for f in fixtures])
        show(stdout, MIME"text/plain"(), book_health; allrows = true, allcols = true)
        println("\n")

        wall_now = Dates.now(Dates.UTC)
        if wall_now >= R12_AS_OF_UTC
            nrow(book_health) == length(fixtures) || error(
                "pre-flight: only $(nrow(book_health)) of $(length(fixtures)) fixtures have a " *
                "Betfair crosswalk/book health row.")
            all(Int.(book_health.markets) .> 0) || error(
                "pre-flight: one or more fixtures have no mapped Betfair market.")
            all(Int.(book_health.book_rows) .> 0) || error(
                "pre-flight: one or more fixtures have no archived order-book tick.")
        else
            println("  Notice: Before T-25 ($R12_AS_OF_UTC UTC), Betfair orderbook ticks are being verified.")
            println("  Current crosswalk linked fixtures: $(nrow(book_health)) / $(length(fixtures)).")
        end

        println("=== G-C LINEUP READINESS ===")
        lineup_health = r12_lineup_health(conn, [f.m_id for f in fixtures])
        show(stdout, MIME"text/plain"(), lineup_health; allrows = true, allcols = true)
        println("\n")
        return fixtures
    finally
        close(conn)
    end
end

fixtures = r12_preflight()

# %%
# ===================================================================
# 4. Data snapshot, canonical posterior, and convergence gate
# ===================================================================
@info "loading ScottishLower DataStore (uses .cache/ if warm)"
ds = DD.load_datastore_cached(DD.ScottishLower())

storage = TT.PostgresStorage(R12_EXPERIMENT)
canonical = MD.canonical_fit(storage, string(R12_RUN_UUID); require_converged = false)
MD.matchday_fit_report(canonical)
canonical.n_folds >= 43 || error(
    "pre-flight: Run $R12_RUN_UUID has $(canonical.n_folds) folds; Fold 43 is required.")

if Dates.now(Dates.UTC) < R12_AS_OF_UTC
    selected, team_coverage = verify_fold_team_coverage(canonical.fit, ds, fixtures)
    println("=== G-D FOLD $(selected.idx) TEAM-MAP COVERAGE ===")
    show(stdout, MIME"text/plain"(), team_coverage; allrows = true, allcols = true)
    println("\n")
    selected.idx == 43 || error("expected serving fold 43, selected fold $(selected.idx)")
end

# %%
# ===================================================================
# 5. Live sources and Option B recipe
# ===================================================================
spec = MD.MatchDaySpec(
    fixtures = MD.ExplicitFixtures(fixtures),
    identity = MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
    lineups = MD.SourceChain(MD.BBCLineupSource(ds = ds), MD.ProvisionalDB(), MD.LastHistorical(ds)),
    book = MD.ArchivedOrderBook(max_age = Hour(2)),
    instrument = MD.BestOfBackLay(),
    rounding = MD.FloorOrDrop(minimum = 1.0),
    gate = MD.GateChain(MD.IdentityResolved(),
                        MD.MaxBookAge(Minute(10)),
                        MD.MaxSpread(0.08),
                        MD.MinMatched(minimum = 20.0)),
    markets = MD.canonical_markets(),
)

calibrator = MD.option_b_calibrator()
system = MD.option_b_system()

println("=== G-E OPTION B RECIPE ===")
println("  calibrator : ", calibrator)
println("  book hash  : ", PF.portfolio_spec_hash(system.book))
println("  policy hash: ", PF.portfolio_spec_hash(system.policy))
println("  λ / cap    : ", system.policy.risk.lambda, " / ", system.policy.cap.cap)
println("  shrink     : ", system.book.shrink.k)

wall_before_pricing = Dates.now(Dates.UTC)
if wall_before_pricing < R12_AS_OF_UTC
    R12_MODE === :commit && error(
        "commit refused at $wall_before_pricing UTC: the T-25 book does not exist until " *
        "$R12_AS_OF_UTC UTC (14:35 BST).")
    println("\nEARLY DRY RUN PRE-FLIGHT COMPLETE — MultiScaleGRW Fold-43 verified (100% team coverage). " *
            "Pricing will execute at 14:35 BST tomorrow.")
    exit(0)
end

# %%
# ===================================================================
# 6. T-25 calibrated pricing
# ===================================================================
slate = MD.price_slate(spec, system, DD.ScottishLower(), canonical, ds;
                       as_of = R12_AS_OF_UTC,
                       bankroll = R12_BANKROLL,
                       account_id = R12_ACCOUNT,
                       calibrator = calibrator)

slate.fold_idx == 43 || error(
    "pre-flight: MatchDay selected fold $(slate.fold_idx), expected Fold 43. " *
    "Do not commit from an earlier posterior.")
isempty(slate.blocked) || error(
    "pre-flight: $(length(slate.blocked)) fixtures were blocked by gates. " *
    "Read the blocked report before commitment.")

println("\n=== G-F BATCH HEADER (EXPOSURE FIRST) ===")
for (key, value) in pairs(MD.slate_batch_summary(slate))
    println("  ", rpad(string(key), 20), value)
end

println("\n=== COMPLETE MULTISCALE GRW STAKE SHEET ===")
show(stdout, MIME"text/plain"(), slate.sheet[:,
    [:match_id, :group, :line, :selection, :side, :venue_selection, :venue_odds,
     :p_model, :p_market, :edge, :frac, :risk, :venue_stake,
     :depth_touch, :depth_book, :expected_fill, :expected_vwap,
     :expected_slippage, :fill_confidence]]; allrows = true, allcols = true)
println("\n")

println("==========================================================================================================")
println("  MANUAL BETTING SLIP — MULTISCALE GRW SHADOW (15:00 BST KICKOFF)")
println("==========================================================================================================")
fix_names = Dict(c.fixture.m_id => "$(c.fixture.home) v $(c.fixture.away)" for c in slate.cards)
@printf("%-4s %-32s %-12s %-12s %-6s %-8s %-8s %-8s %-8s\n",
        "#", "MATCH", "MARKET", "SELECTION", "SIDE", "ODDS", "STAKE", "EDGE", "DEPTH")
println(repeat("-", 106))
for (i, row) in enumerate(eachrow(slate.sheet))
    mname = get(fix_names, row.match_id, string(row.match_id))
    mkt = row.group == "OverUnder" ? "O/U $(row.line)" : string(row.group)
    sel = string(row.selection)
    side = string(row.side)
    odds_str = @sprintf("%.2f", row.venue_odds)
    stake_str = @sprintf("£%.2f", row.venue_stake)
    edge_str = @sprintf("%+.1f%%", 100 * row.edge)
    depth_str = @sprintf("£%.0f", row.depth_touch)
    @printf("%-4d %-32s %-12s %-12s %-6s %-8s %-8s %-8s %-8s\n",
            i, mname, mkt, sel, side, odds_str, stake_str, edge_str, depth_str)
end
println(repeat("=", 106), "\n")

if R12_MODE === :dry_run
    println("DRY RUN COMPLETE — no ledger write performed.")
    exit(0)
end

# %%
# ===================================================================
# 7. Commit gate and atomic paper_runbook reservation
# ===================================================================
wall_utc = Dates.now(Dates.UTC)
(R12_AS_OF_UTC <= wall_utc <= R12_COMMIT_BY_UTC) || error(
    "commit refused at $wall_utc UTC. Run in [$R12_AS_OF_UTC, $R12_COMMIT_BY_UTC] UTC.")
assert_console_port_free(R12_PORT)

conn = MD.paper_connection()
MD.migrate_paper_schema!(conn; schema = R12_SCHEMA)
account = MD.ensure_account!(conn,
    MD.PaperAccount(account_id = R12_ACCOUNT, opening_balance = R12_BANKROLL,
                    balance = R12_BANKROLL, max_slate_exposure = 0.25, is_live = false);
    schema = R12_SCHEMA)

run_name = canonical.run_name
git_commit = try readchomp(`git rev-parse HEAD`) catch "" end
slate_id = MD.insert_slate!(conn, slate;
    schema = R12_SCHEMA,
    run_name = run_name,
    model_run_id = R12_RUN_UUID,
    book_spec_hash = PF.portfolio_spec_hash(system.book),
    policy_spec_hash = PF.portfolio_spec_hash(system.policy),
    git_commit = git_commit)
orders = MD.orders_to_paper(slate; slate_id = slate_id)
MD.insert_orders!(conn, orders; schema = R12_SCHEMA)

reservation = MD.execute_slate_batch!(conn, R12_ACCOUNT, slate_id; schema = R12_SCHEMA)
reservation.status === MD.RESERVED || error("reservation refused: $(reservation.reason)")

fills = MD.submit_slate!(conn, slate_id, slate.books, MD.LadderSweep(max_slippage = 0.02); schema = R12_SCHEMA)
reconciliation = MD.reconcile_account(conn, R12_ACCOUNT; schema = R12_SCHEMA)
reconciliation.ok || error("post-submission account reconciliation failed: $reconciliation")

println("\n=== COMMITTED MULTISCALE GRW PAPER BATCH ===")
println("  slate_id : ", slate_id)
@printf("  reserved : £%.2f\n", reservation.reserved)
println("  fills    : matched ", fills.n_matched, " partial ", fills.n_partial, " unfilled ", fills.n_unfilled)

# %%
# ===================================================================
# 8. Live operator console — port 8087
# ===================================================================
state = MD.ConsoleState(
    () -> begin
        status = MD._parse_batch(String(first(MD.slate_row(conn, slate_id; schema = R12_SCHEMA)).batch_status))
        MD.slate_snapshot(slate, MD.account_row(conn, R12_ACCOUNT; schema = R12_SCHEMA); status = status)
    end,
    on_execute = () -> begin
        result = MD.execute_slate_batch!(conn, R12_ACCOUNT, slate_id; schema = R12_SCHEMA)
        (ok = result.status === MD.RESERVED, note = "reservation state $(result.status)", error = result.reason)
    end,
    on_kill = () -> begin
        result = MD.kill_slate!(conn, slate_id; schema = R12_SCHEMA)
        (ok = true, note = "killed, released $(abs(result.reserved))", error = nothing)
    end,
)

MD.serve_console(state; host = R12_HOST, port = R12_PORT)
println("\nOption B MultiScaleGRW LIVE console: http://localhost:", R12_PORT)

try
    wait()
catch error
    error isa InterruptException || rethrow()
finally
    MD.stop_console!(state)
    close(conn)
    println("\nLive GRW console stopped.")
end
