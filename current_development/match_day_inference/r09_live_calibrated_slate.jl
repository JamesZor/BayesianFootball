# r09_live_calibrated_slate.jl
#
# ===================================================================
# WHAT THIS IS
# ===================================================================
# The live Scottish Lower Option B card for Saturday 2026-09-05. It prices one simultaneous
# 15:00 BST settlement window at 14:35 BST (T-25) from the completed Fold-43 m12 posterior,
# applies the validated T-25 generative-rate calibrator, and sizes only the audited Option B
# basket.
#
# This is execution plumbing, not a training run and not a replay. Nothing samples here.
# Historical validation belongs on port 8086 / schema paper_replay; this runner may write only
# when invoked with `--commit`, and then only to betdb.paper_runbook.
#
# TIME CONTRACT. MatchDay stores UTC-naive DateTimes. 15:00 BST is 14:00 UTC, so the internal
# pricing instant is 13:35 UTC. Labelling `DateTime(2026,9,5,14,35)` as T-25 would actually hand
# the pipeline an in-play T+35 book.
#
# USAGE
#   julia --project -t 8 current_development/match_day_inference/r09_live_calibrated_slate.jl --dry-run
#   julia --project -t 8 current_development/match_day_inference/r09_live_calibrated_slate.jl --commit
#
# `--dry-run` performs no ledger migration, account creation, slate insertion, reservation, fill
# or submission. `--commit` performs the whole paper_runbook transaction chain and serves the
# resulting live operator console on port 8085.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning, LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using DataFrames, Dates, Printf
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
const R09_DAY           = Date(2026, 9, 5)
const R09_KICKOFF_UTC   = DateTime(2026, 9, 5, 14, 0)   # 15:00 BST
const R09_AS_OF_UTC     = DateTime(2026, 9, 5, 13, 35)  # 14:35 BST, exactly T-25
const R09_COMMIT_BY_UTC = DateTime(2026, 9, 5, 13, 58)  # commit allowed before 15:00 BST kickoff
const R09_TOURNAMENTS   = [56, 57]
const R09_BANKROLL      = 2_400.0
const R09_ACCOUNT       = "live_scottish"
const R09_SCHEMA        = "paper_runbook"
const R09_EXPERIMENT    = "scottish_lower_joint_player_2426"
const R09_RUN_NAME      = "m12_joint_hybrid_synergy"
const R09_HOST          = "0.0.0.0"
const R09_PORT          = 8085

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

function r09_mode(args)
    flags = Set(args)
    unknown = setdiff(flags, Set(["--dry-run", "--commit"]))
    isempty(unknown) || error("unknown argument(s): $(join(sort!(collect(unknown)), ", ")). " *
                              "Use exactly one of --dry-run or --commit.")
    xor("--dry-run" in flags, "--commit" in flags) || error(
        "choose exactly one mode: --dry-run (no writes) or --commit (paper_runbook write).")
    return "--commit" in flags ? :commit : :dry_run
end

const R09_MODE = r09_mode(ARGS)

println("\n" * "="^82)
println("  Scottish Lower Option B — LIVE slate")
println("  decision : 2026-09-05 14:35 BST / ", R09_AS_OF_UTC, " UTC (T-25)")
println("  kickoff  : 2026-09-05 15:00 BST / ", R09_KICKOFF_UTC, " UTC")
println("  model    : ", R09_EXPERIMENT, " / ", R09_RUN_NAME)
println("  mode     : ", R09_MODE)
println("  ledger   : ", R09_SCHEMA, R09_MODE === :dry_run ? " (NO WRITES)" : " (COMMIT)")
println("="^82 * "\n")

# %%
# ===================================================================
# 3. Read-only operational pre-flight
# ===================================================================
function r09_load_fixtures(conn)
    lo = Int(round(datetime2unix(DateTime(R09_DAY))))
    hi = Int(round(datetime2unix(DateTime(R09_DAY) + Day(1))))
    frame = DataFrame(LibPQ.execute(conn, """
        SELECT match_id, home_team, away_team, start_timestamp, tournament_id, status_type
        FROM sofascore.events
        WHERE tournament_id = ANY(\$1) AND start_timestamp >= \$2 AND start_timestamp < \$3
        ORDER BY start_timestamp, tournament_id, match_id;
    """, (R09_TOURNAMENTS, lo, hi)))
    fixtures = MD.Fixture[
        MD.Fixture(Int(row.match_id), String(row.home_team), String(row.away_team),
                   unix2datetime(row.start_timestamp), Int(row.tournament_id))
        for row in eachrow(frame)
    ]
    return frame, fixtures
end

function r09_book_health(conn, match_ids)
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

function r09_lineup_health(conn, match_ids)
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

function r09_preflight()
    conn = MD.paper_connection()
    try
        fixture_frame, fixtures = r09_load_fixtures(conn)
        println("=== G-A FIXTURE INVENTORY ===")
        show(stdout, MIME"text/plain"(), fixture_frame; allrows = true, allcols = true)
        println("\n")
        length(fixtures) == 10 || error(
            "pre-flight: expected 10 Scottish Lower fixtures on $R09_DAY, " *
            "found $(length(fixtures)).")
        all(f -> f.kickoff == R09_KICKOFF_UTC, fixtures) || error(
            "pre-flight: the card does not form one 14:00 UTC settlement window: " *
            join(sort!(unique(string(f.kickoff) for f in fixtures)), ", "))

        println("=== G-B BETFAIR TICK HEALTH ===")
        book_health = r09_book_health(conn, [f.m_id for f in fixtures])
        show(stdout, MIME"text/plain"(), book_health; allrows = true, allcols = true)
        println("\n")
        nrow(book_health) == length(fixtures) || error(
            "pre-flight: only $(nrow(book_health)) of $(length(fixtures)) fixtures have a " *
            "Betfair crosswalk/book health row.")
        all(Int.(book_health.markets) .> 0) || error(
            "pre-flight: one or more fixtures have no mapped Betfair market.")
        all(Int.(book_health.book_rows) .> 0) || error(
            "pre-flight: one or more fixtures have no archived order-book tick.")

        println("=== G-C LINEUP READINESS ===")
        lineup_health = r09_lineup_health(conn, [f.m_id for f in fixtures])
        show(stdout, MIME"text/plain"(), lineup_health; allrows = true, allcols = true)
        println("\n")
        ready_count = count(>(0), Int.(lineup_health.player_rows))
        if ready_count == 0
            @warn "no SofaScore provisional XI has landed; the T-25 SourceChain will query " *
                  "BBC confirmed lineups first, then fall back to LastHistorical only if both " *
                  "live sources have no complete 11v11 answer"
        else
            @info "provisional lineups visible" fixtures = ready_count total = length(fixtures)
        end
        return fixtures
    finally
        close(conn)
    end
end

fixtures = r09_preflight()

# %%
# ===================================================================
# 4. Data snapshot, canonical posterior, and convergence gate
# ===================================================================
@info "loading ScottishLower DataStore (uses .cache/ if warm)"
ds = DD.load_datastore_cached(DD.ScottishLower())

storage = TT.PostgresStorage(R09_EXPERIMENT)
canonical = MD.canonical_fit(storage, R09_RUN_NAME; require_converged = true)
MD.matchday_fit_report(canonical)
canonical.n_folds >= 43 || error(
    "pre-flight: $R09_RUN_NAME has $(canonical.n_folds) folds; Fold 43 is required.")

# Before the decision instant there is no T-25 book to price, but the feature build can still
# prove that the exact serving fold recognises every team. At/after T-25 the full pipeline repeats
# this gate on the materialised cards, so do not spend the execution window building it twice.
if Dates.now(Dates.UTC) < R09_AS_OF_UTC
    selected, team_coverage = verify_fold_team_coverage(canonical.fit, ds, fixtures)
    println("=== G-D FOLD $(selected.idx) TEAM-MAP COVERAGE ===")
    show(stdout, MIME"text/plain"(), team_coverage; allrows = true, allcols = true)
    println("\n")
    selected.idx == 43 || error("expected serving fold 43, selected fold $(selected.idx)")
end

# %%
# ===================================================================
# 5. Live sources and the frozen Option B recipe
# ===================================================================
# BBC's confirmed XI is the primary source. SofaScore's provisional table is retained as the
# second rung, and `LastHistorical(ds)` is the explicit final fallback rather than a neutral
# player pillar. Whichever live source answers is absorbed by the same
# LineupAggregateFromRAPM materialiser used throughout MatchDay.
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

# A future T-25 book does not exist yet. Refuse to manufacture one from an older ladder and make
# an early dry-run a successful PRE-FLIGHT only; running the same command at/after 14:35 BST goes
# through the full calibrated pricing path below.
wall_before_pricing = Dates.now(Dates.UTC)
if wall_before_pricing < R09_AS_OF_UTC
    R09_MODE === :commit && error(
        "commit refused at $wall_before_pricing UTC: the T-25 book does not exist until " *
        "$R09_AS_OF_UTC UTC (14:35 BST).")
    println("\nEARLY DRY RUN PRE-FLIGHT COMPLETE — fixtures, collector, lineup state and the " *
            "converged 43-fold run are available. Calibrated stake generation is deliberately deferred until the " *
            "T-25 book exists at 14:35 BST. No paper_runbook write was performed.")
    exit(0)
end

# %%
# ===================================================================
# 6. T-25 calibrated pricing and coverage gate
# ===================================================================
# A successful full-card extraction is the strongest coverage check: MatchDay materialises every
# per-fixture map and then `check_coverage` refuses all absent clubs by name before extraction.
# Fold 43 is expected to cover Ross County and Airdrieonians; no hard-coded exclusion is allowed.
slate = MD.price_slate(spec, system, DD.ScottishLower(), canonical, ds;
                       as_of = R09_AS_OF_UTC,
                       bankroll = R09_BANKROLL,
                       account_id = R09_ACCOUNT,
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

println("\n=== COMPLETE OPTION B STAKE SHEET ===")
show(stdout, MIME"text/plain"(), slate.sheet[:,
    [:match_id, :group, :line, :selection, :side, :venue_selection, :venue_odds,
     :p_model, :p_market, :edge, :frac, :risk, :venue_stake,
     :depth_touch, :depth_book, :expected_fill, :expected_vwap,
     :expected_slippage, :fill_confidence]]; allrows = true, allcols = true)
println("\n")

println("=== G-G CAPACITY / EXPOSURE SUMMARY ===")
@printf("  bankroll             £%.2f\n", slate.bankroll)
@printf("  total liability      £%.2f (%.2f%%)\n", slate.total_risk,
        100 * slate.total_risk / slate.bankroll)
@printf("  expected three-level £%.2f\n", sum(slate.sheet.expected_fill))
println("  capped               ", slate.capped)
println("  low-confidence legs  ", count(==(:low), slate.sheet.fill_confidence))
println("  gated basket check   ", all(row ->
    (row.group == "1X2" && row.selection in (:home, :draw, :away)) ||
    (row.group == "OverUnder" && row.line == 2.5 && row.selection == :under_25) ||
    (row.group == "OverUnder" && row.line == 1.5 && row.selection == :over_15),
    eachrow(slate.sheet)))

blocked = MD.blocked_report(MD.MatchDayResult(slate.sheet, slate.cards, slate.blocked,
                                               slate.odds, slate.instruments, slate.as_of))
println("\n=== G-H BLOCKED REPORT ===")
show(stdout, MIME"text/plain"(), blocked; allrows = true, allcols = true)
println("\n")

if R09_MODE === :dry_run
    println("DRY RUN COMPLETE — no paper_runbook schema migration or ledger write was performed.")
    exit(0)
end

# %%
# ===================================================================
# 7. Commit gate and atomic paper_runbook reservation
# ===================================================================
wall_utc = Dates.now(Dates.UTC)
(R09_AS_OF_UTC <= wall_utc <= R09_COMMIT_BY_UTC) || error(
    "commit refused at $wall_utc UTC. A slate priced from the T-25 snapshot may be committed " *
    "only in [$R09_AS_OF_UTC, $R09_COMMIT_BY_UTC] UTC; after that its executable book is stale. " *
    "Run --dry-run outside that window.")
assert_console_port_free(R09_PORT)

conn = MD.paper_connection()
MD.migrate_paper_schema!(conn; schema = R09_SCHEMA)
account = MD.ensure_account!(conn,
    MD.PaperAccount(account_id = R09_ACCOUNT, opening_balance = R09_BANKROLL,
                    balance = R09_BANKROLL, max_slate_exposure = 0.25, is_live = false);
    schema = R09_SCHEMA)
MD.equity(account) == R09_BANKROLL || error(
    "commit refused: account '$R09_ACCOUNT' has equity £$(MD.equity(account)), but this slate " *
    "was priced at £$R09_BANKROLL. Re-price from the account of record; never rescale a solved vector.")

run_id = TT.Inference._run_uuid(storage, R09_RUN_NAME)
git_commit = try
    readchomp(`git rev-parse HEAD`)
catch
    ""
end
slate_id = MD.insert_slate!(conn, slate;
    schema = R09_SCHEMA,
    run_name = R09_RUN_NAME,
    model_run_id = run_id,
    book_spec_hash = PF.portfolio_spec_hash(system.book),
    policy_spec_hash = PF.portfolio_spec_hash(system.policy),
    git_commit = git_commit)
orders = MD.orders_to_paper(slate; slate_id = slate_id)
MD.insert_orders!(conn, orders; schema = R09_SCHEMA)

reservation = MD.execute_slate_batch!(conn, R09_ACCOUNT, slate_id; schema = R09_SCHEMA)
reservation.status === MD.RESERVED || error(
    "reservation refused: $(reservation.reason). No submission was attempted.")

# `TouchOnly` is the honest live paper model: rest at the touch and let unmatched size expire.
fills = MD.submit_slate!(conn, slate_id, slate.books, MD.TouchOnly(); schema = R09_SCHEMA)
reconciliation = MD.reconcile_account(conn, R09_ACCOUNT; schema = R09_SCHEMA)
reconciliation.ok || error("post-submission account reconciliation failed: $reconciliation")

println("\n=== COMMITTED PAPER BATCH ===")
println("  slate_id : ", slate_id)
@printf("  reserved : £%.2f\n", reservation.reserved)
println("  admitted : ", reservation.n_admitted, "  refused: ", reservation.n_refused)
println("  fills    : matched ", fills.n_matched, " partial ", fills.n_partial,
        " unfilled ", fills.n_unfilled)
println("  reconcile: OK")

# %%
# ===================================================================
# 8. Live operator console — port 8085
# ===================================================================
state = MD.ConsoleState(
    () -> begin
        status = MD._parse_batch(String(first(MD.slate_row(conn, slate_id;
                                                            schema = R09_SCHEMA)).batch_status))
        MD.slate_snapshot(slate, MD.account_row(conn, R09_ACCOUNT; schema = R09_SCHEMA);
                          status = status)
    end,
    on_execute = () -> begin
        result = MD.execute_slate_batch!(conn, R09_ACCOUNT, slate_id; schema = R09_SCHEMA)
        (ok = result.status === MD.RESERVED,
         note = "reservation state $(result.status)", error = result.reason)
    end,
    on_kill = () -> begin
        result = MD.kill_slate!(conn, slate_id; schema = R09_SCHEMA)
        (ok = true, note = "killed, released $(abs(result.reserved))", error = nothing)
    end,
)

MD.serve_console(state; host = R09_HOST, port = R09_PORT)
println("\n" * "="^82)
println("  Option B LIVE console: http://localhost:", R09_PORT)
println("  schema: ", R09_SCHEMA, "  account: ", R09_ACCOUNT, "  slate: ", slate_id)
println("  Ctrl+C stops the console; the committed ledger remains durable.")
println("="^82 * "\n")

try
    wait()
catch error
    error isa InterruptException || rethrow()
finally
    MD.stop_console!(state)
    close(conn)
    println("\nLive Option B console stopped.")
end
