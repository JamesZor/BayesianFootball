module QSPaper2627

# 26/27 paper trade (TODO 045): the frozen wave-5 config replayed on this season's L1/L2 matches.
# Order-book odds and closes, the 26/27 market-rate inversion, walk-forward plans for the
# in-memory fold extensions, the m12 fold inventory and the 9-cell replay.
# Read-only: SELECT/SHOW on betdb through a read-only session, load_fit and SELECT/SHOW on
# mcmc_experiments. Outputs go to files only. Sampling lives in l01_sampling.jl.
# Comments only below; the static checker scans every non-comment line for write verbs.

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import LibPQ
import Random
import Serialization
import SHA
import Statistics
import UUIDs

const PF = BF.Portfolio
const MD = BF.MatchDay
const CAL = BF.Calibration
const Data = BF.Data
const P0 = Main.QSBacktestPhase0          # read_query, audit_snapshot
const P3 = Main.QSBacktestPhase3          # goal_model, cell_tables, bet_ledger
const W5 = Main.QSWave5                   # day_problems, solve_days, basket, policies, with_audit
const QMA = Main.QSMarketArm              # wave-1 market arm: SNAPSHOT_SHA, panel_config
const MM = QMA.MM                         # gated_close, gate_book, invert_panel (TODO 039 code)

# ===================================================================
# 1. Constants
# ===================================================================
const SEASON = "26/27"
const TOURNAMENTS = (56, 57)
const FIRST_DAY, LAST_DAY = Dates.Date(2026, 8, 1), Dates.Date(2026, 9, 19)
const N_FIXTURES = 79
const AS_OF = -25.0                       # minutes before the sofascore kick-off
const STALE = 15.0                        # minutes; collector cadence is about 3 min
const CLOSE_WINDOW = (-20.0, 0.0)
const OUT_DIR = "/root/BF_runs/qs_2627/out"
const DS_FILE = joinpath(OUT_DIR, "datastore_ScottishLower_2627.jls")
const OB_FILE = joinpath(OUT_DIR, "orderbook_2627.jls")
const W2_DIR = normpath(joinpath(@__DIR__, "..", "06_qs_joint_and_market_observation", "results"))
const W2_TABLE = joinpath(W2_DIR, "market_rates.csv")
const W2_TABLE_SHA = joinpath(W2_DIR, "market_rates.sha256")
const INVERSION_TOL = 1e-12               # max |delta log rate| on full-book rows; flags exact
const QS = (label = "qs_marketobs", ns = "scottish_lower_qs_wave2_2426", run = "81fdf817-f82d-4f66-9bfd-5f456c4fe29f")
const M12 = (label = "m12", ns = "scottish_lower_joint_player_2426", run = "132df5c2-c742-4e95-8693-3aeb2b2cbaef")
const N_PERSISTED = 40
const MARKET_TYPES = ["MATCH_ODDS", "OVER_UNDER_05", "OVER_UNDER_15", "OVER_UNDER_25", "OVER_UNDER_35",
                      "BOTH_TEAMS_TO_SCORE"]
const MKEY = [:match_id, :market_name, :market_line]
const KEY = [:match_id, :market_name, :market_line, :selection]

# The frozen basket and lambdas (FREEZE.md 04fdba71); the engine gate proves the spelling.
const B1 = W5.basket("B1_canonical", [W5.H, W5.D, W5.A, W5.U25])
const FROZEN_LAMBDA, FALLBACK_LAMBDA = 12.0, 23.0
const GATE_REF = Dict("frozen_B1_l12" => (1.6241209548752367, 494), "fallback_B1_l23" => (1.3208208815934512, 496),
                      "m12_canonical_l23" => (1.327330130248303, 557))

sha_file(path) = bytes2hex(open(SHA.sha256, path))

# ===================================================================
# 2. betdb (read-only session) and the 26/27 DataStore
# ===================================================================
# The connection string comes from .env and is never printed; a failed connect reports its type.
function betdb_connect()
    env = joinpath(pkgdir(BF), ".env")
    isfile(env) && BF.DotEnv.load!(ENV, env)
    haskey(ENV, "BF_DB_URL") || error("BF_DB_URL is not set")
    opts = Dict(LibPQ.CONNECTION_OPTION_DEFAULTS..., "default_transaction_read_only" => "on")
    conn = try
        LibPQ.Connection(ENV["BF_DB_URL"]; options = opts)
    catch err
        error("betdb connection failed: $(typeof(err))")
    end
    value = String(P0.read_query(conn, "SHOW default_transaction_read_only")[1, 1])
    println("BETDB_SESSION_DEFAULT_TRANSACTION_READ_ONLY=", value)
    value == "on" || (close(conn); error("betdb session is not read-only"))
    return conn
end

# Built once (src get_datastore over the read-only session), then pinned by sha256.
function datastore_2627(; expected_sha = nothing)
    if !isfile(DS_FILE)
        expected_sha === nothing || error("pinned DataStore $(DS_FILE) is missing")
        conn = betdb_connect()
        ds = try
            Data.get_datastore(conn, Data.ScottishLower())
        finally
            close(conn)
        end
        mkpath(OUT_DIR)
        Serialization.serialize(DS_FILE, ds)
        println("DATASTORE_2627_BUILT file=", DS_FILE)
    end
    sha = sha_file(DS_FILE)
    expected_sha === nothing || sha == expected_sha || error("26/27 DataStore digest differs: $sha")
    ds = Serialization.deserialize(DS_FILE)
    println("DATASTORE_2627 sha256=", sha, " matches=", DF.nrow(ds.matches), " last=", maximum(ds.matches.match_date))
    return ds, sha
end

function pinned_datastore()
    cache = joinpath(pkgdir(BF), ".cache", "datastore_ScottishLower.jls")
    sha = sha_file(cache)
    sha == QMA.SNAPSHOT_SHA || error("pinned ScottishLower snapshot differs: $sha")
    println("PINNED_DATASTORE sha256=", sha)
    return Serialization.deserialize(cache)
end

# ===================================================================
# 3. The 79 fixtures, results and transition clubs
# ===================================================================
function fixtures_2627(ds)
    m = filter(r -> !ismissing(r.season) && r.season == SEASON && Int(r.tournament_id) in TOURNAMENTS, ds.matches)
    m = DF.sort!(DF.select(m, :match_id, :match_date, :tournament_id, :home_team, :away_team, :home_score,
                           :away_score), [:match_date, :match_id])
    m.match_id = Int.(m.match_id)
    played = .!ismissing.(m.home_score) .& .!ismissing.(m.away_score)
    println("FIXTURES_2627 rows=", DF.nrow(m), " played=", count(played), " first=", minimum(m.match_date),
            " last=", maximum(m.match_date))
    DF.nrow(m) == N_FIXTURES && all(played) || error("expected $N_FIXTURES played 26/27 L1/L2 fixtures")
    minimum(m.match_date) == FIRST_DAY && maximum(m.match_date) == LAST_DAY || error("26/27 date range differs")
    tiers = Data.club_season_tiers(ds.matches)
    moved(c) = get(tiers, (c, "25/26"), 0) != get(tiers, (c, SEASON), -1)
    m.home_moved = moved.(String.(m.home_team))
    m.away_moved = moved.(String.(m.away_team))
    m.transition = m.home_moved .| m.away_moved
    clubs = sort!(unique(vcat(String.(m.home_team[m.home_moved]), String.(m.away_team[m.away_moved]))))
    println("TRANSITION_CLUBS ", join(clubs, ","), " fixtures=", count(m.transition))
    return m, clubs
end

# ===================================================================
# 4. The live order book (copied and widened from proto/clv-napkin l04 c4_markets/c4_snapshots)
# ===================================================================
# One market per (fixture, market type) through the verified crosswalk; the kick-off is the
# sofascore start_timestamp. All canonical market types, not MATCH_ODDS only.
function ob_markets(conn, ids)
    md = P0.read_query(conn, """
        SELECT md.market_id, md.event_id, md.market_type, mm.match_id, mm.is_verified, e.start_timestamp
        FROM betfair_live.market_metadata md
        JOIN betfair.match_meta mm ON mm.betfair_event_id = md.event_id
        JOIN sofascore.events e ON e.match_id = mm.match_id
        WHERE mm.match_id = ANY(\$1) AND md.market_type = ANY(\$2)""", (collect(Int, ids), MARKET_TYPES))
    md.market_id = String.(md.market_id)
    md.market_type = String.(md.market_type)
    md.match_id = Int.(md.match_id)
    md.kickoff = Dates.unix2datetime.(Int.(md.start_timestamp))
    dup = DF.combine(DF.groupby(md, [:match_id, :market_type]), DF.nrow => :n)
    all(==(1), dup.n) || error("more than one market per (fixture, market type)")
    return md
end

_first(v, scale) = (ismissing(v) || isempty(v) || ismissing(v[1])) ? NaN : v[1] / scale

# Every pre-off snapshot: best back (bid) / lay (ask) price and size, best-3 back depth, LTP.
# Prices and volumes arrive x10^4. minutes are relative to the sofascore kick-off.
function ob_snapshots(conn, markets)
    raw = P0.read_query(conn, """
        SELECT market_id, symbol, ts, bid_prices, bid_volumes, ask_prices, ask_volumes,
               total_matched, market_matched, last_price_traded
        FROM betfair_live.order_book_1m WHERE market_id = ANY(\$1)""", (markets.market_id,))
    info = Dict(r.market_id => r for r in eachrow(markets))
    rows = NamedTuple[]
    for r in eachrow(raw)
        mk = info[String(r.market_id)]
        key = MD.betfair_to_key(mk.market_type, String(r.symbol))
        key === nothing && error("unmapped runner $(r.symbol) in $(mk.market_type)")
        ts = Dates.DateTime(r.ts)
        minutes = Dates.value(ts - mk.kickoff) / 60_000
        minutes <= 0.0 || continue
        bv = r.bid_volumes
        push!(rows, (; match_id = mk.match_id, market_type = mk.market_type, market_name = key.group,
                     market_line = Float64(key.line), selection = key.selection, ts, minutes,
                     back = _first(r.bid_prices, 1e4), back_size = _first(bv, 1e4),
                     back_depth3 = ismissing(bv) ? 0.0 : sum(x / 1e4 for x in bv[1:min(3, length(bv))]; init = 0.0),
                     lay = _first(r.ask_prices, 1e4), lay_size = _first(r.ask_volumes, 1e4),
                     ltp = ismissing(r.last_price_traded) ? NaN : r.last_price_traded / 1e4,
                     market_matched = ismissing(r.market_matched) ? NaN : r.market_matched / 1e4))
    end
    snaps = DF.DataFrame(rows)
    snaps.mid = ifelse.(isfinite.(snaps.back) .& isfinite.(snaps.lay), (snaps.back .+ snaps.lay) ./ 2, NaN)
    DF.sort!(snaps, vcat(KEY, [:ts]))
    return snaps
end

# Pulled once and pinned to a file, so later phases read the same rows.
function orderbook_2627(ds, ids; expected_sha = nothing)
    if !isfile(OB_FILE)
        expected_sha === nothing || error("pinned order book $(OB_FILE) is missing")
        conn = betdb_connect()
        markets, snaps = try
            mk = ob_markets(conn, ids)
            (mk, ob_snapshots(conn, mk))
        finally
            close(conn)
        end
        Serialization.serialize(OB_FILE, (; markets, snaps))
        println("ORDERBOOK_2627_BUILT file=", OB_FILE)
    end
    sha = sha_file(OB_FILE)
    expected_sha === nothing || sha == expected_sha || error("26/27 order-book digest differs: $sha")
    ob = Serialization.deserialize(OB_FILE)
    println("ORDERBOOK_2627 sha256=", sha, " markets=", DF.nrow(ob.markets), " snapshots=", DF.nrow(ob.snaps),
            " fixtures=", length(unique(ob.markets.match_id)))
    return ob, sha
end

# The entry book at AS_OF (generalised c4_entry_book / _c4_book_at): per market, every runner's
# last snapshot at or before AS_OF must be two-sided and at most STALE minutes old, the market
# complete, and the mid overround in [0.90, 1.10]. Refusals are kept with their reason.
function entry_book(snaps; as_of = AS_OF, stale = STALE)
    rows, refused = NamedTuple[], NamedTuple[]
    for g in DF.groupby(snaps, MKEY)
        want = CAL.expected_selection_count(first(g.market_name), first(g.market_line))
        pick, reason = NamedTuple[], ""
        for s in DF.groupby(g, :selection)
            vis = s[s.minutes .<= as_of, :]
            if DF.nrow(vis) == 0
                reason = "no snapshot before as-of"
                break
            end
            r = vis[end, :]
            age = as_of - r.minutes
            if !isfinite(r.mid)
                reason = "one-sided"
                break
            elseif age > stale
                reason = "stale"
                break
            end
            push!(pick, (; match_id = r.match_id, market_name = r.market_name, market_line = r.market_line,
                         selection = r.selection, ts = r.ts, entry_minutes = r.minutes, age, back = r.back,
                         back_size = r.back_size, back_depth3 = r.back_depth3, lay = r.lay, lay_size = r.lay_size,
                         mid = r.mid, ltp = r.ltp, market_matched = r.market_matched))
        end
        isempty(reason) && length(pick) != want && (reason = "incomplete market")
        over = isempty(reason) ? sum(1 / p.mid for p in pick) : NaN
        isempty(reason) && !(0.90 <= over <= 1.10) && (reason = "mid overround outside [0.90, 1.10]")
        if !isempty(reason)
            push!(refused, (; match_id = first(g.match_id), market_name = first(g.market_name),
                            market_line = first(g.market_line), reason))
            continue
        end
        back_over = sum(1 / p.back for p in pick)
        append!(rows, [(; p..., mid_overround = over, back_overround = back_over, p_mid_fair = (1 / p.mid) / over)
                       for p in pick])
    end
    return DF.DataFrame(rows), DF.DataFrame(refused)
end

# Wave-4 snapshot-frame columns for PF.build_books_reported: the price sits in :odds_close.
# A market enters a price's frame only if every runner has that price (> 1).
function odds_frame(entry, price::Symbol)
    out = DF.DataFrame[]
    for g in DF.groupby(entry, MKEY)
        p = Float64.(g[!, price])
        all(x -> isfinite(x) && x > 1.0, p) || continue
        push!(out, DF.DataFrame(match_id = Int.(g.match_id), market_name = String.(g.market_name),
                                market_line = Float64.(g.market_line), selection = Symbol.(g.selection),
                                odds_close = p))
    end
    return isempty(out) ? DF.DataFrame(match_id = Int[], market_name = String[], market_line = Float64[],
                                       selection = Symbol[], odds_close = Float64[]) : reduce(vcat, out)
end

# Close = TWA over CLOSE_WINDOW of the mid, through the TODO 039 closing-book arithmetic
# (MM.closing_book's summarize_odds call on the mid instead of odds_history LTP), then MM.gate_book:
# complete markets, overround in [0.90, 1.10], multiplicative de-vig.
function close_book(snaps)
    long = DF.DataFrame(match_id = snaps.match_id, market_name = snaps.market_name, market_line = snaps.market_line,
                        selection = snaps.selection, minutes_to_kickoff = snaps.minutes, traded_price = snaps.mid)
    filter!(r -> isfinite(r.traded_price) && r.traded_price > 1.0, long)
    raw = Data.summarize_odds(long, Data.TWAEstimator(); window = CLOSE_WINDOW)
    prices = DF.DataFrame(match_id = Int.(raw.match_id), market_name = String.(raw.market_name),
                          market_line = Float64.(raw.market_line), selection = Symbol.(raw.selection),
                          odds_close = Float64.(raw.odds))
    filter!(r -> isfinite(r.odds_close) && r.odds_close > 1.0, prices)
    return MM.gate_book(prices)
end

close_index(book) = Dict((Int(r.match_id), String(r.market_name), Float64(r.market_line), Symbol(r.selection)) =>
                         Float64(r.prob_fair_close) for r in eachrow(book))

# Coverage by match day (CLV napkin section 13.1 columns plus the canonical markets).
function coverage_table(fx, ob, entry, closebook)
    has = (ids, f) -> Set(Int.(ids[f]))
    market = Set(ob.markets.match_id[ob.markets.market_type .== "MATCH_ODDS"])
    s1 = ob.snaps[ob.snaps.market_name .== "1X2", :]
    pre = Set(s1.match_id)
    complete(name, line) = Set(entry.match_id[(entry.market_name .== name) .& (entry.market_line .== line)])
    t25 = complete("1X2", 0.0)
    ou25 = complete("OverUnder", 2.5)
    canon = intersect(t25, ou25, complete("OverUnder", 0.5), complete("OverUnder", 1.5), complete("OverUnder", 3.5),
                      complete("BTTS", 0.0))
    e1 = entry[entry.market_name .== "1X2", :]
    ltp1 = Set(g.match_id[1] for g in DF.groupby(e1, :match_id) if all(isfinite, g.ltp))
    close1 = Set(closebook.match_id[closebook.market_name .== "1X2"])
    rows = NamedTuple[]
    for g in DF.groupby(fx, :match_date; sort = true)
        ids = Set(g.match_id)
        n(s) = length(intersect(ids, s))
        push!(rows, (; date = string(first(g.match_date)), played = DF.nrow(g), market = n(market), pre_off_snapshots = n(pre),
                     t25_1x2 = n(t25), t25_1x2_ou25 = n(intersect(t25, ou25)), t25_all_canonical = n(canon),
                     t25_1x2_ltp = n(ltp1), close_1x2 = n(close1), transition = count(g.transition)))
    end
    tab = DF.DataFrame(rows)
    tot = (; date = "total", played = sum(tab.played), market = sum(tab.market),
           pre_off_snapshots = sum(tab.pre_off_snapshots), t25_1x2 = sum(tab.t25_1x2),
           t25_1x2_ou25 = sum(tab.t25_1x2_ou25), t25_all_canonical = sum(tab.t25_all_canonical),
           t25_1x2_ltp = sum(tab.t25_1x2_ltp), close_1x2 = sum(tab.close_1x2), transition = sum(tab.transition))
    push!(tab, tot)
    return tab
end

# ===================================================================
# 5. Market-rate observations (ruling 3)
# ===================================================================
# Wave-2 table rule (06_.../l01_market_table.jl prepare): full book = accepted and >= 5 selections;
# uncovered rows carry finite zero dummies.
function rates_table(rates, config)
    full = rates.accepted .& (rates.n_selections .>= config.min_selections_ladder)
    t = DF.DataFrame(match_id = Int.(rates.match_id),
        log_lambda_h = [full[i] ? log(rates.lambda_h[i]) : 0.0 for i in 1:DF.nrow(rates)],
        log_lambda_a = [full[i] ? log(rates.lambda_a[i]) : 0.0 for i in 1:DF.nrow(rates)],
        full_book = full)
    return DF.sort!(t, :match_id)
end

function committed_table()
    sha = sha_file(W2_TABLE)
    sha == strip(read(W2_TABLE_SHA, String)) || error("wave-2 market-rate table digest differs")
    println("W2_MARKET_TABLE sha256=", sha)
    return CSV.read(W2_TABLE, DF.DataFrame)
end

# Gate: the unchanged inversion on the pinned odds_history closes reproduces the committed table.
function inversion_gate(pinned)
    config = QMA.panel_config()
    book, _ = MM.gated_close(pinned, config)
    rates = MM.invert_panel(pinned, book; config)
    fresh = rates_table(rates, config)
    ref = committed_table()
    fresh.match_id == Int.(ref.match_id) || error("inversion gate: match IDs differ")
    season = Dict(Int(r.match_id) => String(r.season) for r in eachrow(rates))
    rows = NamedTuple[]
    for s in vcat(sort!(unique(values(season))), ["all"])
        idx = [i for i in 1:DF.nrow(ref) if s == "all" || season[Int(ref.match_id[i])] == s]
        flags = fresh.full_book[idx] == Bool.(ref.full_book[idx])
        full = [i for i in idx if ref.full_book[i]]
        dh = isempty(full) ? 0.0 : maximum(abs.(fresh.log_lambda_h[full] .- ref.log_lambda_h[full]))
        da = isempty(full) ? 0.0 : maximum(abs.(fresh.log_lambda_a[full] .- ref.log_lambda_a[full]))
        zero_ok = all(i -> ref.full_book[i] || (fresh.log_lambda_h[i] == 0.0 && fresh.log_lambda_a[i] == 0.0), idx)
        push!(rows, (; season = s, n = length(idx), n_full = length(full), flags_equal = flags,
                     max_abs_dlog_h = dh, max_abs_dlog_a = da, zero_dummies_equal = zero_ok,
                     tolerance = INVERSION_TOL, pass = flags && zero_ok && max(dh, da) <= INVERSION_TOL))
    end
    gate = DF.DataFrame(rows)
    for r in eachrow(gate)
        println("INVERSION_GATE season=", r.season, " n=", r.n, " full=", r.n_full, " flags_equal=", r.flags_equal,
                " max_dlog=", max(r.max_abs_dlog_h, r.max_abs_dlog_a), " pass=", r.pass)
    end
    all(gate.pass) || error("inversion gate failed")
    return gate, ref
end

# 26/27 rates from the order-book mid close with the same inversion; source change logged.
function rates_2627(ds, closebook)
    config = QMA.panel_config()
    rates = MM.invert_panel(ds, closebook; config, seasons = [SEASON])
    table = rates_table(rates, config)
    println("RATES_2627 fixtures=", DF.nrow(table), " accepted=", count(rates.accepted), " full_book=",
            count(table.full_book), " refusals=", join(["$(isempty(k) ? "accepted" : k)=$(v)" for (k, v) in
            sort!(collect(pairs(Dict(r => count(==(r), rates.reason) for r in unique(rates.reason)))))], ";"))
    return rates, table
end

function table_with_2627(ref, table)
    isempty(intersect(Set(Int.(ref.match_id)), Set(table.match_id))) || error("26/27 rows overlap the committed table")
    ext = vcat(DF.DataFrame(match_id = Int.(ref.match_id), log_lambda_h = Float64.(ref.log_lambda_h),
                            log_lambda_a = Float64.(ref.log_lambda_a), full_book = Bool.(ref.full_book)), table)
    return DF.sort!(ext, :match_id)
end

# ===================================================================
# 6. qs_marketobs: the fit's model with the extended feature table; walk-forward plan
# ===================================================================
const FEAT = BF.Features

# Same components (identical objects) except the observation's feature table.
function qs_model_2627(fit, ref, ext)
    m = fit.config.model
    obs = m.observation
    obs isa BF.MarketRateObservation || error("qs fit observation is $(typeof(obs))")
    d_fit = FEAT.market_rates_digest(obs.feature)
    d_ref = FEAT.market_rates_digest(FEAT.MarketRatesFeature(ref))
    d_fit == d_ref || error("fit feature digest differs from the committed table")
    feat = FEAT.MarketRatesFeature(ext)
    old = in.(feat.match_ids, Ref(Set(obs.feature.match_ids)))
    feat.match_ids[old] == obs.feature.match_ids && feat.log_lambda_h[old] == obs.feature.log_lambda_h &&
        feat.log_lambda_a[old] == obs.feature.log_lambda_a && feat.full_book[old] == obs.feature.full_book ||
        error("extended feature changes a fitted row")
    model = BF.PoissonCountModel(m.interception, m.dynamics, m.home_advantage, m.covariates,
                                 BF.MarketRateObservation(feature = feat, sigma_prior = obs.sigma_prior), m.guard)
    println("QS_MODEL_2627 fit_feature_sha=", d_fit, " ext_feature_sha=", FEAT.market_rates_digest(feat),
            " rows=", length(feat.match_ids), " new_rows=", count(.!old), " new_full=", count(feat.full_book[.!old]))
    return model
end

# c4_scope_extended (copied): the run's DataScope with SEASON appended to target_seasons.
function scope_extended(s::Data.DataScope, season::AbstractString = SEASON)
    return Data.DataScope(name = s.name, train_tournaments = s.train_tournaments, cups = s.cups,
        target_tournaments = s.target_tournaments, monitor_tournaments = s.monitor_tournaments,
        clock_tournaments = s.clock_tournaments, target_seasons = vcat(s.target_seasons, [season]),
        history_seasons = s.history_seasons, dynamics_col = s.dynamics_col)
end

# Held-out set of each persisted fold on the pinned store with the persisted splitter (wave 4's
# goal_model rule).
fitted_heldout(fit, pinned) =
    Dict(fd.fold => Set(Int.(Data.get_next_matches(pinned, fd.meta, fit.config.splitter).match_id)) for fd in fit.folds)

# c4_extension_plan generalised: positions 1..N_PERSISTED of the extended splitter must hold out
# exactly the persisted fixtures, position by position; new positions are those > N_PERSISTED.
function walkforward_plan(label, fit, ds, splitter, persisted_ids, by_fold)
    boundaries = Data.create_id_boundaries(ds, splitter)
    length(boundaries) > N_PERSISTED || error("$label: no new positions")
    rows = NamedTuple[]
    union_old = Set{Int}()
    for i in 1:N_PERSISTED
        ho = Set(Int.(Data.get_next_matches(ds, boundaries[i], splitter).match_id))
        union!(union_old, ho)
        ref = get(by_fold, i, Set{Int}())
        push!(rows, (; model = label, position = i, n_heldout = length(ho), n_persisted = length(ref), equal = ho == ref))
    end
    pos = DF.DataFrame(rows)
    ok = all(pos.equal) && union_old == persisted_ids
    println("POSITIONS_1_40 model=", label, " equal=", count(pos.equal), "/", N_PERSISTED, " union_equal=",
            union_old == persisted_ids, " n_union=", length(union_old), " pass=", ok)
    ok || error("$label: the extended splitter moves a historical fold's held-out set")
    delta = collect((N_PERSISTED + 1):length(boundaries))
    oos = Any[Data.get_next_matches(ds, boundaries[i], splitter) for i in delta]
    return (; label, splitter, ds, boundaries, delta, oos, positions = pos)
end

# c4_filtration (copied; pxG coverage replaced by market-rate coverage for qs): no train/held-out
# overlap; every training fixture on a calendar day strictly before the first held-out day;
# training tournaments inside the allowed set.
function filtration(plan, allowed; full_book = nothing)
    meta = Dict(Int(r.match_id) => r for r in eachrow(plan.ds.matches))
    rows = NamedTuple[]
    for (k, i) in enumerate(plan.delta)
        bd, m = plan.boundaries[i]
        train = Int.(vcat(bd.history_match_ids, bd.target_match_ids))
        ho = plan.oos[k]
        ho_ids = Int.(ho.match_id)
        overlap = !isempty(intersect(Set(train), Set(ho_ids)))
        scope_ok = all(id -> Int(meta[id].tournament_id) in allowed, train)
        last_train = maximum(Dates.Date(meta[id].match_date) for id in train)
        first_ho = minimum(Dates.Date.(ho.match_date))
        cur = [id for id in train if String(meta[id].season) == SEASON]
        l12 = ho[in.(Int.(ho.tournament_id), Ref(TOURNAMENTS)), :]
        push!(rows, (; model = plan.label, fold = i, time_step = m.time_step, n_train = length(train),
                     train_last = last_train, heldout_first = first_ho, heldout_last = maximum(Dates.Date.(ho.match_date)),
                     n_heldout = length(ho_ids), n_heldout_l1l2 = DF.nrow(l12),
                     heldout_days = join(string.(sort!(unique(Dates.Date.(l12.match_date)))), " "),
                     n_train_2627 = length(cur),
                     n_train_2627_full_book = full_book === nothing ? missing : count(in(full_book), cur),
                     overlap, scope_ok, strictly_before = last_train < first_ho,
                     pass = !overlap && scope_ok && last_train < first_ho))
    end
    tab = DF.DataFrame(rows)
    for r in eachrow(tab)
        println("FILTRATION model=", r.model, " fold=", r.fold, " heldout=", r.heldout_days, " n_heldout=", r.n_heldout,
                " train_last=", r.train_last, " n_train=", r.n_train, " n_train_2627=", r.n_train_2627,
                " pass=", r.pass)
    end
    all(tab.pass) || error("$(plan.label): filtration failed")
    return tab
end

# ===================================================================
# 7. m12: the persisted 26/27 folds (ruling 4)
# ===================================================================
function m12_inventory(mconn, fit, ds, fx)
    db = P0.read_query(mconn, """
        SELECT fr.fold_idx, fr.n_matches, fr.first_match_date::text AS first_date,
               fr.last_match_date::text AS last_date, fr.runtime_seconds, fr.r_hat_max,
               fr.ess_tail_min, fr.divergences, count(ml.match_id) AS n_latents
        FROM fold_results fr LEFT JOIN match_latents ml ON ml.fold_id = fr.fold_id
        WHERE fr.run_id = \$1::uuid AND fr.fold_idx > \$2
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8 ORDER BY 1""", (M12.run, N_PERSISTED))
    lat = P0.read_query(mconn, """
        SELECT fr.fold_idx, ml.match_id FROM match_latents ml JOIN fold_results fr ON fr.fold_id = ml.fold_id
        WHERE fr.run_id = \$1::uuid AND fr.fold_idx > \$2""", (M12.run, N_PERSISTED))
    fold_of = Dict(Int(r.match_id) => Int(r.fold_idx) for r in eachrow(lat))
    boundaries = Data.create_id_boundaries(ds, fit.config.splitter)
    meta = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
    rows = NamedTuple[]
    for r in eachrow(db)
        k = Int(r.fold_idx)
        ids = sort!([id for (id, f) in fold_of if f == k])
        played = [id for id in ids if id in Set(fx.match_id)]
        fd = only(filter(f -> f.fold == k, fit.folds))
        in_store = k <= length(boundaries)
        train_last = missing
        heldout_store = missing
        if in_store
            bd, m = boundaries[k]
            m.time_step == fd.meta.time_step && m.target_season == fd.meta.target_season ||
                error("m12 fold $k: store boundary differs from the persisted meta")
            train = Int.(vcat(bd.history_match_ids, bd.target_match_ids))
            train_last = maximum(Dates.Date(meta[id].match_date) for id in train)
            heldout_store = length(Data.get_next_matches(ds, boundaries[k], fit.config.splitter).match_id)
        end
        first_ho = isempty(played) ? missing : minimum(Dates.Date(meta[id].match_date) for id in played)
        push!(rows, (; fold = k, season = fd.meta.target_season, time_step = fd.meta.time_step,
                     db_n_matches = r.n_matches, db_first = r.first_date, db_last = r.last_date,
                     runtime_seconds = r.runtime_seconds, r_hat_max = r.r_hat_max, n_latents = Int(r.n_latents),
                     n_latents_played = length(played), heldout_in_store = heldout_store,
                     train_last_reconstructed = train_last, heldout_first = first_ho,
                     strictly_before = (ismissing(train_last) || ismissing(first_ho)) ? missing : train_last < first_ho))
    end
    inv = DF.DataFrame(rows)
    covered = Set(id for (id, f) in fold_of if id in Set(fx.match_id))
    missing_days = sort!(unique(fx.match_date[.!in.(fx.match_id, Ref(covered))]))
    batch = length(unique(inv.runtime_seconds)) < DF.nrow(inv)
    println("M12_DB_FOLDS n=", DF.nrow(inv), " latents_played=", length(covered), "/", N_FIXTURES,
            " uncovered_days=", join(string.(missing_days), " "), " shared_runtime_batches=", batch)
    for r in eachrow(inv)
        println("M12_DB_FOLD fold=", r.fold, " step=", r.time_step, " db=", r.db_first, "..", r.db_last, " n=", r.db_n_matches,
                " latents=", r.n_latents, " played=", r.n_latents_played, " runtime=", r.runtime_seconds,
                " train_last=", r.train_last_reconstructed, " strictly_before=", r.strictly_before)
    end
    return inv, covered, missing_days
end

# ===================================================================
# 8. The replay: 3 cells per price (frozen, fallback, m12 canonical)
# ===================================================================
function build_books(model, odds, ds)
    spec = W5.canonical_spec()
    if model.source isa BF.Training.Fit
        return PF.build_books_reported(spec, model.source, odds, ds; require_result = true,
                                       require_converged = false, quiet = true)
    end
    return PF.build_books_reported(spec, model.source, odds, ds; require_result = true,
                                   converged = model.converged, quiet = true)
end

function replay_cells(qs, m12, odds, ds)
    qb, qr = build_books(qs, odds, ds)
    mb, mr = build_books(m12, odds, ds)
    days = W5.day_problems(qs, qb)
    slates, diag = W5.solve_days(days, s -> W5.in_basket(B1, s))
    DF.nrow(diag) == length(days) || error("the frozen basket dropped a day")
    books = PF.MatchBook[b for sl in slates for b in sl.books]
    cells = [
        (; cell = "frozen_B1_l12", role = "frozen", model = qs.label, lambda = FROZEN_LAMBDA,
           result = W5.simulate_slates(W5.basket_policy(B1.trust, FROZEN_LAMBDA), slates, qr), books, report = qr, diag),
        (; cell = "fallback_B1_l23", role = "fallback", model = qs.label, lambda = FALLBACK_LAMBDA,
           result = W5.simulate_slates(W5.basket_policy(B1.trust, FALLBACK_LAMBDA), slates, qr), books, report = qr, diag),
        (; cell = "m12_canonical_l23", role = "benchmark", model = m12.label, lambda = 23.0,
           result = W5.simulate_books(MD.canonical_scottish_lower_policy(), mb, mr), books = mb, report = mr,
           diag = nothing),
    ]
    return (; cells, qbooks = qb, mbooks = mb, days)
end

# Ruling 9: the replay code on wave 5's 25/26 T-25m inputs reproduces the frozen cell bit-identically.
function engine_gate(freeze_path)
    W5.release!(freeze_path)
    inputs = W5.load_inputs([W5.T25])
    odds = W5.half_odds(inputs.ctx, inputs.books[W5.T25.label], :evaluation)
    rep = replay_cells(inputs.models["qs_marketobs"], inputs.models["m12"], odds, inputs.ctx.ds)
    rows = NamedTuple[]
    for c in rep.cells
        w, n = c.result.trajectory.bankroll[end], c.result.summary.n_bets
        ref = GATE_REF[c.cell]
        push!(rows, (; cell = c.cell, wealth = w, n_bets = n, ref_wealth = ref[1], ref_bets = ref[2],
                     bit_identical = w === ref[1] && n == ref[2]))
        println("ENGINE_GATE cell=", c.cell, " wealth=", repr(w), " bets=", n, " ref=", repr(ref[1]), "/", ref[2],
                " bit_identical=", rows[end].bit_identical)
    end
    gate = DF.DataFrame(rows)
    all(gate.bit_identical) || error("engine gate failed")
    return gate
end

end # module
