module QSBacktestPhase3

# Phase 3: the five-model x four-snapshot x five-policy betting grid on DailySlate.
# Read-only: saved fits by UUID, the saved datastore cache, C0's saved latents. No fitting,
# no tuning; every policy parameter is fixed by DECISIONS_LOG rulings 1-3.
# Comments only below; the static checker scans every non-comment line for write verbs.

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import Random
import Serialization
import SHA
import Statistics
import UUIDs

const PF = BF.Portfolio
const MD = BF.MatchDay
const CAL = BF.Calibration
const P0 = Main.QSBacktestPhase0          # read_query, audit_snapshot, snapshot, historical_system
const SC = Main.QSBacktestScenario        # scenario_books (Phase 2 daily adapter)

const SEASONS = ["24/25", "25/26"]
const GATE_WEALTH = 2.515204600600047
const GATE_BETS = 1127
const GATE_TOLERANCE = 0.00005
const B_REPS = 999
const BLOCK_WEEKS = 8
const BOOT_SEED = 20261009

const GOAL_MODELS = [
    (label = "qs_marketobs", ns = "scottish_lower_qs_wave2_2426", run = "81fdf817-f82d-4f66-9bfd-5f456c4fe29f"),
    (label = "m12", ns = "scottish_lower_joint_player_2426", run = "132df5c2-c742-4e95-8693-3aeb2b2cbaef"),
    (label = "grw_joint", ns = "scottish_lower_qs_wave2_2426", run = "98447840-e95c-420f-92b4-2db6545506a3"),
    (label = "fusion_qs_nobias", ns = "scottish_lower_qs_wave3_2426", run = "5afdfeb3-abad-4d23-82fb-1bfe6babe797"),
]
const C0_DIR = "/root/BF_runs/qs_experiment_out/market_grid"
const MODEL_ORDER = ["qs_marketobs", "m12", "grw_joint", "fusion_qs_nobias", "market_c0"]

# Phase 0 committed the four snapshot books; Phase 3 rebuilds them and requires equality.
const SNAPSHOTS = [
    (label = "T-25m_s90", minutes = -25.0, staleness = 90.0, stem = "phase0_t25"),
    (label = "T-6h_s90", minutes = -360.0, staleness = 90.0, stem = "phase0_t360"),
    (label = "T-24h_s90", minutes = -1440.0, staleness = 90.0, stem = "phase0_t1440"),
    (label = "T-24h_s360", minutes = -1440.0, staleness = 360.0, stem = "phase0_t1440_sensitivity360"),
]
const POLICY_ORDER = ["historical_exp07", "canonical", "option_b_live", "plain_kelly", "scenario_kelly"]

# ===================================================================
# 1. Policies (rulings 2 and 3; no parameter differs from policy_inventory.csv)
# ===================================================================
function systems(out)
    hist = P0.historical_system(out)
    canonical_spec = PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb())
    option_b = MD.option_b_system()
    plain_spec = PF.BookSpec(
        markets = BF.Data.MarketConfig(BF.Data.AbstractMarket[
            BF.Data.Market1X2(), BF.Data.MarketOverUnder(2.5), BF.Data.MarketBTTS()]),
        price = PF.DeArb(), allocator = PF.KellyLogUtility(), shrink = PF.FractionalKelly(0.25),
        exec = PF.ExecutionConfig(commission = PF.PerBetCommission(0.02), budget = 0.99,
                                  min_selection_stake = 0.0001, max_selection_stake = 0.50))
    plain_policy = PF.PolicySpec(trust = PF.FlatTrust(1.0), risk = PF.SlateDrawdown(23.0),
                                 cap = PF.FixedCap(0.25), grouping = PF.DailySlate())
    return Dict("historical_exp07" => (hist.book, hist.policy),
                "canonical" => (canonical_spec, MD.canonical_scottish_lower_policy()),
                "option_b_live" => (option_b.book, option_b.policy),
                "plain_kelly" => (plain_spec, plain_policy))
end

# ===================================================================
# 2. Saved predictions and posterior/fold witnesses
# ===================================================================
# A witness names the run, the fold that predicted the fixture and the draw layout. Two
# fixtures share draw column j only if their witnesses are equal; the adapter refuses a day
# whose fixtures disagree.
function goal_model(m, ds, ids)
    raw = BF.Training.load_fit(BF.Training.PostgresStorage(m.ns), UUIDs.UUID(m.run))
    latents = CAL.restrict_latents(raw.latents, ids)
    Set(Int.(BF.Models.latent_match_ids(latents))) == ids || error("$(m.label): not the 710 fixtures")
    fit = BF.Training.Fit(raw.config, raw.folds, latents, raw.diagnostics, raw.metadata, raw.save_path)
    nd = BF.Models.n_draws(latents)
    fold = Dict{Int,Int}()
    for fd in raw.folds
        size(fd.chain, 1) * size(fd.chain, 3) == nd || error("$(m.label) fold $(fd.fold): draw layout differs")
        for id in Int.(BF.Data.get_next_matches(ds, fd.meta, raw.config.splitter).match_id)
            id in ids || continue
            haskey(fold, id) && error("$(m.label): fixture $id held out by two folds")
            fold[id] = fd.fold
        end
    end
    length(fold) == length(ids) || error("$(m.label): fold map covers $(length(fold)) fixtures")
    layout = string(size(raw.folds[1].chain, 3), "chains_x_", size(raw.folds[1].chain, 1))
    witness = Dict(id => string(m.label, "|", m.run, "|fold=", fold[id], "|", layout) for id in ids)
    return (; label = m.label, run = m.run, source = fit, latents, witness, fold, nd, layout)
end

function market_model(ds, ids)
    full = Serialization.deserialize(joinpath(C0_DIR, "market_latents.jls"))
    latents = CAL.restrict_latents(full, ids)
    Set(Int.(BF.Models.latent_match_ids(latents))) == ids || error("market_c0: not the 710 fixtures")
    fold = Dict{Int,Int}()
    for d in filter(d -> startswith(d, "fold_"), readdir(C0_DIR))
        part = Serialization.deserialize(joinpath(C0_DIR, d, "latents.jls"))
        size(part.λ_home, 2) == 512 || error("market_c0 $d: draw layout differs")
        for id in Int.(part.match_ids)
            haskey(fold, id) && error("market_c0: fixture $id in two folds")
            fold[id] = parse(Int, d[6:end])
        end
    end
    all(id -> haskey(fold, id), ids) || error("market_c0: fold map incomplete")
    layout = "theta128_x_state4"            # market_arm.jl forecast_fold: column = (theta j, rep)
    witness = Dict(id => string("market_c0|fold=", fold[id], "|", layout) for id in ids)
    return (; label = "market_c0", run = "c0_files:" * C0_DIR, source = latents, latents,
            witness, fold, nd = BF.Models.n_draws(latents), layout)
end

# ===================================================================
# 3. Snapshot books (Phase 0 inputs, equality required) and the T-0 close
# ===================================================================
function snapshot_books(ctx, phase0_dir)
    books = Dict{String,DF.DataFrame}()
    for s in SNAPSHOTS
        book, _ = P0.snapshot(ctx, s.minutes; staleness = s.staleness)
        ref = CSV.read(joinpath(phase0_dir, s.stem * "_book.csv"), DF.DataFrame)
        ref.selection = Symbol.(ref.selection)
        isequal(ref, book) || error("snapshot $(s.label) differs from its committed Phase 0 book")
        books[s.label] = book
        println("INPUT_PARITY_PASS snapshot=", s.label, " selections=", DF.nrow(book),
                " fixtures=", length(unique(book.match_id)))
    end
    return books
end

# Napkin close (proto/clv-napkin l01 cn_close_book): TWA over (-20, 0], complete markets
# only, overround in [0.90, 1.10], multiplicative de-vig within (match, market, line).
function close_index(ds, ids)
    odds = filter(r -> Int(r.match_id) in ids, ds.betfair_odds)
    raw = BF.Data.summarize_odds(odds, BF.Data.TWAEstimator(); window = (-20.0, 0.0),
                                 overround_limits = (0.0, Inf))
    prices = DF.DataFrame(match_id = Int.(raw.match_id), market_name = String.(raw.market_name),
                          market_line = Float64.(raw.market_line), selection = Symbol.(raw.selection),
                          odds_close = Float64.(coalesce.(raw.odds, NaN)))
    filter!(r -> isfinite(r.odds_close) && r.odds_close > 1.0, prices)
    index = Dict{Tuple{Int,String,Float64,Symbol},Float64}()
    for g in DF.groupby(prices, [:match_id, :market_name, :market_line])
        want = CAL.expected_selection_count(first(g.market_name), first(g.market_line))
        (want > 0 && DF.nrow(g) == want) || continue
        implied = 1.0 ./ g.odds_close
        over = sum(implied)
        0.90 <= over <= 1.10 || continue
        for (k, r) in enumerate(eachrow(g))
            index[(r.match_id, r.market_name, r.market_line, r.selection)] = implied[k] / over
        end
    end
    return index
end

# ===================================================================
# 4. One cell's tables
# ===================================================================
market_label(s) = s.group == "OverUnder" ? string("OU", s.line) : s.group

function bet_ledger(result, books, close)
    t = result.trajectory
    sels = Dict((b.m_id, s.family) => s for b in books for s in b.sels)
    open_bank = Dict(d => t.bankroll[k] for (k, d) in enumerate(t.dates))
    rows = NamedTuple[]
    for r in eachrow(t.bets)
        s = sels[(r.match_id, r.family)]
        pc = get(close, (Int(r.match_id), s.group, s.line, s.selection), NaN)
        bank = open_bank[r.date]
        push!(rows, (; match_id = Int(r.match_id), date = r.date, market = market_label(s),
                     family = s.family, selection = String(s.selection), odds_quoted = s.odds_quoted,
                     odds_used = r.odds, stake_frac = r.stake, pnl_frac = r.pnl,
                     stake_unit = r.stake * bank, pnl_unit = r.pnl * bank,
                     p_model = r.p_model, p_market = r.p_market, p_close = pc,
                     clv_pp = 100 * (pc - 1 / s.odds_quoted), clv_pct = 100 * (s.odds_quoted * pc - 1)))
    end
    isempty(rows) && return DF.DataFrame(match_id = Int[], date = Dates.Date[], market = String[],
        family = String[], selection = String[], odds_quoted = Float64[], odds_used = Float64[],
        stake_frac = Float64[], pnl_frac = Float64[], stake_unit = Float64[], pnl_unit = Float64[],
        p_model = Float64[], p_market = Float64[], p_close = Float64[], clv_pp = Float64[],
        clv_pct = Float64[])
    return DF.DataFrame(rows)
end

function clv_stats(bets)
    c = filter(r -> isfinite(r.clv_pp), bets)
    n = DF.nrow(c)
    return (; clv_n = n, clv_coverage = DF.nrow(bets) == 0 ? NaN : n / DF.nrow(bets),
            clv_mean_pp = n == 0 ? NaN : Statistics.mean(c.clv_pp),
            clv_mean_pct = n == 0 ? NaN : Statistics.mean(c.clv_pct),
            clv_beat_pct = n == 0 ? NaN : 100 * Statistics.mean(c.clv_pp .> 0))
end

function cell_tables(key, result, books, report, close, season_of)
    s = result.summary
    t = result.trajectory
    bets = bet_ledger(result, books, close)
    money_stake, money_pnl = sum(bets.stake_unit; init = 0.0), sum(bets.pnl_unit; init = 0.0)
    summary = (; key..., status = "COMPUTED", reason = "",
        final_wealth = t.bankroll[end], mean_daily_log_growth = s.growth_per_slate,
        roi_frac_pct = s.roi, roi_money_pct = money_stake > 0 ? 100 * money_pnl / money_stake : NaN,
        n_bets = s.n_bets, turnover_frac = s.total_stake, turnover_unit = money_stake,
        pnl_unit = money_pnl, mdd_pct = s.mdd, sharpe = s.sharpe, sharpe_ann = s.sharpe_ann,
        calmar = s.calmar, n_slates = s.n_slates, n_books = length(books),
        n_build_errors = report === nothing ? 0 : length(report.errored),
        converged = report === nothing ? missing : something(report.converged, missing), clv_stats(bets)...)
    markets = NamedTuple[]
    for g in DF.groupby(bets, :market; sort = true)
        st, pl = sum(g.stake_unit), sum(g.pnl_unit)
        push!(markets, (; key..., market = first(g.market), n_bets = DF.nrow(g),
            turnover_unit = st, pnl_unit = pl, roi_money_pct = 100 * pl / st,
            turnover_frac = sum(g.stake_frac), roi_frac_pct = 100 * sum(g.pnl_frac) / sum(g.stake_frac),
            clv_stats(g)...))
    end
    states = PF.states_frame(result)
    states.date == t.dates || error("daily states and trajectory dates differ")
    daily = [(; key..., date = d, season = season_of[d], n_fixtures = states.n_fixtures[k],
                         n_bets = states.n_bets[k], stake_frac = states.stake_frac[k],
                         pnl_frac = t.slate_pl[k], log_growth = log1p(t.slate_pl[k]),
                         bank_open = t.bankroll[k], bank_close = t.bankroll[k + 1])
             for (k, d) in enumerate(t.dates)]
    bets_long = DF.hcat(DF.DataFrame([key for _ in 1:DF.nrow(bets)]), bets)
    return (; summary, markets, daily, bets = bets_long)
end

# ===================================================================
# 5. Scenario Kelly: canonical books, joint daily solve, canonical overlay
# ===================================================================
function scenario_slates(model, spec, policy, books)
    slates = PF.group(policy.grouping, books)
    w = PF.BookWorkspace(spec, model.latents; quiet = true)
    row = Dict(Int(id) => i for (i, id) in enumerate(BF.Models.latent_match_ids(model.latents)))
    out, diag = PF.Slate[], NamedTuple[]
    for sl in slates
        grids = Array{Float64,3}[]
        for b in sl.books
            PF.price_fixture!(w, model.latents, row[b.m_id])
            g = w.S[:, :, :]
            p = vec(Statistics.mean(g, dims = 3)[:, :, 1])
            p ./= sum(p)
            p == b.p_grid || error("draw grid does not reproduce book p_grid: $(b.m_id)")
            push!(grids, g)
        end
        groups = [model.witness[b.m_id] for b in sl.books]
        solved = SC.scenario_books(PF.ScenarioKelly(), sl, grids, groups, spec.exec; native = true)
        push!(out, solved.slate)
        push!(diag, (; date = sl.window, n_books = length(sl.books),
                     n_legs = size(solved.problem.R, 2), n_scenarios = size(solved.problem.R, 1),
                     draw_group = solved.problem.draw_group,
                     raw_total_joint = sum(solved.solution.a),
                     raw_total_per_match = sum(sum(b.a_kelly) for b in sl.books),
                     kkt = solved.solution.kkt, converged = solved.solution.converged))
    end
    return out, diag
end

# ===================================================================
# 6. One (model, snapshot) group: four stock policies plus scenario Kelly
# ===================================================================
function run_group(model, snap, book, ds, sys, close, season_of)
    tables = NamedTuple[]
    status = NamedTuple[]
    scenario_diag = NamedTuple[]
    canonical_books = nothing
    canonical_report = nothing
    for pol in POLICY_ORDER[1:4]
        key = (; model = model.label, snapshot = snap.label, policy = pol)
        spec, policy = sys[pol]
        try
            result, books, report = PF.run_portfolio_simulation(spec, policy, model.source, book, ds;
                initial_bankroll = 1.0, bootstrap = false, quiet = true)
            push!(tables, cell_tables(key, result, books, report, close, season_of))
            if pol == "canonical"
                canonical_books, canonical_report = books, report
            end
        catch err
            push!(status, (; key..., status = "ERROR", reason = sprint(showerror, err)[1:min(end, 300)]))
        end
    end
    key = (; model = model.label, snapshot = snap.label, policy = "scenario_kelly")
    if canonical_books === nothing
        push!(status, (; key..., status = "NOT_AVAILABLE", reason = "canonical books failed to build"))
    else
        try
            spec, policy = sys["canonical"]
            slates, diag = scenario_slates(model, spec, policy, canonical_books)
            result = PF.simulate_portfolio(policy, slates; initial_bankroll = 1.0, bootstrap = false,
                converged = canonical_report.converged, failed_gates = canonical_report.failed_gates)
            books = PF.MatchBook[b for sl in slates for b in sl.books]
            push!(tables, cell_tables(key, result, books, canonical_report, close, season_of))
            append!(scenario_diag, [(; model = model.label, snapshot = snap.label, d...) for d in diag])
        catch err
            push!(status, (; key..., status = "ERROR", reason = sprint(showerror, err)[1:min(end, 300)]))
        end
    end
    println("GROUP_DONE model=", model.label, " snapshot=", snap.label, " cells=", length(tables),
            " errors=", length(status))
    flush(stdout)
    return (; tables, status, scenario_diag)
end

# ===================================================================
# 7. Moving-block bootstrap (wave 2 scheme: 8-week noncircular blocks within season)
# ===================================================================
function block_bootstrap(dates, values, season_of; B = B_REPS, block = BLOCK_WEEKS, seed = BOOT_SEED)
    weeks = Dict{Tuple{String,Dates.Date},Vector{Int}}()
    for (i, d) in enumerate(dates)
        push!(get!(weeks, (season_of[d], Dates.firstdayofweek(d)), Int[]), i)
    end
    seasons = Dict{String,Vector{Vector{Int}}}()
    for key in sort!(collect(keys(weeks)))
        push!(get!(seasons, key[1], Vector{Int}[]), weeks[key])
    end
    rng = Random.Xoshiro(seed)
    stats = zeros(B)
    for b in 1:B
        total, n = 0.0, 0
        for season in sort!(collect(keys(seasons)))
            ws = seasons[season]
            W = length(ws)
            L = min(block, W)
            chosen = Int[]
            while length(chosen) < W
                s = rand(rng, 1:(W - L + 1))
                append!(chosen, s:(s + L - 1))
            end
            for w in chosen[1:W], i in ws[w]
                total += values[i]
                n += 1
            end
        end
        stats[b] = total / n
    end
    return (; point = Statistics.mean(values), boot_mean = Statistics.mean(stats),
            lo90 = Statistics.quantile(stats, 0.05), hi90 = Statistics.quantile(stats, 0.95),
            n_days = length(values), n_weeks = length(weeks))
end

function uncertainty(daily::DF.DataFrame, season_of)
    single, paired = NamedTuple[], NamedTuple[]
    for g in DF.groupby(daily, [:model, :snapshot, :policy]; sort = true)
        bs = block_bootstrap(g.date, g.log_growth, season_of)
        push!(single, (; model = first(g.model), snapshot = first(g.snapshot), policy = first(g.policy), bs...))
    end
    for g in DF.groupby(daily, [:snapshot, :policy]; sort = true)
        ref = filter(r -> r.model == "m12", g)
        DF.nrow(ref) == 0 && continue
        refmap = Dict(zip(ref.date, ref.log_growth))
        for m in setdiff(unique(g.model), ["m12"])
            x = filter(r -> r.model == m, g)
            xmap = Dict(zip(x.date, x.log_growth))
            days = sort!(collect(union(keys(refmap), keys(xmap))))
            unmatched = count(d -> !(haskey(refmap, d) && haskey(xmap, d)), days)
            diff = [get(xmap, d, 0.0) - get(refmap, d, 0.0) for d in days]
            bs = block_bootstrap(days, diff, season_of)
            push!(paired, (; model = m, reference = "m12", snapshot = first(g.snapshot),
                            policy = first(g.policy), bs..., unmatched_days = unmatched))
        end
    end
    return DF.DataFrame(single), DF.DataFrame(paired)
end

# ===================================================================
# 8. Gate, grid, uncertainty; outputs to files only
# ===================================================================
function run_grid(out, phase0_dir)
    cache = joinpath(pkgdir(BF), ".cache", "datastore_ScottishLower.jls")
    isfile(cache) || error("Phase 3 requires the saved datastore cache; no SQL fallback")
    println("DATASTORE_CACHE sha256=", bytes2hex(open(SHA.sha256, cache)))
    ds = Serialization.deserialize(cache)
    ids = Set(Int.(ds.matches.match_id[in.(ds.matches.season, Ref(SEASONS))]))
    models = Dict{String,Any}()
    for m in GOAL_MODELS
        models[m.label] = goal_model(m, ds, ids)
    end
    models["market_c0"] = market_model(ds, ids)
    universe = sort!(collect(Set(Int.(BF.Models.latent_match_ids(models["m12"].latents)))))
    ref_ids = Int.(CSV.read(joinpath(phase0_dir, "phase0_fixture_universe.csv"), DF.DataFrame).match_id)
    universe == sort(ref_ids) && length(universe) == 710 || error("held-out universe differs from Phase 0")
    for label in MODEL_ORDER
        m = models[label]
        println("MODEL ", label, " run=", m.run, " draws=", m.nd, " layout=", m.layout,
                " folds=", length(unique(values(m.fold))))
    end
    season_of = Dict{Dates.Date,String}()
    for r in eachrow(ds.matches)
        Int(r.match_id) in ids || continue
        d = Dates.Date(r.match_date)
        get!(season_of, d, String(r.season)) == String(r.season) || error("date $d spans two seasons")
    end
    ctx = (; ds, ids)
    books = snapshot_books(ctx, phase0_dir)
    closing = close_index(ds, ids)
    println("CLOSE_BOOK selections=", length(closing))
    sys = systems(out)

    # --- gate: m12 x T-25m x historical_exp07 must reproduce before anything else runs
    spec, policy = sys["historical_exp07"]
    result, gbooks, greport = PF.run_portfolio_simulation(spec, policy, models["m12"].source,
        books["T-25m_s90"], ds; initial_bankroll = 1.0, bootstrap = false, quiet = true)
    wealth = result.trajectory.bankroll[end]
    passed = abs(wealth - GATE_WEALTH) <= GATE_TOLERANCE && result.summary.n_bets == GATE_BETS
    CSV.write(joinpath(out, "phase3_gate.csv"), DF.DataFrame([(; model = "m12", snapshot = "T-25m_s90",
        policy = "historical_exp07", final_wealth = wealth, n_bets = result.summary.n_bets,
        n_slates = result.summary.n_slates, n_books = length(gbooks), expected_wealth = GATE_WEALTH,
        expected_bets = GATE_BETS, abs_gap = abs(wealth - GATE_WEALTH), bit_identical = wealth == GATE_WEALTH,
        status = passed ? "PASS" : "FAIL")]))
    println("GATE wealth=", repr(wealth), " bets=", result.summary.n_bets, " slates=",
            result.summary.n_slates, " status=", passed ? "PASS" : "FAIL")
    flush(stdout)
    passed || error("PHASE3_GATE_FAILED: m12 T-25 historical cell did not reproduce; stop")

    # --- grid: one task per (model, snapshot); each task runs its five policies in order
    jobs = [(models[label], s) for label in MODEL_ORDER for s in SNAPSHOTS]
    tasks = [Threads.@spawn run_group(m, s, books[s.label], ds, sys, closing, season_of) for (m, s) in jobs]
    groups = fetch.(tasks)

    summaries = NamedTuple[]
    markets, daily, status, scen = NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    bets = DF.DataFrame[]
    for g in groups
        append!(status, g.status)
        append!(scen, g.scenario_diag)
        for t in g.tables
            push!(summaries, t.summary)
            append!(markets, t.markets)
            append!(daily, t.daily)
            push!(bets, t.bets)
        end
    end
    summary = DF.DataFrame(summaries)
    isempty(status) || DF.append!(summary, DF.DataFrame(status); cols = :union, promote = true)
    gate = filter(r -> r.model == "m12" && r.snapshot == "T-25m_s90" && r.policy == "historical_exp07", summary)
    DF.nrow(gate) == 1 && gate.final_wealth[1] == wealth || error("grid gate cell differs from the standalone gate run")
    order(df) = sort(df, [DF.order(:model, by = m -> findfirst(==(m), MODEL_ORDER)),
                          DF.order(:snapshot, by = s -> findfirst(x -> x.label == s, SNAPSHOTS)),
                          DF.order(:policy, by = p -> findfirst(==(p), POLICY_ORDER))])
    summary = order(summary)
    daily_df = DF.DataFrame(daily)
    single, paired = uncertainty(daily_df, season_of)
    summary = DF.leftjoin(summary, DF.rename(DF.select(single, :model, :snapshot, :policy, :lo90, :hi90, :n_weeks),
                          :lo90 => :growth_lo90, :hi90 => :growth_hi90); on = [:model, :snapshot, :policy])
    summary = DF.leftjoin(summary, DF.rename(DF.select(paired, :model, :snapshot, :policy, :point, :lo90, :hi90, :unmatched_days),
                          :point => :diff_vs_m12, :lo90 => :diff_vs_m12_lo90, :hi90 => :diff_vs_m12_hi90);
                          on = [:model, :snapshot, :policy])
    summary = order(summary)
    CSV.write(joinpath(out, "phase3_cells.csv"), summary)
    CSV.write(joinpath(out, "phase3_markets.csv"), order(DF.DataFrame(markets)))
    CSV.write(joinpath(out, "phase3_daily.csv"), order(daily_df))
    CSV.write(joinpath(out, "phase3_bootstrap_growth.csv"), order(single))
    CSV.write(joinpath(out, "phase3_paired_vs_m12.csv"), order(paired))
    CSV.write(joinpath(out, "phase3_scenario_days.csv"), DF.DataFrame(scen))
    CSV.write(joinpath(out, "phase3_bets.csv.gz"), reduce(vcat, bets; cols = :union); compress = true)
    println("GRID_DONE cells=", DF.nrow(summary), " computed=", count(==("COMPUTED"), summary.status),
            " other=", count(!=("COMPUTED"), summary.status))
    return nothing
end

# Same safety frame as Phase 0: static read check, then a before/after audit that also runs on
# failure exits; any audit change stops the phase.
function run(out, phase0_dir)
    mkpath(out)
    Base.run(`python3 $(joinpath(@__DIR__, "check_read_paths.py")) l00_phase0_inventory.jl l01_scenario_kelly.jl l02_phase3_grid.jl r02_phase3_grid.jl`)
    conn = BF.Training.Inference._db_connect(BF.Training.PostgresStorage(GOAL_MODELS[2].ns))
    try
        value = String(P0.read_query(conn, "SHOW default_transaction_read_only")[1, 1])
        println("SESSION_DEFAULT_TRANSACTION_READ_ONLY=", value)
        before = P0.audit_snapshot(conn)
        CSV.write(joinpath(out, "phase3_db_before.csv"), before)
        println("DB_AUDIT_BEFORE tables=", length(unique(before.table_name)))
        flush(stdout)
        try
            run_grid(out, phase0_dir)
        finally
            after = P0.audit_snapshot(conn)
            CSV.write(joinpath(out, "phase3_db_after.csv"), after)
            tables = sort!(unique(vcat(String.(before.table_name), String.(after.table_name))))
            checks = DF.DataFrame([(; table_name = t,
                unchanged = isequal(filter(r -> r.table_name == t, before),
                                    filter(r -> r.table_name == t, after))) for t in tables])
            CSV.write(joinpath(out, "phase3_db_audit.csv"), checks)
            println("DB_AUDIT_AFTER unchanged=", all(checks.unchanged), " tables=", length(tables))
            flush(stdout)
            all(checks.unchanged) || error("before/after database audit changed; stop")
        end
    finally
        close(conn)
    end
    println("PHASE3_PASS")
    return nothing
end

end # module
