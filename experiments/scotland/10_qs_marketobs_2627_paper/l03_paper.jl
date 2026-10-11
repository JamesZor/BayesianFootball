module QSPaper2627Paper

# Phase 2 (TODO 045): the 9 paper-trade cells (ruling 6) on the 26/27 days with a T-25m book, the
# per-bet ledger with order-book sizes and CLV, the fill check (ruling 7), the drawdown watch and
# slate-clustered intervals (ruling 8). Read-only: the serialised Phase 1 extensions and the pinned
# Phase 0 inputs; the replay is Q.replay_cells, the code that passed the engine gate (ruling 9).
# Comments only below; the static checker scans every non-comment line for write verbs.

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import Random
import Serialization
import Statistics

const Q = Main.QSPaper2627
const S = Main.QSPaper2627Sampling
const R = Main.QSPaper2627Predictions
const P3 = Main.QSBacktestPhase3
const W5 = Main.QSWave5
const PF = BF.Portfolio

# Phase 1 extension files (logs/extend_qs.log, logs/extend_m12.log).
const EXT_SHA = Dict("qs_marketobs" => "5f6a7e280ab80c254562473dd5f79f85d0d1c18c63310da229528699ab1c04ab",
                     "m12" => "f7328646d8eac9ec9a831aa67a61e816d169a8b56a2fc6f9a69efee218c97a1d")
const PRICES = (:back, :mid, :ltp)
const NOTIONALS = (1000.0, 200.0)
const DD_WATCH = -20.39                      # ruling 6: wave-5 drawdown watch, percent
const BOOT_B, BOOT_SEED = R.BOOT_B, R.BOOT_SEED
const PHASE0_DIR = joinpath(@__DIR__, "results", "phase0")
const W5_SUMMARY = joinpath(@__DIR__, "..", "09_qs_marketobs_wave5_config", "results", "phase2", "phase2_summary.csv")

# ===================================================================
# 1. Inputs: the two 26/27 models and the three price frames
# ===================================================================
# The model NamedTuple that W5.day_problems / Q.replay_cells expect (as P3.goal_model returns),
# built from the in-memory extension. The witness groups draws by fold, as goal_model does.
function model_2627(label)
    path = joinpath(Q.OUT_DIR, "ext_$(label)_2627.jls")
    Q.sha_file(path) == EXT_SHA[label] || error("$label: extension file differs from Phase 1")
    ext = Serialization.deserialize(path)
    ids = Int.(BF.Models.latent_match_ids(ext.latents))
    length(ids) == Q.N_FIXTURES || error("$label: extension holds $(length(ids)) fixtures")
    shapes = unique((size(f.chain, 3), size(f.chain, 1)) for f in ext.folds)
    length(shapes) == 1 || error("$label: folds differ in draw layout")
    layout = string(only(shapes)[1], "chains_x_", only(shapes)[2])
    fold_of, _ = R.fold_maps(ext)
    witness = Dict(id => string(label, "|", ext.run, "+mem2627|fold=", fold_of[id], "|", layout) for id in ids)
    used = filter(r -> r.attempt == ext.attempt_used[r.fold], ext.diagnostics)
    converged = all(used.max_rhat .<= S.RHAT_RERUN)
    nd = BF.Models.n_draws(ext.latents)
    println("MODEL_2627 ", label, " run=", ext.run, " fixtures=", length(ids), " draws=", nd, " layout=", layout,
            " converged=", converged)
    return (; label, run = ext.run, source = ext.latents, latents = ext.latents, witness, fold = fold_of, nd,
            layout, converged)
end

read_ref(path) = CSV.read(path, DF.DataFrame; stringtype = String)

# Entry book and the back / mid / LTP odds frames, checked equal to the committed Phase 0 frames.
function price_frames(inp)
    entry, _ = Q.entry_book(inp.ob.snaps)
    DF.nrow(entry) == DF.nrow(read_ref(joinpath(PHASE0_DIR, "phase0_entry_book.csv"))) ||
        error("entry book differs from Phase 0")
    odds = Dict{Symbol,DF.DataFrame}()
    for p in PRICES
        f = Q.odds_frame(entry, p)
        ref = read_ref(joinpath(PHASE0_DIR, "phase0_odds_$(p).csv"))
        ref.selection = Symbol.(ref.selection)
        isequal(ref, f) || error("odds frame $p differs from Phase 0")
        println("ODDS_PARITY_PASS price=", p, " rows=", DF.nrow(f), " fixtures=", length(unique(f.match_id)))
        odds[p] = f
    end
    return entry, odds
end

# ===================================================================
# 2. One cell: P3.cell_tables plus the order-book columns and the opening-point MDD
# ===================================================================
function entry_index(entry)
    return Dict((Int(r.match_id), String(r.market_name), Float64(r.market_line), Symbol(r.selection)) => r
                for r in eachrow(entry))
end

function cell_ledger(bets, books, eidx, fx, open_bank)
    km = Dict((b.m_id, s.family) => (String(s.group), Float64(s.line), s.odds_used) for b in books for s in b.sels)
    team = Dict(r.match_id => (r.home_team, r.away_team, r.transition) for r in eachrow(fx))
    rows = NamedTuple[]
    for r in eachrow(bets)
        g, l, _ = km[(r.match_id, r.family)]
        e = eidx[(r.match_id, g, l, Symbol(r.selection))]
        h, a, tr = team[r.match_id]
        push!(rows, (; market_name = g, market_line = l, home_team = h, away_team = a, transition = tr,
                     bank_open = open_bank[r.date], back = Float64(e.back), back_size = Float64(e.back_size),
                     back_depth3 = Float64(e.back_depth3), lay = Float64(e.lay), mid = Float64(e.mid),
                     ltp = Float64(e.ltp), entry_age_min = Float64(e.age), won = r.pnl_frac > 0))
    end
    extra = isempty(rows) ? DF.DataFrame() : DF.DataFrame(rows)
    return DF.hcat(bets, extra)
end

function score_cell(price, c, eidx, close, fx, season_of)
    key = (; price = String(price), cell = c.cell, role = c.role, model = c.model, basket = c.role == "benchmark" ?
           "canonical" : "B1_canonical", lambda = c.lambda)
    t = P3.cell_tables(key, c.result, c.books, c.report, close, season_of)
    tr = c.result.trajectory
    open_bank = Dict(d => tr.bankroll[k] for (k, d) in enumerate(tr.dates))
    bets = cell_ledger(t.bets, c.books, eidx, fx, open_bank)
    cr = W5.curve_row((;), c.result, c.diag)
    mdd_open = W5.drawdown_pct(tr.bankroll)
    summary = (; t.summary..., first_day = isempty(tr.dates) ? missing : first(tr.dates),
               mdd_open_pct = mdd_open, dd_watch_pct = DD_WATCH, dd_watch_breached = mdd_open < DD_WATCH,
               n_fixtures_bet = length(unique(bets.match_id)),
               n_price_dearbed = count(bets.odds_used .< bets.odds_quoted),
               mean_exposure = cr.mean_exposure, max_exposure = cr.max_exposure, cap_share = cr.cap_share,
               risk_active_share = cr.risk_active_share, budget_share = cr.budget_share)
    println("CELL price=", price, " cell=", c.cell, " wealth=", repr(summary.final_wealth), " g=",
            round(summary.mean_daily_log_growth; sigdigits = 4), " mdd_open=", round(mdd_open; digits = 2),
            " bets=", summary.n_bets, " days=", summary.n_slates, " books=", summary.n_books,
            " clv_pp=", round(summary.clv_mean_pp; digits = 2))
    diag = c.diag === nothing ? nothing : DF.hcat(DF.DataFrame([key for _ in 1:DF.nrow(c.diag)]), c.diag)
    return (; summary, markets = t.markets, daily = t.daily, bets, diag, books = length(c.books))
end

function run_cells(qs, m12, odds, entry, inp)
    eidx = entry_index(entry)
    close = Q.close_index(inp.closebook)
    season_of = Dict(Dates.Date(d) => Q.SEASON for d in inp.fx.match_date)
    acc = (; summary = NamedTuple[], markets = NamedTuple[], daily = NamedTuple[], bets = DF.DataFrame[],
           diag = DF.DataFrame[], books = NamedTuple[])
    for p in PRICES
        rep = Q.replay_cells(qs, m12, odds[p], inp.ds)
        push!(acc.books, (; price = String(p), qs_books = length(rep.qbooks), m12_books = length(rep.mbooks),
                          days = length(rep.days), first_day = first(rep.days).slate.window,
                          fixtures_0801 = count(b -> b.date == Q.FIRST_DAY, rep.qbooks)))
        println("PRICE_BOOKS price=", p, " qs_books=", length(rep.qbooks), " m12_books=", length(rep.mbooks),
                " days=", length(rep.days), " first_day=", first(rep.days).slate.window)
        for c in rep.cells
            s = score_cell(p, c, eidx, close, inp.fx, season_of)
            push!(acc.summary, s.summary)
            append!(acc.markets, s.markets)
            append!(acc.daily, s.daily)
            push!(acc.bets, s.bets)
            s.diag === nothing || push!(acc.diag, s.diag)
        end
    end
    return (; summary = DF.DataFrame(acc.summary), markets = DF.DataFrame(acc.markets), daily = DF.DataFrame(acc.daily),
            bets = reduce(vcat, acc.bets; cols = :union), diag = reduce(vcat, acc.diag), books = DF.DataFrame(acc.books))
end

# ===================================================================
# 3. Slate-clustered intervals (ruling 8): match days resampled with replacement
# ===================================================================
function cluster_boot(dates, stat; B = BOOT_B, seed = BOOT_SEED)
    days = sort!(unique(dates))
    rows_of = Dict(d => findall(==(d), dates) for d in days)
    rng = Random.MersenneTwister(seed)
    reps = Float64[]
    for _ in 1:B
        idx = reduce(vcat, [rows_of[days[rand(rng, 1:length(days))]] for _ in days])
        v = stat(idx)
        isfinite(v) && push!(reps, v)
    end
    point = stat(collect(eachindex(dates)))
    isempty(reps) && return (; point, lo95 = NaN, hi95 = NaN, n_days = length(days))
    return (; point, lo95 = Statistics.quantile(reps, 0.025), hi95 = Statistics.quantile(reps, 0.975),
            n_days = length(days))
end

ratio(num, den) = idx -> (d = sum(view(den, idx)); d > 0 ? sum(view(num, idx)) / d : NaN)

function uncertainty(cells)
    rows = NamedTuple[]
    for s in eachrow(cells.summary)
        dl = filter(r -> r.price == s.price && r.cell == s.cell, cells.daily)
        bt = filter(r -> r.price == s.price && r.cell == s.cell, cells.bets)
        ct = filter(r -> isfinite(r.clv_pp), bt)
        g = cluster_boot(dl.date, idx -> Statistics.mean(view(dl.log_growth, idx)))
        w = cluster_boot(dl.date, idx -> exp(sum(view(dl.log_growth, idx))))
        roi = cluster_boot(bt.date, ratio(100 .* bt.pnl_unit, bt.stake_unit))
        clv = DF.nrow(ct) == 0 ? (; point = NaN, lo95 = NaN, hi95 = NaN) :
              cluster_boot(ct.date, idx -> Statistics.mean(view(ct.clv_pp, idx)))
        push!(rows, (; price = s.price, cell = s.cell, n_days = DF.nrow(dl), n_bets = DF.nrow(bt),
                     g = g.point, g_lo95 = g.lo95, g_hi95 = g.hi95, wealth = w.point, wealth_lo95 = w.lo95,
                     wealth_hi95 = w.hi95, roi_money_pct = roi.point, roi_lo95 = roi.lo95, roi_hi95 = roi.hi95,
                     clv_mean_pp = clv.point, clv_lo95 = clv.lo95, clv_hi95 = clv.hi95))
    end
    return DF.DataFrame(rows)
end

# Paired daily log growth, frozen minus each other cell at the same price; days missing from a
# cell's path (no book) count as zero growth. Reported to show the width, not to rank.
function paired(cells)
    rows = NamedTuple[]
    for p in unique(cells.daily.price)
        dp = filter(r -> r.price == p, cells.daily)
        base = Dict(r.date => r.log_growth for r in eachrow(filter(r -> r.cell == "frozen_B1_l12", dp)))
        for other in ("fallback_B1_l23", "m12_canonical_l23")
            o = Dict(r.date => r.log_growth for r in eachrow(filter(r -> r.cell == other, dp)))
            days = sort!(collect(union(keys(base), keys(o))))
            d = [get(base, x, 0.0) - get(o, x, 0.0) for x in days]
            ci = cluster_boot(days, idx -> Statistics.mean(view(d, idx)))
            push!(rows, (; price = p, cell = "frozen_B1_l12", reference = other, n_days = length(days),
                         d_g = ci.point, d_g_lo95 = ci.lo95, d_g_hi95 = ci.hi95,
                         share_days_frozen_ahead = Statistics.mean(d .> 0)))
        end
    end
    return DF.DataFrame(rows)
end

# ===================================================================
# 4. Bet sets across prices (D12): each price cell decides at its own price
# ===================================================================
function bet_sets(cells)
    rows = NamedTuple[]
    for cell in unique(cells.bets.cell)
        k = Dict(p => Set(zip(r.match_id, r.family)) for p in ("back", "mid", "ltp")
                 for r in (filter(x -> x.cell == cell && x.price == p, cells.bets),))
        fxs = Dict(p => Set(first.(collect(k[p]))) for p in keys(k))
        push!(rows, (; cell, n_back = length(k["back"]), n_mid = length(k["mid"]), n_ltp = length(k["ltp"]),
                     back_and_mid = length(intersect(k["back"], k["mid"])),
                     back_and_ltp = length(intersect(k["back"], k["ltp"])),
                     all_three = length(intersect(k["back"], k["mid"], k["ltp"])),
                     back_not_mid = length(setdiff(k["back"], k["mid"])), mid_not_back = length(setdiff(k["mid"], k["back"])),
                     back_not_ltp = length(setdiff(k["back"], k["ltp"])), ltp_not_back = length(setdiff(k["ltp"], k["back"])),
                     fixtures_back = length(fxs["back"]), fixtures_mid = length(fxs["mid"]), fixtures_ltp = length(fxs["ltp"])))
    end
    return DF.DataFrame(rows)
end

# ===================================================================
# 5. Fill check (ruling 7) on the frozen back-price bets
# ===================================================================
# Sequential re-run of the bankroll path at a notional bankroll, with each bet's money stake
# (stake fraction x opening bankroll x notional) capped at the order-book size. The policy's
# stake fractions are kept from the uncapped run (the SlateDrawdown state is not re-solved on the
# capped path): an approximation, stated in the report. cap = :none must reproduce the cell.
function capped_path(bets, notional, cap::Symbol)
    days = unique(bets.date)
    bank, path = 1.0, [1.0]
    staked, filled, n_short = 0.0, 0.0, 0
    for d in days
        pl = 0.0
        for i in findall(==(d), bets.date)
            want = bets.stake_frac[i] * bank * notional
            avail = cap === :none ? Inf : cap === :best ? bets.back_size[i] : bets.back_depth3[i]
            share = want <= avail ? 1.0 : avail / want
            share < 1.0 && (n_short += 1)
            staked += want
            filled += want * share
            pl += share == 1.0 ? bets.pnl_frac[i] : bets.pnl_frac[i] * share
        end
        bank *= (1.0 + pl)
        push!(path, bank)
    end
    return (; wealth = bank, mdd_pct = W5.drawdown_pct(path), n_short, staked_gbp = staked, filled_gbp = filled)
end

function fill_check(cells, ref_wealth)
    fb = filter(r -> r.price == "back" && r.cell == "frozen_B1_l12", cells.bets)
    issorted(fb.date) || error("frozen back ledger is not in date order")
    base = capped_path(fb, 1.0, :none)
    base.wealth === ref_wealth || error("fill re-run does not reproduce the frozen back cell")
    println("FILL_REPRODUCES wealth=", repr(base.wealth))
    rows, per_bet = NamedTuple[], DF.DataFrame(match_id = fb.match_id, date = fb.date, family = fb.family)
    for n in NOTIONALS
        gbp = fb.stake_frac .* fb.bank_open .* n
        tag = string(Int(n))
        per_bet[!, "stake_gbp_$tag"] = gbp
        per_bet[!, "exceeds_best_$tag"] = gbp .> fb.back_size
        per_bet[!, "exceeds_depth3_$tag"] = gbp .> fb.back_depth3
        for cap in (:none, :best, :depth3)
            c = capped_path(fb, n, cap)
            push!(rows, (; notional_gbp = n, cap = String(cap), n_bets = DF.nrow(fb),
                         share_exceeds_best = Statistics.mean(gbp .> fb.back_size),
                         share_exceeds_depth3 = Statistics.mean(gbp .> fb.back_depth3),
                         median_stake_gbp = Statistics.median(gbp), max_stake_gbp = maximum(gbp),
                         n_short_on_path = c.n_short, staked_gbp = c.staked_gbp, filled_gbp = c.filled_gbp,
                         fill_rate = c.filled_gbp / c.staked_gbp, wealth = c.wealth, mdd_pct = c.mdd_pct))
            println("FILL notional=", Int(n), " cap=", cap, " wealth=", round(c.wealth; digits = 4), " short=", c.n_short,
                    " fill_rate=", round(rows[end].fill_rate; digits = 3), " exceeds_best=",
                    round(rows[end].share_exceeds_best; digits = 3), " exceeds_depth3=",
                    round(rows[end].share_exceeds_depth3; digits = 3))
        end
    end
    return DF.DataFrame(rows), per_bet
end

# ===================================================================
# 6. Wave-5 reference rows and the paper ledger
# ===================================================================
function wave5_reference()
    w = read_ref(W5_SUMMARY)
    keep = filter(r -> startswith(r.snapshot, "T-25m") &&
                       r.cell in ("frozen_B1_l12", "fallback_B1_l23", "m12_canonical_l23"), w)
    return DF.select(keep, :snapshot, :half, :cell, :final_wealth, :mean_daily_log_growth, :mdd_pct, :n_bets,
                     :n_slates, :roi_money_pct, :clv_mean_pp, :clv_beat_pct, :mean_exposure,
                     [:n_bets, :n_slates] => DF.ByRow((b, s) -> b / s) => :bets_per_day)
end

function paper_ledger(cells, per_bet)
    fb = filter(r -> r.price == "back" && r.cell == "frozen_B1_l12", cells.bets)
    j = DF.leftjoin(fb, per_bet, on = [:match_id, :date, :family])
    DF.nrow(j) == DF.nrow(fb) || error("paper ledger join changed the row count")
    DF.sort!(j, [:date, :match_id, :market, :selection])
    return DF.select(j, :date, :match_id, :home_team, :away_team, :transition, :market, :selection,
                     :odds_quoted => :back_price, :odds_used => :price_settled, :back_size, :back_depth3, :lay, :mid,
                     :ltp, :entry_age_min, :p_model, :p_market, :bank_open, :stake_frac, :stake_gbp_1000,
                     :exceeds_best_1000, :exceeds_depth3_1000, :won, :pnl_frac,
                     [:pnl_frac, :bank_open] => DF.ByRow((p, b) -> p * b * 1000) => :pnl_gbp_1000,
                     :p_close, :clv_pp, :clv_pct)
end

end # module
