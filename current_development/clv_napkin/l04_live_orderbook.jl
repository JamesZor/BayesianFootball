# l04_live_orderbook.jl — loader for brief 04: grw_spfl_joint on 26/27 League One/Two,
# priced from the betfair_live order book (see BRIEF_04.md, r04_extend_grw_spfl_joint.jl,
# r04_live_orderbook.jl).
#
# Two halves:
#   §1  an IN-MEMORY walk-forward extension of a persisted fit. It mirrors
#       `Training.extend_fit` (src/training/inference/extension.jl) step for step —
#       same boundaries, same `_ExtensionSampler`, `run_folds`, `audit_convergence`,
#       `_extension_extract_latents` — but stops before the transaction: nothing is written to
#       `mcmc_experiments`. (`extend_fit` itself also refuses `ScopedWalkForwardCV`.)
#   §2  the live 1-minute order book (`betfair_live.order_book_1m`): best back / lay and sizes
#       per snapshot, entry books at an instant, mid-price and LTP closes, Betfair ticks.
#
# Definitions (brief 04, on top of BRIEF.md):
#   entry     best available BACK price in the last snapshot at or before T−n
#             (staleness ≤ C4_STALE minutes); size = the size offered at that price
#   mid       (best back + best lay) / 2 in odds (MatchDay.MidPrice); de-vigged across the 3 runners
#   edge_pp   100·(p_model − p_mid_fair)                 (the mid book at the entry minute)
#   close     TWA over (−20, 0] of the mid odds, de-vigged (same TWA code as the harness close)
#   close_ltp TWA over (−20, 0] of last_price_traded, de-vigged — the §2 definition, cross-check
#   clv_pp    100·(p_close − 1/odds_entry);  bets: 1X2 back, edge ≥ 2 pp, 1 unit, 2% commission
#
# Only `SELECT`s touch a database.

include(joinpath(@__DIR__, "l01_clv_napkin.jl"))


# ==============================================================================
# 1. IN-MEMORY WALK-FORWARD EXTENSION
# ==============================================================================

const C4_RUN = UUID("82e01a3b-d686-4e29-a269-3702b96fc5ba")     # grw_spfl_joint (W1)
const C4_EXPERIMENT = "scottish_dynamics_scope_matrix_w1"
const C4_SEASON = "26/27"
const C4_OUT = "/root/BF_runs/clv_napkin_dev/out"

"The run's DataScope with `season` appended to its target seasons (everything else identical)."
function c4_scope_extended(s::Data.DataScope, season::AbstractString)
    return Data.DataScope(name = s.name, train_tournaments = s.train_tournaments, cups = s.cups,
        target_tournaments = s.target_tournaments, monitor_tournaments = s.monitor_tournaments,
        clock_tournaments = s.clock_tournaments, target_seasons = vcat(s.target_seasons, [season]),
        history_seasons = s.history_seasons, dynamics_col = s.dynamics_col)
end

"""
    c4_extension_plan(fit, raw_ds; season) -> NamedTuple

Boundaries of the extended scope, the positions the fit does not yet hold (`delta`) and their
held-out fixtures. Refuses if positions 1..n of the extended splitter do not hold out exactly
the fixtures the persisted fit holds out (the extension must not move a historical fold).
"""
function c4_extension_plan(fit, raw_ds; season::AbstractString = C4_SEASON)
    scope = c4_scope_extended(fit.config.splitter.scope, season)
    splitter = Data.ScopedWalkForwardCV(scope)
    ds = Data.apply_scope(raw_ds, scope)
    boundaries = Data.create_id_boundaries(ds, splitter)
    done = Set(f.fold for f in fit.folds)
    old = Set{Int}()
    for i in sort!(collect(done))
        union!(old, Int.(Data.get_next_matches(ds, boundaries[i], splitter).match_id))
    end
    old == Set(Int.(fit.latents.match_ids)) || error(
        "c4_extension_plan: the extended splitter moves a historical fold's held-out set")
    delta = [i for i in eachindex(boundaries) if i ∉ done]
    oos = Any[Data.get_next_matches(ds, boundaries[i], splitter) for i in delta]
    return (; scope, splitter, ds, boundaries, delta, oos)
end

"""
    c4_filtration(plan) -> DataFrame

One row per new fold. Hard checks: no train/held-out overlap; every training fixture is on a
calendar day STRICTLY before the first held-out day (the Date-vs-DateTime lesson from the
LastHistorical note — compare days, not stamps); training tournaments inside the scope.
Also reports pxG (BBC live-text) coverage of the fold's 26/27 training rows.
"""
function c4_filtration(plan)
    meta = Dict(Int(r.match_id) => r for r in eachrow(plan.ds.matches))
    bbc = Set(Int.(plan.ds.bbc_events.match_id))
    allowed = Set(plan.scope.train_tournaments)
    rows = NamedTuple[]
    for (k, i) in enumerate(plan.delta)
        bd, m = plan.boundaries[i]
        train = Int.(vcat(bd.history_match_ids, bd.target_match_ids))
        ho = plan.oos[k]
        ho_ids = Int.(ho.match_id)
        isempty(intersect(Set(train), Set(ho_ids))) || error("fold $i: train/held-out overlap")
        all(id -> Int(meta[id].tournament_id) in allowed, train) || error("fold $i: out-of-scope training row")
        last_train = maximum(Date(meta[id].match_date) for id in train)
        first_ho = minimum(Date.(ho.match_date))
        last_train < first_ho || error("fold $i: training day $last_train not before held-out day $first_ho")
        cur = [id for id in train if String(meta[id].season) == m.target_season]
        l12 = ho[in.(Int.(ho.tournament_id), Ref((56, 57))), :]
        push!(rows, (; fold = i, time_step = m.time_step, n_train = length(train),
                     train_last = last_train, heldout_first = first_ho,
                     heldout_last = maximum(Date.(ho.match_date)),
                     n_heldout = length(ho_ids), n_heldout_l1l2 = nrow(l12),
                     heldout_days = join(string.(sort!(unique(Date.(l12.match_date)))), " "),
                     n_train_2627 = length(cur), n_train_2627_pxg = count(in(bbc), cur)))
    end
    return DataFrame(rows)
end

"""
    c4_extend_inmemory(fit, plan; sampler, execution) -> NamedTuple

`Training.extend_fit` minus the database: sample the missing positions with the run's own
model and sampler, audit them against the run's thresholds, extract held-out CountLatents.
"""
function c4_extend_inmemory(fit, plan; sampler = fit.config.sampler,
                            execution = fit.config.execution)
    I = Training.Inference
    model = fit.config.model
    started = time()
    selected = [plan.boundaries[i] for i in plan.delta]
    feature_sets = Features.create_features(selected, plan.ds, model, plan.splitter)
    resolved = I.resolve_execution(execution, sampler)
    results = I.run_folds(model, I._ExtensionSampler(sampler, plan.delta), resolved, feature_sets;
                          on_progress = I._inf_progress(started))
    ok = findall(!isnothing, results)
    length(ok) == length(plan.delta) ||
        @warn "c4_extend_inmemory: $(length(plan.delta) - length(ok)) fold(s) failed to sample"
    new_folds = I._inf_narrow(Training.FoldFit[Training.FoldFit(plan.delta[i], results[i], feature_sets[i][2])
                                               for i in ok])
    diagnostics = I.audit_convergence(new_folds; thresholds = fit.diagnostics.thresholds,
                                      max_depth = I.sampler_max_depth(sampler))
    per_fold, latents = I._extension_extract_latents(model, new_folds, feature_sets[ok], plan.oos[ok])
    return (; run_id = C4_RUN, scope = plan.scope, sampler, delta = plan.delta, folds = new_folds,
            diagnostics, per_fold_latents = per_fold, latents, elapsed = time() - started,
            oos_ids = Dict(plan.delta[i] => Int.(plan.oos[i].match_id) for i in ok))
end

"Per-fold convergence table (R̂, ESS, divergence rate) with the brief's rerun trigger."
function c4_convergence_table(ext)
    rows = NamedTuple[]
    for f in ext.diagnostics.folds
        rows = push!(rows, (; fold = f.fold, max_rhat = f.max_rhat, worst_rhat = String(f.worst_rhat_param),
                             min_ess_bulk = f.min_ess_bulk, min_ess_tail = f.min_ess_tail,
                             n_divergent = f.n_divergent, n_transitions = f.n_transitions,
                             divergence_pct = 100 * f.divergence_rate, treedepth_pct = 100 * f.treedepth_rate,
                             min_bfmi = f.min_bfmi,
                             rerun = f.max_rhat > 1.05 || f.divergence_rate > 0.001))
    end
    return DataFrame(rows)
end


# ==============================================================================
# 2. THE LIVE ORDER BOOK
# ==============================================================================

const C4_TOURNAMENTS = (56, 57)
const C4_STALE = 15.0          # minutes; the collector's cadence is ~3 min
const C4_SELS = Dict("home" => :home, "draw" => :draw, "away" => :away)

"L1/L2 MATCH_ODDS markets for `season` with the verified crosswalk match_id and sofascore kick-off."
function c4_markets(ds; season::AbstractString = C4_SEASON)
    md = BayesianFootball.MatchDay._query("""
        SELECT md.market_id, md.event_id, md.event_name, md.competition, md.open_date,
               mm.match_id, mm.is_verified, e.start_timestamp
        FROM betfair_live.market_metadata md
        JOIN betfair.match_meta mm ON mm.betfair_event_id = md.event_id
        JOIN sofascore.events e ON e.match_id = mm.match_id
        WHERE md.market_type = 'MATCH_ODDS' AND e.tournament_id = ANY(\$1)""",
        (collect(C4_TOURNAMENTS),))
    md.match_id = Int.(md.match_id)
    md.kickoff = unix2datetime.(Int.(md.start_timestamp))
    md.open_utc = DateTime.(md.open_date)          # the connection reports +00:00
    meta = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
    filter!(r -> haskey(meta, r.match_id) && String(meta[r.match_id].season) == season, md)
    md.market_id = String.(md.market_id)
    return select(md, :market_id, :event_id, :event_name, :competition, :match_id, :is_verified,
                  :kickoff, :open_utc)
end

_c4_first(v, scale) = (ismissing(v) || isempty(v) || ismissing(v[1])) ? NaN : v[1] / scale

"""
    c4_snapshots(markets) -> DataFrame

Every pre-off snapshot (ts ≤ kick-off) of the markets, one row per (market, runner, ts): best
back / lay price and size, runner and market matched volume, LTP. Prices and volumes arrive
×10 000 (`MatchDay._unscale`). `minutes` is relative to the sofascore kick-off.
"""
function c4_snapshots(markets::AbstractDataFrame)
    raw = BayesianFootball.MatchDay._query("""
        SELECT market_id, symbol, ts, bid_prices, bid_volumes, ask_prices, ask_volumes,
               total_matched, market_matched, last_price_traded
        FROM betfair_live.order_book_1m WHERE market_id = ANY(\$1)""", (markets.market_id,))
    ko = Dict(r.market_id => r.kickoff for r in eachrow(markets))
    mid = Dict(r.market_id => r.match_id for r in eachrow(markets))
    n = nrow(raw)
    out = DataFrame(market_id = String.(raw.market_id), match_id = [mid[String(x)] for x in raw.market_id],
                    selection = [C4_SELS[String(s)] for s in raw.symbol],
                    ts = DateTime.(raw.ts))
    out.minutes = [Dates.value(out.ts[i] - ko[out.market_id[i]]) / 60_000 for i in 1:n]
    out.back = [_c4_first(v, 1e4) for v in raw.bid_prices]
    out.back_size = [_c4_first(v, 1e4) for v in raw.bid_volumes]
    out.lay = [_c4_first(v, 1e4) for v in raw.ask_prices]
    out.lay_size = [_c4_first(v, 1e4) for v in raw.ask_volumes]
    out.back_depth3 = [ismissing(v) ? 0.0 : sum(x / 1e4 for x in v[1:min(3, length(v))]; init = 0.0) for v in raw.bid_volumes]
    out.runner_matched = [ismissing(x) ? NaN : x / 1e4 for x in raw.total_matched]
    out.market_matched = [ismissing(x) ? NaN : x / 1e4 for x in raw.market_matched]
    out.ltp = [ismissing(x) ? NaN : x / 1e4 for x in raw.last_price_traded]
    filter!(r -> r.minutes <= 0.0, out)
    out.mid = ifelse.(isfinite.(out.back) .& isfinite.(out.lay), (out.back .+ out.lay) ./ 2, NaN)
    sort!(out, [:match_id, :selection, :ts])
    return out
end

# Betfair price ladder: (upper bound, increment).
const C4_LADDER = ((2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2), (20.0, 0.5),
                   (30.0, 1.0), (50.0, 2.0), (100.0, 5.0), (1000.0, 10.0))

"Position of a price on the Betfair ladder (1.01 → 1)."
function c4_tick_index(p::Real)
    isfinite(p) || return NaN
    idx, lo = 0.0, 1.0
    for (hi, step) in C4_LADDER
        if p <= hi + 1e-9
            return idx + round((p - lo) / step)
        end
        idx += round((hi - lo) / step)
        lo = hi
    end
    return idx + round((p - lo) / 10.0)
end

c4_spread_ticks(back, lay) = (isfinite(back) && isfinite(lay)) ? c4_tick_index(lay) - c4_tick_index(back) : NaN

"""
    c4_entry_book(snaps, as_of; stale = C4_STALE) -> DataFrame

The last snapshot at or before `as_of` minutes per runner; complete (3 runners, all two-sided,
each ≤ `stale` minutes old) markets only; mid de-vigged across the runners.
`as_of = :earliest` takes, per market, the first snapshot at which all three runners are two-sided.
"""
function c4_entry_book(snaps::AbstractDataFrame, as_of; stale::Real = C4_STALE, latest::Real = -25.0)
    rows = NamedTuple[]
    for g in groupby(snaps, :match_id)
        if as_of === :earliest
            # the first snapshot time at which all three runners are two-sided AND the mid book
            # passes the overround gate
            ok = combine(groupby(g[isfinite.(g.mid) .& (g.minutes .<= latest), :], :ts), nrow => :n)
            for t in sort!(ok[ok.n .== 3, :ts])
                b = _c4_book_at(g, t, nothing, stale)
                b === nothing && continue
                append!(rows, b)
                break
            end
        else
            b = _c4_book_at(g, nothing, as_of, stale)
            b === nothing || append!(rows, b)
        end
    end
    return DataFrame(rows)
end

"One fixture's book at snapshot time `t` (or the last snapshot ≤ `as_of` minutes); `nothing` if refused."
function _c4_book_at(g, t, as_of, stale)
    pick = NamedTuple[]
    for s in groupby(g, :selection)
        vis = t === nothing ? s[s.minutes .<= as_of, :] : s[s.ts .<= t, :]
        isempty(vis) && return nothing
        r = vis[end, :]
        age = t === nothing ? as_of - r.minutes : 0.0
        (isfinite(r.mid) && age <= stale) || return nothing
        push!(pick, (; match_id = r.match_id, market_name = "1X2", market_line = 0.0,
                     selection = r.selection, ts = r.ts, entry_minutes = r.minutes, age,
                     back = r.back, back_size = r.back_size, back_depth3 = r.back_depth3,
                     lay = r.lay, lay_size = r.lay_size, mid = r.mid, ltp = r.ltp,
                     market_matched = r.market_matched,
                     spread_ticks = c4_spread_ticks(r.back, r.lay)))
    end
    length(pick) == 3 || return nothing
    implied = [1 / p.mid for p in pick]
    over = sum(implied)
    (0.90 <= over <= 1.10) || return nothing
    return [merge(p, (; p_mid_fair = implied[k] / over, mid_overround = over,
                      back_overround = sum(1 / q.back for q in pick))) for (k, p) in enumerate(pick)]
end

"""
    c4_close(snaps, col) -> DataFrame

The harness close on the order book: `Data.summarize_odds(TWAEstimator; window = (−20, 0))`
over the snapshots' `col` (`:mid` or `:ltp`), then `cn_devig` (complete, overround-gated).
"""
function c4_close(snaps::AbstractDataFrame, col::Symbol)
    long = DataFrame(match_id = snaps.match_id, market_name = fill("1X2", nrow(snaps)),
                     market_line = zeros(nrow(snaps)), selection = snaps.selection,
                     minutes_to_kickoff = snaps.minutes, traded_price = snaps[!, col])
    filter!(r -> isfinite(r.traded_price) && r.traded_price > 1.0, long)
    raw = Data.summarize_odds(long, Data.TWAEstimator(); window = (-20.0, 0.0),
                              overround_limits = (0.0, Inf))
    prices = DataFrame(match_id = Int.(raw.match_id), market_name = String.(raw.market_name),
                       market_line = Float64.(raw.market_line), selection = Symbol.(raw.selection),
                       odds_close = Float64.(coalesce.(raw.odds, NaN)))
    book, _ = cn_devig(prices; price_col = :odds_close)
    rename!(book, :p_fair => :p_close)
    return book
end

"""
    c4_frame(probs, entry, close, outcomes; price = :back) -> DataFrame

The l01 selection frame with the entry priced at `price` (`:back` executable, `:ltp`, `:mid`)
while `edge_pp` is ALWAYS measured against the de-vigged mid book, so the bet set is the same
whatever the price column.
"""
function c4_frame(probs, entry, close, outcomes; price::Symbol = :back)
    e = select(entry, CN_KEY..., price => :odds_entry, :p_mid_fair => :p_entry_fair, :entry_minutes,
               :back, :back_size, :back_depth3, :lay, :mid, :ltp, :spread_ticks, :market_matched, :age)
    filter!(r -> isfinite(r.odds_entry) && r.odds_entry > 1.0, e)
    f = cn_selection_frame(probs, e, close, outcomes)
    return leftjoin(f, select(e, Not([:odds_entry, :p_entry_fair, :entry_minutes])); on = CN_KEY)
end
