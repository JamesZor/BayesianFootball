# src/MatchDay/ledger/settle.jl
#
# Grading, PnL, and closing-line value.
#
# Settlement is the only transition that moves money in a direction the reservation did not
# authorise: `reserved` falls by the filled risk and `balance` rises by the gross return net of
# commission. Everything before it is a pair of equal and opposite movements.
#
# CLV is computed here rather than at pricing time because the closing price does not exist yet
# when the bet is placed -- which is the whole reason it is the metric with power. On ~100 legs a
# slate the ROI bootstrap interval spans +/-40pp and the ordering of two policies is a coin toss;
# leg-weighted CLV separates them. Judge the EXECUTION layer on CLV and the MODEL on log score,
# and never mix the two.

export settle_order, settle_order_legacy_v1, settle_slate!, clv_for_order,
       grade_selection, mark_to_market

"""
    grade_selection(group, line, selection, home_goals, away_goals) -> Symbol

`:win`, `:lose` or `:void` for the MODEL's position.

Graded on `selection`, never on `venue_selection`: a synthetic expresses the model's position by
trading the complement, so grading the runner the order touched would invert every synthetic.
That is the same distinction `Instrument.venue_key` exists for, appearing one last time at the
end of the lifecycle.
"""
function grade_selection(group::AbstractString, line::Real, selection::Symbol,
                         home_goals::Integer, away_goals::Integer)
    total = home_goals + away_goals
    if group == "1X2"
        selection === :home && return home_goals > away_goals ? :win : :lose
        selection === :draw && return home_goals == away_goals ? :win : :lose
        selection === :away && return away_goals > home_goals ? :win : :lose
    elseif group == "OverUnder"
        s = String(selection)
        startswith(s, "over")  && return total > line  ? :win : :lose
        startswith(s, "under") && return total < line  ? :win : :lose
    elseif group == "BTTS"
        both = home_goals > 0 && away_goals > 0
        selection === :btts_yes && return both  ? :win : :lose
        selection === :btts_no  && return !both ? :win : :lose
    end
    return :void
end

"""
    settle_order(order, fills, home_goals, away_goals, commission_rate) -> NamedTuple

The money, in RISK units, reconstructed from the exact child-fill cashflows.

For a winning back, gross return is `Σ(size * price)`. For a winning lay, the backer stakes
`Σsize` are won and the price-specific filled liability is returned. Commission is charged on
this order's win when this function is used alone; `settle_slate!` first computes zero-commission
cashflows and then nets commission across all positions in the same exchange market.

`risk_settled` is the FILLED risk, not the ordered risk. A partially matched leg settles the part
that filled; the remainder was released back to `balance` at execution and settling it here would
create money.
"""
function settle_order(o::PaperOrder, fills::AbstractVector{Fill},
                      home_goals::Integer, away_goals::Integer, commission_rate::Real)
    risk_settled = filled_risk(fills)
    outcome = grade_selection(o.market_group, o.market_line, o.selection, home_goals, away_goals)
    if risk_settled <= 1e-9 || outcome === :void
        # Nothing filled, or an ungradeable market: stake back, no PnL. Not a loss.
        return (; outcome = :void, risk_settled, gross_return = risk_settled,
                commission = 0.0, net_pnl = 0.0)
    end
    if outcome === :win
        if o.side === :back
            gross_return = sum(f -> f.size * f.price, fills)
            win = gross_return - risk_settled
        elseif o.side === :lay
            win = sum(f -> f.size, fills)
            gross_return = risk_settled + win
        else
            error("settle_order: side must be :back or :lay, got $(o.side)")
        end
        comm = Float64(commission_rate) * win
        return (; outcome = :win, risk_settled, gross_return,
                commission = comm, net_pnl = win - comm)
    end
    return (; outcome = :lose, risk_settled, gross_return = 0.0, commission = 0.0,
            net_pnl = -risk_settled)
end

"""
    settle_order_legacy_v1(order, fills, home_goals, away_goals, commission_rate)

Reconstruct the original touch-odds settlement convention for persisted `:ladder_sweep_v1`
rows. This is an audit compatibility path only; new settlement must use `settle_order`, whose
cashflows come from each child fill's actual price.
"""
function settle_order_legacy_v1(o::PaperOrder, fills::AbstractVector{Fill},
                                home_goals::Integer, away_goals::Integer,
                                commission_rate::Real)
    all(f -> f.model === :ladder_sweep_v1, fills) || error(
        "settle_order_legacy_v1 requires only :ladder_sweep_v1 fills")
    risk_settled = filled_risk(fills)
    outcome = grade_selection(o.market_group, o.market_line, o.selection,
                              home_goals, away_goals)
    if risk_settled <= 1e-9 || outcome === :void
        return (; outcome = :void, risk_settled, gross_return = risk_settled,
                commission = 0.0, net_pnl = 0.0)
    elseif outcome === :win
        win = risk_settled * (o.effective_odds - 1.0)
        commission = Float64(commission_rate) * win
        return (; outcome = :win, risk_settled, gross_return = risk_settled + win,
                commission, net_pnl = win - commission)
    end
    return (; outcome = :lose, risk_settled, gross_return = 0.0, commission = 0.0,
            net_pnl = -risk_settled)
end

"Apply exchange commission to a vector of zero-commission settlements from one market."
function _net_market_commission(settlements::AbstractVector, commission_rate::Real)
    isempty(settlements) && return NamedTuple[]
    net_market_win = sum(s -> s.net_pnl, settlements)
    market_commission = max(0.0, net_market_win) * Float64(commission_rate)
    gross_wins = sum(s -> max(0.0, s.net_pnl), settlements)

    return [begin
                commission = s.net_pnl > 0.0 && gross_wins > 0.0 ?
                             market_commission * s.net_pnl / gross_wins : 0.0
                merge(s, (; commission, net_pnl = s.net_pnl - commission))
            end for s in settlements]
end

"""
    settle_slate!(conn, slate_id, results; schema, at) -> NamedTuple

Grade every filled leg of a slate and book the PnL.

`results` maps `match_id => (home_goals, away_goals)`. A fixture absent from it is left alone
rather than voided -- an unavailable result is not a void, and treating it as one would release
liability on a bet that is still running.

Commission is assessed on net winnings per `(match_id, market_group, market_line)`, matching the
exchange market contract, then allocated pro rata across that market's winning orders for the
per-order audit rows. The account movement per leg is a single `SETTLE`: `reserved` down by the
filled risk, `balance` up by the gross return less allocated commission. After the last one,
`reserved` attributable to the batch is zero, which is the invariant `reconcile_account` checks.
"""
function settle_slate!(conn, slate_id::UUID, results::Dict{Int,Tuple{Int,Int}};
                       schema::AbstractString = PAPER_SCHEMA, at::DateTime = now(),
                       result_source::AbstractString = "sofascore.events")
    orders = slate_orders(conn, slate_id; schema = schema)
    isempty(orders) && error("settle_slate!: no orders for slate $slate_id.")
    account = account_row(conn, first(orders).account_id; schema = schema)
    fills_df = fill_rows(conn, slate_id; schema = schema)

    # Build every exact cashflow first. Commission cannot be correct while orders are handled
    # one at a time because losses in the same exchange market offset wins.
    prepared = NamedTuple[]
    for o in orders
        haskey(results, o.match_id) || continue
        (o.state == MATCHED || o.state == PARTIALLY_MATCHED) || continue
        h, a = results[o.match_id]
        sub = filter(r -> String(r.order_id) == string(o.order_id), fills_df)
        fills = Fill[Fill(order_id = o.order_id, filled_at = DateTime(r.filled_at),
                          price = Float64(r.price), size = Float64(r.size),
                          risk_filled = Float64(r.risk_filled),
                          model = Symbol(r.fill_model), levels_used = Int(r.level_depth))
                     for r in eachrow(sub)]
        settlement = settle_order(o, fills, h, a, 0.0)
        push!(prepared, (; order = o, home_goals = h, away_goals = a, settlement))
    end

    market_groups = Dict{Tuple{Int,String,Float64},Vector{Int}}()
    for (i, p) in enumerate(prepared)
        o = p.order
        push!(get!(market_groups, (o.match_id, o.market_group, o.market_line), Int[]), i)
    end
    for indices in values(market_groups)
        netted = _net_market_commission(
            [prepared[i].settlement for i in indices], account.commission_rate)
        for (i, settlement) in zip(indices, netted)
            prepared[i] = merge(prepared[i], (; settlement))
        end
    end

    n_settled = 0; total_pnl = 0.0
    for p in prepared
        o, h, a, s = p.order, p.home_goals, p.away_goals, p.settlement
        LibPQ.execute(conn, """
            INSERT INTO $schema.paper_settlements
                (order_id, settled_at, result_source, home_goals, away_goals, outcome,
                 gross_return, commission, net_pnl)
            VALUES (\$1,\$2,\$3,\$4,\$5,\$6,\$7,\$8,\$9)
            ON CONFLICT (order_id) DO NOTHING;""",
            (string(o.order_id), at, String(result_source), h, a, uppercase(String(s.outcome)),
             s.gross_return, s.commission, s.net_pnl))

        # Clamp to remaining reserved on the account to absorb penny rounding
        # between individual order limits and batch reservation.
        acct_now = account_row(conn, o.account_id; schema = schema)
        delta_res = -min(acct_now.reserved, s.risk_settled)

        post_ledger!(conn, LedgerDelta(kind = :SETTLE, account_id = o.account_id,
                                       delta_balance = s.gross_return - s.commission,
                                       delta_reserved = delta_res,
                                       order_id = o.order_id, slate_id = slate_id,
                                       note = "settle $(s.outcome)"); schema = schema)
        update_order_state!(conn, o.order_id, SETTLED, "settled $(s.outcome)";
                            schema = schema, at = at)
        n_settled += 1; total_pnl += s.net_pnl
    end

    set_batch_status!(conn, slate_id, BATCH_SETTLED; schema = schema, at = at)
    return (; n_settled, total_pnl,
            account = account_row(conn, first(orders).account_id; schema = schema))
end

"""
    clv_for_order(order, fills, close_prob, close_ts; source) -> NamedTuple

Closing-line value in probability points, positive when we beat the close.

`entry_prob` is `1 / effective_odds` at the volume-weighted fill, NOT at the quoted price: a leg
that filled two ticks worse than it was priced has spent that difference, and measuring CLV
against the price we wanted rather than the one we got would credit execution with a fill it did
not achieve.

`close_prob` must be a DE-VIGGED closing probability. The book's raw `1/best_back` sums above one
and would make every leg look better than it was, uniformly.
"""
function clv_for_order(o::PaperOrder, fills::AbstractVector{Fill}, close_prob::Real,
                       close_ts::DateTime; source::AbstractString = "order_book_1m_mid")
    vwap = fill_vwap(fills)
    eff  = isnan(vwap) ? o.effective_odds :
           (o.side === :lay ? lay_to_back(vwap) : vwap)
    entry = 1.0 / eff
    clv   = Float64(close_prob) - entry
    filled_at = isempty(fills) ? o.quote_ts : minimum(f.filled_at for f in fills)
    return (; entry_prob = entry, close_prob = Float64(close_prob), close_ts,
            close_source = String(source), clv, clv_pct = clv / entry,
            beat_close = clv > 0,
            entry_lead_min = Int(round(Dates.value(o.kickoff - filled_at) / 60_000)))
end

"""
    mark_to_market(order, fills, best_back, best_lay; conservative = true) -> Float64

Unrealised PnL on an open position, in currency.

Marked in **probability space**, which is the only denomination in which one formula covers both
instruments: a position of `risk` at effective odds `D` has expected value `risk * (p * D - 1)`
at fair probability `p`, and that is exactly zero at entry where `p = 1/D`. The morphism already
made `D` the effective odds of the position whichever side it was taken on, so this does not
branch on `side` -- the last place it might have had to.

`best_back` and `best_lay` are the **model selection's** book, not the venue runner's. A held
position is closed by LAYING it, so `conservative = true` marks against `1 / best_lay`: the price
at which the position could actually be flattened right now. Marking against the mid, let alone
against the side we entered on, flatters every position by the spread -- on Scottish League Two's
5-tick 1X2 book that is 4-5% of the position, which is larger than the edge being measured.
"""
function mark_to_market(o::PaperOrder, fills::AbstractVector{Fill},
                        best_back::Real, best_lay::Real; conservative::Bool = true)
    risk = filled_risk(fills)
    risk <= 1e-9 && return 0.0
    bb, bl = Float64(best_back), Float64(best_lay)
    p_exit = if conservative
        bl > 1.0 ? 1.0 / bl : NaN
    else
        (bb > 1.0 && bl > 1.0) ? (1.0 / bb + 1.0 / bl) / 2 : NaN
    end
    isnan(p_exit) && return 0.0
    return risk * (o.effective_odds * p_exit - 1.0)
end
