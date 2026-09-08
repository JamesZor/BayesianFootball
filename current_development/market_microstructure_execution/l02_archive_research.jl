module ArchiveMicrostructureResearch

"""
Read-only extraction and causal archive-replay helpers for the 2026-09-05 live slate.

This module deliberately does not construct fits, write a paper ledger, or infer a lineup
state. It consumes the frozen `paper_runbook` order fields as order provenance and the
Betfair one-minute archive as execution opportunity. The two clocks are reported apart:
the operator ledger's submitted/fill timestamps are not archive execution timestamps.
"""

import Dates
import DataFrames
import LibPQ
import Statistics
import CSV

const DAY = Dates.Date(2026, 9, 5)
const LEDGER_SCHEMA = "paper_runbook"
const PRICE_SCALE = 10_000.0
const SIZE_SCALE = 10_000.0

"A typed archive row after decoding the collector fixed-point arrays."
struct ArchiveSnapshot
    order_id::String
    market_id::String
    symbol::String
    ts::Dates.DateTime
    kickoff::Dates.DateTime
    back::NTuple{3,Float64}
    back_size::NTuple{3,Float64}
    lay::NTuple{3,Float64}
    lay_size::NTuple{3,Float64}
    market_matched::Float64
end

"""
Open a read-only, repeatable-read extraction transaction in UTC with bounded statements.

The transaction is intentionally left open for the caller's complete extraction batch; closing
without commit rolls it back. This function never prints the credential-bearing connection URL.
"""
function connect_readonly()
    conn = LibPQ.Connection(ENV["BF_DB_URL"])
    try
        LibPQ.execute(conn, "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;")
        LibPQ.execute(conn, "SET LOCAL TIME ZONE 'UTC';")
        LibPQ.execute(conn, "SET LOCAL statement_timeout = '60s';")
        return conn
    catch
        close(conn)
        rethrow()
    end
end

function _datetime(value)
    value isa Dates.DateTime && return value
    text = string(value)
    return Dates.DateTime(first(text, 19))
end

_float(value) = ismissing(value) || value === nothing ? NaN : Float64(value)
_string(value) = ismissing(value) || value === nothing ? "" : String(value)

function _fixed_tuple(values, scale)
    raw = collect(values)
    return ntuple(i -> i > length(raw) || ismissing(raw[i]) ? 0.0 : Float64(raw[i]) / scale, 3)
end

"Frozen order/fill/settlement facts for the actual Saturday slate, one row per order."
function extract_live_ledger(conn; day::Dates.Date = DAY,
                             slate_id::AbstractString = "0fd23606-80bf-4fcd-be44-66d708562fe6")
    sql = """
    SELECT o.order_id::text, o.match_id, o.kickoff, o.bf_market_id, o.market_group,
           o.market_line, o.selection, o.venue_selection, o.side, o.venue_odds,
           o.leverage, o.effective_odds, o.p_model, o.p_market, o.edge,
           o.stake_fraction, o.risk, o.venue_stake, o.quote_ts, o.state, o.reason,
           o.submitted_at, o.terminal_at,
           f.filled_at, f.price AS fill_price, f.size AS fill_size,
           f.risk_filled, f.fill_model, f.level_depth,
           st.outcome, st.gross_return, st.commission AS settlement_commission,
           st.net_pnl, s.slate_id::text, s.account_id, s.as_of AS slate_as_of,
           s.run_name, s.bankroll AS slate_bankroll, s.total_risk AS slate_total_risk,
           s.n_legs, s.batch_status
    FROM paper_runbook.paper_orders o
    JOIN paper_runbook.paper_slates s ON s.slate_id = o.slate_id
    LEFT JOIN paper_runbook.paper_fills f ON f.order_id = o.order_id
    LEFT JOIN paper_runbook.paper_settlements st ON st.order_id = o.order_id
    WHERE s.slate_window = \$1 AND s.slate_id::text = \$2
    ORDER BY o.order_id, f.fill_id;
    """
    frame = DataFrames.DataFrame(LibPQ.execute(conn, sql, (string(day), slate_id)))
    DataFrames.nrow(frame) > 0 || error("no orders found for explicit slate UUID $slate_id on $day")
    length(unique(frame.slate_id)) == 1 || error("explicit slate selection returned multiple slate UUIDs")
    DataFrames.nrow(frame) == length(unique(frame.order_id)) ||
        error("expected one recorded fill at most per order; found $(DataFrames.nrow(frame)) rows for $(length(unique(frame.order_id))) orders")
    return frame
end

"Actual Scottish 56/57 fixture inventory for the day; this is slate context, not inference."
function extract_fixture_inventory(conn; day::Dates.Date = DAY)
    sql = """
    SELECT e.match_id, e.home_team, e.away_team,
           to_timestamp(e.start_timestamp) AT TIME ZONE 'UTC' AS kickoff,
           e.tournament_id
    FROM sofascore.events e
    WHERE e.tournament_id IN (56, 57)
      AND to_timestamp(e.start_timestamp) >= \$1::date
      AND to_timestamp(e.start_timestamp) < \$1::date + INTERVAL '1 day'
    ORDER BY e.start_timestamp;
    """
    return DataFrames.DataFrame(LibPQ.execute(conn, sql, (string(day),)))
end

"Map a frozen order to its exact Betfair market and venue runner. No fallback runner is allowed."
function extract_order_snapshots(conn, orders; start_minutes::Int = 25, end_minutes::Int = 5)
    start_minutes >= end_minutes >= 0 || error("invalid archive window")
    isempty(orders) && return DataFrames.DataFrame()
    ids = join(["'" * replace(String(id), "'" => "''") * "'" for id in orders.order_id], ",")
    sql = """
    WITH frozen AS (
        SELECT o.order_id::text, o.match_id, o.kickoff, o.bf_market_id, o.market_group,
               o.selection, o.venue_selection, o.side, o.venue_odds,
               o.effective_odds, o.p_model, o.risk, o.venue_stake,
               o.quote_ts, s.bankroll
        FROM paper_runbook.paper_orders o
        JOIN paper_runbook.paper_slates s ON s.slate_id = o.slate_id
        WHERE o.order_id::text IN ($ids)
    ), mapped AS (
        SELECT f.*, m.market_id, m.market_type
        FROM frozen f
        JOIN betfair.match_meta mm ON mm.match_id = f.match_id
        JOIN betfair_live.market_metadata m ON m.event_id = mm.betfair_event_id
          AND (NULLIF(f.bf_market_id, '') IS NULL OR m.market_id = f.bf_market_id)
          AND ((f.market_group = '1X2' AND m.market_type = 'MATCH_ODDS')
            OR (f.market_group = 'OverUnder' AND m.market_type = 'OVER_UNDER_25'))
    )
    SELECT m.*, b.symbol, b.ts, b.bid_prices, b.bid_volumes,
           b.ask_prices, b.ask_volumes, b.market_matched
    FROM mapped m
    LEFT JOIN betfair_live.order_book_1m b
      ON b.market_id = m.market_id
     AND (lower(b.symbol) = lower(m.venue_selection)
          OR (m.market_group = 'OverUnder' AND m.venue_selection = 'under_25'
              AND lower(b.symbol) = 'under 2.5 goals')
          OR (m.market_group = 'OverUnder' AND m.venue_selection = 'over_25'
              AND lower(b.symbol) = 'over 2.5 goals'))
     AND b.ts >= m.kickoff - INTERVAL '$start_minutes minutes 90 seconds'
     AND b.ts <= m.kickoff - INTERVAL '$end_minutes minutes'
    ORDER BY m.order_id, b.ts;
    """
    return DataFrames.DataFrame(LibPQ.execute(conn, sql))
end

"Decode a database row to an archive snapshot. Returns `nothing` for unmapped/archive-null rows."
function decode_snapshot(row)
    (ismissing(row.ts) || ismissing(row.symbol)) && return nothing
    return ArchiveSnapshot(
        String(row.order_id), String(row.market_id), String(row.symbol), _datetime(row.ts),
        _datetime(row.kickoff), _fixed_tuple(row.bid_prices, PRICE_SCALE),
        _fixed_tuple(row.bid_volumes, SIZE_SCALE), _fixed_tuple(row.ask_prices, PRICE_SCALE),
        _fixed_tuple(row.ask_volumes, SIZE_SCALE), _float(row.market_matched) / SIZE_SCALE,
    )
end

"Residual three-level depth after shared absolute-price consumption."
function residual_sizes(market_id::AbstractString, symbol::AbstractString, side::Symbol,
                        prices::NTuple{3,Float64}, displayed::NTuple{3,Float64}, depleted)
    return ntuple(i -> max(0.0, displayed[i] - get(depleted, (String(market_id), String(symbol), side, prices[i]), 0.0)), 3)
end

"Record consumed venue stake by market, runner, side and absolute price."
function consume_sizes!(depleted, market_id::AbstractString, symbol::AbstractString, side::Symbol,
                        prices::NTuple{3,Float64}, fills::NTuple{3,Float64}; stamp = nothing)
    for i in 1:3
        fills[i] > 0 || continue
        key = stamp === nothing ? (String(market_id), String(symbol), side, prices[i]) :
                                  (String(market_id), String(symbol), side, prices[i], stamp)
        depleted[key] = get(depleted, key, 0.0) + fills[i]
    end
    return depleted
end

"Residual depth under no-replenishment or timestamp-local refreshed-display accounting."
function replay_residual(snapshot::ArchiveSnapshot, side::Symbol, depleted; refreshed::Bool)
    displayed = side === :back ? snapshot.back_size : snapshot.lay_size
    prices = side === :back ? snapshot.back : snapshot.lay
    return ntuple(i -> begin
        key = refreshed ? (snapshot.market_id, snapshot.symbol, side, prices[i], snapshot.ts) :
                          (snapshot.market_id, snapshot.symbol, side, prices[i])
        max(0.0, displayed[i] - get(depleted, key, 0.0))
    end, 3)
end

"Convert a decoded archive snapshot to the execution kernel's ladder."
function execution_ladder(snapshot::ArchiveSnapshot, ME)
    return ME.Ladder(snapshot.ts, snapshot.back, snapshot.back_size, snapshot.lay,
                     snapshot.lay_size, snapshot.market_matched)
end

"Construct a fixed parent order, rejecting an ambiguous synthetic lay before replay."
function replay_order(row, ME; commission::Float64, hurdle::Float64)
    side = Symbol(row.side)
    side === :lay && String(row.selection) == String(row.venue_selection) &&
        error("lay order $(row.order_id) has selection == venue_selection; p_model is not an explicit favorable-event probability")
    return ME.ExecutionOrder(side, Float64(row.p_model), Float64(row.risk),
                             Float64(row.venue_odds), Float64(row.slate_bankroll);
                             commission = commission, hurdle = hurdle)
end

"""
Replay a policy in causal clock order, returning parent, child, and sequential-gate diagnostics.

At every decision minute, parent UUIDs establish a deterministic collision priority. In the
refreshed-display scenario, depth still depletes across orders sharing the same stored archive
stamp; only a later archive timestamp may display fresh depth. Level reasons are sequential
binding diagnostics, not an additive unfilled-risk decomposition.
"""
function replay_policy(orders, rows, policy, ME;
                       start_minutes::Int, end_minutes::Int,
                       commission::Float64, hurdle::Float64, refreshed::Bool)
    states = Dict{String,Any}()
    parents = Dict{String,Any}()
    for row in eachrow(orders)
        order_id = String(row.order_id)
        parents[order_id] = row
        states[order_id] = ME.ExecutionState()
    end
    children = NamedTuple[]
    diagnostics = NamedTuple[]
    depleted = Dict{Tuple,Float64}()
    clocks = sort(unique([_datetime(row.kickoff) - Dates.Minute(m) for row in eachrow(orders)
                               for m in start_minutes:-1:end_minutes]))
    for at in clocks
        eligible = [row for row in eachrow(orders) if _datetime(row.kickoff) - Dates.Minute(start_minutes) <= at <= _datetime(row.kickoff) - Dates.Minute(end_minutes)]
        sort!(eligible; by = row -> String(row.order_id))
        for (priority, row) in enumerate(eligible)
            order_id = String(row.order_id)
            state = states[order_id]
            snapshot = snapshot_at(rows, order_id, at)
            if snapshot === nothing
                push!(diagnostics, (order_id = order_id, at = at, archive_ts = missing,
                                    status = :no_archive_quote, level_1_reason = :no_archive_quote,
                                    level_2_reason = :no_archive_quote, level_3_reason = :no_archive_quote,
                                    filled_risk = 0.0, priority))
                continue
            end
            order = replay_order(row, ME; commission, hurdle)
            available = replay_residual(snapshot, order.side, depleted; refreshed)
            book = execution_ladder(snapshot, ME)
            fills = ME.execute_snapshot!(state, policy, order, book, at, _datetime(row.kickoff);
                                         available_sizes = available)
            prices = order.side === :back ? snapshot.back : snapshot.lay
            stamp = refreshed ? snapshot.ts : nothing
            consume_sizes!(depleted, snapshot.market_id, snapshot.symbol, order.side, prices, fills; stamp)
            filled_risk = sum(price_risk(order.side, prices[i], fills[i]) for i in 1:3)
            reasons = state.level_reasons
            push!(diagnostics, (order_id = order_id, at = at, archive_ts = snapshot.ts,
                                status = state.last_status, level_1_reason = reasons[1],
                                level_2_reason = reasons[2], level_3_reason = reasons[3],
                                filled_risk, priority))
            for level in 1:3
                fills[level] > 0 || continue
                push!(children, (order_id = order_id, at, archive_ts = snapshot.ts,
                                 market_id = snapshot.market_id, symbol = snapshot.symbol,
                                 level, price = prices[level], venue_stake = fills[level],
                                 risk = price_risk(order.side, prices[level], fills[level])))
            end
        end
    end
    parent_rows = NamedTuple[]
    for row in eachrow(orders)
        order_id = String(row.order_id)
        state = states[order_id]
        vwap = ME.arithmetic_vwap(state)
        push!(parent_rows, (order_id, match_id = Int(row.match_id), side = String(row.side),
                            target_risk = Float64(row.risk), simulated_risk = state.risk,
                            risk_fill_pct = Float64(row.risk) == 0 ? 0.0 : state.risk / Float64(row.risk),
                            simulated_venue_stake = state.venue_size,
                            arithmetic_vwap = vwap === nothing ? missing : vwap,
                            final_status = state.last_status))
    end
    return DataFrames.DataFrame(parent_rows), DataFrames.DataFrame(children), DataFrames.DataFrame(diagnostics)
end

"The archive snapshot at or before a requested absolute time; `nothing` if unavailable."
function snapshot_at(rows, order_id::AbstractString, at::Dates.DateTime)
    candidates = ArchiveSnapshot[]
    for row in eachrow(rows)
        String(row.order_id) == order_id || continue
        snapshot = decode_snapshot(row)
        snapshot === nothing || snapshot.ts <= at && push!(candidates, snapshot)
    end
    isempty(candidates) && return nothing
    return candidates[argmax(getfield.(candidates, :ts))]
end

"Archive coverage per frozen order: mapping, exact T−25, working-window rows, and stale age."
function coverage_report(orders, snapshots; start_minutes::Int = 25, end_minutes::Int = 5)
    out = DataFrames.DataFrame(order_id = String[], mapped_market = Bool[], mapped_runner = Bool[],
                                exact_t25 = Bool[], window_rows = Int[], newest_age_seconds = Union{Missing,Int}[])
    for order in eachrow(orders)
        order_id = String(order.order_id)
        rows = snapshots[snapshots.order_id .== order_id, :]
        mapped_market = DataFrames.nrow(rows) > 0 && any(.!ismissing.(rows.market_id))
        usable = [decode_snapshot(row) for row in eachrow(rows)]
        usable = [x for x in usable if x !== nothing]
        kickoff = _datetime(order.kickoff)
        target = kickoff - Dates.Minute(start_minutes)
        ages = [Dates.value(target - x.ts) ÷ 1000 for x in usable if x.ts <= target]
        timely = !isempty(ages) && minimum(ages) <= 90
        push!(out, (order_id, mapped_market, !isempty(usable), timely, length(usable),
                    isempty(ages) ? missing : minimum(ages)))
    end
    return out
end

"All runner snapshots for MATCH_ODDS, OVER_UNDER_25 and BOTH_TEAMS_TO_SCORE on the day."
function extract_whole_slate_snapshots(conn; day::Dates.Date = DAY,
                                       start_minutes::Int = 25, end_minutes::Int = 5)
    sql = """
    SELECT e.match_id, m.market_id, m.market_type, b.symbol, b.ts,
           to_timestamp(e.start_timestamp) AT TIME ZONE 'UTC' AS kickoff,
           b.bid_prices, b.bid_volumes, b.ask_prices, b.ask_volumes, b.market_matched
    FROM sofascore.events e
    JOIN betfair.match_meta mm ON mm.match_id = e.match_id
    JOIN betfair_live.market_metadata m ON m.event_id = mm.betfair_event_id
      AND m.market_type IN ('MATCH_ODDS', 'OVER_UNDER_25', 'BOTH_TEAMS_TO_SCORE')
    JOIN betfair_live.order_book_1m b ON b.market_id = m.market_id
      AND b.ts >= (to_timestamp(e.start_timestamp) AT TIME ZONE 'UTC') - INTERVAL '$start_minutes minutes 90 seconds'
      AND b.ts <= (to_timestamp(e.start_timestamp) AT TIME ZONE 'UTC') - INTERVAL '$end_minutes minutes'
    WHERE e.tournament_id IN (56, 57)
      AND to_timestamp(e.start_timestamp) >= \$1::date
      AND to_timestamp(e.start_timestamp) < \$1::date + INTERVAL '1 day'
    ORDER BY e.match_id, m.market_id, b.symbol, b.ts;
    """
    return DataFrames.DataFrame(LibPQ.execute(conn, sql, (string(day),)))
end

"Price-specific risk: stake for backs, liability for lays."
price_risk(side::Symbol, price::Real, stake::Real) = side === :back ? Float64(stake) : Float64(stake) * (Float64(price) - 1.0)

"Gross per-order P&L before exchange commission, using known outcome and simulated child fills."
function simulated_gross_pnl(side::Symbol, outcome::AbstractString, prices, stakes)
    won = uppercase(outcome) == "WIN"
    gross = 0.0
    for (price, stake) in zip(prices, stakes)
        gross += side === :back ? (won ? stake * (price - 1.0) : -stake) :
                                 (won ? stake : -stake * (price - 1.0))
    end
    return gross
end

"Apply commission once to positive *market* gross, supplied by the caller's market grouping."
net_market_pnl(gross::Real, commission::Real) = gross > 0 ? gross * (1 - commission) : gross

"Correlation only; no threshold selection. Returns `missing` for insufficient/constant pairs."
function safe_correlation(x, y)
    keep = [isfinite(a) && isfinite(b) for (a, b) in zip(x, y)]
    sum(keep) >= 3 || return missing
    xx, yy = x[keep], y[keep]
    (Statistics.std(xx) == 0 || Statistics.std(yy) == 0) && return missing
    return Statistics.cor(xx, yy)
end

"Write a DataFrame as a reproducible CSV artefact."
function write_csv(path::AbstractString, frame)
    mkpath(dirname(path))
    CSV.write(path, frame)
    return path
end

end # module
