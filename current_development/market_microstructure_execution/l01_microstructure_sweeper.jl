module MicrostructureExecution

import Dates

export AbstractExecutionPolicy, TouchOnly, MultiLevelSweep, StagedTWAP,
       ExecutionOrder, ExecutionState, Ladder, reservation_price, wom,
       execute_snapshot!, arithmetic_vwap, matched_velocity, execution_signal

abstract type AbstractExecutionPolicy end

"One observation at T−25, level 1 only; same price/risk guards as the sweep."
Base.@kwdef struct TouchOnly <: AbstractExecutionPolicy
    max_slip::Float64 = 0.01
end

"Three-level taker. max_slip bounds adverse cumulative arithmetic venue-stake VWAP."
Base.@kwdef struct MultiLevelSweep <: AbstractExecutionPolicy
    max_slip::Float64 = 0.01
end

"21 cumulative taker tranches by default; no inferred passive fills or deadline override."
Base.@kwdef struct StagedTWAP <: AbstractExecutionPolicy
    max_slip::Float64 = 0.01
    start_minutes::Int = 25
    end_minutes::Int = 5
end

"Archive prices available TO back/lay; each size is backer stake, not lay liability."
struct Ladder
    ts::Dates.DateTime
    back::NTuple{3,Float64}
    back_size::NTuple{3,Float64}
    lay::NTuple{3,Float64}
    lay_size::NTuple{3,Float64}
    matched::Float64
end

"""
p_win means the probability of the POSITION winning: venue event for a back,
complement of venue event for a lay. For a synthetic binary lay, this is the model
selection probability. Never substitute p(venue event) for p_win on a lay.
Parent target_risk is immutable. Commission here is a standalone positive-payoff
haircut, not exact market-netted commission on correlated positions.
"""
struct ExecutionOrder
    side::Symbol
    p_win::Float64
    target_risk::Float64
    reference_odds::Float64
    bankroll::Float64
    commission::Float64
    hurdle::Float64
    function ExecutionOrder(side::Symbol, p_win::Real, target_risk::Real,
                            reference_odds::Real, bankroll::Real;
                            commission::Real = 0.02, hurdle::Real = 0.02)
        side in (:back, :lay) || throw(ArgumentError("side must be :back or :lay"))
        all(isfinite, (p_win, target_risk, reference_odds, bankroll, commission, hurdle)) ||
            throw(ArgumentError("order inputs must be finite"))
        0 < p_win < 1 || throw(ArgumentError("p_win must lie strictly in (0,1)"))
        0 <= target_risk < bankroll || throw(ArgumentError("require 0 <= target_risk < bankroll"))
        reference_odds > 1 || throw(ArgumentError("decimal odds must exceed 1"))
        0 <= commission < 1 || throw(ArgumentError("commission must lie in [0,1)"))
        hurdle >= 0 || throw(ArgumentError("hurdle must be nonnegative"))
        new(side, p_win, target_risk, reference_odds, bankroll, commission, hurdle)
    end
end

"Simulated fills only. Production must update from acknowledged fills, not intentions."
Base.@kwdef mutable struct ExecutionState
    risk::Float64 = 0.0
    net_win::Float64 = 0.0
    venue_size::Float64 = 0.0
    notional::Float64 = 0.0
    last_ts::Dates.DateTime = Dates.DateTime(1)
    last_status::Symbol = :not_attempted
    level_reasons::NTuple{3,Symbol} = (:not_attempted, :not_attempted, :not_attempted)
end

"Minimum accepted BACK odds; maximum accepted LAY odds (edge per unit liability)."
function reservation_price(order::ExecutionOrder)
    p = order.p_win
    c = order.commission
    a = order.hurdle
    return order.side === :back ? 1 + (1 - p + a) / (p * (1 - c)) :
                                 1 + p * (1 - c) / (1 - p + a)
end

arithmetic_vwap(s::ExecutionState) = s.venue_size > 0 ? s.notional / s.venue_size : nothing

"Displayed available-to-back imbalance (3 levels); nothing for zero/invalid depth."
function wom(book::Ladder)
    all(x -> isfinite(x) && x >= 0, (book.back_size..., book.lay_size...)) || return nothing
    b = sum(book.back_size)
    l = sum(book.lay_size)
    return b + l > 0 ? b / (b + l) : nothing
end

"Market-wide matched increments per second, not runner trade volume; reset => nothing."
function matched_velocity(previous::Ladder, current::Ladder)
    dt = Dates.value(current.ts - previous.ts) / 1000
    delta = current.matched - previous.matched
    return dt > 0 && isfinite(delta) && delta >= 0 ? delta / dt : nothing
end

"""
Research annotation only: imbalance sign is NOT a validated shortening predictor.
No maker fill is inferred from this signal. Acceleration must use past observations.
"""
function execution_signal(book::Ladder; threshold::Real = 0.65)
    0.5 < threshold < 1 || throw(ArgumentError("threshold must be in (0.5,1)"))
    w = wom(book)
    return w === nothing ? :unknown : w > threshold ? :back_depth_heavy :
           w < 1 - threshold ? :lay_depth_heavy : :balanced
end

window(::TouchOnly) = (25, 5)
window(::MultiLevelSweep) = (25, 5)
window(p::StagedTWAP) = (p.start_minutes, p.end_minutes)
slippage(p::Union{TouchOnly,MultiLevelSweep,StagedTWAP}) = p.max_slip
levels(::TouchOnly) = 1
levels(::Union{MultiLevelSweep,StagedTWAP}) = 3
single_shot(::Union{TouchOnly,MultiLevelSweep}) = true
single_shot(::StagedTWAP) = false

function refuse_snapshot!(state, reason)
    state.last_status = reason
    state.level_reasons = (reason, reason, reason)
    return (0.0, 0.0, 0.0)
end

function due_risk(::Union{TouchOnly,MultiLevelSweep}, order, at, kickoff)
    return order.target_risk
end
function due_risk(policy::StagedTWAP, order, at, kickoff)
    elapsed = Dates.value(at - (kickoff - Dates.Minute(policy.start_minutes))) / 60000
    tranche_count = policy.start_minutes - policy.end_minutes + 1
    return order.target_risk * clamp((floor(elapsed) + 1) / tranche_count, 0.0, 1.0)
end

"""
    execute_snapshot!(state, policy, order, book, at, kickoff; available_sizes=nothing)

Simulated taker fills, returned as a 3-tuple of venue stakes. All guards are causal:
quote <= decision, age <= 90s, monotone quote time, execution within the window.
Single-shot policies attempt only the start minute. Invalid/crossed/unsorted books
fail closed. Missing levels must be encoded as (price=0,size=0).
`state.last_status` records snapshot rejection or `:evaluated`. `level_reasons`
records a binding constraint (:depth, :reservation, :slip, :kelly, :target,
:policy_depth), including partial fills. These are sequential binding diagnostics,
NOT an additive causal decomposition of unfilled pounds. Ties prioritize depth,
then target, then Kelly, then slip. The slip anchor is the immutable parent
reference, NOT each tick's touch; all policies share the same anchor and tolerance.

Caller owns a shared per-(market, runner, side, absolute price) depletion map and
passes residual available_sizes. A new timestamp alone does NOT prove replenishment.
Without that map, repeated-snapshot TWAP is an explicitly optimistic bound.

The scalar Kelly derivative includes frozen fills on THIS position only. It is not
Portfolio's joint SlateDrawdown certificate. Do not promote this kernel to live
execution without a full-slate conditional-risk solve and durable child-order state.
"""
function execute_snapshot!(state::ExecutionState, policy::AbstractExecutionPolicy,
                           order::ExecutionOrder, book::Ladder,
                           at::Dates.DateTime, kickoff::Dates.DateTime;
                           available_sizes::Union{Nothing,NTuple{3,Float64}} = nothing)
    start_minutes, end_minutes = window(policy)
    start_minutes > end_minutes >= 0 || throw(ArgumentError("invalid working window"))
    slip = slippage(policy)
    isfinite(slip) && 0 <= slip < 1 || throw(ArgumentError("max_slip must be in [0,1)"))
    empty_fill = (0.0, 0.0, 0.0)
    start = kickoff - Dates.Minute(start_minutes)
    deadline = kickoff - Dates.Minute(end_minutes)
    start <= at <= deadline || return refuse_snapshot!(state, :outside_window)
    single_shot(policy) && at != start && return refuse_snapshot!(state, :not_scheduled)
    book.ts <= at || return refuse_snapshot!(state, :future_quote)
    at - book.ts <= Dates.Second(90) || return refuse_snapshot!(state, :stale_quote)
    book.ts > state.last_ts || return refuse_snapshot!(state, :reused_quote)
    prices = order.side === :back ? book.back : book.lay
    displayed = order.side === :back ? book.back_size : book.lay_size
    sizes = available_sizes === nothing ? displayed : available_sizes
    all(x -> isfinite(x) && x >= 0, (sizes..., displayed...)) || return refuse_snapshot!(state, :invalid_book)
    book.back[1] > 1 && book.lay[1] > 1 && book.back[1] > book.lay[1] && return refuse_snapshot!(state, :crossed_book)
    previous = order.side === :back ? Inf : 1.0
    for i in 1:3
        sizes[i] <= displayed[i] + 1e-9 || throw(ArgumentError("residual exceeds displayed depth"))
        displayed[i] == 0 && continue
        price = prices[i]
        isfinite(price) && price > 1 || return refuse_snapshot!(state, :invalid_book)
        (order.side === :back ? price <= previous : price >= previous) || return refuse_snapshot!(state, :unsorted_book)
        previous = price
    end
    state.last_ts = book.ts
    state.last_status = :evaluated
    state.level_reasons = (:policy_depth, :policy_depth, :policy_depth)
    target = due_risk(policy, order, at, kickoff)
    reserve = reservation_price(order)
    boundary = order.reference_odds * (order.side === :back ? 1 - slip : 1 + slip)
    filled = empty_fill
    for i in 1:levels(policy)
        price = prices[i]
        state.level_reasons = Base.setindex(state.level_reasons, :target, i)
        target - state.risk > 1e-10 || continue
        state.level_reasons = Base.setindex(state.level_reasons, :depth, i)
        sizes[i] > 0 || continue
        if !(order.side === :back ? price >= reserve : price <= reserve)
            state.level_reasons = Base.setindex(state.level_reasons, :reservation, i)
            continue
        end
        risk_per_stake = order.side === :back ? 1.0 : price - 1
        win_per_risk = order.side === :back ? (price - 1) * (1 - order.commission) :
                                            (1 - order.commission) / (price - 1)
        # Concave one-position expected log growth: stop where marginal utility is zero.
        kelly_room = order.p_win * (order.bankroll - state.risk) -
                     (1 - order.p_win) * (order.bankroll + state.net_win) / win_per_risk
        taken = sizes[i]
        reason = :depth
        target_size = max(0.0, min(target, order.target_risk) - state.risk) / risk_per_stake
        if target_size < taken
            taken = target_size
            reason = :target
        end
        kelly_size = max(0.0, kelly_room) / risk_per_stake
        if kelly_size < taken
            taken = kelly_size
            reason = :kelly
        end
        slip_size = Inf
        if order.side === :back && price < boundary
            slip_size = max(0.0, (state.notional - boundary * state.venue_size) / (boundary - price))
        elseif order.side === :lay && price > boundary
            slip_size = max(0.0, (boundary * state.venue_size - state.notional) / (price - boundary))
        end
        if slip_size < taken
            taken = slip_size
            reason = :slip
        end
        state.level_reasons = Base.setindex(state.level_reasons, reason, i)
        taken > 1e-10 || continue
        risk = taken * risk_per_stake
        state.risk += risk
        state.net_win += risk * win_per_risk
        state.venue_size += taken
        state.notional += taken * price
        filled = Base.setindex(filled, taken, i)
    end
    return filled
end

end # module
