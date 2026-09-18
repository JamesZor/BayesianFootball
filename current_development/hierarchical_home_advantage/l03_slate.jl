# ==============================================================================
# Task 008 Phase 1 loader — counterfactual re-pricing of the 2026-09-12 card
# ==============================================================================
#
# Definitions only. `r05_slate_repricing.jl` executes.
#
# WHY THE REPLAY ENGINE AND NOT `MD.price_slate`. The card is re-priced after it was
# played, from a DataStore that already holds the XIs that took the field.
# `MD.matchday_latents` materialises only `MD.INJECTABLE_KEYS`, which omits
# `:player_lineup_ratings_map`, so a hybrid priced through `price_slate` today would read
# the PLAYED teamsheet. The replay engine (`current_development/match_day_inference/
# replay_state.jl`) exists to close exactly that leak — `slot_latents` materialises
# `REPLAY_INJECTABLE_KEYS` from the XI visible at `as_of` — and it reads the book and the
# lineup scrapes with `searchsortedlast` / `scraped_at <= as_of`. Every stage here is the
# replay console's, so a number here is a claim about the live pipeline.
#
# WHY SLOTS ARE LOADED BY RUN ID. The replay registry addresses runs by name, and
# `m12_joint_hybrid_synergy` resolves to two completed runs. `hha_load_pinned_slot!` is
# `load_slot!` with the key passed through to `canonical_fit`, so the flat arm is Run 67 —
# the run that priced the live card — and nothing else.
#
# NOTHING HERE WRITES. `ReplayState` refuses any schema but `paper_replay`, and this
# loader never calls an execute or settle transition. The live ledger is read, not written.
# ==============================================================================

if !isdefined(@__MODULE__, :HHAConfig)
    include(joinpath(@__DIR__, "l01_loader.jl"))
end
if !isdefined(@__MODULE__, :ReplayCard)
    include(joinpath(@__DIR__, "..", "match_day_inference", "replay_state.jl"))
end

# ==============================================================================
# 1. The card
# ==============================================================================

"""
    hha_played_card(card) -> (ReplayCard, Vector{Fixture})

The same card holding only fixtures with a recorded full-time score.

`load_replay_card` reads every event on the day, including the postponed fixture the live
runner filtered out on `status_type`. A fixture with no score can be neither staked
comparably nor settled, so it is removed from the card before any model sees it, and
returned so the runner can name it.
"""
function hha_played_card(card::ReplayCard)
    played = MD.Fixture[f for f in card.fixtures if haskey(card.results, f.m_id)]
    dropped = MD.Fixture[f for f in card.fixtures if !haskey(card.results, f.m_id)]
    filtered = ReplayCard(card.day, played, card.kickoff, card.identities, card.book,
                          card.lineups, card.results, card.lineup_drop, card.book_span)
    return filtered, dropped
end

# ==============================================================================
# 2. Pinned model slots
# ==============================================================================

"""
    hha_load_pinned_slot!(slot, key, ds, card) -> ModelSlot

`load_slot!` with the `canonical_fit` key supplied rather than read from `slot.run_name`.
Unlike `load_slot!` it THROWS: a counterfactual with a missing arm is not a result.
"""
function hha_load_pinned_slot!(slot::ModelSlot, key, ds, card::ReplayCard)
    t0 = time()
    fit = MD.canonical_fit(TT.PostgresStorage(slot.experiment), key; require_converged = true)
    slot.fit = fit
    slot.run_name = fit.run_name
    slot.boundaries = DD.create_id_boundaries(ds, fit.config.splitter)
    slot.fcol = FE.create_features(slot.boundaries, ds, fit.config.model, fit.config.splitter)
    rebind_slot!(slot, ds, card)
    slot.status = :ready
    slot.error = ""
    slot.load_seconds = time() - t0
    return slot
end

# ==============================================================================
# 3. The live ledger, read only
# ==============================================================================

"""
    hha_live_ledger(conn, account, day; schema = "paper_runbook") -> NamedTuple

What the live account actually did on `day`: the slate row, every order, and each order's
settlement. `realised_net` is the sum of `net_pnl` over settled legs — the realised figure
the work package quotes, computed rather than copied.
"""
function hha_live_ledger(conn, account::AbstractString, day::Date; schema::AbstractString = "paper_runbook")
    slates = DataFrame(LibPQ.execute(conn, """
        SELECT slate_id, as_of, bankroll, total_risk, fold_idx, run_name
        FROM $schema.paper_slates
        WHERE account_id = \$1 AND as_of::date = \$2
        ORDER BY as_of;""", (String(account), day)))
    nrow(slates) == 1 || error("expected exactly one $schema slate for $account on $day; found $(nrow(slates))")
    sid = slates.slate_id[1] isa UUID ? slates.slate_id[1] : UUID(String(slates.slate_id[1]))
    orders = MD.slate_orders(conn, sid; schema = schema)
    settlements = DataFrame(LibPQ.execute(conn, """
        SELECT s.order_id, s.home_goals, s.away_goals, s.outcome,
               s.gross_return, s.commission, s.net_pnl
        FROM $schema.paper_settlements s
        JOIN $schema.paper_orders o ON o.order_id = s.order_id
        WHERE o.slate_id = \$1;""", (string(sid),)))
    by_order = Dict(String(r.order_id) => r for r in eachrow(settlements))
    # `paper_settlements` carries no filled-risk column; the fills table is the record of it.
    fills = MD.fill_rows(conn, sid; schema = schema)
    filled = Dict{String,Float64}()
    for r in eachrow(fills)
        filled[String(r.order_id)] = get(filled, String(r.order_id), 0.0) + Float64(r.risk_filled)
    end
    legs = DataFrame(
        match_id = [o.match_id for o in orders],
        group = [o.market_group for o in orders],
        line = [o.market_line for o in orders],
        selection = [o.selection for o in orders],
        side = [o.side for o in orders],
        effective_odds = [o.effective_odds for o in orders],
        p_model = [o.p_model for o in orders],
        p_market = [o.p_market for o in orders],
        risk = [o.risk for o in orders],
        risk_filled = [get(filled, string(o.order_id), 0.0) for o in orders],
        net_pnl = [haskey(by_order, string(o.order_id)) ?
                   Float64(by_order[string(o.order_id)].net_pnl) : 0.0 for o in orders],
    )
    account_row = MD.account_row(conn, account; schema = schema)
    return (; slate = slates[1, :], legs, realised_net = sum(legs.net_pnl),
              commission_rate = Float64(account_row.commission_rate))
end

# ==============================================================================
# 4. Settling a sheet
# ==============================================================================

"""
    hha_settle_sheet(slate, results, commission_rate; label) -> DataFrame

Every leg of a priced slate, settled at its PLANNED risk against the full-time score.

`grade_selection` grades the model's selection (never the venue runner), and a win pays
`risk × (effective_odds − 1)` net of commission — `settle_order`'s arithmetic, with the
fill assumed complete. The live account filled through `LadderSweep` and some legs only
partly, so this is the comparable figure for two re-priced sheets; the realised ledger
figure is reported beside it, not in place of it.
"""
function hha_settle_sheet(slate::MD.PricedSlate, results::AbstractDict, commission_rate::Real;
                          label::AbstractString)
    orders = MD.orders_to_paper(slate)
    rows = NamedTuple[]
    for o in orders
        hg, ag = results[o.match_id]
        outcome = MD.grade_selection(o.market_group, o.market_line, o.selection, hg, ag)
        net = outcome === :win ? o.risk * (o.effective_odds - 1.0) * (1.0 - commission_rate) :
              outcome === :lose ? -o.risk : 0.0
        push!(rows, (; arm = String(label), match_id = o.match_id, group = o.market_group,
                       line = o.market_line, selection = o.selection, side = o.side,
                       effective_odds = o.effective_odds, p_model = o.p_model,
                       p_market = o.p_market, edge = o.edge, risk = o.risk,
                       score = "$hg-$ag", outcome, net_pnl = net))
    end
    isempty(rows) && return DataFrame(arm = String[], match_id = Int[], risk = Float64[], net_pnl = Float64[])
    return DataFrame(rows)
end

"""
    hha_reproduction(repriced, live) -> NamedTuple

How closely a re-priced sheet reproduces the live orders, keyed on
`(match_id, group, line, selection)`. Reports the key sets both ways and the worst risk and
`p_model` gaps on shared legs; it decides nothing, the runner prints it.
"""
function hha_reproduction(repriced::DataFrame, live::DataFrame)
    key(r) = (Int(r.match_id), String(r.group), round(Float64(r.line); digits = 2), Symbol(r.selection))
    rk = Dict(key(r) => r for r in eachrow(repriced))
    lk = Dict(key(r) => r for r in eachrow(live))
    shared = intersect(Set(keys(rk)), Set(keys(lk)))
    only_repriced = setdiff(Set(keys(rk)), Set(keys(lk)))
    only_live = setdiff(Set(keys(lk)), Set(keys(rk)))
    risk_gap = isempty(shared) ? NaN : maximum(abs(rk[k].risk - lk[k].risk) for k in shared)
    p_gap = isempty(shared) ? NaN : maximum(abs(rk[k].p_model - lk[k].p_model) for k in shared)
    return (; n_live = nrow(live), n_repriced = nrow(repriced), n_shared = length(shared),
              only_repriced = sort!(collect(only_repriced)), only_live = sort!(collect(only_live)),
              max_risk_gap = risk_gap, max_p_model_gap = p_gap)
end
