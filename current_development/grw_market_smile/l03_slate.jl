# ==============================================================================
# Task 015 loader — counterfactual re-pricing of the 2026-09-12 card
# ==============================================================================
#
# Definitions only. `r05_slate_repricing.jl` executes.
#
# PROVENANCE. §1–§4 are Task 008's `hierarchical_home_advantage/l03_slate.jl` (branch
# `feat/hierarchical-home-advantage`, not merged into this one), carried with a `gms_` prefix
# and no behavioural change: played-card filter, pinned slot loading, read-only live ledger,
# full-fill settlement and the reproduction diff. §5 is new.
#
# WHY THE REPLAY ENGINE. It is the live pipeline with only the clock- and network-reading
# sources swapped for point-in-time twins: the book is served by `searchsortedlast(ts, as_of)`,
# so no tick after T−25 is reachable. A number here is a claim about the live pipeline.
#
# SMILE × OPTION B CAVEAT. `MD.calibrate_matchday_latents` replaces the `λ_h`/`λ_a` draw
# columns and leaves every other column alone. For a smile rung that means 1X2 and BTTS price
# off the CALIBRATED grid while O/U prices off the UNCALIBRATED `λ_tot · φ`. The smile
# rung's Option B arm is therefore a mixed object; the raw arms are the clean comparison.
#
# NOTHING HERE WRITES. `paper_runbook` is read; no execute or settle transition is called.
# ==============================================================================

if !isdefined(@__MODULE__, :GMSConfig)
    include(joinpath(@__DIR__, "l01_loader.jl"))
end
if !isdefined(@__MODULE__, :ReplayCard)
    include(joinpath(@__DIR__, "..", "match_day_inference", "replay_state.jl"))
end

# ==============================================================================
# 1. The card
# ==============================================================================

"The card holding only fixtures with a recorded full-time score, and the ones removed."
function gms_played_card(card::ReplayCard)
    played = MD.Fixture[f for f in card.fixtures if haskey(card.results, f.m_id)]
    dropped = MD.Fixture[f for f in card.fixtures if !haskey(card.results, f.m_id)]
    filtered = ReplayCard(card.day, played, card.kickoff, card.identities, card.book,
                          card.lineups, card.results, card.lineup_drop, card.book_span)
    return filtered, dropped
end

# ==============================================================================
# 2. Pinned model slots
# ==============================================================================

"`load_slot!` with the `canonical_fit` key supplied; THROWS rather than marking `:failed`."
function gms_load_pinned_slot!(slot::ModelSlot, key, ds, card::ReplayCard)
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

"""
    gms_clone_slot!(dst, src, ds, card) -> ModelSlot

`dst` shares `src`'s fit, boundaries and feature collection and differs only in its
calibrator. The feature build is a function of `(ds, model, splitter)` alone, so building it
twice for the raw and Option B arms of one run would cost minutes to produce an identical
object.
"""
function gms_clone_slot!(dst::ModelSlot, src::ModelSlot, ds, card::ReplayCard)
    src.status === :ready || error("cannot clone slot $(src.key): it is $(src.status)")
    dst.fit = src.fit
    dst.run_name = src.run_name
    dst.boundaries = src.boundaries
    dst.fcol = src.fcol
    rebind_slot!(dst, ds, card)
    dst.status = :ready
    dst.error = ""
    dst.load_seconds = 0.0
    return dst
end

# ==============================================================================
# 3. The live ledger, read only
# ==============================================================================

function gms_live_ledger(conn, account::AbstractString, day::Date; schema::AbstractString = "paper_runbook")
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

"Every leg of a priced slate settled at PLANNED risk (full fill) against the final score."
function gms_settle_sheet(slate::MD.PricedSlate, results::AbstractDict, commission_rate::Real;
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
    isempty(rows) && return gms_empty_legs()
    return DataFrame(rows)
end

"A settled sheet with no legs — the answer for an arm that staked nothing."
gms_empty_legs() = DataFrame(arm = String[], match_id = Int[], group = String[],
                             line = Float64[], selection = Symbol[], side = Symbol[],
                             effective_odds = Float64[], p_model = Float64[],
                             p_market = Float64[], edge = Float64[], risk = Float64[],
                             score = String[], outcome = Symbol[], net_pnl = Float64[])

"How closely a re-priced sheet reproduces the live orders, keyed on (match, group, line, selection)."
function gms_reproduction(repriced::DataFrame, live::DataFrame)
    key(r) = (Int(r.match_id), String(r.group), round(Float64(r.line); digits = 2), Symbol(r.selection))
    rk = Dict(key(r) => r for r in eachrow(repriced))
    lk = Dict(key(r) => r for r in eachrow(live))
    shared = intersect(Set(keys(rk)), Set(keys(lk)))
    risk_gap = isempty(shared) ? NaN : maximum(abs(rk[k].risk - lk[k].risk) for k in shared)
    p_gap = isempty(shared) ? NaN : maximum(abs(rk[k].p_model - lk[k].p_model) for k in shared)
    return (; n_live = nrow(live), n_repriced = nrow(repriced), n_shared = length(shared),
              only_repriced = sort!(collect(setdiff(Set(keys(rk)), Set(keys(lk))))),
              only_live = sort!(collect(setdiff(Set(keys(lk)), Set(keys(rk))))),
              max_risk_gap = risk_gap, max_p_model_gap = p_gap)
end

# ==============================================================================
# 5. Arm summaries
# ==============================================================================

"One tearsheet row per arm: legs, risk, P&L, and the away / home / Under 2.5 split."
function gms_arm_summary(arm::AbstractString, legs::DataFrame)
    is_1x2 = legs.group .== "1X2"
    away = legs[is_1x2 .& (legs.selection .== :away), :]
    home = legs[is_1x2 .& (legs.selection .== :home), :]
    return (; arm = String(arm), n_legs = nrow(legs),
              risk = sum(legs.risk; init = 0.0), net_pnl = sum(legs.net_pnl; init = 0.0),
              n_away = nrow(away), away_risk = sum(away.risk; init = 0.0),
              away_net = sum(away.net_pnl; init = 0.0),
              n_home = nrow(home), home_risk = sum(home.risk; init = 0.0),
              home_net = sum(home.net_pnl; init = 0.0),
              n_under25 = count(==(:under_25), legs.selection),
              n_wins = count(==(:win), legs.outcome))
end
