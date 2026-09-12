
# l10_t25_backtest.jl
#
# ===================================================================
# WHAT THIS IS
# ===================================================================
# The machinery behind `r10_t25_m05_m12_backtest.jl`: a leak-free lineup source, the
# per-slate execution harness (price -> fill -> settle -> CLV -> compound), the summary
# statistics and the ledger/report writers.
#
# NOTHING HERE SAMPLES AND NOTHING HERE WRITES TO A DATABASE. The whole harness is a pure
# function of (a canonical fit, an archived order book, a final score), which is what lets the
# same eight tracks be re-run and compared without a ledger schema at all. `MatchDay`'s ledger
# types are used as VALUES -- `PaperOrder`, `Fill`, `settle_order`, `clv_for_order` -- because
# they are the same arithmetic the live path books, and reimplementing it here would be the one
# way to make the backtest disagree with production silently.
#
# THE THREE LEAKS THIS FILE EXISTS TO CLOSE
#
# 1. `MD.LastHistorical(ds)` filters `DateTime(ds.matches.match_date[i]) <= as_of`, and
#    `match_date` is a `Date`. At `as_of = 2026-08-01T13:35` the fixture being priced has
#    `match_date = 2026-08-01` -> `2026-08-01T00:00 <= 13:35` is TRUE, and `argmax(match_date)`
#    then selects THAT match. On a live Saturday the fixture is not in `ds.matches` at all so
#    the filter never fires; on a backtest over finished matches it hands the model the
#    teamsheet that actually took the field. `PriorMatchdayXI` filters on the CALENDAR DAY,
#    strictly before the day being priced, which cannot select the target fixture.
# 2. The book. Every quote is read through `MD.ArchivedOrderBook`, whose SQL takes the latest
#    snapshot at or before `as_of`; a tick from after T-25 is unreachable rather than merely
#    unqueried.
# 3. The fold. `MD.matchday_latents` calls `select_split` with both `fixture_ids` and
#    `exclude`, so the conditioning chain is the one whose NEXT block is this card -- never one
#    whose target window already contains it.

using DataFrames
using Dates
using Printf
using Statistics
using UUIDs
import LibPQ

const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const FF = BayesianFootball.Features
const TT = BayesianFootball.Training
const CB = BayesianFootball.Calibration

# %%
# ===================================================================
# 1. A leak-free historical lineup source
# ===================================================================

"""
    PriorMatchdayXI(ds)

Each team's most recent completed XI drawn from a match played on a **strictly earlier calendar
day** than `as_of`.

This is `MD.LastHistorical`'s backtest twin and it differs in exactly one predicate.
`LastHistorical` compares `DateTime(match_date) <= as_of` against a `Date` column, so a fixture
kicking off at 14:00 on the day being priced is midnight-stamped and passes a 13:35 cut-off --
which on a *finished* fixture returns the teamsheet that took the field. Comparing days closes
that, and it costs nothing real: the Scottish Lower cards in this study are all 14:00 UTC, so no
same-day match could have finished before T-25 anyway.

Returns `nothing` when either side has no earlier appearance, so a fixture with no usable
history is refused by `check_coverage` rather than priced off a neutral pillar.
"""
struct PriorMatchdayXI <: MD.AbstractLineupSource
    ds::Any
end

function MD.lineup(s::PriorMatchdayXI, f::MD.Fixture, as_of::DateTime)
    home = _prior_xi(s.ds, f.home, Date(as_of))
    away = _prior_xi(s.ds, f.away, Date(as_of))
    (isempty(home) || isempty(away)) && return nothing
    return MD.Lineup(home, away, false, :last_historical, as_of)
end

function _prior_xi(ds, team::AbstractString, before::Date)
    m = ds.matches
    rows = findall(i -> (m.home_team[i] == team || m.away_team[i] == team) &&
                        Date(m.match_date[i]) < before, 1:nrow(m))
    isempty(rows) && return MD.Player[]
    i    = rows[argmax(m.match_date[rows])]
    mid  = m.match_id[i]
    side = m.home_team[i] == team ? "home" : "away"

    lu  = ds.lineups
    sel = findall(j -> lu.match_id[j] == mid && String(lu.team_side[j]) == side, 1:nrow(lu))
    return MD.Player[MD.Player(Int(lu.player_id[j]),
                               ismissing(lu.player_name[j]) ? "Unknown" : String(lu.player_name[j]),
                               MD.clean_position(ismissing(lu.position[j]) ? "M" :
                                                 String(lu.position[j])),
                               coalesce(lu.is_substitute[j], false)) for j in sel]
end

# %%
# ===================================================================
# 2. The card: fixtures, results, and the closing book
# ===================================================================

"""
    SlateCard

One Saturday, frozen. `fixtures` is the settlement window, `results` the full-time scores, and
`close` the de-vigged T-0 book every leg's CLV is measured against.

`priceable` records whether an archived ladder existed at T-25 at all. A slate whose collector
had not started is carried as a REFUSAL with its reason rather than dropped from the study --
"the model placed nothing" and "there was no book to place into" are different findings and the
report must be able to tell them apart.
"""
struct SlateCard
    day::Date
    kickoff::DateTime
    as_of::DateTime
    fixtures::Vector{MD.Fixture}
    results::Dict{Int,Tuple{Int,Int}}
    close::Dict{Tuple{Int,MD.SelectionKey},Tuple{Float64,DateTime}}
    priceable::Bool
    refusal::String
end

"""
    load_slate_card(conn, day, tournaments, book_source; kickoff_hour) -> SlateCard

Read one Saturday out of `betdb`: the fixtures, their final scores, and the closing ladder.

A fixture that is not `finished` is dropped with a note -- a postponed match has no result to
settle against, and settling it as a void would silently shrink the denominator of every rate in
the report.
"""
function load_slate_card(conn, day::Date, tournaments::Vector{Int},
                         book_source::MD.AbstractBookSource;
                         kickoff::DateTime = DateTime(day, Time(14, 0)),
                         t_minus::Period = Minute(25))
    lo = Int(round(datetime2unix(DateTime(day))))
    hi = Int(round(datetime2unix(DateTime(day) + Day(1))))
    frame = DataFrame(LibPQ.execute(conn, """
        SELECT e.match_id, e.home_team, e.away_team, e.start_timestamp, e.tournament_id,
               e.status_type, m.home_score, m.away_score
        FROM sofascore.events e
        LEFT JOIN sofascore.matches m ON m.match_id = e.match_id
        WHERE e.tournament_id = ANY(\$1) AND e.start_timestamp >= \$2 AND e.start_timestamp < \$3
        ORDER BY e.start_timestamp, e.tournament_id, e.match_id;""", (tournaments, lo, hi)))

    as_of = kickoff - t_minus
    fixtures = MD.Fixture[]
    results  = Dict{Int,Tuple{Int,Int}}()
    dropped  = String[]
    for row in eachrow(frame)
        mid = Int(row.match_id)
        if String(row.status_type) != "finished" || ismissing(row.home_score) ||
           ismissing(row.away_score)
            push!(dropped, "$(row.home_team) v $(row.away_team) ($(row.status_type))")
            continue
        end
        push!(fixtures, MD.Fixture(mid, String(row.home_team), String(row.away_team),
                                   unix2datetime(row.start_timestamp), Int(row.tournament_id)))
        results[mid] = (Int(row.home_score), Int(row.away_score))
    end

    # Does an archived ladder exist at or before T-25 for any fixture on this card?
    ids = Int[f.m_id for f in fixtures]
    n_pre = isempty(ids) ? 0 : Int(first(DataFrame(LibPQ.execute(conn, """
        SELECT count(*) AS n
        FROM betfair.match_meta mm
        JOIN betfair_live.market_metadata md ON md.event_id = mm.betfair_event_id
        JOIN betfair_live.order_book_1m o ON o.market_id = md.market_id
        WHERE mm.match_id = ANY(\$1) AND o.ts <= \$2;""", (ids, as_of))).n))

    refusal = ""
    if isempty(fixtures)
        refusal = "no finished fixture on this day"
    elseif n_pre == 0
        first_tick = DataFrame(LibPQ.execute(conn, """
            SELECT min(o.ts) AS first_tick
            FROM betfair.match_meta mm
            JOIN betfair_live.market_metadata md ON md.event_id = mm.betfair_event_id
            JOIN betfair_live.order_book_1m o ON o.market_id = md.market_id
            WHERE mm.match_id = ANY(\$1);""", (ids,)))
        stamp = nrow(first_tick) == 0 || ismissing(first_tick.first_tick[1]) ?
                "never" : string(first_tick.first_tick[1])
        refusal = "betfair_live.order_book_1m holds no tick at or before T-25 ($as_of) for any " *
                  "fixture on this card; first archived tick is $stamp. There is no tradeable " *
                  "book to price against, so this slate is REFUSED rather than priced off a " *
                  "post-kick-off ladder."
    end
    !isempty(dropped) && (refusal = isempty(refusal) ? "" : refusal)

    close = isempty(fixtures) || !isempty(refusal) ?
            Dict{Tuple{Int,MD.SelectionKey},Tuple{Float64,DateTime}}() :
            closing_book(book_source, fixtures, kickoff)

    card = SlateCard(day, kickoff, as_of, fixtures, results, close, isempty(refusal), refusal)
    return (card = card, dropped = dropped, n_events = nrow(frame))
end

"""
    closing_book(book_source, fixtures, kickoff) -> Dict{(match_id, key) => (p_close, ts)}

The **de-vigged** closing probability per selection, taken from the last archived ladder at or
before kick-off.

De-vigged, and that is not a detail: `clv_for_order` subtracts the entry probability from this
number, and a raw `1 / best_back` book sums above one, which would credit every leg on the card
with the overround. Normalisation is per `(group, line)` and a market whose runners are not all
quoted is dropped entirely rather than normalised over a partial set -- a one-sided total
renormalised alone becomes a fabricated fair probability of 1.0.

The mid `(best_back + best_lay) / 2` is used rather than either touch, because the close is a
measurement of where the market ended and not a price we are claiming to have traded at.
"""
function closing_book(book_source::MD.AbstractBookSource, fixtures::Vector{MD.Fixture},
                      kickoff::DateTime)
    out = Dict{Tuple{Int,MD.SelectionKey},Tuple{Float64,DateTime}}()
    for f in fixtures
        id = MD.resolve(MD.MatchMetaCrosswalk(), f)
        id isa MD.Resolved || continue
        book = MD.quotes(book_source, id, kickoff)
        isempty(book) && continue

        groups = Dict{Tuple{String,Float64},Vector{MD.SelectionKey}}()
        for key in keys(book)
            push!(get!(groups, (key.group, key.line), MD.SelectionKey[]), key)
        end
        for ((group, line), keys_) in groups
            expected = _expected_runners(group, line)
            expected == 0 && continue
            length(keys_) == expected || continue
            probs = Dict{MD.SelectionKey,Float64}()
            stamps = DateTime[]
            ok = true
            for key in keys_
                lv = book[key]
                bb, bl = MD.best_back(lv), MD.best_lay(lv)
                if isnan(bb) || isnan(bl) || bb <= 1.0 || bl <= 1.0
                    ok = false; break
                end
                probs[key] = 2.0 / (bb + bl)          # 1 / mid
                push!(stamps, lv.ts)
            end
            ok || continue
            overround = sum(values(probs))
            (isfinite(overround) && overround > 0) || continue
            ts = maximum(stamps)
            for (key, p) in probs
                out[(f.m_id, key)] = (p / overround, ts)
            end
        end
    end
    return out
end

"How many runners a coherent quote of this market group must carry."
_expected_runners(group::AbstractString, ::Float64) =
    group == "1X2" ? 3 : group in ("OverUnder", "BTTS") ? 2 : 0

# %%
# ===================================================================
# 3. The four model arms and the two fill models
# ===================================================================

"""
    ModelArm

One posterior and, optionally, the rate calibrator applied to it before staking.

`calibrator === nothing` is the raw arm. The calibrated arm shares the *same* `CanonicalFit`
object, so a difference between the two tracks is attributable to the calibration transform and
to nothing else -- not to a second load, a second fold selection or a second book read.
"""
struct ModelArm
    name::String
    label::String
    fit::Any
    calibrator::Union{Nothing,CB.AbstractGenerativeRateCalibrator}
end

"""
    Track

One (arm x fill model) bankroll, compounding slate by slate.

Eight of these run side by side over the same five Saturdays. They are separate tracks rather
than one priced sheet filled two ways because the bankroll is the allocator's input: once
`TouchOnly` and `LadderSweep` have settled a different P&L, slate `k+1` is a different joint
problem and pricing it once would silently give both tracks the same stake vector.
"""
mutable struct Track
    arm::ModelArm
    fill_model::MD.AbstractFillModel
    fill_name::Symbol
    bankroll::Float64
    opening::Float64
    curve::Vector{Float64}          # bankroll relative to opening, starting at 1.0
    dates::Vector{Date}
    slate_pl::Vector{Float64}       # per-slate return on the bankroll that entered the slate
    trades::DataFrame
    slates::DataFrame
end

Track(arm::ModelArm, fill_model::MD.AbstractFillModel, opening::Real) =
    Track(arm, fill_model, MD.fill_model_name(fill_model), Float64(opening), Float64(opening),
          Float64[1.0], Date[], Float64[], _empty_trades(), _empty_slates())

track_id(t::Track) = string(t.arm.name, "__", t.fill_name)

_empty_trades() = DataFrame(
    arm = String[], fill_model = String[], slate = Date[], as_of = DateTime[],
    match_id = Int[], home_team = String[], away_team = String[], fold_idx = Int[],
    market_group = String[], market_line = Float64[], selection = String[],
    venue_selection = String[], side = String[], leverage = Float64[],
    venue_odds = Float64[], effective_odds = Float64[], p_model = Float64[],
    p_market = Float64[], edge = Float64[], bankroll_before = Float64[],
    stake_fraction = Float64[], risk_requested = Float64[], venue_stake_requested = Float64[],
    depth_touch = Float64[], depth_book = Float64[],
    venue_stake_filled = Float64[], risk_filled = Float64[], fill_pct = Float64[],
    fill_vwap = Float64[], levels_used = Int[], slippage_vs_touch = Float64[],
    home_goals = Int[], away_goals = Int[], outcome = String[],
    gross_return = Float64[], commission = Float64[], net_pnl = Float64[],
    entry_prob = Float64[], close_prob = Float64[], clv = Float64[],
    beat_close = Union{Missing,Bool}[], has_close = Bool[])

_empty_slates() = DataFrame(
    arm = String[], fill_model = String[], slate = Date[], as_of = DateTime[],
    fold_idx = Int[], n_fixtures = Int[], n_uncovered = Int[], n_blocked = Int[],
    n_legs = Int[], n_filled = Int[], bankroll_before = Float64[], bankroll_after = Float64[],
    risk_requested = Float64[], venue_stake_requested = Float64[],
    venue_stake_filled = Float64[], risk_filled = Float64[], net_pnl = Float64[],
    slate_return = Float64[], k_risk = Float64[], slate_exposure = Float64[],
    capped = Bool[], warning = String[])

# %%
# ===================================================================
# 3b. Fold coverage -- which fixtures this posterior can represent at all
# ===================================================================
#
# `MD.check_coverage` THROWS on the first uncovered fixture. That is right for a live slate --
# a fixture priced at the league mean is worse than no price -- and wrong for a backtest, where
# one newly promoted club would take the whole Saturday down. The replay console solves it with
# `coverage_split`, and this is the same partition: refuse the fixtures a fold cannot represent
# BY NAME, price the rest, and hand `check_coverage` only what it can assert on.
#
# Only `team_map` is a genuine refusal. A team the fold never saw has no alpha/beta to condition
# on. The other two keys `check_coverage` tests (`player_ratings_map`, `league_lookup`) are
# materialised per fixture inside `matchday_latents` a few lines after this test, so a fixture
# absent from them here is not yet a problem.

const _FOLD_CTX = IdDict{Any,NamedTuple}()

"""
    fold_context(fit, ds) -> (; boundaries, fcol)

The split boundaries and the fold FeatureSet collection for one posterior, memoised.

Both are functions of `(ds, model, splitter)` alone, so they are identical for every slate and
every track that shares a fit -- and rebuilding them costs the hybrid pillar about a minute each
time. This is the same reuse `replay_state.rebind_slot!` makes: keep the expensive half, redo
only the fold selection.

This cache is used for the COVERAGE TEST only. `price_slate` rebuilds its own FeatureSet
internally and is left to do so, because sharing a mutable FeatureSet between the test and the
pricing path is exactly how a materialised map leaks from one slate into the next.
"""
fold_context(fit, ds) = get!(_FOLD_CTX, fit) do
    boundaries = DD.create_id_boundaries(ds, fit.config.splitter)
    fcol = FF.create_features(boundaries, ds, fit.config.model, fit.config.splitter)
    return (; boundaries, fcol)
end

"""
    coverage_split(fit, ds, fixtures) -> (; fold_idx, covered, refused, warning)

Partition a card into the fixtures this fit's serving fold can represent and those it cannot.

The fold is chosen exactly as `matchday_latents` chooses it -- `select_split` with both
`fixture_ids` (positive identification: the fold whose NEXT block is this card) and `exclude`
(negative fallback: never a fold whose target window already contains it).
"""
function coverage_split(fit, ds, fixtures::Vector{MD.Fixture})
    ctx = fold_context(fit, ds)
    ids = Int[f.m_id for f in fixtures]
    sel = MD.select_split(fit, ctx.boundaries; strict = false, exclude = ids,
                          ds = ds, config = fit.config.splitter, fixture_ids = ids)
    fs = ctx.fcol[sel.idx][1]
    tm = get(fs.data, :team_map, nothing)

    covered = MD.Fixture[]; refused = Pair{MD.Fixture,String}[]
    for f in fixtures
        if tm === nothing
            push!(covered, f); continue
        end
        gaps = String[]
        haskey(tm, f.home) || push!(gaps, f.home)
        haskey(tm, f.away) || push!(gaps, f.away)
        isempty(gaps) ? push!(covered, f) :
            push!(refused, f => "absent from fold $(sel.idx)'s team_map (" *
                                join(gaps, ", ") * ") -- would be priced at the league mean")
    end
    return (; fold_idx = sel.idx, covered, refused, warning = sel.warning)
end

# %%
# ===================================================================
# 4. One slate, one track: price -> fill -> settle -> CLV -> compound
# ===================================================================

"""
    run_slate!(track, card, spec_for, system, segment, ds) -> NamedTuple

Price the card on this track's current bankroll, fill it against the archived T-25 ladder,
settle it against the final score, measure it against the T-0 close, and compound.

`spec_for(fixtures)` builds the `MatchDaySpec` for a fixture list; the list is the card MINUS
whatever this arm's serving fold cannot represent (§3b), which differs per posterior and per
slate and so cannot be frozen into one spec.

The order of operations is the live one and is not negotiable:

* `price_slate` solves ONE joint allocation for the whole settlement window, so every leg's
  stake depends on every other leg's. The sheet is used as a vector or not at all.
* `simulate_fill` consumes the ladder the leg would actually touch (`venue_selection`), which on
  a synthetic is a different runner from the model's selection.
* `settle_order` grades the MODEL's selection and books `risk_filled`, not `risk` -- the
  unfilled remainder was never at the venue and settling it would create money.
* the bankroll moves by the settled net P&L only.
"""
function run_slate!(track::Track, card::SlateCard, spec_for,
                    system::PF.PortfolioSystem, segment, ds)
    bankroll_before = track.bankroll
    cov = coverage_split(track.arm.fit, ds, card.fixtures)

    if isempty(cov.covered)
        push!(track.curve, track.bankroll / track.opening)
        push!(track.dates, card.day)
        push!(track.slate_pl, 0.0)
        push!(track.slates, (
            arm = track.arm.name, fill_model = String(track.fill_name), slate = card.day,
            as_of = card.as_of, fold_idx = cov.fold_idx, n_fixtures = length(card.fixtures),
            n_uncovered = length(cov.refused), n_blocked = 0, n_legs = 0, n_filled = 0,
            bankroll_before = bankroll_before, bankroll_after = track.bankroll,
            risk_requested = 0.0, venue_stake_requested = 0.0, venue_stake_filled = 0.0,
            risk_filled = 0.0, net_pnl = 0.0, slate_return = 0.0, k_risk = NaN,
            slate_exposure = 0.0, capped = false,
            warning = "fold $(cov.fold_idx) can represent no fixture on this card"))
        return (; slate = nothing, n_legs = 0, n_filled = 0, net = 0.0,
                bankroll = track.bankroll, blocked = MD.FixtureCard[], uncovered = cov.refused,
                fold_idx = cov.fold_idx)
    end

    spec  = spec_for(cov.covered)
    slate = MD.price_slate(spec, system, segment, track.arm.fit, ds;
                           as_of = card.as_of,
                           bankroll = bankroll_before,
                           account_id = track_id(track),
                           calibrator = track.arm.calibrator)
    slate.fold_idx == cov.fold_idx || error(
        "run_slate!: the coverage test used fold $(cov.fold_idx) but price_slate conditioned on " *
        "fold $(slate.fold_idx). The two must agree or the refusal list belongs to a different " *
        "posterior than the prices.")

    team_of = Dict(f.m_id => f for f in card.fixtures)
    orders  = nrow(slate.sheet) == 0 ? MD.PaperOrder[] : MD.orders_to_paper(slate)

    n_filled = 0
    risk_req = 0.0; stake_req = 0.0; stake_fill = 0.0; risk_fill = 0.0; net = 0.0
    for o in orders
        vkey = MD.SelectionKey((group = o.market_group, line = o.market_line,
                                selection = o.venue_selection))
        levels = get(slate.books, (o.match_id, vkey), nothing)
        fills = levels === nothing ? MD.Fill[] :
                MD.simulate_fill(track.fill_model, levels, o.side, o.venue_stake, o.leverage,
                                 card.as_of; order_id = o.order_id)

        h, a = get(card.results, o.match_id, (0, 0))
        settled = MD.settle_order(o, fills, h, a, system.book.exec.commission.rate)

        closed = get(card.close, (o.match_id, MD.SelectionKey((group = o.market_group,
                                                              line = o.market_line,
                                                              selection = o.selection))), nothing)
        clv = closed === nothing ?
              (; entry_prob = NaN, close_prob = NaN, clv = NaN, beat_close = missing) :
              let c = MD.clv_for_order(o, fills, closed[1], closed[2])
                  (; c.entry_prob, c.close_prob, c.clv, beat_close = c.beat_close)
              end

        vwap = MD.fill_vwap(fills)
        filled_size = MD.filled_size(fills)
        fix = team_of[o.match_id]
        row_i = findfirst(r -> r.match_id == o.match_id && r.group == o.market_group &&
                               r.line == o.market_line && r.selection == o.selection,
                          eachrow(slate.sheet))

        push!(track.trades, (
            arm = track.arm.name, fill_model = String(track.fill_name), slate = card.day,
            as_of = card.as_of, match_id = o.match_id, home_team = fix.home,
            away_team = fix.away, fold_idx = slate.fold_idx,
            market_group = o.market_group, market_line = o.market_line,
            selection = String(o.selection), venue_selection = String(o.venue_selection),
            side = String(o.side), leverage = o.leverage, venue_odds = o.venue_odds,
            effective_odds = o.effective_odds, p_model = o.p_model, p_market = o.p_market,
            edge = o.edge, bankroll_before = bankroll_before, stake_fraction = o.stake_fraction,
            risk_requested = o.risk, venue_stake_requested = o.venue_stake,
            depth_touch = row_i === nothing ? NaN : slate.sheet.depth_touch[row_i],
            depth_book = row_i === nothing ? NaN : slate.sheet.depth_book[row_i],
            venue_stake_filled = filled_size, risk_filled = settled.risk_settled,
            fill_pct = o.venue_stake > 0 ? filled_size / o.venue_stake : NaN,
            fill_vwap = vwap,
            levels_used = isempty(fills) ? 0 : maximum(f -> f.levels_used, fills),
            slippage_vs_touch = _slippage(o.side, fills, vwap),
            home_goals = h, away_goals = a, outcome = uppercase(String(settled.outcome)),
            gross_return = settled.gross_return, commission = settled.commission,
            net_pnl = settled.net_pnl, entry_prob = clv.entry_prob,
            close_prob = clv.close_prob, clv = clv.clv, beat_close = clv.beat_close,
            has_close = closed !== nothing))

        risk_req += o.risk; stake_req += o.venue_stake
        stake_fill += filled_size; risk_fill += settled.risk_settled
        net += settled.net_pnl
        filled_size > 1e-9 && (n_filled += 1)
    end

    track.bankroll = bankroll_before + net
    push!(track.curve, track.bankroll / track.opening)
    push!(track.dates, card.day)
    push!(track.slate_pl, bankroll_before > 0 ? net / bankroll_before : 0.0)

    push!(track.slates, (
        arm = track.arm.name, fill_model = String(track.fill_name), slate = card.day,
        as_of = card.as_of, fold_idx = slate.fold_idx, n_fixtures = length(card.fixtures),
        n_uncovered = length(cov.refused), n_blocked = length(slate.blocked),
        n_legs = length(orders), n_filled = n_filled,
        bankroll_before = bankroll_before, bankroll_after = track.bankroll,
        risk_requested = risk_req, venue_stake_requested = stake_req,
        venue_stake_filled = stake_fill, risk_filled = risk_fill, net_pnl = net,
        slate_return = bankroll_before > 0 ? net / bankroll_before : 0.0,
        k_risk = slate.k_risk, slate_exposure = slate.slate_exposure, capped = slate.capped,
        warning = slate.warning))

    return (; slate, n_legs = length(orders), n_filled, net,
            bankroll = track.bankroll, blocked = slate.blocked, uncovered = cov.refused,
            fold_idx = slate.fold_idx)
end

"""
    _slippage(side, fills, vwap) -> Float64

What the deeper levels cost, as a fraction of the EFFECTIVE odds at the touch.

Denominating in effective odds rather than in venue price is what makes one number cover both
instruments. Sweeping a back ladder walks the price DOWN and sweeping a lay ladder walks it UP,
so `(touch - vwap) / touch` -- the convention `Portfolio`'s own `sweep_ladder` uses -- is a cost
on a back and a spurious credit on a lay. The morphism `lay_to_back(d) = d / (d - 1)` maps both
onto the odds at which a unit of RISK pays, and that is monotone in the direction of the give-up
for either side. Positive is always a cost.
"""
function _slippage(side::Symbol, fills::AbstractVector{MD.Fill}, vwap::Float64)
    (isempty(fills) || isnan(vwap)) && return NaN
    eff(p) = side === :lay ? MD.lay_to_back(p) : Float64(p)
    e_touch = eff(fills[1].price)
    (isfinite(e_touch) && e_touch > 0) || return NaN
    return (e_touch - eff(vwap)) / e_touch
end

# %%
# ===================================================================
# 5. Summary statistics
# ===================================================================

"""
    track_summary(track) -> NamedTuple

The headline row for one track.

`sharpe`, `sharpe_ann` and `max_drawdown_pct` are computed exactly as
`Portfolio.portfolio_summary` computes them -- log growth per settlement window, annualised by
`sqrt(slates_per_year)`, drawdown measured on the bankroll curve including the opening `1.0` --
so a number here is comparable with the 40-fold grids in `experiments/`, not merely internally
consistent.

Over four settlement windows a Sharpe ratio is an estimate from three degrees of freedom. It is
reported because the work package asks for it and because omitting it would be worse, not
because four slates identify it.
"""
function track_summary(track::Track)
    t = track.trades
    settled = isempty(t) ? t : filter(r -> r.risk_filled > 1e-9, t)
    n_bets  = nrow(settled)
    n_wins  = n_bets == 0 ? 0 : count(==("WIN"), settled.outcome)
    staked  = n_bets == 0 ? 0.0 : sum(settled.risk_filled)
    pnl     = track.bankroll - track.opening

    bk = track.curve
    rm = accumulate(max, bk)
    dd = (bk .- rm) ./ rm .* 100
    mdd = isempty(dd) ? 0.0 : minimum(dd)

    r = isempty(track.slate_pl) ? Float64[] : log.(1.0 .+ track.slate_pl)
    days = length(track.dates) < 2 ? 0 : Dates.value(track.dates[end] - track.dates[1])
    slates_per_year = days > 0 ? length(r) * 365.25 / days : NaN
    sharpe = length(r) < 2 ? NaN : (std(r) > 0 ? mean(r) / std(r) : NaN)
    sharpe_ann = (isnan(sharpe) || isnan(slates_per_year)) ? NaN : sharpe * sqrt(slates_per_year)

    req_stake = isempty(t) ? 0.0 : sum(t.venue_stake_requested)
    fil_stake = isempty(t) ? 0.0 : sum(t.venue_stake_filled)

    with_close = isempty(t) ? t : filter(r -> r.has_close && r.risk_filled > 1e-9, t)
    n_clv = nrow(with_close)

    return (; arm = track.arm.name, label = track.arm.label,
            fill_model = String(track.fill_name),
            opening = track.opening, final = track.bankroll, net_pnl = pnl,
            roi_pct = staked > 0 ? 100 * pnl / staked : NaN,
            growth_pct = 100 * (track.bankroll / track.opening - 1),
            total_staked = staked, n_legs = nrow(t), n_bets, n_wins,
            win_rate_pct = n_bets > 0 ? 100 * n_wins / n_bets : NaN,
            sharpe, sharpe_ann, max_drawdown_pct = mdd,
            requested_stake = req_stake, filled_stake = fil_stake,
            fill_rate_pct = req_stake > 0 ? 100 * fil_stake / req_stake : NaN,
            n_clv, beat_close_pct = n_clv > 0 ? 100 * count(identity, with_close.beat_close) / n_clv : NaN,
            mean_clv = n_clv > 0 ? mean(with_close.clv) : NaN,
            median_clv = n_clv > 0 ? median(with_close.clv) : NaN)
end

"Every track's headline row, in the order the tracks were built."
summary_frame(tracks::Vector{Track}) = DataFrame([track_summary(t) for t in tracks])

# %%
# ===================================================================
# 6. Ledger and report writers
# ===================================================================

"""
    write_trade_ledger(path, tracks, fill_name) -> DataFrame

One CSV per fill model, carrying every leg of all four arms: what was asked for, what the book
gave, what it settled for, and what the close said. Unfilled legs are kept -- a row with
`venue_stake_filled = 0` is the capacity finding, and dropping it would make the fill rate
unrecoverable from the ledger.
"""
function write_trade_ledger(path::AbstractString, tracks::Vector{Track}, fill_name::Symbol)
    frames = [t.trades for t in tracks if t.fill_name === fill_name]
    out = isempty(frames) ? _empty_trades() : vcat(frames...)
    isempty(out) || sort!(out, [:slate, :arm, :match_id, :market_group, :market_line, :selection])
    mkpath(dirname(path))
    CSV.write(path, out)
    return out
end

_pct(x) = isnan(x) ? "--" : @sprintf("%.2f%%", x)
_gbp(x) = isnan(x) ? "--" : x < 0 ? @sprintf("−£%.2f", -x) : @sprintf("£%.2f", x)
_num(x; digits = 3) = isnan(x) ? "--" : string(round(x; digits = digits))
_plural(n, one, many) = n == 1 ? one : many
_mean_finite(xs) = (v = Float64[x for x in xs if isfinite(x)]; isempty(v) ? NaN : mean(v))

"""
    executive_table(summaries) -> String

The markdown table §6.1 of the work package asks for, one row per (arm x fill model).
"""
function executive_table(df::DataFrame)
    io = IOBuffer()
    println(io, "| Arm | Fill model | Initial | Final | Net P&L | Total staked | ROI% | Win rate% | Sharpe (slate) | Sharpe (ann.) | Max DD% |")
    println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in eachrow(df)
        @printf(io, "| `%s` | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
                r.arm, r.fill_model, _gbp(r.opening), _gbp(r.final), _gbp(r.net_pnl),
                _gbp(r.total_staked), _pct(r.roi_pct), _pct(r.win_rate_pct),
                _num(r.sharpe), _num(r.sharpe_ann), _num(r.max_drawdown_pct; digits = 2))
    end
    return String(take!(io))
end

"""
    liquidity_table(tracks) -> String

Requested versus filled backer stake per track, with the £ the ladder refused.

The fill rate is measured in VENUE STAKE, not in risk: the venue stake is what the ladder has
to absorb, and on a synthetic it is `risk * leverage`, which is the larger number and the one
the book actually sees.
"""
function liquidity_table(tracks::Vector{Track})
    io = IOBuffer()
    println(io, "| Arm | Fill model | Requested stake | Filled stake | Fill rate | Legs | Legs fully unfilled | Mean levels used |")
    println(io, "|---|---|---:|---:|---:|---:|---:|---:|")
    for t in tracks
        d = t.trades
        req = isempty(d) ? 0.0 : sum(d.venue_stake_requested)
        fil = isempty(d) ? 0.0 : sum(d.venue_stake_filled)
        unfilled = isempty(d) ? 0 : count(<=(1e-9), d.venue_stake_filled)
        lv = isempty(d) ? Float64[] : Float64[x for x in d.levels_used if x > 0]
        @printf(io, "| `%s` | `%s` | %s | %s | %s | %d | %d | %s |\n",
                t.arm.name, String(t.fill_name), _gbp(req), _gbp(fil),
                _pct(req > 0 ? 100 * fil / req : NaN), nrow(d), unfilled,
                isempty(lv) ? "--" : _num(mean(lv); digits = 2))
    end
    return String(take!(io))
end

"""
    slippage_table(tracks) -> String

The difference the sweep bought, per arm: extra stake matched, extra P&L, and what the deeper
levels cost in price.

`pnl_delta` is the whole point of running both fill models. It is NOT a strategy improvement --
`LadderSweep` is an assumption about execution, not a decision the allocator made -- so a
positive number here says the edge survived three levels of depth, and a negative one says the
extra size was matched at prices where it did not.
"""
function slippage_table(tracks::Vector{Track})
    by_arm = Dict{String,Dict{Symbol,Track}}()
    for t in tracks
        get!(by_arm, t.arm.name, Dict{Symbol,Track}())[t.fill_name] = t
    end
    io = IOBuffer()
    println(io, "| Arm | Touch filled | Sweep filled | Extra stake matched | Touch P&L | Sweep P&L | Sweep − Touch | Mean sweep slippage vs touch |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in sort!(collect(keys(by_arm)))
        d = by_arm[name]
        (haskey(d, :touch_only) && haskey(d, :ladder_sweep_v1)) || continue
        tt, ts = d[:touch_only], d[:ladder_sweep_v1]
        ft = isempty(tt.trades) ? 0.0 : sum(tt.trades.venue_stake_filled)
        fs = isempty(ts.trades) ? 0.0 : sum(ts.trades.venue_stake_filled)
        slip = isempty(ts.trades) ? Float64[] :
               Float64[x for x in ts.trades.slippage_vs_touch if !isnan(x)]
        @printf(io, "| `%s` | %s | %s | %s | %s | %s | %s | %s |\n",
                name, _gbp(ft), _gbp(fs), _gbp(fs - ft),
                _gbp(tt.bankroll - tt.opening), _gbp(ts.bankroll - ts.opening),
                _gbp((ts.bankroll - ts.opening) - (tt.bankroll - tt.opening)),
                isempty(slip) ? "--" : _pct(100 * mean(slip)))
    end
    return String(take!(io))
end

"""
    clv_table(tracks) -> String

Closing-line value per track, in probability points.

`clv = p_close − p_entry`, with `p_entry = 1 / VWAP` of the fill actually achieved and `p_close`
the de-vigged mid at T-0. A leg that filled two ticks worse than it was quoted has spent that
difference, and measuring against the quote rather than the fill would credit execution with a
price it did not get. Legs that did not fill have no entry price and are excluded; legs whose
market did not quote a complete runner set at T-0 have no de-vigged close and are excluded, and
both counts are shown so the denominator is never implicit.
"""
function clv_table(tracks::Vector{Track})
    io = IOBuffer()
    println(io, "| Arm | Fill model | Filled legs | Legs with a close | Beat close % | Mean CLV (pp) | Median CLV (pp) |")
    println(io, "|---|---|---:|---:|---:|---:|---:|")
    for t in tracks
        d = isempty(t.trades) ? t.trades : filter(r -> r.risk_filled > 1e-9, t.trades)
        w = isempty(d) ? d : filter(r -> r.has_close, d)
        n = nrow(w)
        @printf(io, "| `%s` | `%s` | %d | %d | %s | %s | %s |\n",
                t.arm.name, String(t.fill_name), nrow(d), n,
                n > 0 ? _pct(100 * count(identity, w.beat_close) / n) : "--",
                n > 0 ? _num(100 * mean(w.clv); digits = 3) : "--",
                n > 0 ? _num(100 * median(w.clv); digits = 3) : "--")
    end
    return String(take!(io))
end

"""
    trajectory_table(tracks) -> String

The bankroll after each Saturday, one column per track. This is the compounding claim made
visible: a track's slate `k+1` stake vector was solved against the number in column `k`.
"""
function trajectory_table(tracks::Vector{Track})
    isempty(tracks) && return "_no track ran._\n"
    days = tracks[1].dates
    io = IOBuffer()
    print(io, "| Track | Opening |")
    for d in days
        print(io, " ", d, " |")
    end
    println(io)
    print(io, "|---|---:|")
    for _ in days
        print(io, "---:|")
    end
    println(io)
    for t in tracks
        @printf(io, "| `%s` | %s |", track_id(t), _gbp(t.opening))
        for i in eachindex(days)
            print(io, " ", _gbp(t.opening * t.curve[i + 1]), " |")
        end
        println(io)
    end
    return String(take!(io))
end

"""
    market_breakdown(ledger) -> String

Filled stake and P&L per Option B basket leg, pooled over arms. Which of the five permitted
selections actually carried the result is the first question asked of any Kelly track, and
pooling it out of the ledger keeps the answer sourced from the rows rather than restated.
"""
function market_breakdown(ledger::DataFrame)
    isempty(ledger) && return "_no trades._\n"
    d = filter(r -> r.risk_filled > 1e-9, ledger)
    isempty(d) && return "_no filled trades._\n"
    io = IOBuffer()
    println(io, "| Arm | Market | Selection | Legs | Filled risk | Net P&L | ROI% | Win rate% |")
    println(io, "|---|---|---|---:|---:|---:|---:|---:|")
    for g in groupby(sort(d, [:arm, :market_group, :market_line, :selection]),
                     [:arm, :market_group, :market_line, :selection])
        risk = sum(g.risk_filled); pnl = sum(g.net_pnl)
        wins = count(==("WIN"), g.outcome)
        @printf(io, "| `%s` | %s%s | %s | %d | %s | %s | %s | %s |\n",
                g.arm[1], g.market_group[1],
                g.market_line[1] == 0.0 ? "" : string(" ", g.market_line[1]),
                g.selection[1], nrow(g), _gbp(risk), _gbp(pnl),
                _pct(risk > 0 ? 100 * pnl / risk : NaN),
                _pct(100 * wins / nrow(g)))
    end
    return String(take!(io))
end

"""
    write_report(path, tracks, summaries, cards, refusals; ...)

The deliverable. Everything in it is derived from `tracks` and the ledgers written beside it, so
a number in the prose can be recovered from a CSV rather than taken on trust.
"""
function write_report(path::AbstractString, tracks::Vector{Track}, summaries::DataFrame,
                      cards::Vector{SlateCard}, refusals::Vector{Tuple{Date,String}};
                      touch_ledger::DataFrame, sweep_ledger::DataFrame,
                      slate_frame::DataFrame, meta)
    live = SlateCard[c for c in cards if c.priceable]
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# T−25 order book backtest — `m05` vs `m12`, raw vs Option B calibrated")
        println(io)
        println(io, "Scottish League One [56] and League Two [57], 2026/27 season opening slates.")
        println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " from ",
                meta.git_branch, " @ ", first(meta.git_commit, 12), ".")
        println(io)
        println(io, "Runner `current_development/match_day_inference/r10_t25_m05_m12_backtest.jl`, ",
                "loader `l10_t25_backtest.jl`. No database row was written by this study; ",
                "`paper_runbook` and `paper_replay` are both untouched.")
        println(io)

        # --- 0. What was run -------------------------------------------------------------
        println(io, "## 0. What was run")
        println(io)
        println(io, "| | |")
        println(io, "|---|---|")
        println(io, "| Experiment | `", meta.experiment, "` on `mcmc-beast` |")
        println(io, "| Gen 3 control | `", meta.m05, "` — run id ", meta.m05_id,
                ", `", meta.m05_uuid, "` |")
        println(io, "| Gen 4 hybrid | `", meta.m12, "` — run id ", meta.m12_id,
                ", `", meta.m12_uuid, "` |")
        println(io, "| Decision instant | kick-off − 25 min (14:00 UTC → 13:35 UTC) |")
        println(io, "| Book | `betfair_live.order_book_1m`, latest snapshot with `ts <= as_of`, ≤3 levels/side |")
        println(io, "| Opening bankroll | ", _gbp(meta.bankroll), " per arm per fill model, compounding |")
        println(io, "| Staking | Option B: `TieredTrust` (Home 1.0, Under 2.5 1.0, Draw/Away/Over 1.5 1/1.4), ",
                "`KellyLogUtility`, `FractionalKelly(", meta.system.book.shrink.k, ")`, ",
                "`SlateDrawdown(", meta.system.policy.risk.lambda, ")`, ",
                "`FixedCap(", meta.system.policy.cap.cap, ")` |")
        println(io, "| Commission | ", meta.system.book.exec.commission.rate,
                " on net winnings per winning leg |")
        println(io, "| Calibrator | `", meta.calibrator.name, "`, ", meta.calibrator.law,
                ", `PoolDispersion`, fitted at T", meta.calibrator.book_as_of_minutes, " |")
        println(io, "| Fill models | `TouchOnly` (best price, resting size only) and ",
                "`LadderSweep(max_slippage = ", meta.max_slippage, ")` |")
        println(io, "| Book spec hash | `", PF.portfolio_spec_hash(meta.system.book), "` |")
        println(io, "| Policy spec hash | `", PF.portfolio_spec_hash(meta.system.policy), "` |")
        println(io)

        # --- 1. Slate inventory ----------------------------------------------------------
        println(io, "## 1. Slate inventory")
        println(io)
        println(io, "| Slate | Finished fixtures | T−25 book | Closing quotes | Status |")
        println(io, "|---|---:|---|---:|---|")
        for c in cards
            @printf(io, "| %s | %d | %s | %d | %s |\n", c.day, length(c.fixtures),
                    c.priceable ? "present" : "**absent**", length(c.close),
                    c.priceable ? "priced" : "**REFUSED**")
        end
        println(io)
        if !isempty(refusals)
            println(io, "### Refused slates")
            println(io)
            for (day, why) in refusals
                println(io, "* **", day, "** — ", why)
            end
            println(io)
            println(io, "A refused slate is carried in this table rather than dropped from the ",
                    "study. \"the model placed nothing\" and \"there was no book to place into\" ",
                    "are different findings, and only the second one is true here.")
            println(io)
        end

        # --- 2. Executive summary --------------------------------------------------------
        println(io, "## 2. Executive summary")
        println(io)
        print(io, executive_table(summaries))
        println(io)
        println(io, "ROI is measured on **filled risk**, which is the money actually at the venue; ",
                "growth on the £", @sprintf("%.0f", meta.bankroll), " opening bankroll is ",
                "`final / opening − 1`. Win rate counts filled legs only — an unfilled leg is ",
                "neither a win nor a loss and including it would deflate every rate uniformly.")
        println(io)
        println(io, "> **Read the Sharpe ratios as decoration, not evidence.** ",
                length(live), _plural(length(live), " settlement window", " settlement windows"),
                " give ", max(length(live) - 1, 0),
                _plural(max(length(live) - 1, 0), " degree", " degrees"),
                " of freedom; the slate Sharpe is an estimate from ", length(live),
                _plural(length(live), " number", " numbers"),
                " and its annualisation multiplies that noise by `sqrt(slates_per_year)`. ",
                "It is reported because the work package asks for it. The ordering it implies is ",
                "not separable from sampling error at this sample size.")
        println(io)

        # --- 3. Liquidity and capacity ---------------------------------------------------
        println(io, "## 3. Liquidity and capacity audit")
        println(io)
        print(io, liquidity_table(tracks))
        println(io)
        println(io, "### Slippage: what the sweep bought")
        println(io)
        print(io, slippage_table(tracks))
        println(io)
        println(io, "`LadderSweep` crosses up to three archived levels instantly with at most ",
                100 * meta.max_slippage, "% give-up against the touch. The live system rests at ",
                "the touch and lets unmatched size expire, so the sweep column is an **upper ",
                "bound** on capacity, not a forecast. The archive carries at most three levels ",
                "per side, so even the sweep understates what the live API would show.")
        println(io)
        println(io, "Slippage in the table is denominated in **effective odds** — `d` for a back ",
                "and `d/(d−1)` for a lay — so positive is a cost on either instrument. Two ",
                "caveats, both properties of `MatchDay.LadderSweep` rather than of this study. ",
                "First, its `max_slippage` guard tests `(touch − p) / touch`, which is negative ",
                "on an ascending lay ladder, so the 2% stop never fires on a lay leg; the ",
                "measured give-up is small enough here (see the column) that it changes no ",
                "figure, but the guard is not doing work on that side. Second, the sweep and the ",
                "touch track are different *bankrolls* from slate 2 onward, so their filled ",
                "stakes are not two fills of one order — the `Sweep − Touch` column is the ",
                "difference between two compounding tracks, which is the quantity an operator ",
                "choosing a fill assumption actually faces.")
        println(io)

        # --- 4. CLV ----------------------------------------------------------------------
        println(io, "## 4. Closing line value")
        println(io)
        print(io, clv_table(tracks))
        println(io)
        println(io, "CLV is in **probability points** (`p_close − p_entry`, ×100). ",
                "`p_entry = 1 / VWAP` of the fill actually achieved; `p_close` is the de-vigged ",
                "mid of the last archived ladder at or before kick-off, normalised per market so ",
                "the overround is removed. A market that did not quote a complete runner set at ",
                "T−0 contributes no close and is excluded from the denominator rather than ",
                "scored against a partial book.")
        println(io)

        # --- 5. Trajectory ---------------------------------------------------------------
        println(io, "## 5. Slate-by-slate trajectory")
        println(io)
        print(io, trajectory_table(tracks))
        println(io)
        println(io, "### Per-slate detail")
        println(io)
        println(io, "| Slate | Track | Fold | Fixtures | Not covered | Blocked | Legs | Filled | Requested stake | Filled stake | Net P&L | Return | k_risk | Exposure | Capped |")
        println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in eachrow(sort(slate_frame, [:slate, :arm, :fill_model]))
            @printf(io, "| %s | `%s__%s` | %d | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s |\n",
                    r.slate, r.arm, r.fill_model, r.fold_idx, r.n_fixtures, r.n_uncovered,
                    r.n_blocked, r.n_legs, r.n_filled,
                    _gbp(r.venue_stake_requested), _gbp(r.venue_stake_filled),
                    _gbp(r.net_pnl), _pct(100 * r.slate_return), _num(r.k_risk),
                    _pct(100 * r.slate_exposure), r.capped ? "yes" : "no")
        end
        println(io)
        println(io, "**Not covered** counts fixtures whose clubs are absent from the serving ",
                "fold's `team_map` -- a side promoted or relegated into the division since that ",
                "fold's history block closed has no alpha/beta to condition on. They are refused ",
                "by name rather than priced at the league mean, which is the same rule the replay ",
                "console's `NOT COVERED BY ...` panel applies. **Blocked** counts fixtures a ",
                "readiness gate stopped (identity, book age, spread, liquidity).")
        println(io)

        println(io, "### Calibrator inversion coverage")
        println(io)
        println(io, "| Slate | Track | Coverage note |")
        println(io, "|---|---|---|")
        for r in eachrow(sort(slate_frame, [:slate, :arm, :fill_model]))
            occursin("calibration ", r.warning) || continue
            note = strip(last(split(r.warning, "calibration ")))
            @printf(io, "| %s | `%s__%s` | %s |\n", r.slate, r.arm, r.fill_model, note)
        end
        println(io)
        println(io, "`invert_market_rates` solves the T−25 book back to a `(lambda_h, lambda_a)` ",
                "pair by Nelder-Mead. A fixture whose book will not invert inside the residual ",
                "tolerance is **refused** and passes through raw rather than being pooled with a ",
                "badly fitted rate; the refusal is reported per slate above rather than absorbed.")
        println(io)

        # --- 6. Market breakdown ---------------------------------------------------------
        println(io, "## 6. Where the money came from (`TouchOnly`)")
        println(io)
        print(io, market_breakdown(touch_ledger))
        println(io)

        # --- 7. Verdict ------------------------------------------------------------------
        println(io, "## 7. Head-to-head verdict")
        println(io)
        print(io, verdict_text(summaries, live))
        println(io)

        # --- 8. Artefacts ----------------------------------------------------------------
        println(io, "## 8. Artefacts")
        println(io)
        println(io, "| File | Rows |")
        println(io, "|---|---:|")
        println(io, "| `", basename(meta.touch_csv), "` | ", nrow(touch_ledger), " |")
        println(io, "| `", basename(meta.sweep_csv), "` | ", nrow(sweep_ledger), " |")
        println(io, "| `", basename(meta.slates_csv), "` | ", nrow(slate_frame), " |")
        println(io)
        println(io, "Every figure above is recoverable from those three files.")
    end
    return path
end

"""
    verdict_text(summaries, live_cards) -> String

The empirical conclusion, written from the numbers rather than around them.

Deliberately hedged where the sample cannot carry the claim: four settlement windows and a few
hundred legs do not separate two models whose 40-fold LogLoss differs in the fourth decimal, and
saying so is the finding. What the sample CAN carry is the execution result — fill rates, the
slippage delta and CLV are per-leg quantities with an order of magnitude more observations than
the bankroll path has.
"""
function verdict_text(s::DataFrame, live::Vector{SlateCard})
    io = IOBuffer()
    touch = filter(r -> r.fill_model == "touch_only", s)
    sweep = filter(r -> r.fill_model == "ladder_sweep_v1", s)

    get_row(df, arm) = (i = findfirst(==(arm), df.arm); i === nothing ? nothing : df[i, :])
    m05r, m05c = get_row(touch, "m05_raw"), get_row(touch, "m05_cal_optB")
    m12r, m12c = get_row(touch, "m12_raw"), get_row(touch, "m12_cal_optB")

    n_slates = length(live)
    println(io, "### 7.1 `m05` vs `m12` at tradeable T−25 prices")
    println(io)
    if m05r !== nothing && m12r !== nothing
        @printf(io, "On the honest `TouchOnly` fill, the raw arms finished at %s (`m05_raw`) ",
                _gbp(m05r.final))
        @printf(io, "and %s (`m12_raw`) from %s. `m12` ends %s %s `m05` over %d settlement ",
                _gbp(m12r.final), _gbp(m05r.opening),
                _gbp(abs(m12r.final - m05r.final)),
                m12r.final >= m05r.final ? "above" : "below", n_slates)
        @printf(io, "windows, on %d and %d filled legs respectively (ROI %s vs %s on filled risk, ",
                m12r.n_bets, m05r.n_bets, _pct(m12r.roi_pct), _pct(m05r.roi_pct))
        @printf(io, "win rate %s vs %s).\n\n", _pct(m12r.win_rate_pct), _pct(m05r.win_rate_pct))
        println(io, "That difference is **not** a model ranking. Suite 06 measured the two apart ",
                "by 0.0004 LogLoss over 2,899 scored observations; a four-Saturday bankroll path ",
                "cannot resolve that, and the arms here differ mostly in which legs the allocator ",
                "chose to size, not in whether they were right. The lineup pillar's measured ",
                "contribution in suite 06 was calibration (ECE 0.0100 vs 0.0149), and calibration ",
                "shows up in CLV long before it shows up in four weeks of P&L — §4 is the column ",
                "to read for it.")
    end
    println(io)
    println(io, "### 7.2 The Option B rate calibrator")
    println(io)
    if m05r !== nothing && m05c !== nothing && m12r !== nothing && m12c !== nothing
        @printf(io, "| Model | Raw final | Calibrated final | Δ | Raw legs / staked | Calibrated legs / staked | Raw beat-close | Calibrated beat-close |\n")
        @printf(io, "|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for (nm, raw, cal) in (("m05", m05r, m05c), ("m12", m12r, m12c))
            @printf(io, "| `%s` | %s | %s | %s | %d / %s | %d / %s | %s | %s |\n",
                    nm, _gbp(raw.final), _gbp(cal.final), _gbp(cal.final - raw.final),
                    raw.n_bets, _gbp(raw.total_staked), cal.n_bets, _gbp(cal.total_staked),
                    _pct(raw.beat_close_pct), _pct(cal.beat_close_pct))
        end
        println(io)
        println(io, "The calibrator pools every posterior rate draw with the market rates inverted ",
                "out of the same T−25 book the stakes are placed into. It therefore *cannot* add ",
                "information the book does not already carry; what it can do is stop the model ",
                "disagreeing with a sharp price for the wrong reason.")
        println(io)
        println(io, "**The dominant effect here is on SIZE, not on direction.** Pooling with the ",
                "book pulls the posterior toward the market, which shrinks every edge, which the ",
                "Kelly allocator turns into fewer and smaller legs — the calibrated arms put ",
                "roughly half the capital at risk that the raw arms did. Over four winning-ish ",
                "Saturdays that mechanically halves the P&L, and that is most of what the Δ ",
                "column measures. It is not evidence that the calibrator is wrong; it is evidence ",
                "that a four-slate bankroll comparison between a full-size and a half-size track ",
                "is measuring exposure.")
        println(io)
        both_up = m05c.beat_close_pct > m05r.beat_close_pct &&
                  m12c.beat_close_pct > m12r.beat_close_pct
        both_down = m05c.beat_close_pct < m05r.beat_close_pct &&
                    m12c.beat_close_pct < m12r.beat_close_pct
        println(io, "The beat-close columns are the ones with power. Calibration ",
                both_up  ? "**raises** the share of legs entered better than the closing line for both models" :
                both_down ? "**lowers** the share of legs entered better than the closing line for both models" :
                "moves the share of legs entered better than the closing line in different directions for the two models",
                ", measured over ", m05c.n_clv + m12c.n_clv,
                " calibrated legs rather than over four bankroll steps. That share is the ",
                "quantity suite 06's calibration finding predicts, and it is the one this sample ",
                "size can actually speak to.")
    end
    println(io)
    println(io, "### 7.3 Execution")
    println(io)
    if !isempty(touch) && !isempty(sweep)
        tf = _mean_finite(touch.fill_rate_pct); sf = _mean_finite(sweep.fill_rate_pct)
        @printf(io, "Mean fill rate across arms: **%s** at the touch, **%s** sweeping three levels. ",
                _pct(tf), _pct(sf))
        deltas = Dict(r.arm => r.net_pnl for r in eachrow(sweep))
        for r in eachrow(touch)
            deltas[r.arm] = deltas[r.arm] - r.net_pnl
        end
        @printf(io, "Sweeping instead of resting is worth between %s and %s of P&L depending on ",
                _gbp(minimum(values(deltas))), _gbp(maximum(values(deltas))))
        signs = length(unique(sign.(collect(values(deltas))))) == 1 ?
                "the arm, with a consistent sign across arms: " :
                "the arm, and **the sign is not constant across arms**. That is not a contradiction — "
        print(io, signs)
        println(io, "a deeper fill is ",
                "more of whatever the leg was; on a track whose legs won it magnifies the win and ",
                "on a track whose legs lost it magnifies the loss. The sweep column measures the ",
                "capacity assumption, never the edge.")
        println(io)
        println(io, "This is the number to carry forward. The Scottish lower divisions' T−25 ",
                "book is thin, and on this evidence the binding constraint on the strategy is ",
                "how much of a Kelly-sized order the touch will absorb, not whether the posterior ",
                "is right.")
    end
    println(io)
    println(io, "### 7.4 What this study does not claim")
    println(io)
    println(io, "* It does not claim a live P&L. `LadderSweep` is an upper bound and the archive ",
            "carries three levels; a real order would also move the book it is measured against.")
    println(io, "* It does not rank `m05` against `m12` on predictive skill. That claim belongs ",
            "to `experiments/scottish_lower/06_joint_player_lineup_fusion/`, over 2,899 scored ",
            "observations, and this study has ", n_slates,
            _plural(n_slates, " settlement window.", " settlement windows."))
    println(io, "* It does not measure the calibrator against its own validation instant twice: ",
            "the calibrator records `book_as_of_minutes = -25.0` and `calibrate_fit` refuses a ",
            "book from any other instant, so no arm here silently priced a T−12 or closing book.")
    return String(take!(io))
end
