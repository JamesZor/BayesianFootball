# l07_halftime.jl — loader for brief 07: half-time re-pricing vs the in-play market.
#
# Plumbing for r07_halftime.jl:
#   §1  the half-time break per match (BBC live-text `half_end` → second-half `kick_off`
#       timestamps; an empirical fallback window where BBC is absent) and the HT in-play book
#       (median LTP of the ticks inside the break, de-vigged; the archive is LTP only);
#   §2  HT state (sofascore HT score, first-half red cards from sofascore incidents) and
#       first-half performance from BBC live text (shots, on target, corners, proxy xG from the
#       src shot-cell table fitted on training matches only);
#   §3  the second-half model: team-row Poisson GLM with offset log λ_pre (IRLS), a full-time draw
#       inflation δ, FT 1X2 / O/U 2.5 probabilities given HT, a log-pool blend with the market;
#   §4  the 26/27 order book during the break (best back/lay and sizes).
#
# Only SELECTs touch a database. No sampling; the only fits are GLM/1-D maximum likelihood.

include(joinpath(@__DIR__, "l05_market_structure.jl"))

const C7_TIDS = (54, 55, 56, 57)
const C7_ANCHOR_GRID = (:bp_com, (0.08, 1.08))      # §14's best grid
const C7_ANCHOR_LINES = (1.5, 2.5, 3.5)
const C7_MAXG = 10                                   # second-half goals per side


# ==============================================================================
# 1. THE HALF-TIME BREAK AND THE HT BOOK
# ==============================================================================

"Match kick-off (sofascore `start_timestamp`, UTC DateTime) for the tournaments."
function c7_kickoffs(tids = C7_TIDS)
    k = BayesianFootball.MatchDay._query(
        "SELECT match_id, start_timestamp FROM sofascore.matches WHERE tournament_id = ANY(\$1)", (collect(tids),))
    return Dict(Int(r.match_id) => c7_utc(r.start_timestamp) for r in eachrow(k) if !ismissing(r.start_timestamp))
end

c7_utc(x) = x isa DateTime ? x :
            DateTime(Data.TimeZones.astimezone(x, Data.TimeZones.TimeZone("UTC")))
_c7_min(t, ko) = Dates.value(t - ko) / 60_000

"""
    c7_ht_windows(kickoffs) -> (Dict{Int,Tuple{Float64,Float64,Symbol}}, fallback)

Half-time break in minutes after the sofascore kick-off. BBC: first `half_end` post + 1 min
to the first `kick_off` post after it − 0.5 min; kept if the start lies in [44, 62] and the
break lasts 8–25 min. Fallback for every other match: (90th percentile of BBC break starts,
10th percentile of BBC break ends), a window inside the break for ~80% of matches.
"""
function c7_ht_windows(kickoffs::Dict{Int,DateTime})
    ev = BayesianFootball.MatchDay._query("""
        SELECT l.match_id, l.event_type, l.published
        FROM bbc.live_text l JOIN sofascore.matches m USING (match_id)
        WHERE l.event_type IN ('kick_off', 'half_end') AND m.tournament_id = ANY(\$1)
        ORDER BY l.match_id, l.published""", (collect(C7_TIDS),))
    out = Dict{Int,Tuple{Float64,Float64,Symbol}}()
    for g in groupby(ev, :match_id)
        mid = Int(first(g.match_id))
        haskey(kickoffs, mid) || continue
        he = findfirst(==("half_end"), g.event_type)
        he === nothing && continue
        t_he = c7_utc(g.published[he])
        k2 = findfirst(i -> g.event_type[i] == "kick_off" && c7_utc(g.published[i]) > t_he, 1:nrow(g))
        k2 === nothing && continue
        lo = _c7_min(t_he, kickoffs[mid]) + 1.0
        hi = _c7_min(c7_utc(g.published[k2]), kickoffs[mid]) - 0.5
        (44.0 <= lo <= 62.0 && 8.0 <= hi - lo <= 25.0) && (out[mid] = (lo, hi, :bbc))
    end
    los = [v[1] for v in values(out)]; his = [v[2] for v in values(out)]
    fb = (quantile(los, 0.9), quantile(his, 0.1))
    for mid in keys(kickoffs)
        haskey(out, mid) || (out[mid] = (fb[1], fb[2], :fallback))
    end
    return out, fb
end

"""
    c7_ht_book(odds, windows) -> DataFrame

1X2 and O/U 2.5 at half-time: per runner, the MEDIAN last-traded price of the ticks inside the
break. A market with every runner but one traded gets the missing runner imputed as
1 − Σ(other implied), floored at 0.005 (flagged). Then multiplicative de-vig within market
(overround gate 0.9–1.1).
"""
function c7_ht_book(odds::AbstractDataFrame, windows)
    ip = odds[(odds.minutes_to_kickoff .> 40.0) .& (odds.minutes_to_kickoff .< 80.0) .&
              ((odds.market_name .== "1X2") .| ((odds.market_name .== "OverUnder") .& (odds.market_line .== 2.5))), :]
    rows = NamedTuple[]
    for g in groupby(ip, [:match_id, :market_name, :market_line])
        mid = Int(first(g.match_id))
        haskey(windows, mid) || continue
        lo, hi, src = windows[mid]
        w = g[(lo .<= g.minutes_to_kickoff .<= hi) .& (g.traded_price .> 1.0), :]
        name = String(first(g.market_name))
        sels = name == "1X2" ? [:home, :draw, :away] : [:over_25, :under_25]
        med = Dict{Symbol,Tuple{Float64,Int}}()
        for s in groupby(w, :selection)
            med[Symbol(first(s.selection))] = (median(s.traded_price), nrow(s))
        end
        missing_sels = [s for s in sels if !haskey(med, s)]
        length(missing_sels) > 1 && continue
        imputed = false
        if length(missing_sels) == 1
            rest = sum(1 / med[s][1] for s in sels if haskey(med, s))
            rest > 0.995 && continue
            med[missing_sels[1]] = (1 / max(1 - rest, 0.005), 0)
            imputed = true
        end
        over = sum(1 / med[s][1] for s in sels)
        0.9 <= over <= 1.1 || continue
        for s in sels
            push!(rows, (; match_id = mid, market_name = name, market_line = Float64(first(g.market_line)),
                         selection = s, odds_ht = med[s][1], n_ticks = med[s][2], p_ht = (1 / med[s][1]) / over,
                         overround = over, imputed, window = src, win_lo = lo, win_hi = hi))
        end
    end
    return DataFrame(rows)
end


# ==============================================================================
# 2. HT STATE AND FIRST-HALF PERFORMANCE
# ==============================================================================

"First-half red cards per (match, is_home) from sofascore incidents (red or second yellow, minute ≤ 45)."
function c7_first_half_reds()
    r = BayesianFootball.MatchDay._query("""
        SELECT i.match_id, i.is_home, count(*) n
        FROM sofascore.match_incidents i JOIN sofascore.matches m USING (match_id)
        WHERE m.tournament_id = ANY(\$1) AND i.incident_type = 'card'
          AND i.data->>'incidentClass' IN ('red', 'yellowRed') AND i.time <= 45
        GROUP BY 1, 2""", (collect(C7_TIDS),))
    return Dict((Int(x.match_id), Bool(x.is_home)) => Int(x.n) for x in eachrow(r) if !ismissing(x.is_home))
end

"Matches with any sofascore incident rows (the red-card source's coverage)."
c7_incident_matches() = Set(Int.(BayesianFootball.MatchDay._query("""
    SELECT DISTINCT i.match_id FROM sofascore.match_incidents i JOIN sofascore.matches m USING (match_id)
    WHERE m.tournament_id = ANY(\$1)""", (collect(C7_TIDS),)).match_id))

"BBC first-half corners per (match, is_home) with the fetcher's three-way `-fc`-normalised side map."
function c7_first_half_corners()
    r = BayesianFootball.MatchDay._query("""
        SELECT lt.match_id,
               CASE WHEN regexp_replace(lt.team, '-fc\$', '') = regexp_replace(mm.bbc_home_slug, '-fc\$', '') THEN true
                    WHEN regexp_replace(lt.team, '-fc\$', '') = regexp_replace(mm.bbc_away_slug, '-fc\$', '') THEN false
                    ELSE NULL END AS is_home, count(*) n
        FROM bbc.live_text lt JOIN sofascore.matches m ON m.match_id = lt.match_id
        JOIN bbc.match_meta mm ON mm.match_id = lt.match_id
        WHERE m.tournament_id = ANY(\$1) AND lt.event_type = 'corner' AND lt.time <= 45
        GROUP BY 1, 2""", (collect(C7_TIDS),))
    return Dict((Int(x.match_id), Bool(x.is_home)) => Int(x.n) for x in eachrow(r) if !ismissing(x.is_home))
end

"""
    c7_first_half_perf(ds, fit_ids) -> DataFrame

Per match with BBC shot commentary: first-half (minute ≤ 45, incl. added time) shots, shots
on target, proxy xG (`Features.fit_shot_xg` on shots from `fit_ids` matches only, then
`predict_xg`) and corners, home and away.
"""
function c7_first_half_perf(ds, fit_ids::Set{Int})
    shots = BayesianFootball.Features.build_shots(ds)
    shots = shots[.!ismissing.(shots.is_home) .& .!ismissing.(shots.time), :]
    model = BayesianFootball.Features.fit_shot_xg(shots[in.(shots.match_id, Ref(fit_ids)), :])
    shots.xg = BayesianFootball.Features.predict_xg(model, shots)
    fh = shots[shots.time .<= 45, :]
    corners = c7_first_half_corners()
    covered = Set(Int.(shots.match_id))
    rows = NamedTuple[]
    for mid in covered
        g = fh[fh.match_id .== mid, :]
        h = g[g.is_home .=== true, :]; a = g[g.is_home .=== false, :]
        push!(rows, (; match_id = mid, sh_h = nrow(h), sh_a = nrow(a),
                     sot_h = count(h.is_on_target), sot_a = count(a.is_on_target),
                     pxg_h = sum(h.xg; init = 0.0), pxg_a = sum(a.xg; init = 0.0),
                     cor_h = get(corners, (mid, true), 0), cor_a = get(corners, (mid, false), 0)))
    end
    return DataFrame(rows), model
end


# ==============================================================================
# 3. THE SECOND-HALF MODEL
# ==============================================================================

"""
    c7_poisson(X, y, offset; ridge = 1e-6) -> (β, se)

Poisson GLM with log link and offset by IRLS (y may be non-integer: quasi-Poisson point
estimates). Standard errors from the inverse Fisher information.
"""
function c7_poisson(X::AbstractMatrix, y::AbstractVector, offset::AbstractVector; ridge::Real = 1e-6)
    β = zeros(size(X, 2))
    β[1] = log(max(mean(y), 1e-6) / mean(exp.(offset)))
    for _ in 1:100
        μ = exp.(offset .+ X * β)
        H = X' * (X .* μ) + ridge * I
        step = H \ (X' * (y .- μ) - ridge * β)
        β .+= step
        maximum(abs, step) < 1e-10 && break
    end
    μ = exp.(offset .+ X * β)
    se = sqrt.(max.(diag(LinearAlgebra.inv(Symmetric(X' * (X .* μ)))), 0.0))
    return β, se
end

"""
    c7_ft_probs(hh, ha, μh, μa; δ = 0.0) -> (pH, pD, pA, p_under25)

Full-time 1X2 and under 2.5 given the HT score (hh, ha) and independent Poisson second-half
means, with full-time-draw cells inflated by (1 + δ) and the grid renormalised.
"""
function c7_ft_probs(hh::Integer, ha::Integer, μh::Real, μa::Real; δ::Real = 0.0)
    ph = [exp(-μh) * μh^k / factorial(k) for k in 0:C7_MAXG]
    pa = [exp(-μa) * μa^k / factorial(k) for k in 0:C7_MAXG]
    H = D = A = U = 0.0
    for i in 0:C7_MAXG, j in 0:C7_MAXG
        fh, fa = hh + i, ha + j
        p = ph[i+1] * pa[j+1]
        (fh + fa) <= 2 && (U += p)                       # δ acts on the 1X2 only
        p *= fh == fa ? 1 + δ : 1.0
        fh > fa ? (H += p) : fh == fa ? (D += p) : (A += p)
    end
    Z = H + D + A
    return (H / Z, D / Z, A / Z, U)
end

"Mean log-loss of full-time 1X2 outcomes under probability triples `P` (rows home, draw, away)."
c7_ll(P::AbstractMatrix, res::AbstractVector{Int}) =
    [-log(max(P[i, res[i]], 1e-12)) for i in eachindex(res)]

"Logarithmic pool of two 1X2 probability matrices with weight w on `A`."
function c7_pool(A::AbstractMatrix, B::AbstractMatrix, w::Real)
    P = (A .^ w) .* (B .^ (1 - w))
    return P ./ sum(P; dims = 2)
end

"Paired cluster-bootstrap mean difference (a − b) with 95% interval and the SE."
function c7_paired(a::AbstractVector, b::AbstractVector, clusters::AbstractVector; B::Int = 2_000, seed::Int = 20260928)
    d = a .- b
    m = c5_mean_ci(d, clusters; B, seed)
    ukeys = unique(clusters)
    pos = Dict(k => i for (i, k) in enumerate(ukeys))
    S = zeros(length(ukeys)); N = zeros(length(ukeys))
    for i in eachindex(d)
        S[pos[clusters[i]]] += d[i]; N[pos[clusters[i]]] += 1
    end
    rng = MersenneTwister(seed + 1)
    boots = [begin
        idx = rand(rng, 1:length(ukeys), length(ukeys)); sum(S[idx]) / sum(N[idx])
    end for _ in 1:B]
    return (; mean = m.mean, lo = m.lo, hi = m.hi, se = std(boots), n = m.n)
end


# ==============================================================================
# 4. THE 26/27 ORDER BOOK DURING THE BREAK
# ==============================================================================

"""
    c7_live_ht_book(windows) -> DataFrame

MATCH_ODDS order-book snapshots inside each 26/27 match's HT window: best back/lay price and
size per runner (×10⁴ units unscaled). `windows` keyed by match_id.
"""
function c7_live_ht_book(ds, windows, kickoffs)
    md = BayesianFootball.MatchDay._query("""
        SELECT md.market_id, mm.match_id
        FROM betfair_live.market_metadata md
        JOIN betfair.match_meta mm ON mm.betfair_event_id = md.event_id
        JOIN sofascore.events e ON e.match_id = mm.match_id
        WHERE md.market_type = 'MATCH_ODDS' AND e.tournament_id = ANY(\$1)""", (collect(C7_TIDS),))
    seasons = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
    filter!(r -> get(seasons, Int(r.match_id), "") == "26/27" && haskey(kickoffs, Int(r.match_id)), md)
    mid = Dict(String(r.market_id) => Int(r.match_id) for r in eachrow(md))
    raw = BayesianFootball.MatchDay._query("""
        SELECT market_id, symbol, ts, bid_prices, bid_volumes, ask_prices, ask_volumes, last_price_traded
        FROM betfair_live.order_book_1m WHERE market_id = ANY(\$1)""", (collect(keys(mid)),))
    rows = NamedTuple[]
    for r in eachrow(raw)
        m = mid[String(r.market_id)]
        mins = _c7_min(DateTime(r.ts), kickoffs[m])
        lo, hi, src = windows[m]
        lo <= mins <= hi || continue
        push!(rows, (; match_id = m, selection = Symbol(r.symbol), minutes = mins, window = src,
                     back = _c4_first(r.bid_prices, 1e4), back_size = _c4_first(r.bid_volumes, 1e4),
                     lay = _c4_first(r.ask_prices, 1e4), lay_size = _c4_first(r.ask_volumes, 1e4),
                     ltp = ismissing(r.last_price_traded) ? NaN : r.last_price_traded / 1e4))
    end
    return DataFrame(rows)
end

"""
    c7_invert_ht(hh, ha, p, u) -> (μh, μa)

Second-half means that make `c7_ft_probs` reproduce the HT market's de-vigged 1X2 `p`
(and under-2.5 `u` when finite) given the HT score: minimum summed KL, Nelder–Mead on logs.
"""
function c7_invert_ht(hh::Integer, ha::Integer, p::NTuple{3,Float64}, u::Float64)
    function f(z)
        q = c7_ft_probs(hh, ha, exp(z[1]), exp(z[2]))
        s = sum(_c5_xlogy(p[i], q[i]) for i in 1:3)
        isfinite(u) && (s += _c5_kl2(u, q[4]))
        return s
    end
    r = optimize(f, [log(0.7), log(0.6)], NelderMead(), Optim.Options(x_abstol = 1e-8, f_reltol = 1e-12, iterations = 2_000))
    z = Optim.minimizer(r)
    return (exp(z[1]), exp(z[2]))
end
