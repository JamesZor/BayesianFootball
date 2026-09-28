# l05_market_structure.jl — loader for brief 05: how the market rates teams, and its score grid.
#
# Everything here is plumbing for r05_market_structure.jl:
#   §1  score grids: independent Poisson, Dixon–Coles (ρ), bivariate Poisson (λ₃), negative
#       binomial (κ), Conway–Maxwell–Poisson (ν), diagonal inflation (δ); the markets they imply;
#   §2  per-match inversion: the grid's two per-match parameters fitted to the market's de-vigged
#       1X2 + O/U probabilities by minimum summed KL; the global shape parameter profiled over a
#       grid of values, then chosen by the correct-score KL (cluster bootstrap by match date);
#   §3  fit targets: the archive close (TWA (−20, 0], de-vigged; l01), a partial correct-score book
#       (last traded price at kick-off, ≤ 120 min stale, compared on the runners present), the
#       26/27 betfair_live order-book mid close, and model probabilities;
#   §4  market ratings: per league-season least squares on log λ with a per-team random walk over
#       the team's match sequence (sparse ridge), filtered/smoothed, and opponent-adjusted
#       per-match ("implied") ratings;
#   §5  features for the season-start regression (tables, tiers, wealth, turnover) and small OLS.
#
# Only SELECTs touch a database. No sampling. The only fits are ≤ 3-parameter per-match MLE-style
# minimisations and linear least squares.

include(joinpath(@__DIR__, "l04_live_orderbook.jl"))
using LinearAlgebra
using Optim
using SparseArrays
using StatsBase: countmap

# ==============================================================================
# 1. SCORE GRIDS
# ==============================================================================

const C5_G = 12                                  # goals 0..12 per side
const C5_N = C5_G + 1
const C5_CS_SCORES = [(h, a) for h in 0:3 for a in 0:3]
const C5_CS_SELS = vcat([Symbol("cs_$(h)$(a)") for (h, a) in C5_CS_SCORES],
                        [:cs_any_other_home, :cs_any_other_draw, :cs_any_other_away])
const C5_CS_INDEX = Dict(s => i for (i, s) in enumerate(C5_CS_SELS))
const C5_KINDS = (:poisson, :dc, :bp, :nb, :com, :diag)
const C5_KIND_LABEL = Dict(:poisson => "(a) independent Poisson", :dc => "(b) Dixon–Coles ρ",
                           :dc_com => "(b+d′) Dixon–Coles ρ + COM ν", :bp_com => "(c+d′) bivariate λ₃ + COM ν",
                           :bp => "(c) bivariate Poisson λ₃", :nb => "(d) negative binomial κ",
                           :com => "(d′) Conway–Maxwell–Poisson ν", :diag => "(e) diagonal inflation δ")
const C5_THETA = Dict(:poisson => [0.0],
                      :dc => collect(-0.30:0.01:0.12),
                      :bp => collect(0.0:0.01:0.30),
                      :nb => collect(0.0:0.01:0.30),
                      :com => collect(0.80:0.02:1.40),
                      :diag => collect(-0.10:0.01:0.30),
                      :dc_com => [(ρ, ν) for ν in 0.90:0.02:1.30 for ρ in -0.12:0.01:0.02],
                      :bp_com => [(l, ν) for ν in 0.90:0.02:1.30 for l in 0.0:0.01:0.18])
const C5_COMBOS = (:dc_com, :bp_com)

function _c5_pois!(v, λ)
    v[1] = exp(-λ)
    @inbounds for k in 1:C5_G
        v[k+1] = v[k] * λ / k
    end
    return v
end

"Negative binomial with mean m and Var = m + κm² (κ = 0 is Poisson)."
function _c5_nb!(v, m, κ)
    κ <= 1e-9 && return _c5_pois!(v, m)
    r = 1 / κ
    p = r / (r + m)
    v[1] = p^r
    @inbounds for k in 1:C5_G
        v[k+1] = v[k] * (k - 1 + r) / k * (1 - p)
    end
    return v
end

"Conway–Maxwell–Poisson with rate λ and dispersion ν (ν > 1 under-dispersed), truncated at C5_G."
function _c5_com!(v, λ, ν)
    lw = 0.0
    v[1] = 0.0
    @inbounds for k in 1:C5_G
        lw += log(λ) - ν * log(k)
        v[k+1] = lw
    end
    mx = maximum(v)
    s = 0.0
    @inbounds for k in 1:C5_N
        v[k] = exp(v[k] - mx)
        s += v[k]
    end
    v ./= s
    return v
end

"""
    c5_grid!(P, kind, x1, x2, θ, u, v, w) -> Bool

Fill the (C5_N × C5_N) home-by-away score grid. `x1, x2` are the per-match parameters (the goal
means for :poisson/:dc/:nb/:diag; the component rates for :bp; the COM rates for :com); `θ` is the
global shape parameter. Returns `false` if the grid is invalid (a negative cell).
"""
function c5_grid!(P, kind::Symbol, x1, x2, θ, u, v, w)
    # combinations carry θ = (ρ or λ₃, ν): COM marginals plus DC τ / a common Poisson component
    ν = kind === :com ? θ : (kind === :dc_com || kind === :bp_com) ? θ[2] : 1.0
    s1 = θ isa Tuple ? θ[1] : θ
    base = kind === :dc_com ? :dc : kind === :bp_com ? :bp : kind
    if kind === :nb
        _c5_nb!(u, x1, θ); _c5_nb!(v, x2, θ)
    elseif ν != 1.0
        _c5_com!(u, x1, ν); _c5_com!(v, x2, ν)
    else
        _c5_pois!(u, x1); _c5_pois!(v, x2)
    end
    if base === :bp
        _c5_pois!(w, s1)
        @inbounds for a in 0:C5_G, h in 0:C5_G
            s = 0.0
            for k in 0:min(h, a)
                s += w[k+1] * u[h-k+1] * v[a-k+1]
            end
            P[h+1, a+1] = s
        end
    else
        @inbounds for a in 1:C5_N, h in 1:C5_N
            P[h, a] = u[h] * v[a]
        end
        if base === :dc          # τ: (0,0) 1−λμρ, (0,1) 1+λρ, (1,0) 1+μρ, (1,1) 1−ρ
            P[1, 1] *= 1 - x1 * x2 * s1
            P[1, 2] *= 1 + x1 * s1
            P[2, 1] *= 1 + x2 * s1
            P[2, 2] *= 1 - s1
        elseif base === :diag
            @inbounds for i in 1:C5_N
                P[i, i] *= 1 + s1
            end
        end
    end
    s = 0.0
    @inbounds for i in eachindex(P)
        P[i] < 0 && return false
        s += P[i]
    end
    P ./= s
    return true
end

"Home-win, draw, away-win probabilities of a grid."
function c5_1x2(P)
    h = d = a = 0.0
    @inbounds for j in 1:C5_N, i in 1:C5_N
        i > j ? (h += P[i, j]) : i == j ? (d += P[i, j]) : (a += P[i, j])
    end
    return (h, d, a)
end

"P(total goals < line)."
function c5_under(P, line)
    s = 0.0
    @inbounds for j in 1:C5_N, i in 1:C5_N
        (i + j - 2) < line && (s += P[i, j])
    end
    return s
end

c5_btts(P) = 1.0 - sum(@view P[1, :]) - sum(@view P[:, 1]) + P[1, 1]

"The 19 Betfair correct-score runners (C5_CS_SELS order) implied by a grid."
function c5_cs(P)
    out = zeros(19)
    @inbounds for (k, (h, a)) in enumerate(C5_CS_SCORES)
        out[k] = P[h+1, a+1]
    end
    @inbounds for j in 1:C5_N, i in 1:C5_N
        (i <= 4 && j <= 4) && continue
        out[i > j ? 17 : i == j ? 18 : 19] += P[i, j]
    end
    return out
end

"Goal means of a grid."
function c5_means(P)
    mh = ma = 0.0
    @inbounds for j in 1:C5_N, i in 1:C5_N
        mh += (i - 1) * P[i, j]
        ma += (j - 1) * P[i, j]
    end
    return (mh, ma)
end

_c5_xlogy(q, p) = q <= 0 ? 0.0 : q * log(q / max(p, 1e-12))
_c5_kl2(q, p) = _c5_xlogy(q, p) + _c5_xlogy(1 - q, 1 - p)


# ==============================================================================
# 2. TARGETS AND PER-MATCH INVERSION
# ==============================================================================

"""
One match's de-vigged market (or model) probabilities. `under[line]` = P(under line);
`cs_idx`/`cs_q` = the correct-score runners present (C5_CS_SELS positions) and their
probabilities renormalised over that set (conditional on the set).
"""
struct C5Target
    match_id::Int
    x12::NTuple{3,Float64}
    under::Dict{Float64,Float64}
    btts::Float64
    cs_idx::Vector{Int}
    cs_q::Vector{Float64}
end

c5_has_cs(t::C5Target) = !isempty(t.cs_idx)

"""
    c5_targets(book; pcol, cs = nothing, min_cs = 16) -> Dict{Int,C5Target}

Targets from a long book (match_id, market_name, market_line, selection, `pcol`). The 1X2 and
the O/U 2.5 must both be present. `cs` (optional) is a long partial correct-score frame with
`selection` and `q` (already renormalised over the runners present); if `nothing`, the book's own
CorrectScore rows are used when complete.
"""
function c5_targets(book::AbstractDataFrame; pcol::Symbol, cs = nothing, min_cs::Int = 16)
    out = Dict{Int,C5Target}()
    csmap = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    if cs !== nothing
        for g in groupby(cs, :match_id)
            nrow(g) >= min_cs || continue
            idx = [C5_CS_INDEX[s] for s in g.selection]
            all(k -> k in idx, 1:6) || continue      # 0-0 … 1-1 region must be present
            p = sortperm(idx)
            csmap[Int(first(g.match_id))] = (idx[p], Float64.(g.q)[p])
        end
    end
    for g in groupby(book, :match_id)
        mid = Int(first(g.match_id))
        x = Dict(Symbol(r.selection) => Float64(r[pcol]) for r in eachrow(g) if r.market_name == "1X2")
        length(x) == 3 || continue
        under = Dict{Float64,Float64}()
        for r in eachrow(g)
            if r.market_name == "OverUnder" && startswith(String(r.selection), "under")
                under[Float64(r.market_line)] = Float64(r[pcol])
            end
        end
        haskey(under, 2.5) || continue
        by = [Float64(r[pcol]) for r in eachrow(g) if r.market_name == "BTTS" && r.selection == :btts_yes]
        if cs === nothing
            cr = g[g.market_name .== "CorrectScore", :]
            if nrow(cr) == 19
                idx = [C5_CS_INDEX[Symbol(s)] for s in cr.selection]
                p = sortperm(idx)
                csmap[mid] = (idx[p], Float64.(cr[!, pcol])[p] ./ sum(cr[!, pcol]))
            end
        end
        ci, cq = get(csmap, mid, (Int[], Float64[]))
        out[mid] = C5Target(mid, (x[:home], x[:draw], x[:away]), under,
                            isempty(by) ? NaN : by[1], ci, cq)
    end
    return out
end

"Summed KL of the target's 1X2 and O/U `lines` (those present) from the grid."
function c5_loss(t::C5Target, P, lines)
    ph, pd, pa = c5_1x2(P)
    s = _c5_xlogy(t.x12[1], ph) + _c5_xlogy(t.x12[2], pd) + _c5_xlogy(t.x12[3], pa)
    for L in lines
        q = get(t.under, L, NaN)
        isnan(q) || (s += _c5_kl2(q, c5_under(P, L)))
    end
    return s
end

"KL(market CS ‖ grid CS) over the runners present, both renormalised over that set."
function c5_kl_cs(t::C5Target, pcs::AbstractVector)
    c5_has_cs(t) || return NaN
    z = sum(pcs[k] for k in t.cs_idx)
    s = 0.0
    for (k, q) in zip(t.cs_idx, t.cs_q)
        s += _c5_xlogy(q, pcs[k] / z)
    end
    return s
end

"The grid's CS probabilities on the target's runner set, renormalised (for residuals)."
function c5_cs_on(t::C5Target, pcs::AbstractVector)
    out = fill(NaN, 19)
    c5_has_cs(t) || return out
    z = sum(pcs[k] for k in t.cs_idx)
    for k in t.cs_idx
        out[k] = pcs[k] / z
    end
    return out
end

"The target's CS probabilities as a 19-vector (NaN where the runner is absent)."
function c5_cs_q(t::C5Target)
    out = fill(NaN, 19)
    for (k, q) in zip(t.cs_idx, t.cs_q)
        out[k] = q
    end
    return out
end

"A starting point: total from the O/U 2.5 (Poisson), split by the 1X2 supremacy."
function c5_start(t::C5Target, kind::Symbol, θ)
    q = t.under[2.5]
    T = 2.6
    for _ in 1:30                                    # solve P(Pois(T) ≤ 2) = q
        f = exp(-T) * (1 + T + T^2 / 2) - q
        df = -exp(-T) * T^2 / 2
        T = clamp(T - f / df, 0.5, 6.0)
    end
    sh = clamp(0.5 + 0.55 * (t.x12[1] - t.x12[3]), 0.1, 0.9)
    lh, la = T * sh, T * (1 - sh)
    kind === :bp && return (max(lh - θ, 0.05), max(la - θ, 0.05))
    kind === :bp_com && return (max(lh - θ[1], 0.05), max(la - θ[1], 0.05))
    return (lh, la)
end

"""
    c5_fit(t, kind, θ, lines; x0 = nothing) -> NamedTuple

Minimise `c5_loss` over the two per-match parameters (log scale, Nelder–Mead). Returns the
parameters, the loss, the grid's goal means and its CS vector.
"""
function c5_fit(t::C5Target, kind::Symbol, θ, lines; x0 = nothing)
    P = zeros(C5_N, C5_N)
    u, v, w = zeros(C5_N), zeros(C5_N), zeros(C5_N)
    f(z) = c5_grid!(P, kind, exp(z[1]), exp(z[2]), θ, u, v, w) ? c5_loss(t, P, lines) : 1e6
    s = x0 === nothing ? c5_start(t, kind, θ) : x0
    r = optimize(f, [log(s[1]), log(s[2])], NelderMead(),
                 Optim.Options(x_abstol = 1e-8, f_reltol = 1e-14, iterations = 2_000))
    z = Optim.minimizer(r)
    c5_grid!(P, kind, exp(z[1]), exp(z[2]), θ, u, v, w)
    mh, ma = c5_means(P)
    return (; x1 = exp(z[1]), x2 = exp(z[2]), loss = Optim.minimum(r), mh, ma,
            cs = c5_cs(P), x12 = c5_1x2(P), P = copy(P))
end

"""
    c5_profile(targets, kind, lines; θs = C5_THETA[kind]) -> NamedTuple

For every target and every θ: the per-match fit to 1X2 + O/U `lines`, its loss, and the CS KL.
Matrices are (n targets × n θ). Threaded over targets; warm-started along θ.
"""
function c5_profile(targets::Vector{C5Target}, kind::Symbol, lines; θs = C5_THETA[kind])
    n, m = length(targets), length(θs)
    KL = fill(NaN, n, m); L = fill(NaN, n, m); MH = fill(NaN, n, m); MA = fill(NaN, n, m)
    Threads.@threads for i in 1:n
        prev = nothing
        for j in 1:m
            r = c5_fit(targets[i], kind, θs[j], lines; x0 = prev)
            prev = (r.x1, r.x2)
            L[i, j] = r.loss
            KL[i, j] = c5_kl_cs(targets[i], r.cs)
            MH[i, j] = r.mh; MA[i, j] = r.ma
        end
    end
    return (; kind, θs, KL, L, MH, MA)
end

"""
    c5_argmin_ci(M, clusters, θs; B, seed) -> NamedTuple

argmin over θ of the column mean of `M` (rows NaN-free), with a cluster bootstrap percentile
interval of the argmin, plus the mean curve.
"""
function c5_argmin_ci(M::AbstractMatrix, clusters::AbstractVector, θs; B::Int = 2_000, seed::Int = 20260928)
    ukeys = unique(clusters)
    pos = Dict(k => i for (i, k) in enumerate(ukeys))
    S = zeros(length(ukeys), size(M, 2)); N = zeros(length(ukeys))
    for i in axes(M, 1)
        c = pos[clusters[i]]
        S[c, :] .+= @view M[i, :]
        N[c] += 1
    end
    curve = vec(sum(S; dims = 1)) ./ sum(N)
    j = argmin(curve)
    rng = MersenneTwister(seed)
    boot = eltype(θs)[]
    acc = zeros(size(M, 2))
    for _ in 1:B
        fill!(acc, 0.0)
        for _ in 1:length(ukeys)
            acc .+= @view S[rand(rng, 1:length(ukeys)), :]
        end
        push!(boot, θs[argmin(acc)])
    end
    q(p) = θs[1] isa Tuple ? Tuple(quantile(getindex.(boot, i), p) for i in 1:length(θs[1])) :
                            quantile(boot, p)
    return (; θ = θs[j], lo = q(0.025), hi = q(0.975), curve, j, at_edge = j == 1 || j == length(θs))
end

"Cluster-bootstrap mean and 95% interval of `x` (NaN dropped)."
function c5_mean_ci(x::AbstractVector, clusters::AbstractVector; B::Int = 2_000, seed::Int = 20260928)
    keep = .!isnan.(x)
    x, cl = x[keep], clusters[keep]
    isempty(x) && return (; mean = NaN, lo = NaN, hi = NaN, n = 0)
    ukeys = unique(cl)
    pos = Dict(k => i for (i, k) in enumerate(ukeys))
    S = zeros(length(ukeys)); N = zeros(length(ukeys))
    for i in eachindex(x)
        S[pos[cl[i]]] += x[i]; N[pos[cl[i]]] += 1
    end
    rng = MersenneTwister(seed)
    boot = Float64[]
    for _ in 1:B
        s = n = 0.0
        for _ in 1:length(ukeys)
            c = rand(rng, 1:length(ukeys))
            s += S[c]; n += N[c]
        end
        push!(boot, s / n)
    end
    return (; mean = mean(x), lo = quantile(boot, 0.025), hi = quantile(boot, 0.975), n = length(x))
end


# ==============================================================================
# 3. BOOKS: ARCHIVE, PARTIAL CORRECT SCORE, LIVE ORDER BOOK, OTHER LEAGUES
# ==============================================================================

"""
    c5_cs_pit(odds; stale = 120.0) -> DataFrame

Partial correct-score book: each runner's last traded price at or before kick-off, kept if at
most `stale` minutes old; `q` = implied probabilities renormalised over the runners present.
(The archive holds LTPs only; the complete TWA (−20, 0] CS book exists for < 200 Scottish matches.)
"""
function c5_cs_pit(odds::AbstractDataFrame; stale::Real = 120.0)
    cs = odds[odds.market_name .== "CorrectScore", :]
    cfg = Calibration.PointInTimeBookConfig(as_of_minutes = 0.0)
    p = Calibration.point_in_time_prices(cs; config = cfg)
    filter!(r -> r.staleness_minutes <= stale && isfinite(r.odds_close) && r.odds_close > 1.0 &&
                 haskey(C5_CS_INDEX, Symbol(r.selection)), p)
    out = DataFrame(match_id = Int.(p.match_id), selection = Symbol.(p.selection),
                    implied = 1.0 ./ Float64.(p.odds_close))
    transform!(groupby(out, :match_id), :implied => (x -> x ./ sum(x)) => :q,
               :implied => sum => :cs_overround, nrow => :cs_runners)
    return out
end

const C5_LIVE_TYPES = ["MATCH_ODDS", "OVER_UNDER_15", "OVER_UNDER_25", "OVER_UNDER_35",
                       "BOTH_TEAMS_TO_SCORE", "CORRECT_SCORE"]

"Live order-book symbol → (market_name, line, selection)."
function c5_live_symbol(mtype::AbstractString, sym::AbstractString)
    if mtype == "MATCH_ODDS"
        return ("1X2", 0.0, Symbol(sym))
    elseif startswith(mtype, "OVER_UNDER_")
        line = parse(Float64, replace(mtype, "OVER_UNDER_" => "")) / 10
        tag = replace(@sprintf("%.1f", line), "." => "")
        return ("OverUnder", line, Symbol(startswith(sym, "Over") ? "over_" : "under_", tag))
    elseif mtype == "BOTH_TEAMS_TO_SCORE"
        return ("BTTS", 0.0, sym == "Yes" ? :btts_yes : :btts_no)
    elseif mtype == "CORRECT_SCORE"
        m = match(r"^(\d+) - (\d+)$", sym)
        m !== nothing && return ("CorrectScore", 0.0, Symbol("cs_", m[1], m[2]))
        sym == "Any Other Home Win" && return ("CorrectScore", 0.0, :cs_any_other_home)
        sym == "Any Other Away Win" && return ("CorrectScore", 0.0, :cs_any_other_away)
        (sym == "Any Other Draw" || sym == "draw") && return ("CorrectScore", 0.0, :cs_any_other_draw)
    end
    return ("", 0.0, Symbol(sym))
end

"""
    c5_live_close(ds; season = "26/27", tournaments = (54, 55, 56, 57)) -> (book, cs)

The 26/27 order-book close: best-back/best-lay mid per snapshot (both sides required), TWA over
(−20, 0] minutes from the sofascore kick-off (`Data.summarize_odds`), then `cn_devig` for 1X2,
O/U and BTTS; CS returned as a partial book (runners with a two-sided mid in the window).
"""
function c5_live_close(ds; season::AbstractString = "26/27", tournaments = (54, 55, 56, 57))
    md = BayesianFootball.MatchDay._query("""
        SELECT md.market_id, md.market_type, mm.match_id, e.start_timestamp
        FROM betfair_live.market_metadata md
        JOIN betfair.match_meta mm ON mm.betfair_event_id = md.event_id
        JOIN sofascore.events e ON e.match_id = mm.match_id
        WHERE md.market_type = ANY(\$1) AND e.tournament_id = ANY(\$2)""",
        (C5_LIVE_TYPES, collect(tournaments)))
    seasons = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
    filter!(r -> get(seasons, Int(r.match_id), "") == season, md)
    info = Dict(String(r.market_id) => (String(r.market_type), Int(r.match_id),
                                        unix2datetime(Int(r.start_timestamp))) for r in eachrow(md))
    raw = BayesianFootball.MatchDay._query("""
        SELECT market_id, symbol, ts, bid_prices, ask_prices
        FROM betfair_live.order_book_1m WHERE market_id = ANY(\$1)""", (collect(keys(info)),))
    rows = NamedTuple[]
    for r in eachrow(raw)
        mt, mid, ko = info[String(r.market_id)]
        name, line, sel = c5_live_symbol(mt, String(r.symbol))
        isempty(name) && continue
        mins = Dates.value(DateTime(r.ts) - ko) / 60_000
        -20.0 <= mins <= 0.0 || continue
        b, l = _c4_first(r.bid_prices, 1e4), _c4_first(r.ask_prices, 1e4)
        (isfinite(b) && isfinite(l) && b > 1.0) || continue
        push!(rows, (; match_id = mid, market_name = name, market_line = line, selection = sel,
                     minutes_to_kickoff = mins, traded_price = (b + l) / 2))
    end
    long = DataFrame(rows)
    rawc = Data.summarize_odds(long, Data.TWAEstimator(); window = (-20.0, 0.0),
                               overround_limits = (0.0, Inf))
    prices = DataFrame(match_id = Int.(rawc.match_id), market_name = String.(rawc.market_name),
                       market_line = Float64.(rawc.market_line), selection = Symbol.(rawc.selection),
                       odds_close = Float64.(coalesce.(rawc.odds, NaN)))
    filter!(r -> isfinite(r.odds_close) && r.odds_close > 1.0, prices)
    book, _ = cn_devig(prices[prices.market_name .!= "CorrectScore", :]; price_col = :odds_close)
    rename!(book, :p_fair => :p_close)
    cs = prices[prices.market_name .== "CorrectScore", :]
    cs = DataFrame(match_id = cs.match_id, selection = cs.selection, implied = 1.0 ./ cs.odds_close)
    transform!(groupby(cs, :match_id), :implied => (x -> x ./ sum(x)) => :q,
               :implied => sum => :cs_overround, nrow => :cs_runners)
    return book, cs
end

"""
    c5_other_league_odds(tournament_id) -> (odds, matches)

Archive odds (1X2, O/U 0.5–4.5, BTTS, CS only) and kick-off dates for one non-Scottish
tournament, through the `Data` Betfair fetcher with a market-type filter. Read-only.
"""
function c5_other_league_odds(tournament_id::Integer)
    types = ["MATCH_ODDS", "OVER_UNDER_05", "OVER_UNDER_15", "OVER_UNDER_25", "OVER_UNDER_35",
             "OVER_UNDER_45", "BOTH_TEAMS_TO_SCORE", "CORRECT_SCORE"]
    conn = Data.connect_to_db(Data.DBConfig(ENV["BF_DB_URL"]))
    try
        raw = DataFrame(Data.LibPQ.execute(conn, """
            SELECT m.match_id, m.start_timestamp, mk.market_type, o.odds_data
            FROM sofascore.matches m
            INNER JOIN betfair.match_meta mm ON m.match_id = mm.match_id
            INNER JOIN betfair.odds_history o ON m.match_id = o.match_id
            INNER JOIN betfair.markets mk ON o.market_id = mk.market_id
            WHERE m.tournament_id = \$1 AND mm.status = 'SUCCESS' AND mk.market_type = ANY(\$2)
            ORDER BY m.match_id""", [Int(tournament_id), types]))
        odds = Data.unpack_betfair_odds(raw)
        matches = DataFrame(Data.LibPQ.execute(conn, """
            SELECT match_id, start_timestamp
            FROM sofascore.matches WHERE tournament_id = \$1""", [Int(tournament_id)]))
        return odds, matches
    finally
        close(conn)
    end
end

"Model probability frame (l01 `cn_model_probs` output) → targets (model CS is complete)."
c5_model_targets(probs::AbstractDataFrame) = c5_targets(probs; pcol = :p_model)


# ==============================================================================
# 4. MARKET RATINGS (per league-season, random walk over each team's match sequence)
# ==============================================================================

"""
    c5_ratings(obs; w, ridge = 1e-4) -> NamedTuple

`obs`: one league-season, rows (match_id, date, home, away, yh = log λ_home, ya = log λ_away),
sorted by (date, match_id). Model
    yh = μ + home + att[h, j_h] − def[a, j_a],   ya = μ + att[a, j_a] − def[h, j_h],
with att/def a random walk over the team's own match index j (penalty weight `w` = (σ_obs/τ)²)
and a tiny ridge on each team's first level (identification). Sparse least squares.
Returns μ, home, per-row (j_h, j_a) and the att/def of both teams at that row.
"""
function c5_ratings(obs::AbstractDataFrame; w::Real, ridge::Real = 1e-4)
    teams = sort(unique(vcat(String.(obs.home), String.(obs.away))))
    cnt = Dict(t => 0 for t in teams)
    jh = Int[]; ja = Int[]
    for r in eachrow(obs)
        cnt[r.home] += 1; push!(jh, cnt[r.home])
        cnt[r.away] += 1; push!(ja, cnt[r.away])
    end
    off = Dict{String,Int}()
    p = 2
    for t in teams
        off[t] = p
        p += 2 * cnt[t]
    end
    att(t, j) = off[t] + j
    dfn(t, j) = off[t] + cnt[t] + j
    I = Int[]; J = Int[]; V = Float64[]; b = Float64[]
    row = 0
    add!(cols, vals, y) = (row += 1; append!(I, fill(row, length(cols))); append!(J, cols);
                           append!(V, vals); push!(b, y))
    for (k, r) in enumerate(eachrow(obs))
        add!([1, 2, att(r.home, jh[k]), dfn(r.away, ja[k])], [1.0, 1.0, 1.0, -1.0], r.yh)
        add!([1, att(r.away, ja[k]), dfn(r.home, jh[k])], [1.0, 1.0, -1.0], r.ya)
    end
    sw, sr = sqrt(w), sqrt(ridge)
    for t in teams
        add!([att(t, 1)], [sr], 0.0); add!([dfn(t, 1)], [sr], 0.0)
        for j in 2:cnt[t]
            add!([att(t, j), att(t, j - 1)], [sw, -sw], 0.0)
            add!([dfn(t, j), dfn(t, j - 1)], [sw, -sw], 0.0)
        end
    end
    A = sparse(I, J, V, row, p)
    x = Matrix(A) \ b                       # small (≤ ~1k columns): dense QR is fine
    n = nrow(obs)
    return (; μ = x[1], home = x[2], jh, ja, cnt,
            att_h = [x[att(obs.home[k], jh[k])] for k in 1:n], def_h = [x[dfn(obs.home[k], jh[k])] for k in 1:n],
            att_a = [x[att(obs.away[k], ja[k])] for k in 1:n], def_a = [x[dfn(obs.away[k], ja[k])] for k in 1:n],
            last = Dict(t => (x[att(t, cnt[t])], x[dfn(t, cnt[t])]) for t in teams if cnt[t] > 0))
end

"""
    c5_one_step(obs; w) -> Vector of squared errors

One-step-ahead check for choosing `w`: for each match date, fit on the strictly earlier dates
and predict that date's (yh, ya) from each team's last rating. Only matches where both teams
have ≥ 3 earlier matches score.
"""
function c5_one_step(obs::AbstractDataFrame; w::Real)
    err = Float64[]
    dates = sort(unique(obs.date))
    for d in dates
        past = obs[obs.date .< d, :]
        nrow(past) < 10 && continue
        fit = c5_ratings(past; w)
        seen = countmap(vcat(String.(past.home), String.(past.away)))
        for r in eachrow(obs[obs.date .== d, :])
            (get(seen, r.home, 0) >= 3 && get(seen, r.away, 0) >= 3) || continue
            ah, dh = fit.last[r.home]; aa, da = fit.last[r.away]
            push!(err, (r.yh - (fit.μ + fit.home + ah - da))^2, (r.ya - (fit.μ + aa - dh))^2)
        end
    end
    return err
end

"""
    c5_implied(obs, fit) -> DataFrame (one row per team-match)

Opponent-adjusted per-match ratings: the market's (or a model's) λ with the opponent's SMOOTHED
rating and μ/home netted out. `net = att + def` (log goal-ratio units).
`obs` may carry a model's (yh, ya) while `fit` is the market's smoothed fit on the same rows.
"""
function c5_implied(obs::AbstractDataFrame, fit)
    rows = NamedTuple[]
    for (k, r) in enumerate(eachrow(obs))
        att_h = r.yh - fit.μ - fit.home + fit.def_a[k]
        def_h = -(r.ya - fit.μ - fit.att_a[k])
        att_a = r.ya - fit.μ + fit.def_h[k]
        def_a = -(r.yh - fit.μ - fit.home - fit.att_h[k])
        push!(rows, (; match_id = r.match_id, date = r.date, team = String(r.home), opp = String(r.away),
                     venue = "H", j = fit.jh[k], att = att_h, def = def_h, net = att_h + def_h,
                     att_s = fit.att_h[k], def_s = fit.def_h[k], lam_for = exp(r.yh), lam_against = exp(r.ya)))
        push!(rows, (; match_id = r.match_id, date = r.date, team = String(r.away), opp = String(r.home),
                     venue = "A", j = fit.ja[k], att = att_a, def = def_a, net = att_a + def_a,
                     att_s = fit.att_a[k], def_s = fit.def_a[k], lam_for = exp(r.ya), lam_against = exp(r.yh)))
    end
    return DataFrame(rows)
end


# ==============================================================================
# 5. FEATURES AND OLS
# ==============================================================================

const C5_TIER = Dict(54 => 1, 55 => 2, 56 => 3, 57 => 4)

"Previous season label: \"22/23\" → \"21/22\"."
function c5_prev_season(s::AbstractString)
    a, b = parse.(Int, split(s, "/"))
    return @sprintf("%02d/%02d", a - 1, b - 1)
end

"Final league table per (tournament, season): points per game, GD per game, rank fraction."
function c5_tables(matches::AbstractDataFrame)
    rows = NamedTuple[]
    m = matches[in.(matches.tournament_id, Ref(keys(C5_TIER))) .& .!ismissing.(matches.home_score), :]
    for g in groupby(m, [:tournament_id, :season])
        pts = Dict{String,Float64}(); gd = Dict{String,Float64}(); n = Dict{String,Int}()
        for r in eachrow(g)
            h, a = String(r.home_team), String(r.away_team)
            hs, as = Int(r.home_score), Int(r.away_score)
            for (t, f, ag) in ((h, hs, as), (a, as, hs))
                pts[t] = get(pts, t, 0.0) + (f > ag ? 3 : f == ag ? 1 : 0)
                gd[t] = get(gd, t, 0.0) + f - ag
                n[t] = get(n, t, 0) + 1
            end
        end
        teams = collect(keys(n))
        order = sortperm([(pts[t], gd[t]) for t in teams]; rev = true)
        for (rank, i) in enumerate(order)
            t = teams[i]
            push!(rows, (; team = t, season = String(first(g.season)), tier = C5_TIER[Int(first(g.tournament_id))],
                         ppg = pts[t] / n[t], gdpg = gd[t] / n[t], rank_frac = (rank - 0.5) / length(teams),
                         played = n[t]))
        end
    end
    return DataFrame(rows)
end

"""
    c5_team_lineup_features(ds, team_matches) -> DataFrame

Per (team, season): mean log production wealth of the starting XI over the team's first 3
league matches (`age_weight(RichardsSigmoid())`, as `ProductionWealthFeature`; missing values
skipped, NaN if no valued starter), and squad turnover = share of those starters who did not
start a league match for the club the previous season.
`team_matches`: rows (team, season, match_id, j) for the league matches.
"""
function c5_team_lineup_features(ds, team_matches::AbstractDataFrame)
    lu = ds.lineups[.!ds.lineups.is_substitute, :]
    meta = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
    kick = Dict(Int(r.match_id) => DateTime(r.match_date) for r in eachrow(ds.matches))
    side_team(mid, side) = String(side == "home" ? meta[mid].home_team : meta[mid].away_team)
    starters = Dict{Tuple{String,String},Set{Int}}()           # (team, season) → player ids
    byteam = Dict{Tuple{Int,String},Vector{Any}}()              # (match, team) → lineup rows
    curve = BayesianFootball.Models.PreGame.Builder.RichardsSigmoid()
    for r in eachrow(lu)
        mid = Int(r.match_id)
        haskey(meta, mid) || continue
        t = side_team(mid, String(r.team_side))
        push!(get!(starters, (t, String(meta[mid].season)), Set{Int}()), Int(r.player_id))
        push!(get!(byteam, (mid, t), Any[]), r)
    end
    rows = NamedTuple[]
    for g in groupby(team_matches[team_matches.j .<= 3, :], [:team, :season])
        t, s = String(first(g.team)), String(first(g.season))
        prev = get(starters, (t, c5_prev_season(s)), Set{Int}())
        lw = Float64[]; new = 0; tot = 0
        for mid in g.match_id
            rs = get(byteam, (Int(mid), t), Any[])
            isempty(rs) && continue
            val = 0.0; nval = 0
            for r in rs
                tot += 1
                Int(r.player_id) in prev || (new += 1)
                v = r.proposed_market_value
                (ismissing(v) || v <= 0) && continue
                age = BayesianFootball.Models.PreGame.Builder._cb_player_age(kick[Int(mid)], r.date_of_birth_timestamp, 26.5)
                val += v * BayesianFootball.Models.PreGame.Builder.age_weight(curve, age)
                nval += 1
            end
            nval >= 6 && push!(lw, log(val * 11 / nval))       # scale to a full XI
        end
        push!(rows, (; team = t, season = s, log_wealth = isempty(lw) ? NaN : mean(lw),
                     turnover = (tot == 0 || isempty(prev)) ? NaN : new / tot))
    end
    return DataFrame(rows)
end

"""
    c5_ols(y, X, names) -> NamedTuple

OLS with an intercept already in `X`; HC1 standard errors; R².
"""
function c5_ols(y::AbstractVector, X::AbstractMatrix, names::AbstractVector)
    n, k = size(X)
    β = X \ y
    e = y .- X * β
    XtXi = inv(Symmetric(X' * X))
    meat = X' * (X .* (e .^ 2))
    V = XtXi * meat * XtXi * n / (n - k)
    se = sqrt.(max.(diag(V), 0.0))
    r2 = 1 - sum(abs2, e) / sum(abs2, y .- mean(y))
    return (; coef = DataFrame(term = names, estimate = β, se = se, t = β ./ se), r2, n,
            rmse = sqrt(mean(abs2, e)))
end
