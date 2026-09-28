# l03_price_movement.jl — loader for brief 03 (REPORT.md §11–§12; r03_leadtime_monitor.jl,
# r03_price_movement.jl).
#
# Builds on l02 (and so l01). Adds a per-selection PRICE-PATH PANEL: for every complete
# market, the last traded price, the tick before it, and a 45-min step-function TWA at the
# market's earliest complete instant and on a fixed grid of instants, plus tick counts. On
# top of it: de-vig per variant, a bounce-free "drop the last tick" price, an entry frame in
# l01's selection-frame shape, slate-clustered OLS and a conditional-logit (softmax) fit of
# outcomes on the close plus a movement term.
#
# Filtration: every panel quantity at instant t reads only ticks at or before t. The close
# (TWA (−20, 0], de-vigged, from l01) and outcomes are joined for scoring only.

include(joinpath(@__DIR__, "l02_clv_leadtime.jl"))

const C3_GRID = [("T−48h", -2880.0), ("T−24h", -1440.0), ("T−12h", -720.0), ("T−6h", -360.0),
                 ("T−3h", -180.0), ("T−2h", -120.0), ("T−90m", -90.0), ("T−60m", -60.0),
                 ("T−45m", -45.0), ("T−25m", -25.0), ("T−10m", -10.0), ("off", 0.0)]
const C3_INSTS = vcat(["earliest"], first.(C3_GRID))
const C3_TIME = Dict(C3_GRID)
const C3_DISCOVERY = Set(["21/22", "22/23", "23/24"])
const C3_CONFIRMATION = Set(["24/25", "25/26"])


# ==============================================================================
# 1. THE PRICE-PATH PANEL
# ==============================================================================

"""
    c3_step_twa(xs, ps, a, b) -> Float64

Time-weighted average over [a, b] of the last-traded-price step function (price ps[i] holds
from xs[i] to xs[i+1]); a tick before `a` carries in, ticks after `b` are never read. NaN if
no tick is at or before `b`.
"""
function c3_step_twa(xs::AbstractVector, ps::AbstractVector, a::Real, b::Real)
    k = searchsortedlast(xs, b)
    k == 0 && return NaN
    num = 0.0; den = 0.0
    for i in max(searchsortedlast(xs, a), 1):k
        lo = max(xs[i], a)
        hi = i < k ? xs[i + 1] : b
        hi > lo || continue
        num += ps[i] * (hi - lo); den += hi - lo
    end
    return den > 0 ? num / den : ps[k]
end

"""
    c3_path_panel(ds, close_book; families, grid = C3_GRID, twa_window = 45.0, latest = -25.0)

One row per selection of every complete pre-off market in `families` whose runners had all
traded by T`latest` (the l01 earliest-book rule). Columns of the matrices are
`C3_INSTS` = ["earliest", grid...]; "earliest" is the market's own first complete instant t0.

- `ltp`   last traded odds at or before the instant (NaN before the runner's first tick)
- `ltime` minutes-to-kick-off of that tick
- `prev`  odds of the tick before it (NaN if none)
- `twa`   step-function TWA of odds over [t − twa_window, t]
- `ticks` cumulative runner ticks at or before the instant; `ticks60` those in (t − 60, t]
- `lag`, `lagtime` last traded odds at or before t − `lag` (15 min) and its time

`keys` carries match/market/selection, t0, league, season, slate date, outcome and the
de-vigged close `p_close` (NaN when the close market is incomplete). Rows of one market are
contiguous; `ranges` indexes them.
"""
function c3_path_panel(ds, close_book::AbstractDataFrame;
                       families = (("1X2", 0.0), ("OverUnder", 2.5)),
                       grid = C3_GRID, twa_window::Real = 45.0, latest::Real = -25.0,
                       match_ids = nothing, lag::Real = 15.0)
    bo = ds.betfair_odds
    keep = match_ids === nothing ? trues(nrow(bo)) : in.(bo.match_id, Ref(Set(match_ids)))
    fam = falses(nrow(bo))
    for (n, l) in families
        fam .|= (bo.market_name .== n) .& (bo.market_line .== l)
    end
    keep .&= fam .& (bo.minutes_to_kickoff .<= 0.0) .& isfinite.(bo.traded_price) .& (bo.traded_price .> 1.0)
    sub = bo[keep, [:match_id, :market_name, :market_line, :selection, :minutes_to_kickoff, :traded_price]]
    nI = length(grid) + 1
    krows = NamedTuple[]
    cols = Dict(k => Vector{Vector{Float64}}() for k in (:ltp, :ltime, :prev, :twa, :ticks, :ticks60, :lag, :lagtime))
    for g in groupby(sub, CN_MKEY)
        want = cn_expected_runners(first(g.market_name))
        sels = groupby(g, :selection)
        length(sels) == want || continue
        t0 = maximum(minimum(s.minutes_to_kickoff) for s in sels)
        t0 <= latest || continue
        ts = vcat([t0], last.(grid))
        for s in sels
            o = sortperm(s.minutes_to_kickoff)
            xs = Float64.(s.minutes_to_kickoff[o]); ps = Float64.(s.traded_price[o])
            v = Dict(k => fill(NaN, nI) for k in keys(cols))
            for (j, t) in enumerate(ts)
                k = searchsortedlast(xs, t)
                if k > 0
                    v[:ltp][j] = ps[k]; v[:ltime][j] = xs[k]
                    k > 1 && (v[:prev][j] = ps[k - 1])
                end
                kl = searchsortedlast(xs, t - lag)
                if kl > 0
                    v[:lag][j] = ps[kl]; v[:lagtime][j] = xs[kl]
                end
                v[:twa][j] = c3_step_twa(xs, ps, t - twa_window, t)
                v[:ticks][j] = k
                v[:ticks60][j] = k - searchsortedlast(xs, t - 60.0)
            end
            for k in keys(cols)
                push!(cols[k], v[k])
            end
            push!(krows, (; match_id = Int(first(g.match_id)), market_name = String(first(g.market_name)),
                          market_line = Float64(first(g.market_line)), selection = Symbol(first(s.selection)),
                          t0 = t0))
        end
    end
    kf = DataFrame(krows)
    oc = cn_outcomes(ds, kf)
    ocd = Dict((r.match_id, r.market_name, r.market_line, r.selection) => r for r in eachrow(oc))
    cl = Dict((r.match_id, r.market_name, r.market_line, r.selection) => r.p_close for r in eachrow(close_book))
    kk = collect(zip(kf.match_id, kf.market_name, kf.market_line, kf.selection))
    kf.won = [ocd[k].won for k in kk]
    kf.date = [ocd[k].date for k in kk]
    kf.tournament_id = [ocd[k].tournament_id for k in kk]
    kf.season = [ocd[k].season for k in kk]
    kf.p_close = [get(cl, k, NaN) for k in kk]
    kf.family = cn_family.(kf.market_name, kf.market_line)
    mat(k) = permutedims(reduce(hcat, cols[k]))
    return c3_finish((; keys = kf, ltp = mat(:ltp), ltime = mat(:ltime), prev = mat(:prev),
                      twa = mat(:twa), ticks = mat(:ticks), ticks60 = mat(:ticks60),
                      lag = mat(:lag), lagtime = mat(:lagtime)))
end

"Attach contiguous per-market row ranges."
function c3_finish(P)
    k = P.keys
    ranges = UnitRange{Int}[]
    i = 1
    while i <= nrow(k)
        j = i
        while j < nrow(k) && k.match_id[j + 1] == k.match_id[i] &&
              k.market_name[j + 1] == k.market_name[i] && k.market_line[j + 1] == k.market_line[i]
            j += 1
        end
        push!(ranges, i:j)
        i = j + 1
    end
    return merge(P, (; ranges))
end

"Whole-market subset of the panel (a market is kept if `keep_market(first row)` is true)."
function c3_subset(P, keep_row::AbstractVector{Bool})
    rows = Int[]
    for r in P.ranges
        keep_row[first(r)] && append!(rows, r)
    end
    return c3_finish((; keys = P.keys[rows, :], ltp = P.ltp[rows, :], ltime = P.ltime[rows, :],
                      prev = P.prev[rows, :], twa = P.twa[rows, :], ticks = P.ticks[rows, :],
                      ticks60 = P.ticks60[rows, :], lag = P.lag[rows, :], lagtime = P.lagtime[rows, :]))
end

c3_subset(P; seasons = nothing, tournaments = nothing, family = nothing) = c3_subset(P,
    (seasons === nothing ? trues(nrow(P.keys)) : in.(P.keys.season, Ref(Set(seasons)))) .&
    (tournaments === nothing ? trues(nrow(P.keys)) : in.(P.keys.tournament_id, Ref(Set(tournaments)))) .&
    (family === nothing ? trues(nrow(P.keys)) : (P.keys.family .== family)))

"Column index of an instant label."
c3_j(inst::AbstractString) = findfirst(==(inst), C3_INSTS)

"Instant time per row: t0 for \"earliest\", else the grid time."
c3_time(P, inst) = inst == "earliest" ? Float64.(P.keys.t0) : fill(C3_TIME[inst], nrow(P.keys))


# ==============================================================================
# 2. PRICES FROM THE PANEL
# ==============================================================================

"""
    c3_devig(P, odds; band = (0.90, 1.10)) -> Vector{Float64}

Multiplicative de-vig within each market of a per-row odds vector. NaN for the whole market
if any runner is missing or the overround is outside `band`.
"""
function c3_devig(P, odds::AbstractVector; band = (0.90, 1.10))
    p = fill(NaN, length(odds))
    for r in P.ranges
        imp = 1.0 ./ view(odds, r)
        all(isfinite, imp) || continue
        over = sum(imp)
        band[1] <= over <= band[2] || continue
        p[r] .= imp ./ over
    end
    return p
end

"De-vigged probability at an instant from price variant `v` ∈ (:ltp, :prev, :twa)."
c3_p(P, v::Symbol, inst) = c3_devig(P, view(getfield(P, v), :, c3_j(inst)))

"""
    c3_bf_move(P, inst_m, inst_a; gap = 0.0) -> Vector{Float64}

Bounce-free move over the window (inst_a, inst_m] on the raw implied scale (1/odds, per runner,
no de-vig — a de-vig would mix other runners' boundary ticks back in):
x = 1/o_end − 1/o_twa(inst_a), where o_end NEVER is the last tick at inst_m, so x shares no tick
with anything measured from that tick (the entry, or y = p_close − 1/o_ltp(inst_m)).
- gap = 0:  o_end is the tick before the last one ("drop the last tick");
- gap = 15: o_end is the last tick at or before inst_m − 15 min (or the tick before the last,
  if that is the same tick).
With no tick inside the window, x = 0 (no movement observed). NaN if the runner has no price.
"""
function c3_bf_move(P, inst_m, inst_a; gap::Real = 0.0)
    jm = c3_j(inst_m); ja = c3_j(inst_a)
    ta = c3_time(P, inst_a)
    x = fill(NaN, nrow(P.keys))
    for i in eachindex(x)
        lt = P.ltime[i, jm]; oa = P.twa[i, ja]
        (isfinite(lt) && isfinite(oa)) || continue
        if lt <= ta[i]
            x[i] = 0.0
            continue
        end
        o_end = gap == 0 ? P.prev[i, jm] : (P.lagtime[i, jm] < lt ? P.lag[i, jm] : P.prev[i, jm])
        x[i] = 1 / o_end - 1 / oa
    end
    return x
end

"Rows whose market is fresh at `inst` under the l02 gate (oldest runner tick ≤ max(90, 25% lead))."
function c3_fresh(P, inst)
    inst == "earliest" && return trues(nrow(P.keys))
    t = C3_TIME[inst]
    gate = cl_max_staleness(t)
    j = c3_j(inst)
    out = falses(nrow(P.keys))
    for r in P.ranges
        st = t .- view(P.ltime, r, j)
        all(isfinite, st) && maximum(st) <= gate && (out[r] .= true)
    end
    return out
end

"Rows whose market had formed (every runner traded) by `inst`."
c3_formed(P, inst) = inst == "earliest" ? trues(nrow(P.keys)) : P.keys.t0 .<= C3_TIME[inst]

"""
    c3_entry_frame(P, inst; mask) -> DataFrame

l01-shaped selection frame for a back at the last traded price at `inst` (fresh-gated,
complete, in band), scored against the de-vigged close: odds_entry, p_entry_fair, p_close,
clv_pp, clv_pct, beat, pnl_gross, won, date. `edge_pp` is filled by the caller.
"""
function c3_entry_frame(P, inst; mask = trues(nrow(P.keys)))
    j = c3_j(inst)
    pf = c3_p(P, :ltp, inst)
    ok = mask .& c3_fresh(P, inst) .& isfinite.(pf) .& isfinite.(P.keys.p_close) .& .!ismissing.(P.keys.won)
    k = P.keys
    f = DataFrame(match_id = k.match_id[ok], market_name = k.market_name[ok], market_line = k.market_line[ok],
                  selection = k.selection[ok], date = k.date[ok], tournament_id = k.tournament_id[ok],
                  season = k.season[ok], family = k.family[ok], won = Bool.(k.won[ok]),
                  odds_entry = P.ltp[ok, j], p_entry_fair = pf[ok], p_close = k.p_close[ok],
                  row = findall(ok))
    f.p_entry = 1.0 ./ f.odds_entry
    f.clv_pp = 100 .* (f.p_close .- f.p_entry)
    f.clv_pct = 100 .* (f.odds_entry .* f.p_close .- 1.0)
    f.beat = f.clv_pp .> 0
    f.pnl_gross = ifelse.(f.won, f.odds_entry .- 1.0, -1.0)
    f.edge_pp = zeros(nrow(f))
    return f
end


# ==============================================================================
# 3. STATISTICS WITH SLATE CLUSTERING
# ==============================================================================

"""
    c3_ols(x, y, slate; B = 1000, seed) -> NamedTuple

OLS slope of y on x (with intercept) over finite pairs, with a slate-clustered bootstrap
(slates resampled with replacement; the slope is recomputed from resampled slate moments).
`p2` is the two-sided bootstrap p (2·min tail share, floor 1/B), `p_le0`/`p_ge0` the one-sided
tail shares (floor 1/B); `z` = slope / bootstrap SD.
"""
function c3_ols(x::AbstractVector, y::AbstractVector, slate::AbstractVector; B::Int = 1000,
                seed::Int = 20260928)
    ok = isfinite.(x) .& isfinite.(y)
    n = count(ok)
    n < 30 && return (; n, n_slates = 0, slope = NaN, lo = NaN, hi = NaN, z = NaN, p2 = NaN,
                      p_le0 = NaN, p_ge0 = NaN, r = NaN)
    xs = x[ok]; ys = y[ok]; ss = slate[ok]
    us = unique(ss); sidx = Dict(s => i for (i, s) in enumerate(us)); S = length(us)
    m = zeros(S, 5)          # n, Σx, Σy, Σxx, Σxy
    for i in eachindex(xs)
        k = sidx[ss[i]]
        m[k, 1] += 1; m[k, 2] += xs[i]; m[k, 3] += ys[i]; m[k, 4] += xs[i]^2; m[k, 5] += xs[i] * ys[i]
    end
    slope(t) = (t[5] / t[1] - t[2] * t[3] / t[1]^2) / (t[4] / t[1] - (t[2] / t[1])^2)
    b = slope(vec(sum(m; dims = 1)))
    rng = MersenneTwister(seed)
    reps = Vector{Float64}(undef, B)
    w = zeros(S)
    for r in 1:B
        fill!(w, 0.0)
        for _ in 1:S
            w[rand(rng, 1:S)] += 1.0
        end
        reps[r] = slope(vec(sum(w .* m; dims = 1)))
    end
    p2 = max(2 * min(mean(reps .<= 0), mean(reps .>= 0)), 1 / B)
    return (; n, n_slates = S, slope = b, lo = quantile(reps, 0.025), hi = quantile(reps, 0.975),
            z = b / std(reps), p2, p_le0 = max(mean(reps .<= 0), 1 / B), p_ge0 = max(mean(reps .>= 0), 1 / B),
            r = cor(xs, ys))
end

"""
    c3_mean(x, slate; B = 1000, seed) -> (n, mean, lo, hi, z, p_le0, p_ge0)

Mean of finite x with a slate-clustered bootstrap CI and one-sided tail shares (floor 1/B).
"""
function c3_mean(x::AbstractVector, slate::AbstractVector; B::Int = 1000, seed::Int = 20260928)
    ok = isfinite.(x)
    n = count(ok)
    n < 10 && return (; n, mean = NaN, lo = NaN, hi = NaN, z = NaN, p_le0 = NaN, p_ge0 = NaN)
    xs = x[ok]; ss = slate[ok]
    us = unique(ss); sidx = Dict(s => i for (i, s) in enumerate(us)); S = length(us)
    sn = zeros(S); sx = zeros(S)
    for i in eachindex(xs)
        sn[sidx[ss[i]]] += 1; sx[sidx[ss[i]]] += xs[i]
    end
    rng = MersenneTwister(seed)
    reps = Vector{Float64}(undef, B)
    w = zeros(S)
    for r in 1:B
        fill!(w, 0.0)
        for _ in 1:S
            w[rand(rng, 1:S)] += 1.0
        end
        reps[r] = sum(w .* sx) / sum(w .* sn)
    end
    μ = mean(xs)
    return (; n, mean = μ, lo = quantile(reps, 0.025), hi = quantile(reps, 0.975), z = μ / std(reps),
            p_le0 = max(mean(reps .<= 0), 1 / B), p_ge0 = max(mean(reps .>= 0), 1 / B))
end

"""
    c3_diff(xa, sa, xb, sb; B = 1000, seed) -> (n_a, n_b, diff, lo, hi, z, p_le0, p_ge0)

mean(xa) − mean(xb) with a joint slate bootstrap (slates of both groups resampled together).
"""
function c3_diff(xa::AbstractVector, sa::AbstractVector, xb::AbstractVector, sb::AbstractVector;
                 B::Int = 1000, seed::Int = 20260928)
    (length(xa) < 10 || length(xb) < 10) &&
        return (; n_a = length(xa), n_b = length(xb), diff = NaN, lo = NaN, hi = NaN, z = NaN, p_le0 = NaN, p_ge0 = NaN)
    us = unique(vcat(sa, sb)); sidx = Dict(s => i for (i, s) in enumerate(us)); S = length(us)
    m = zeros(S, 4)          # n_a, Σx_a, n_b, Σx_b
    for i in eachindex(xa)
        m[sidx[sa[i]], 1] += 1; m[sidx[sa[i]], 2] += xa[i]
    end
    for i in eachindex(xb)
        m[sidx[sb[i]], 3] += 1; m[sidx[sb[i]], 4] += xb[i]
    end
    stat(t) = t[2] / t[1] - t[4] / t[3]
    d = stat(vec(sum(m; dims = 1)))
    rng = MersenneTwister(seed)
    reps = Vector{Float64}(undef, B)
    w = zeros(S)
    for r in 1:B
        fill!(w, 0.0)
        for _ in 1:S
            w[rand(rng, 1:S)] += 1.0
        end
        reps[r] = stat(vec(sum(w .* m; dims = 1)))
    end
    return (; n_a = length(xa), n_b = length(xb), diff = d, lo = quantile(reps, 0.025),
            hi = quantile(reps, 0.975), z = d / std(reps),
            p_le0 = max(mean(reps .<= 0), 1 / B), p_ge0 = max(mean(reps .>= 0), 1 / B))
end

"Holm step-down: adjusted p-values (monotone) for a vector of one-sided p-values."
function c3_holm(p::AbstractVector)
    m = length(p); o = sortperm(p); adj = similar(p, Float64)
    run = 0.0
    for (k, i) in enumerate(o)
        run = max(run, min(1.0, (m - k + 1) * p[i]))
        adj[i] = run
    end
    return adj
end


# ==============================================================================
# 4. OUTCOMES GIVEN THE CLOSE: CONDITIONAL LOGIT
# ==============================================================================
#
# Per market, P(runner i wins) = softmax_i(c·log p_close_i + g·m_i). (c, g) = (1, 0) is the
# close itself; c ≠ 1 is a temperature (favourite–longshot) correction; g ≠ 0 says the
# movement m carries result information the close did not price.

"Market-level design for the conditional logit: (X1, X2, winner index, ranges, slates)."
function c3_clogit_data(P, m::AbstractVector)
    X1 = Float64[]; X2 = Float64[]; win = Int[]; rg = UnitRange{Int}[]; sl = Date[]
    for r in P.ranges
        pc = P.keys.p_close[r]; mm = m[r]; w = P.keys.won[r]
        (all(isfinite, pc) && all(isfinite, mm) && !any(ismissing, w) && count(Bool.(w)) == 1) || continue
        a = length(X1) + 1
        append!(X1, log.(pc)); append!(X2, mm)
        push!(win, a - 1 + findfirst(Bool.(w)))
        push!(rg, a:length(X1)); push!(sl, P.keys.date[first(r)])
    end
    return (; X1, X2, win, rg, sl)
end

"Mean per-market log-likelihood at θ = (c, g), weighted by market weights w."
function c3_clogit_ll(D, θ; w = ones(length(D.rg)))
    ll = 0.0
    for (k, r) in enumerate(D.rg)
        w[k] == 0 && continue
        s = θ[1] .* view(D.X1, r) .+ θ[2] .* view(D.X2, r)
        mx = maximum(s)
        ll += w[k] * (s[D.win[k] - first(r) + 1] - mx - log(sum(exp.(s .- mx))))
    end
    return ll / sum(w)
end

"Newton fit of θ = (c, g); `free` selects the fitted coordinates (others stay at (1, 0))."
function c3_clogit_fit(D; w = ones(length(D.rg)), free = (true, true))
    θ = [1.0, 0.0]
    idx = findall(collect(free))
    for _ in 1:60
        g = zeros(2); H = zeros(2, 2)
        for (k, r) in enumerate(D.rg)
            w[k] == 0 && continue
            x1 = view(D.X1, r); x2 = view(D.X2, r)
            s = θ[1] .* x1 .+ θ[2] .* x2
            e = exp.(s .- maximum(s)); p = e ./ sum(e)
            b1 = sum(p .* x1); b2 = sum(p .* x2)
            iw = D.win[k] - first(r) + 1
            g[1] += w[k] * (x1[iw] - b1); g[2] += w[k] * (x2[iw] - b2)
            for i in eachindex(p)
                d1 = x1[i] - b1; d2 = x2[i] - b2
                H[1, 1] -= w[k] * p[i] * d1 * d1; H[1, 2] -= w[k] * p[i] * d1 * d2
                H[2, 2] -= w[k] * p[i] * d2 * d2
            end
        end
        H[2, 1] = H[1, 2]
        step = H[idx, idx] \ g[idx]
        θ[idx] .-= step
        maximum(abs, step) < 1e-9 && break
    end
    return θ
end

"""
    c3_clogit(D; B = 300, seed) -> NamedTuple

Fits (c only) and (c, g); reports g with a slate-bootstrap CI and the in-sample log-loss gain
(nats per market ×1000) of the close+movement fit over the close and over the temperature-only
fit.
"""
function c3_clogit(D; B::Int = 300, seed::Int = 20260928)
    M = length(D.rg)
    M < 50 && return (; n_markets = M, c_only = NaN, c = NaN, g = NaN, g_lo = NaN, g_hi = NaN, g_z = NaN,
                      gain_vs_close = NaN, gain_vs_temp = NaN)
    θc = c3_clogit_fit(D; free = (true, false))
    θ = c3_clogit_fit(D)
    l0 = c3_clogit_ll(D, [1.0, 0.0]); lc = c3_clogit_ll(D, θc); l = c3_clogit_ll(D, θ)
    us = unique(D.sl); byslate = Dict(s => Int[] for s in us)
    for (k, s) in enumerate(D.sl)
        push!(byslate[s], k)
    end
    rng = MersenneTwister(seed)
    reps = Float64[]
    for _ in 1:B
        w = zeros(M)
        for _ in us
            for k in byslate[us[rand(rng, 1:length(us))]]
                w[k] += 1
            end
        end
        push!(reps, c3_clogit_fit(D; w)[2])
    end
    return (; n_markets = M, c_only = θc[1], c = θ[1], g = θ[2], g_lo = quantile(reps, 0.025),
            g_hi = quantile(reps, 0.975), g_z = θ[2] / std(reps),
            gain_vs_close = 1000 * (l - l0), gain_vs_temp = 1000 * (l - lc))
end


# ==============================================================================
# 5. PRINTING
# ==============================================================================

"""
    c3_bars(labels, series; width = 24, common = true) -> String

Unsigned horizontal bars: one block per series (name, values, fmt). With `common`, every
series shares one scale (the max over all finite values).
"""
function c3_bars(labels, series; width::Int = 24, common::Bool = true)
    io = IOBuffer()
    lw = maximum(length.(labels))
    allv = reduce(vcat, [filter(isfinite, v) for (_, v, _) in series])
    gm = maximum(abs, allv; init = 1e-9)
    println(io, rstrip(rpad("", lw) * "  " * join((rpad(n, width + 9) for (n, _, _) in series), "")))
    for (i, lab) in enumerate(labels)
        line = rpad(lab, lw) * "  "
        for (_, v, fmt) in series
            m = common ? gm : maximum(abs, filter(isfinite, v); init = 1e-9)
            k = isfinite(v[i]) ? clamp(round(Int, abs(v[i]) / m * width), 0, width) : 0
            line *= "▏" * rpad("█"^k, width) * " " * rpad(isfinite(v[i]) ? fmt(v[i]) : "—", 7)
        end
        println(io, rstrip(line))
    end
    return String(take!(io))
end
