# l08_score_grid.jl — loader for brief 08: is our score grid mis-weighted?
#
# Plumbing for r08_score_grid.jl:
#   §1  held-out per-match inputs from a persisted fit: the model's own score grid (src typed
#       latents → `Predictions.compute_score_grid!`, averaged over every posterior draw) and an
#       evenly thinned set of (λh, λa) draws for reshaping;
#   §2  observed vs expected scorelines (0..4 + 5+), draws, totals, goal difference, dispersion,
#       with a parametric bootstrap from the models' own grids;
#   §3  one-shape reshaped grids holding each draw's goal means: Dixon–Coles ρ, bivariate Poisson
#       λ₃ (U = λh − λ₃), COM-Poisson ν (rates solved to hold the means), and bivariate + COM;
#       exact-score MLE on training seasons; held-out exact-score / 1X2 / O-U 2.5 log-loss.
#
# Only SELECTs touch a database. No sampling; the only fits are 1–2 parameter MLEs.

include(joinpath(@__DIR__, "l07_halftime.jl"))

const C8_G = 12                       # reshaped grids: goals 0..12 per side
const C8_N = C8_G + 1
const C8_TAB = 6                      # O/E tables: 0,1,2,3,4,5+
const C8_SHAPES = (:dc, :bp, :com, :bpcom)
const C8_SHAPE_LABEL = Dict(:poisson => "Poisson (as fitted)", :dc => "(a) Dixon–Coles ρ",
                            :bp => "(b) bivariate λ₃, means held", :com => "(c) COM ν, means held",
                            :bpcom => "(d) bivariate λ₃ + COM ν")
const C8_THETA0 = Dict(:poisson => 0.0, :dc => 0.0, :bp => 0.0, :com => 1.0, :bpcom => (0.0, 1.0))


# ==============================================================================
# 1. PER-MATCH INPUTS FROM A FIT
# ==============================================================================

"One held-out match: result, the model's own mean score grid (0..11 × 0..11), thinned λ draws."
struct C8Match
    match_id::Int
    date::Date
    season::String
    tid::Int
    hs::Int
    as::Int
    grid::Matrix{Float64}
    λh::Vector{Float64}
    λa::Vector{Float64}
end

"""
    c8_extract(fit, ids, meta; nthin = 400) -> Vector{C8Match}

The fit's typed latents (`as_typed_latents`), per match the posterior-mean score grid from the
src kernel over ALL draws, and `nthin` evenly spaced (λh, λa) draws. Matches without a final
score are skipped.
"""
function c8_extract(fit, ids, meta; nthin::Int = 400)
    tl = Base.invokelatest(as_typed_latents, fit.latents, fit.config.model)
    tl isa BayesianFootball.Models.CountLatents || error("expected CountLatents, got $(typeof(tl))")
    tl.observation_params === nothing || error("expected a Poisson (observation_params = nothing) family")
    pos = Dict(Int(m) => i for (i, m) in enumerate(tl.match_ids))
    D = size(tl.λ_home, 2)
    sel = unique(round.(Int, range(1, D; length = min(D, nthin))))
    ws = Predictions.GridWorkspace(12)
    S = Predictions.alloc_score_grid(tl, 12)
    out = C8Match[]
    for id in ids
        haskey(pos, id) || continue
        r = meta[id]
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        i = pos[id]
        Predictions.compute_score_grid!(S, ws, tl, i)
        g = dropdims(mean(S; dims = 3); dims = 3)
        push!(out, C8Match(id, Date(r.match_date), String(r.season), Int(r.tournament_id),
                           Int(r.home_score), Int(r.away_score), g ./ sum(g),
                           Float64.(tl.λ_home[i, sel]), Float64.(tl.λ_away[i, sel])))
    end
    return out
end


# ==============================================================================
# 2. OBSERVED VS EXPECTED
# ==============================================================================

_c8_b(k) = min(k, C8_TAB - 1) + 1            # goals → table bucket 1..6

"Expected (Σ grid) and observed 6×6 tables."
function c8_oe(ms::Vector{C8Match})
    E = zeros(C8_TAB, C8_TAB); O = zeros(C8_TAB, C8_TAB)
    for m in ms
        for j in axes(m.grid, 2), i in axes(m.grid, 1)
            E[_c8_b(i - 1), _c8_b(j - 1)] += m.grid[i, j]
        end
        O[_c8_b(m.hs), _c8_b(m.as)] += 1
    end
    return E, O
end

"Per-match summary moments of the grid: P(draw), mean and variance of total goals, P(total = k), P(GD = d)."
function c8_moments(g::AbstractMatrix)
    pd = 0.0; mt = 0.0; m2 = 0.0
    tot = zeros(C8_TAB); gd = zeros(9)                     # GD ≤ −4, −3..+3, ≥ +4
    for j in axes(g, 2), i in axes(g, 1)
        p = g[i, j]; t = i + j - 2; d = (i - 1) - (j - 1)
        i == j && (pd += p)
        mt += p * t; m2 += p * t^2
        tot[min(t, C8_TAB - 1) + 1] += p
        gd[clamp(d, -4, 4) + 5] += p
    end
    return (; pd, mean = mt, var = m2 - mt^2, tot, gd)
end

"""
    c8_bootstrap(ms, E; B, seed) -> NamedTuple

Parametric bootstrap from each match's own grid: B replicate seasons of scorelines. Returns
the replicate χ² (pooled cells, as for the observed), draw counts, total-goal counts, GD counts
and total-goal sample variances.
"""
function c8_bootstrap(ms::Vector{C8Match}, pool::BitMatrix, E::Matrix{Float64}; B::Int = 2_000, seed::Int = 20260928)
    cdfs = [cumsum(vec(m.grid)) for m in ms]
    n = size(ms[1].grid, 1)
    rng = MersenneTwister(seed)
    chi = zeros(B); draws = zeros(B); tots = zeros(B, C8_TAB); gds = zeros(B, 9); vars = zeros(B)
    O = zeros(C8_TAB, C8_TAB); t = zeros(length(ms))
    for b in 1:B
        fill!(O, 0.0)
        for (k, c) in enumerate(cdfs)
            u = rand(rng) * c[end]
            idx = min(searchsortedfirst(c, u), length(c))
            i = (idx - 1) % n; j = (idx - 1) ÷ n
            O[_c8_b(i), _c8_b(j)] += 1
            draws[b] += i == j
            t[k] = i + j
            tots[b, min(i + j, C8_TAB - 1) + 1] += 1
            gds[b, clamp(i - j, -4, 4) + 5] += 1
        end
        chi[b] = c8_chi2(O, E, pool)
        vars[b] = var(t)
    end
    return (; chi, draws, tots, gds, vars)
end

"Pearson χ² over cells with E ≥ 5, the rest pooled into one cell."
function c8_chi2(O, E, pool::BitMatrix)
    s = 0.0
    for i in eachindex(E)
        pool[i] || (s += (O[i] - E[i])^2 / E[i])
    end
    Ep = sum(E[pool]); Op = sum(O[pool])
    Ep > 0 && (s += (Op - Ep)^2 / Ep)
    return s
end

"ASCII table of a 6×6 matrix (rows home goals, cols away goals)."
function c8_ascii(M; fmt = "%+6.2f", title = "")
    io = IOBuffer()
    println(io, title)
    println(io, "        a0      a1      a2      a3      a4     a5+")
    f = Printf.Format(" " * fmt * " ")
    for i in 1:C8_TAB
        print(io, i == C8_TAB ? "h5+ " : "h$(i - 1)  ")
        for j in 1:C8_TAB
            print(io, Printf.format(f, M[i, j]))
        end
        println(io)
    end
    return String(take!(io))
end


# ==============================================================================
# 3. RESHAPED GRIDS HOLDING THE MEANS
# ==============================================================================

@inline function _c8_pois!(v, λ)
    v[1] = exp(-λ)
    @inbounds for k in 1:C8_G
        v[k+1] = v[k] * λ / k
    end
    return v
end

"COM-Poisson pmf with rate r and dispersion ν on 0..C8_G (normalised on the truncation)."
@inline function _c8_com!(v, r, ν)
    lw = 0.0; mx = 0.0
    v[1] = 0.0
    @inbounds for k in 1:C8_G
        lw += log(r) - ν * log(k)
        v[k+1] = lw
        mx = max(mx, lw)
    end
    s = 0.0
    @inbounds for k in 1:C8_N
        v[k] = exp(v[k] - mx); s += v[k]
    end
    v ./= s
    return v
end

"""
    c8_com_table(ν) -> (logμ, logr)

The COM rate that gives mean μ at dispersion ν, on a log grid of μ ∈ [0.01, 9] (Newton on log r;
d mean / d log r = variance). Linear interpolation in logs holds the mean to < 1e-4.
"""
function c8_com_table(ν::Real)
    lμ = collect(range(log(0.01), log(9.0); length = 600))
    lr = similar(lμ)
    v = zeros(C8_N)
    x = log(0.01)
    for (k, target) in enumerate(lμ)
        for _ in 1:60
            _c8_com!(v, exp(x), ν)
            m = sum((i - 1) * v[i] for i in 1:C8_N)
            s2 = sum((i - 1)^2 * v[i] for i in 1:C8_N) - m^2
            step = (log(m) - target) * m / max(s2, 1e-12)
            x -= clamp(step, -2.0, 2.0)
            abs(step) < 1e-11 && break
        end
        lr[k] = x
    end
    return (lμ, lr)
end

@inline function c8_com_rate(tab, μ)
    lμ, lr = tab
    x = clamp(log(μ), lμ[1], lμ[end])
    k = clamp(searchsortedlast(lμ, x), 1, length(lμ) - 1)
    w = (x - lμ[k]) / (lμ[k+1] - lμ[k])
    return exp(lr[k] + w * (lr[k+1] - lr[k]))
end

"""
    c8_draw_grid!(P, kind, θ, λh, λa, tab, u, v, w)

One posterior draw's reshaped grid (C8_N × C8_N), marginal means held at (λh, λa):
  :poisson  independent Poisson;
  :dc       Dixon–Coles τ on 0-0/0-1/1-0/1-1 with ρ = θ (cells clamped at 0, renormalised);
  :bp       X = U + C, Y = V + C, C ~ Pois(λ₃), U ~ Pois(λh − λ₃), λ₃ = min(θ, 0.95·min(λh, λa));
  :com      COM-Poisson marginals with ν = θ, rates solved so the means are λh, λa (`tab`);
  :bpcom    θ = (λ₃, ν): U, V COM with means λh − λ₃, λa − λ₃, plus the common Poisson C.
"""
function c8_draw_grid!(P, kind::Symbol, θ, λh, λa, tab, u, v, w)
    if kind === :poisson || kind === :dc
        _c8_pois!(u, λh); _c8_pois!(v, λa)
        @inbounds for j in 1:C8_N, i in 1:C8_N
            P[i, j] = u[i] * v[j]
        end
        if kind === :dc
            P[1, 1] *= 1 - λh * λa * θ; P[1, 2] *= 1 + λh * θ
            P[2, 1] *= 1 + λa * θ;      P[2, 2] *= 1 - θ
            s = 0.0
            @inbounds for k in eachindex(P)
                P[k] = max(P[k], 0.0); s += P[k]
            end
            P ./= s
        end
    elseif kind === :com
        _c8_com!(u, c8_com_rate(tab, λh), θ); _c8_com!(v, c8_com_rate(tab, λa), θ)
        @inbounds for j in 1:C8_N, i in 1:C8_N
            P[i, j] = u[i] * v[j]
        end
    else
        l3 = kind === :bp ? θ : θ[1]
        l3 = min(l3, 0.95 * min(λh, λa))
        if kind === :bp
            _c8_pois!(u, λh - l3); _c8_pois!(v, λa - l3)
        else
            _c8_com!(u, c8_com_rate(tab, λh - l3), θ[2]); _c8_com!(v, c8_com_rate(tab, λa - l3), θ[2])
        end
        _c8_pois!(w, l3)
        @inbounds for j in 0:C8_G, i in 0:C8_G
            s = 0.0
            for k in 0:min(i, j)
                s += w[k+1] * u[i-k+1] * v[j-k+1]
            end
            P[i+1, j+1] = s
        end
    end
    return P
end

"Posterior-mean reshaped grid of one match (over its thinned draws)."
function c8_match_grid(m::C8Match, kind::Symbol, θ, tab)
    G = zeros(C8_N, C8_N); P = zeros(C8_N, C8_N)
    u, v, w = zeros(C8_N), zeros(C8_N), zeros(C8_N)
    for k in eachindex(m.λh)
        c8_draw_grid!(P, kind, θ, m.λh[k], m.λa[k], tab, u, v, w)
        G .+= P
    end
    return G ./ sum(G)
end

_c8_tab(kind, θ) = kind === :com ? c8_com_table(θ) : kind === :bpcom ? c8_com_table(θ[2]) : nothing

"Exact-score negative log-likelihood (mean per match) of `ms` under a shape."
function c8_nll(ms::Vector{C8Match}, kind::Symbol, θ)
    tab = _c8_tab(kind, θ)
    ll = zeros(length(ms))
    Threads.@threads for k in eachindex(ms)
        m = ms[k]
        P = zeros(C8_N, C8_N); u, v, w = zeros(C8_N), zeros(C8_N), zeros(C8_N)
        s = 0.0
        (m.hs > C8_G || m.as > C8_G) && (ll[k] = log(1e-12); continue)
        for d in eachindex(m.λh)
            c8_draw_grid!(P, kind, θ, m.λh[d], m.λa[d], tab, u, v, w)
            s += P[m.hs+1, m.as+1]
        end
        ll[k] = log(max(s / length(m.λh), 1e-300))
    end
    return -mean(ll)
end

"Fit one shape on training matches: Brent for 1 parameter, Nelder–Mead for (λ₃, ν)."
function c8_fit_shape(ms::Vector{C8Match}, kind::Symbol)
    if kind === :dc
        r = optimize(θ -> c8_nll(ms, :dc, θ), -0.25, 0.25; abs_tol = 1e-5)
    elseif kind === :bp
        r = optimize(θ -> c8_nll(ms, :bp, θ), 0.0, 0.5; abs_tol = 1e-5)
    elseif kind === :com
        r = optimize(θ -> c8_nll(ms, :com, θ), 0.6, 1.6; abs_tol = 1e-5)
    else
        f(z) = c8_nll(ms, :bpcom, (clamp(z[1], 0.0, 0.5), clamp(z[2], 0.6, 1.6)))
        r = optimize(f, [0.05, 1.05], NelderMead(), Optim.Options(x_abstol = 1e-4, f_reltol = 1e-9, iterations = 200))
        z = Optim.minimizer(r)
        return (clamp(z[1], 0.0, 0.5), clamp(z[2], 0.6, 1.6)), Optim.minimum(r)
    end
    return Optim.minimizer(r), Optim.minimum(r)
end

"Per-match held-out scores and probabilities under a shape: exact / 1X2 / O-U 2.5 log-loss, P(draw), P(1X2), P(under 2.5)."
function c8_scores(ms::Vector{C8Match}, kind::Symbol, θ)
    tab = _c8_tab(kind, θ)
    rows = Vector{NamedTuple}(undef, length(ms))
    Threads.@threads for k in eachindex(ms)
        m = ms[k]
        G = c8_match_grid(m, kind, θ, tab)
        ph = pd = pa = pu = 0.0
        for j in 1:C8_N, i in 1:C8_N
            i > j ? (ph += G[i, j]) : i == j ? (pd += G[i, j]) : (pa += G[i, j])
            (i + j - 2) <= 2 && (pu += G[i, j])
        end
        res = m.hs > m.as ? ph : m.hs == m.as ? pd : pa
        under = (m.hs + m.as) <= 2
        pex = (m.hs <= C8_G && m.as <= C8_G) ? G[m.hs+1, m.as+1] : 1e-12
        rows[k] = (; match_id = m.match_id, date = m.date, season = m.season, tid = m.tid,
                   ll_exact = -log(max(pex, 1e-12)), ll_1x2 = -log(max(res, 1e-12)),
                   ll_ou25 = -log(max(under ? pu : 1 - pu, 1e-12)), p_draw = pd, draw = m.hs == m.as,
                   p_home = ph, p_away = pa, p_under = pu)
    end
    return DataFrame(rows)
end

"l01-style model probability frame (1X2 + O/U 2.5) from a `c8_scores` frame."
function c8_probs_frame(sc::AbstractDataFrame)
    rows = NamedTuple[]
    for r in eachrow(sc)
        for (s, p) in ((:home, r.p_home), (:draw, r.p_draw), (:away, r.p_away))
            push!(rows, (; match_id = r.match_id, market_name = "1X2", market_line = 0.0, selection = s, p_model = p))
        end
        push!(rows, (; match_id = r.match_id, market_name = "OverUnder", market_line = 2.5, selection = :under_25, p_model = r.p_under))
        push!(rows, (; match_id = r.match_id, market_name = "OverUnder", market_line = 2.5, selection = :over_25, p_model = 1 - r.p_under))
    end
    return DataFrame(rows)
end
