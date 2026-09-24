# l04 — Dixon–Coles network core shared by r04 (ratings) and r05 (walk-forward pricing).
#
#   log λ_home = μ + comp_c + h·Home + h_cup·Home·Cup + a_{i,s} + d_{j,s}
#   log λ_away = μ + comp_c +                            a_{j,s} + d_{i,s}
#   a_{i,s} = A_i + a′_{i,s},   d_{i,s} = D_i + d′_{i,s}    (club + club-season deviation)
#   P(x, y) = τ_ρ(x, y; λ_h, λ_a) · Pois(x; λ_h) · Pois(y; λ_a)
#
# Penalised MLE (= MAP):  A, D ~ N(0, σ_club²),  a′, d′ ~ N(0, σ_season²).
# Requires _common.jl, Optim and SpecialFunctions to be loaded by the caller.

# Parameter layout: μ, comp×3, h, h_cup, ρ | club A, D | club-season a′, d′.
# Everything the hot loop touches travels in one concretely-typed NamedTuple (no globals).
function build_network(fx)
    clubs = sort(unique(vcat(fx.home_team, fx.away_team)))
    cidx = Dict(c => i for (i, c) in enumerate(clubs))
    cs_keys = sort(unique(vcat(collect(zip(fx.home_team, fx.fs)), collect(zip(fx.away_team, fx.fs)))))
    csidx = Dict(k => i for (i, k) in enumerate(cs_keys))
    nc, ncs, nm = length(clubs), length(cs_keys), nrow(fx)
    NG = 7
    oA, oD, oa, od = NG, NG + nc, NG + 2nc, NG + 2nc + ncs
    X = Vector{Int}(fx.home_goals); Y = Vector{Int}(fx.away_goals)
    D = (; nm, nc, ncs, oA, oD, oa, od,
           hc = [cidx[t] for t in fx.home_team], ac = [cidx[t] for t in fx.away_team],
           hs = [csidx[(t, s)] for (t, s) in zip(fx.home_team, fx.fs)],
           as_ = [csidx[(t, s)] for (t, s) in zip(fx.away_team, fx.fs)],
           comp = [c == "Scottish Cup" ? 1 : c == "League Cup" ? 2 : c == "Challenge Cup" ? 3 : 0 for c in fx.competition],
           H = Float64.(.!fx.neutral), CUP = Float64.(fx.is_cup), X, Y,
           lgX = loggamma.(X .+ 1.0), lgY = loggamma.(Y .+ 1.0))
    catmap = Dict{Tuple{String, Int}, String}()
    for r in eachrow(fx)
        catmap[(r.home_team, r.fs)] = r.home_cat; catmap[(r.away_team, r.fs)] = r.away_cat
    end
    nplayed = countmap(vcat(collect(zip(fx.home_team, fx.fs)), collect(zip(fx.away_team, fx.fs))))
    return (; D, clubs, cidx, cs_keys, csidx, npar = NG + 2nc + 2ncs, catmap, nplayed,
              ybar = mean(vcat(X, Y)))
end

# Objective with analytic gradient.
function dc_fg!(G, p, σc, σs, D)
    (; nm, nc, ncs, oA, oD, oa, od, hc, ac, hs, as_, comp, H, CUP, X, Y, lgX, lgY) = D
    μ = p[1]; cmp = (p[2], p[3], p[4]); h = p[5]; hcup = p[6]; ρ = p[7]
    nll = 0.0
    G !== nothing && fill!(G, 0.0)
    @inbounds for m in 1:nm
        base = μ + (comp[m] == 0 ? 0.0 : cmp[comp[m]])
        lh = base + H[m] * (h + hcup * CUP[m]) + p[oA+hc[m]] + p[oa+hs[m]] + p[oD+ac[m]] + p[od+as_[m]]
        la = base + p[oA+ac[m]] + p[oa+as_[m]] + p[oD+hc[m]] + p[od+hs[m]]
        λ, ν = exp(lh), exp(la); x, y = X[m], Y[m]
        # Dixon–Coles τ and its derivatives (w.r.t. log λ, log ν, ρ)
        τ = 1.0; dτl = 0.0; dτn = 0.0; dτr = 0.0
        if x == 0 && y == 0
            τ = 1 - λ * ν * ρ; dτl = -λ * ν * ρ; dτn = -λ * ν * ρ; dτr = -λ * ν
        elseif x == 0 && y == 1
            τ = 1 + λ * ρ; dτl = λ * ρ; dτr = λ
        elseif x == 1 && y == 0
            τ = 1 + ν * ρ; dτn = ν * ρ; dτr = ν
        elseif x == 1 && y == 1
            τ = 1 - ρ; dτr = -1.0
        end
        τ = max(τ, 1e-10)
        nll -= x * lh - λ - lgX[m] + y * la - ν - lgY[m] + log(τ)
        if G !== nothing
            gl = -(x - λ + dτl / τ); gn = -(y - ν + dτn / τ)
            g0 = gl + gn
            G[1] += g0; comp[m] > 0 && (G[1+comp[m]] += g0)
            G[5] += gl * H[m]; G[6] += gl * H[m] * CUP[m]; G[7] -= dτr / τ
            G[oA+hc[m]] += gl; G[oa+hs[m]] += gl; G[oD+ac[m]] += gl; G[od+as_[m]] += gl
            G[oA+ac[m]] += gn; G[oa+as_[m]] += gn; G[oD+hc[m]] += gn; G[od+hs[m]] += gn
        end
    end
    # Gaussian penalties (MAP)
    @inbounds for k in 1:2nc
        v = p[oA+k]; nll += 0.5v^2 / σc^2; G !== nothing && (G[oA+k] += v / σc^2)
    end
    @inbounds for k in 1:2ncs
        v = p[oa+k]; nll += 0.5v^2 / σs^2; G !== nothing && (G[oa+k] += v / σs^2)
    end
    return nll
end

function fit_dc(N, σc, σs; p0 = nothing)
    p0 === nothing && (p0 = zeros(N.npar); p0[1] = log(N.ybar))
    res = optimize(Optim.only_fg!((F, G, p) -> dc_fg!(G, p, σc, σs, N.D)), p0,
                   LBFGS(m = 20), Optim.Options(iterations = 5000, g_tol = 1e-6))
    Optim.converged(res) || @warn "DC fit did not converge" σc σs
    return Optim.minimizer(res), res
end

# Category of a club-season = its (unique) category in that season.
function rating_table(N, p, lab, window)
    (; D, cs_keys, cidx, csidx, catmap, nplayed) = N
    t = DataFrame(window = window, spec = lab, team = first.(cs_keys), fs = last.(cs_keys),
                  season = fs_label.(last.(cs_keys)),
                  cat = [catmap[k] for k in cs_keys], n = [nplayed[k] for k in cs_keys],
                  a = [p[D.oA+cidx[k[1]]] + p[D.oa+csidx[k]] for k in cs_keys],
                  d = [p[D.oD+cidx[k[1]]] + p[D.od+csidx[k]] for k in cs_keys])
    t.theta = t.a .- t.d; t.old_firm = in.(t.team, Ref(OLD_FIRM))
    return t
end


"""
    predict_dc(N, p, home, away, fs, comp, neutral, cup)

Point-in-time rates for a fixture from a fit on strictly earlier data.  A club-season
seen in the fit uses A_i + a′_{i,s}; an unseen club-season falls back to the club level
A_i (the hierarchical cold start); an unseen club gets 0 (the network mean).
"""
function predict_dc(N, p, home, away, fs, comp, neutral, cup)
    D = N.D
    att(t) = (i = get(N.cidx, t, 0); j = get(N.csidx, (t, fs), 0);
              (i == 0 ? 0.0 : p[D.oA+i]) + (j == 0 ? 0.0 : p[D.oa+j]))
    con(t) = (i = get(N.cidx, t, 0); j = get(N.csidx, (t, fs), 0);
              (i == 0 ? 0.0 : p[D.oD+i]) + (j == 0 ? 0.0 : p[D.od+j]))
    base = p[1] + (comp == "Scottish Cup" ? p[2] : comp == "League Cup" ? p[3] : comp == "Challenge Cup" ? p[4] : 0.0)
    lh = base + (neutral ? 0.0 : p[5] + p[6] * cup) + att(home) + con(away)
    la = base + att(away) + con(home)
    return exp(lh), exp(la)
end

"1X2 probabilities under Dixon–Coles τ_ρ (ρ = 0 ⇒ independent Poisson)."
function probs_1x2(λ, ν, ρ = 0.0; maxg = 12)
    ph = pd = pa = 0.0
    for x in 0:maxg, y in 0:maxg
        τ = x == 0 && y == 0 ? 1 - λ * ν * ρ : x == 0 && y == 1 ? 1 + λ * ρ :
            x == 1 && y == 0 ? 1 + ν * ρ : x == 1 && y == 1 ? 1 - ρ : 1.0
        q = τ * exp(-λ - ν + x * log(λ) + y * log(ν) - loggamma(x + 1.0) - loggamma(y + 1.0))
        x > y ? (ph += q) : x == y ? (pd += q) : (pa += q)
    end
    s = ph + pd + pa
    return ph / s, pd / s, pa / s
end
