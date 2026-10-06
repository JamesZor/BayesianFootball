module LaplaceBookPreflight

import CSV
import DataFrames
import Distributions
import LinearAlgebra
import Optim
import SpecialFunctions
import Statistics

if !isdefined(parentmodule(@__MODULE__), :MarketModel)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l01_market_model.jl"))
end
const MM = parentmodule(@__MODULE__).MarketModel
const DF = DataFrames
const LA = LinearAlgebra

"Complete markets with their Phase A de-vigged probabilities; no new data gates."
function market_vectors(book)
    return [(selections=Symbol.(g.selection), p=Float64.(g.prob_fair_close))
        for g in DF.groupby(book, [:market_name, :market_line])]
end

"Exact Dirichlet density (two selections give exactly a Beta), including normalisers."
function book_logdensity(theta, markets, n)
    probabilities = MM.selection_probabilities(Vector{Float64}(theta))
    value = 0.0
    for market in markets
        q = [probabilities[s] for s in market.selections]
        alpha = n .* q
        value += SpecialFunctions.loggamma(sum(alpha))
        value -= sum(SpecialFunctions.loggamma, alpha)
        value += sum((alpha .- 1) .* log.(market.p))
    end
    return value
end

"The two KL orientations differ for inconsistent books; neither changes the score grid."
function reverse_kl(theta, markets)
    probabilities = MM.selection_probabilities(Vector{Float64}(theta))
    return sum(probabilities[s] * log(probabilities[s] / p)
        for market in markets for (s, p) in zip(market.selections, market.p))
end

"Central finite differences; the step is explicit so the numerical derivative can be checked."
function hessian_fd(f, theta; step=1e-4)
    H = zeros(2, 2)
    h, a = [step, 0.0], [0.0, step]
    centre = f(theta)
    H[1, 1] = (f(theta+h) - 2centre + f(theta-h)) / step^2
    H[2, 2] = (f(theta+a) - 2centre + f(theta-a)) / step^2
    H[1, 2] = H[2, 1] = (f(theta+h+a) - f(theta+h-a) -
        f(theta-h+a) + f(theta-h-a)) / (4step^2)
    return H
end

"Gauss–Hermite nodes/weights for standard-normal expectations, via the Jacobi matrix."
function normal_quadrature(order)
    decomposition = LA.eigen(LA.SymTridiagonal(zeros(order), sqrt.(Float64.(1:order-1))))
    return decomposition.values, decomposition.vectors[1, :].^2
end

"""
Deterministic 2-D quadrature of the EXACT book likelihood times N(mean, spread² I).
An optimised Gaussian proposal only changes coordinates: every integrand evaluation
uses the exact Dirichlet/Beta density, divided by the proposal density. There is no
Laplace approximation to the integrand. Orders 16 and 32 test quadrature stability.
"""
function exact_marginal(theta, markets, n, spread; order=32)
    prior = Distributions.MvNormal(theta, spread^2 * LA.I(2))
    logtarget = z -> book_logdensity(z, markets, n) + Distributions.logpdf(prior, z)
    optimum = Optim.optimize(z -> -logtarget(z), copy(theta), Optim.BFGS(),
        Optim.Options(g_tol=1e-7, iterations=1000))
    mode = Optim.minimizer(optimum)
    precision = hessian_fd(z -> -logtarget(z), mode)
    covariance = inv(LA.Symmetric(precision))
    proposal = Distributions.MvNormal(mode, LA.Symmetric(covariance))
    factor = LA.cholesky(LA.Symmetric(covariance)).L
    nodes, weights = normal_quadrature(order)
    terms = Float64[]
    for j in eachindex(nodes), i in eachindex(nodes)
        z = mode + factor * [nodes[i], nodes[j]]
        push!(terms, log(weights[i]) + log(weights[j]) + logtarget(z) -
            Distributions.logpdf(proposal, z))
    end
    largest = maximum(terms)
    return largest + log(sum(exp.(terms .- largest)))
end

"""
Literal Phase C brief approximation, with its specified forward-cross-entropy H.
For k=2, exp(log L(hat)) exp(-1/2 (theta-hat)' nH (theta-hat))
= exp(log L(hat)) (2pi)^(k/2) det(R)^(1/2) N(theta;hat,R), R=(nH)^-1.
Integrating a normal state prediction gives the convolution N(hat;mean,P+R).
For the brief's supremacy-only rule, replace R by d'R d and k by 1,
exactly retaining its stated n-dependent peak/determinant factor. This is a
DIFFERENT likelihood from the full 2-D Dirichlet likelihood: dropping a direction
cannot preserve its integral against arbitrary Gaussian predictions. Integrating
out the discarded direction with a state-independent Lebesgue measure is not
integrating it against the Gaussian state prediction.

Separately, Stirling's expansion of the exact Dirichlet log density is
-n sum_s q_s log(q_s/p_s) + (S-1)/2 log n + 1/2 sum_s log q_s
- sum_s log p_s - (S-1)/2 log(2pi) + O(1/n).
Thus its large-n maximiser minimises KL(q||p), not KL(p||q). The brief's
forward-cross-entropy Hessian is not generally the exact likelihood curvature.
No corrected model or sampler is implemented in this preflight.
"""
function brief_marginal(theta, markets, n, spread, H; supremacy_only=false)
    R = inv(LA.Symmetric(n .* H))
    peak = book_logdensity(theta, markets, n)
    if supremacy_only
        d = [1.0, -1.0] / sqrt(2)
        variance = LA.dot(d, R*d)
        return peak + 0.5log(2pi*variance) - 0.5log(2pi*(spread^2+variance))
    end
    return peak + log(2pi) + 0.5LA.logdet(R) +
        Distributions.logpdf(Distributions.MvNormal(theta, LA.Symmetric(R + spread^2*LA.I(2))), theta)
end

"Synthetic O/U generated on the unchanged production double-Poisson grid."
function synthetic_total_book(h, a)
    q = MM.selection_probabilities(log.([h, a]))
    return DF.DataFrame(match_id=[1, 1], market_name=["OverUnder", "OverUnder"],
        market_line=[2.5, 2.5], selection=[:under_25, :over_25],
        prob_fair_close=[q[:under_25], q[:over_25]])
end

"Deterministic necessary-condition checks, not a claim to the full production Phase C gate."
function preflight(ds, config, out)
    rates = CSV.read(joinpath(@__DIR__, "results", "A", "rates.csv"), DF.DataFrame)
    rates = DF.filter(r -> r.season in config.seasons && r.tournament in config.tournaments, rates)
    book, _ = MM.gated_close(ds, config)
    books = Dict(Int(first(g.match_id)) => DF.DataFrame(g) for g in DF.groupby(book, :match_id))
    rows = NamedTuple[]
    # All no-complete-1X2 books, not a handpicked subset. Flat optima are retained.
    for r in eachrow(DF.filter(r -> coalesce(r.reason, "") == "no complete 1X2 market", rates))
        b = books[r.match_id]
        fit = MM.solve_kl(b; starts=(MM.STARTS[1],))
        H = hessian_fd(z -> MM.kl_loss(z, b), fit.theta)
        eigen = LA.eigen(LA.Symmetric(H))
        kept = findall(eigen.values .> 1e-6maximum(eigen.values))
        v = eigen.vectors[:, end]
        angle = acosd(clamp(abs(sum(v)/sqrt(2)), 0, 1))
        push!(rows, (; match_id=r.match_id, lines=String(r.lines_quoted),
            lambda_h=exp(fit.theta[1]), lambda_a=exp(fit.theta[2]), kl=fit.kl,
            eigen_min=eigen.values[1], eigen_max=eigen.values[2], kept=length(kept),
            angle_degrees=angle, projection_pass=length(kept) == 1 && angle <= 10))
    end
    projections = DF.DataFrame(rows)
    CSV.write(joinpath(out, "totals_projection_preflight.csv"), projections)

    # Necessary quadrature check at the PRIOR median, not invented posterior medians.
    # Fixed ordered first 30 of each accepted type; no selection on measured errors.
    marginal_rows = NamedTuple[]
    for type in ("full", "1X2-only")
        selected = DF.filter(r -> r.accepted &&
            (type == "full" ? r.n_selections >= config.min_selections_ladder : r.n_selections == 3), rates)
        for r in eachrow(selected[1:30, :])
            b = books[r.match_id]
            markets = market_vectors(b)
            theta = log.([r.lambda_h, r.lambda_a])
            H = hessian_fd(z -> MM.kl_loss(z, b), theta)
            for n in (250.0, 1000.0, 4000.0), spread in (0.05, 0.20)
                exact16 = exact_marginal(theta, markets, n, spread; order=16)
                exact32 = exact_marginal(theta, markets, n, spread; order=32)
                full = brief_marginal(theta, markets, n, spread, H)
                prescribed = brief_marginal(theta, markets, n, spread, H;
                    supremacy_only=type == "1X2-only")
                push!(marginal_rows, (; book_type=type, match_id=r.match_id, n, spread,
                    exact16, exact32, quadrature_delta=abs(exact32-exact16),
                    laplace_full=full, laplace_prescribed=prescribed,
                    error=prescribed-exact32, full_error=full-exact32))
            end
        end
        println("C preflight quadrature completed: ", type)
        flush(stdout)
    end
    marginals = DF.DataFrame(marginal_rows)
    CSV.write(joinpath(out, "laplace_preflight_fixture.csv"), marginals)
    summaries = NamedTuple[]
    for g in DF.groupby(marginals, [:book_type, :n, :spread])
        errors = abs.(g.error)
        med = Statistics.median(errors)
        p95 = Statistics.quantile(errors, 0.95)
        push!(summaries, (; book_type=first(g.book_type), n=first(g.n), spread=first(g.spread),
            books=DF.nrow(g), median_abs_error=med, p95_abs_error=p95,
            max_quadrature_delta=maximum(g.quadrature_delta),
            median_full_abs_error=Statistics.median(abs.(g.full_error)),
            gate_pass=med <= 0.01 && p95 <= 0.05))
    end
    summary = DF.DataFrame(summaries)
    CSV.write(joinpath(out, "laplace_preflight_summary.csv"), summary)
    return (; projections, marginals, summary)
end

end # module
