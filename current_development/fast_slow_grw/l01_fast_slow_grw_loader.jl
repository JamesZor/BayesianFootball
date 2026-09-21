# l01_fast_slow_grw_loader.jl — Fast & Slow GRW Models & Geometric Rate Pooling
# Scottish Lower League Football (Tournaments 56 & 57)
#
# Clean minimal Poisson GRW implementations:
#   1. m01_poisson_grw_tight: Baseline handrail (tight Gaussian priors)
#   2. m02_poisson_grw_loose_var: Loose Candidate A (2.5x variance scale on σ₀)
#   3. m03_poisson_grw_loose_tdist: Loose Candidate B (Student-t(4) innovations)
#
# Also provides:
#   - Geometric Rate Pooling in λ-space
#   - Supremacy slope calculation vs Betfair closing line
#   - Score grid construction and portfolio evaluation hooks

using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, Distributions
import Turing

const BF = BayesianFootball
const PG = BF.Models.PreGame
const Builder = PG.Builder

# ==============================================================================
# 1. Model Builders
# ==============================================================================

"""
Build the baseline "tight" Poisson GRW model (the handrail).
High shrinkage, regularised team separation.
"""
function build_tight_poisson_grw_model(name::Symbol = :m01_poisson_grw_tight)
    dyn = PG.MultiScaleGRW(
        z₀ = Normal(0, 1),
        zₛ = Normal(0, 1),
        zₖ = Normal(0, 1),
        α_σ₀ = Gamma(2, 0.06),
        α_σₛ = Gamma(2, 0.03),
        α_σₖ = Gamma(2, 0.015),
        β_σ₀ = Gamma(2, 0.10),
        β_σₛ = Gamma(2, 0.055),
        β_σₖ = Gamma(2, 0.012),
    )
    b = Builder.CountModelBuilder(name)
    Builder.add!(b, PG.GlobalInterception())
    Builder.add!(b, PG.GlobalHomeAdvantage())
    Builder.add!(b, dyn)
    Builder.add!(b, Builder.PoissonObservation())
    return Builder.build(b)
end

"""
Build the "loose" variance-scaled Poisson GRW model (Candidate A).
Priors on initial team spread σ₀ scaled up ~2.5x to match the market-implied 2.43x spread requirement.
"""
function build_loose_var_poisson_grw_model(name::Symbol = :m02_poisson_grw_loose_var)
    dyn = PG.MultiScaleGRW(
        z₀ = Normal(0, 1),
        zₛ = Normal(0, 1),
        zₖ = Normal(0, 1),
        α_σ₀ = Gamma(2, 0.150),  # 2.5x of 0.06
        α_σₛ = Gamma(2, 0.075),  # 2.5x of 0.03
        α_σₖ = Gamma(2, 0.035),
        β_σ₀ = Gamma(2, 0.250),  # 2.5x of 0.10
        β_σₛ = Gamma(2, 0.1375), # 2.5x of 0.055
        β_σₖ = Gamma(2, 0.030),
    )
    b = Builder.CountModelBuilder(name)
    Builder.add!(b, PG.GlobalInterception())
    Builder.add!(b, PG.GlobalHomeAdvantage())
    Builder.add!(b, dyn)
    Builder.add!(b, Builder.PoissonObservation())
    return Builder.build(b)
end

"""
Build the "loose" heavy-tailed Student-t Poisson GRW model (Candidate B).
Uses TDist(4.0) for initial team spread and season jumps to allow heavy favourites to break out.
"""
function build_loose_tdist_poisson_grw_model(name::Symbol = :m03_poisson_grw_loose_tdist)
    dyn = PG.MultiScaleGRW(
        z₀ = TDist(4.0),
        zₛ = TDist(4.0),
        zₖ = Normal(0, 1),
        α_σ₀ = Gamma(2, 0.06),
        α_σₛ = Gamma(2, 0.03),
        α_σₖ = Gamma(2, 0.015),
        β_σ₀ = Gamma(2, 0.10),
        β_σₛ = Gamma(2, 0.055),
        β_σₖ = Gamma(2, 0.012),
    )
    b = Builder.CountModelBuilder(name)
    Builder.add!(b, PG.GlobalInterception())
    Builder.add!(b, PG.GlobalHomeAdvantage())
    Builder.add!(b, dyn)
    Builder.add!(b, Builder.PoissonObservation())
    return Builder.build(b)
end

# ==============================================================================
# 2. Geometric Rate Pooling (λ-space)
# ==============================================================================

"""
    geometric_rate_pool(lambda_tight, lambda_loose, w::Float64)

Pools two latent goal intensities in log-rate space:
    log(λ_blend) = (1 - w) * log(λ_tight) + w * log(λ_loose)
    λ_blend = (λ_tight)^(1-w) * (λ_loose)^w

Preserves score-grid coherence across 1X2, Totals, and BTTS.
"""
function geometric_rate_pool(lambda_tight::AbstractArray{<:Real}, lambda_loose::AbstractArray{<:Real}, w::Real)
    @assert 0.0 <= w <= 1.0 "Mixing weight w must be in [0.0, 1.0], got $w"
    @assert size(lambda_tight) == size(lambda_loose) "Shape mismatch: $(size(lambda_tight)) vs $(size(lambda_loose))"
    return exp.((1.0 - w) .* log.(max.(lambda_tight, 1e-6)) .+ w .* log.(max.(lambda_loose, 1e-6)))
end

"""
    linear_prob_pool(prob_tight, prob_loose, w::Float64)

Auxiliary baseline: linear mixture of probabilities:
    P_blend = (1 - w) * P_tight + w * P_loose
"""
function linear_prob_pool(prob_tight::AbstractArray{<:Real}, prob_loose::AbstractArray{<:Real}, w::Real)
    @assert 0.0 <= w <= 1.0 "Mixing weight w must be in [0.0, 1.0], got $w"
    return (1.0 - w) .* prob_tight .+ w .* prob_loose
end

# ==============================================================================
# 3. Supremacy & Metrics Helpers
# ==============================================================================

"""
Compute supremacy slope from OLS of model supremacy on market supremacy:
    sup_mkt = log(λ_mkt_h / λ_mkt_a)
    sup_model = log(λ_model_h / λ_model_a)
    sup_model = intercept + slope * sup_mkt
"""
function compute_supremacy_slope(sup_model::Vector{Float64}, sup_mkt::Vector{Float64})
    valid = isfinite.(sup_model) .& isfinite.(sup_mkt)
    n = count(valid)
    if n < 10
        return (slope = NaN, intercept = NaN, r2 = NaN, n = n)
    end
    x = sup_mkt[valid]
    y = sup_model[valid]
    x_mean = mean(x)
    y_mean = mean(y)
    cov_xy = sum((x .- x_mean) .* (y .- y_mean))
    var_x = sum((x .- x_mean).^2)
    var_y = sum((y .- y_mean).^2)
    slope = var_x > 1e-12 ? cov_xy / var_x : NaN
    intercept = y_mean - slope * x_mean
    r2 = (var_x > 1e-12 && var_y > 1e-12) ? (cov_xy^2) / (var_x * var_y) : NaN
    return (slope = slope, intercept = intercept, r2 = r2, n = n)
end

println("Loaded l01_fast_slow_grw_loader.jl successfully.")
