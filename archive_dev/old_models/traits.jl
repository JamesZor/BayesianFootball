# # src/models/traits.jl
#
# export AbstractDynamics, Static, GRW
# export AbstractParameterization, StandardHomeAdvantage, HierarchicalHomeAdvantage 
#
# # --- 1. Dynamics Traits (Time) ---
# abstract type AbstractDynamics end
#
# """
# Static: Parameters are constant (Vector).
# """
# struct Static <: AbstractDynamics end
#
# """
# GRW: Parameters evolve via Random Walk (Matrix).
# """
# struct GRW <: AbstractDynamics end
#
#
# # --- 2. Parameterization Traits (Logic) ---
# abstract type AbstractParameterization end
#
# """
# Standard: Global Home Advantage parameter (γ).
# Equation: λ = exp(μ + γ + ...)
# """
# struct StandardHomeAdvantage <: AbstractParameterization end
#
# """
# Hierarchical: Team-specific Home Advantage (γ_h).
# Equation: λ = exp(μ + γ[h] + ...)
# Prior: γ[h] ~ Normal(γ_mean, σ_γ)
# """
# struct HierarchicalHomeAdvantage <: AbstractParameterization end 
#


# src/models/traits.jl
using Distributions

export AbstractDynamics, Static, GRW
export AbstractParameterization, StandardHomeAdvantage, HierarchicalHomeAdvantage
export AbstractObservation, PoissonObservation, NegBinObservation

# ==============================================================================
# 1. DYNAMICS (Time Evolution)
#    Now holds the priors for the process variance (σ).
# ==============================================================================
abstract type AbstractDynamics end

Base.@kwdef struct Static <: AbstractDynamics
    # Prior for the magnitude of team strength (e.g., how spread out are teams?)
    σ_prior::Distribution = Truncated(Normal(0, 1), 0, Inf) 
end

Base.@kwdef struct GRW <: AbstractDynamics
    # Prior for the random walk step size (e.g., how much can a team change in 1 week?)
    σ_step_prior::Distribution = Truncated(Normal(0, 0.2), 0, Inf)
end


# ==============================================================================
# 2. PARAMETERIZATION (Linear Predictor)
#    Now holds priors for Intercepts (μ) and Home Advantage (γ).
# ==============================================================================
abstract type AbstractParameterization end

Base.@kwdef struct StandardHomeAdvantage <: AbstractParameterization
    μ_prior::Distribution = Normal(0.15, 0.5)
    γ_prior::Distribution = Normal(log(1.3), 0.2)
end

Base.@kwdef struct HierarchicalHomeAdvantage <: AbstractParameterization
    μ_prior::Distribution = Normal(0.15, 0.5)
    
    # In hierarchical, this is the prior for the MEAN home advantage
    γ_bar_prior::Distribution = Normal(log(1.3), 0.2)
    
    # Prior for the variance between stadiums
    σ_γ_prior::Distribution   = Truncated(Normal(0, 0.2), 0, Inf)
end


# ==============================================================================
# 3. OBSERVATION (Likelihood)
#    Now holds priors for dispersion (r) if needed.
# ==============================================================================
abstract type AbstractObservation end

struct PoissonObservation <: AbstractObservation 
    # Poisson has no extra parameters!
end

Base.@kwdef struct NegBinObservation <: AbstractObservation
    # Negative Binomial needs 'r' (dispersion)
    log_r_prior::Distribution = Normal(1.5, 1.0)
end
