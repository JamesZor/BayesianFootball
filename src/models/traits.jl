# src/models/traits.jl

export AbstractDynamics, Static, GRW
export AbstractParameterization, StandardHomeAdvantage, HierarchicalHomeAdvantage 

# --- 1. Dynamics Traits (Time) ---
abstract type AbstractDynamics end

"""
Static: Parameters are constant (Vector).
"""
struct Static <: AbstractDynamics end

"""
GRW: Parameters evolve via Random Walk (Matrix).
"""
struct GRW <: AbstractDynamics end


# --- 2. Parameterization Traits (Logic) ---
abstract type AbstractParameterization end

"""
Standard: Global Home Advantage parameter (γ).
Equation: λ = exp(μ + γ + ...)
"""
struct StandardHomeAdvantage <: AbstractParameterization end

"""
Hierarchical: Team-specific Home Advantage (γ_h).
Equation: λ = exp(μ + γ[h] + ...)
Prior: γ[h] ~ Normal(γ_mean, σ_γ)
"""
struct HierarchicalHomeAdvantage <: AbstractParameterization end 
