# src/models/pregame/implementations/multi_grw_neg_bin.jl


using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics


export MSNegativeBinomial 

Base.@kwdef struct MSNegativeBinomial <: AbstractMultiScaledNegBinModel
    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.5)

    # Standard priors for team strength
    γ::Distribution   = Normal(log(1.3), 0.2)
    
    # Dispersion parameter (Negative Binomial)
    log_r::Distribution = Normal(2.5, 0.5) 

    # --- Dynamic Hyperparameters (Process Noise) ---
    # Adjusted to 0.05 to prevent excessive volatility (factor of ~1.05 per week)
    σₖ::Distribution = Truncated(Normal(0, 0.05), 0, Inf) # micro 
    σₛ::Distribution = Truncated(Normal(0, 0.05), 0, Inf) # macro 
    
    # --- Initial State Hyperparameters (Hierarchical Prior t=0) ---
    σ₀::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

    z_init::Distribution = Normal(0,1)
    z_steps::Distribution = Normal(0,1)
end


