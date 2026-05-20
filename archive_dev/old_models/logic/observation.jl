# # src/models/logic/observation.jl
# using Distributions, Turing
#
# export sample_dispersion, observe_goals
#
# # ==============================================================================
# # 1. SAMPLING HELPER 
# # ==============================================================================
#
# # Even though Poisson does nothing, we make it a @model to satisfy the interface
# @model function sample_dispersion(::PoissonObservation)
#     return nothing 
# end
#
# @model function sample_dispersion(obs::NegBinObservation)
#     log_r ~ obs.log_r_prior
#     return exp(log_r)
# end
#
# # ==============================================================================
# # 2. LIKELIHOOD HELPER
# # ==============================================================================
#
# function observe_goals(::PoissonObservation, goals, λ, r)
#     goals ~ Poisson(λ)
# end
#
# function observe_goals(::NegBinObservation, goals, λ, r)
#     # Negative Binomial parameterization (r, p) where p = r / (r + λ)
#     goals ~ NegativeBinomial(r, r / (r + λ)) 
# end
#

# src/models/logic/observation.jl
using Distributions, Turing

export sample_dispersion, make_observation_dist # <--- Renamed export

# 1. SAMPLING HELPER (Unchanged)
@model function sample_dispersion(::PoissonObservation)
    return nothing 
end

@model function sample_dispersion(obs::NegBinObservation)
    log_r ~ obs.log_r_prior
    return exp(log_r)
end

# 2. DISTRIBUTION FACTORY (The Fix)
# Instead of performing the tilde (~), we just return the object.

function make_observation_dist(::PoissonObservation, λ, r)
    return Poisson(λ)
end

function make_observation_dist(::NegBinObservation, λ, r)
    # Using the (r, p) parameterization for NegBin where mean = λ
    p = r / (r + λ)
    return NegativeBinomial(r, p)
end
