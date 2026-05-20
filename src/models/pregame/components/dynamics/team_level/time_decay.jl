# src/models/pregame/components/dynamics/time_decay.jl

# ==========================================
# 1. CONFIGURATION
# ==========================================
Base.@kwdef struct TimeDecayDynamics <: AbstractDynamicsConfig
    days_half_life::Real = 180
    σ_att::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
    σ_def::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
end

# ==========================================
# 2. TURING SUBMODEL
# ==========================================
@model function build_dynamics(config::TimeDecayDynamics, n_teams::Int)
    # Global variance for attack and defense spread
    σ_a ~ config.σ_att
    σ_d ~ config.σ_def
    
    # Non-centered parameterization (the Z-scores)
    raw_a ~ filldist(Normal(0, 1), n_teams)
    raw_d ~ filldist(Normal(0, 1), n_teams)
    
    # Scale them
    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    
    # Zero-sum constraint (ensures league average is exactly 0)
    α = α_scaled .- mean(α_scaled)
    β = β_scaled .- mean(β_scaled)
    
    return (; α, β)
end

# ==========================================
# 3. EXTRACTOR
# ==========================================
function extract_dynamics(chain::Chains, ::TimeDecayDynamics, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    
    # 1. Extract the global standard deviations
    σ_a = vec(Array(chain[Symbol("$prefix.σ_a")]))
    σ_d = vec(Array(chain[Symbol("$prefix.σ_d")]))
    
    # 2. Extract the raw Z-scores
    raw_a_matrix = zeros(n_samples, n_teams)
    raw_d_matrix = zeros(n_samples, n_teams)
    
    for i in 1:n_teams
        raw_a_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_a[$i]")]))
        raw_d_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_d[$i]")]))
    end
    
    # 3. Reconstruct the scaled parameters
    α_scaled = raw_a_matrix .* σ_a 
    β_scaled = raw_d_matrix .* σ_d
    
    # 4. Apply the Zero-Sum constraint
    α_matrix = α_scaled .- mean(α_scaled, dims=2)
    β_matrix = β_scaled .- mean(β_scaled, dims=2)
    
    return (; α = α_matrix, β = β_matrix)
end
