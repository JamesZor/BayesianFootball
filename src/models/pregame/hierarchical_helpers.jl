# src/models/pregame/hierarchical_helpers.jl

export hierarchical_zero_centered_component, reconstruct_hierarchical_centered

"""
    hierarchical_zero_centered_component(n_items, dist_σ, dist_z)

A Turing submodel that applies Partial Pooling, Non-Centered Parameterization (NCP), 
and a sum-to-zero constraint. Perfect for team strengths, month effects, etc.
"""
@model function hierarchical_zero_centered_component(
        n_items::Int, 
        dist_σ::Distribution, 
        dist_z::Distribution
    )
    
    # 1. Sample the hierarchical variance (Partial Pooling)
    σ ~ dist_σ
    
    # 2. Sample the raw z-scores (NCP)
    z ~ filldist(dist_z, n_items)
    
    # 3. Scale and Center (Sum-to-zero constraint)
    raw = z .* σ
    centered = raw .- mean(raw)
    
    return centered
end

"""
    reconstruct_hierarchical_centered(chain, prefix)

Reconstructs the centered array from the chain using the saved `σ` and `z` values.
Returns an array of shape: [Samples, Items].
"""
function reconstruct_hierarchical_centered(chain::Chains, prefix::String)
    # Extract σ vector: [Samples,]
    σ_vec = vec(Array(chain[Symbol("$prefix.σ")])) 
    
    # Extract z matrix: [Samples, Items]
    # We use `group` to safely grab all elements of the array `z`
    z_raw = Array(group(chain, Symbol("$prefix.z"))) 
    
    # Multiply z by σ (broadcasting σ across columns)
    raw_val = z_raw .* reshape(σ_vec, :, 1)
    
    # Subtract the mean of each row to enforce sum-to-zero
    centered = raw_val .- mean(raw_val, dims=2)
    
    return centered
end
