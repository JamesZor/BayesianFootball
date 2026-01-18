# src/models/logic/parameterization.jl
using Distributions

export sample_hfa_deviations, get_hfa_val

# ==============================================================================
# 1. SAMPLING HELPER (Get the raw deviations)
# ==============================================================================

"""
sample_hfa_deviations(trait, n_teams)

Decides if we need to sample 'raw' home advantages for every team.
"""
function sample_hfa_deviations(::StandardHomeAdvantage, n_teams)
    # Return 'nothing' to signal that this model does NOT need this parameter.
    return nothing
end

function sample_hfa_deviations(::HierarchicalHomeAdvantage, n_teams)
    # Hierarchical model needs a raw deviation for EVERY team.
    # We sample N standard normals (Non-Centered Parameterization).
    return filldist(Normal(0, 1), n_teams)
end

# ==============================================================================
# 2. CALCULATION HELPER (Compute the final Gamma)
# ==============================================================================

"""
get_hfa_val(trait, γ_mean, σ_γ, raw_devs, team_id)

Computes the specific Home Advantage for a specific team.
"""
function get_hfa_val(::StandardHomeAdvantage, γ_mean, σ_γ, raw_devs, team_id)
    # Standard: The "Mean" IS the parameter. 
    # We ignore sigma, deviations, and team_id.
    return γ_mean
end

function get_hfa_val(::HierarchicalHomeAdvantage, γ_mean, σ_γ, raw_devs, team_id)
    # Hierarchical: Start at Mean, add (Deviation * Spread).
    # We look up the specific deviation for this team_id.
    return γ_mean + (raw_devs[team_id] * σ_γ)
end
