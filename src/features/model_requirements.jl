# src/features/model_requirements.jl

# ==============================================================================
# 1. Feature Requirements (Fallbacks)
# ==============================================================================

"""
    required_features(model)
Returns the list of trait symbols (e.g., :team_ids, :goals) that the feature
builder must extract for this model. Concrete models should override this 
in their own engine files.
"""
function required_features(model::AbstractFootballModel)
    return [:team_ids, :time_indices] 
end

# Example fallback for specific abstract types
function required_features(model::AbstractDynamicDixonColesNegBinModel)
    return [:team_ids, :midweek, :month, :is_plastic, :time_indices]
end
