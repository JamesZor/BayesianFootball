# src/features/model_requirements.jl

# ==============================================================================
# 1. Feature Requirements (Fallbacks)
# ==============================================================================

"""
    required_features(model)
Returns the list of feature configurations (AbstractFeatureConfig) that the feature
builder must extract for this model. Concrete models should override this 
in their own engine files.
"""
function required_features(model::AbstractFootballModel)
    return AbstractFeatureConfig[TeamIDsFeature(), TimeIndicesFeature()] 
end

# Example fallback for specific abstract types
function required_features(model::AbstractDynamicDixonColesNegBinModel)
    return AbstractFeatureConfig[
        TeamIDsFeature(), 
        MidweekFeature(), 
        MonthFeature(), 
        PlasticPitchFeature(), 
        TimeIndicesFeature()
    ]
end
