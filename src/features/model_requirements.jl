# src/features/model_requirements.jl

# ==============================================================================
# 1. Feature Requirements
# ==============================================================================

# Fallback: What features does a standard generic model need?
function required_features(model::AbstractFootballModel)
    return [:team_ids, :time_indices] 
end

# Override for MultiScaled models
function required_features(model::AbstractMultiScaledNegBinModel)
    return [:team_ids, :midweek, :month, :is_plastic, :time_indices]
end


function required_features(model::AbstractDynamicDixonColesNegBinModel)
    return [:team_ids, :midweek, :month, :is_plastic, :time_indices]
end

# ==============================================================================
# 2. Model-Specific Preprocessing Hooks
# ==============================================================================

# 1. Fallback Behavior (Covers all Static models automatically!)
# If Julia doesn't find a specific rule for a model, it will default to this.
function apply_model_specific_logic(model::AbstractFootballModel, df::DataFrame)
    return df
end

# 2. Group all dynamic/time-dependent models into a Type Alias
const TimeSortedModels = Union{
    AbstractDynamicPoissonModel,
    AbstractDynamicDixonColesModel,
    AbstractDynamicNegBinModel,
    AbstractDynamicBivariatePoissonModel,
    AbstractFunnelModel,
    AbstractMultiScaledNegBinModel,
    AbstractDynamicDixonColesNegBinModel
}

# 3. Define the sorting logic exactly ONCE for all of them!
function apply_model_specific_logic(model::TimeSortedModels, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end



"""
    target_columns(model)
Returns the list of columns required from the dataframe (e.g. goals, shots)
to ensure we filter out missing rows correctly.
"""
# Default behavior (Existing models): Just needs scores
target_columns(::AbstractFootballModel) = [:home_score, :away_score]

# Funnel behavior: Needs scores + Shot Data
target_columns(::AbstractFunnelModel) = [
    :home_score, :away_score, 
    :HS, :AS,   # Home/Away Shots
    :HST, :AST  # Home/Away Shots on Target
]
