# src/features/types.jl

"""
    AbstractFeatureConfig

Base abstract type for all feature configurations.
"""
abstract type AbstractFeatureConfig end

# --- Core Features ---
struct TeamIDsFeature <: AbstractFeatureConfig end
struct GoalsFeature <: AbstractFeatureConfig end

# --- Stats Features ---
struct XGFeature <: AbstractFeatureConfig end
struct ShotsFeature <: AbstractFeatureConfig end

# --- Market Features ---
struct MarketLambdaFeature <: AbstractFeatureConfig end

# --- Time Features ---
struct TimeIndicesFeature <: AbstractFeatureConfig end
struct DatesFeature <: AbstractFeatureConfig end
struct MonthFeature <: AbstractFeatureConfig end
struct MidweekFeature <: AbstractFeatureConfig end
struct PlasticPitchFeature <: AbstractFeatureConfig end

# --- Player Tracking Features ---
"""
    AbstractRatingTracker

Abstract type for player rating tracking algorithms (e.g., EWMA, Bayesian).
"""
abstract type AbstractRatingTracker end

struct PlayerRatingsFeature{T <: AbstractRatingTracker} <: AbstractFeatureConfig
    tracker::T
end
