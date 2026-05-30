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
abstract type AbstractMarketFeatureConfig <: AbstractFeatureConfig end

# 1. Double Poisson (Independent, ρ = 0)
Base.@kwdef struct DoublePoissonMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = (:result_1x2, :btts, :over_15, :under_15, :over_25, :under_25, :over_35, :under_35)
end

# 2. Dixon-Coles (with ρ)
Base.@kwdef struct DixonColesMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = (:result_1x2, :btts, :over_15, :under_15, :over_25, :under_25, :over_35, :under_35)
end

# 3. Regularized Frank Copula Negative Binomial
Base.@kwdef struct RegularizedFrankCopulaMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = (:result_1x2, :btts, :over_15, :under_15, :over_25, :under_25, :over_35, :under_35)
    prior_r::Float64 = 15.0
    penalty_weight::Float64 = 0.05
end

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
