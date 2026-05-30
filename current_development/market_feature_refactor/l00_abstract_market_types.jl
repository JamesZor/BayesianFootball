# current_development/market_feature_refactor/l00_abstract_market_types.jl

# ==============================================================================
# Feature Configuration Types
# ==============================================================================
# We define an abstract type to sit within the project's existing feature tree.
# We assume `AbstractFeatureConfig` is defined somewhere in `src/features/types.jl`.
# For the purpose of this isolated prototype, we mock it:
abstract type AbstractFeatureConfig end

abstract type AbstractMarketFeatureConfig <: AbstractFeatureConfig end

# ------------------------------------------------------------------------------
# 1. Dixon-Coles Market Feature
# ------------------------------------------------------------------------------
# Note the use of `Tuple{Vararg{Symbol}}` for `lines`. Tuples are stack-allocated
# and type-stable, which is critical for AD-safety when looping over them in the optimizer.
Base.@kwdef struct DixonColesMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = (:result_1x2, :btts, :uo_25)
end

# ------------------------------------------------------------------------------
# 2. Frank Copula Market Feature
# ------------------------------------------------------------------------------
Base.@kwdef struct FrankCopulaMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = (:result_1x2, :btts, :uo_25)
    # We could eventually add fixed dispersion bounds or priors here.
end

