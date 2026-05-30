# current_development/market_feature_refactor/l02_market_extractor.jl

# We include the math layer
include("l01_market_inverse_math.jl")

# ==============================================================================
# The Generalized Extractor
# ==============================================================================

"""
    add_feature!(F_data::Dict, config::AbstractMarketFeatureConfig, ordered_ids::Vector{Int}, mock_db::Dict)

This generalized extractor takes ANY market feature configuration, dynamically calculates the required
inverse market parameters, and unrolls them into the flat `F_data` dictionary. 
It strictly enforces AD-safety by padding missing match data with `NaN`.
"""
function add_feature!(F_data::Dict, config::AbstractMarketFeatureConfig, ordered_ids::Vector{Int}, mock_db::Dict)
    
    # 1. We mock the threaded extraction. In reality this loops over `ds.Matches` and calls `fit_market_implied_parameters`
    market_map = Dict{Int, NamedTuple}()
    for id in ordered_ids
        if haskey(mock_db, id)
            # Find the targets for this match
            targets = mock_db[id]
            # Run the dynamic optimizer
            params = fit_market_implied_parameters(targets, config)
            market_map[id] = params
        else
            # Missing data! (We just skip adding it to the map)
        end
    end
    
    # 2. Dynamic Unrolling into F_data
    # We need to know what keys to expect. We can safely get a "dummy" tuple by running
    # `extract_parameters` on the initial guess.
    dummy_params = extract_parameters(config, get_initial_guess(config))
    
    for key in keys(dummy_params)
        dict_key = Symbol("flat_market_", key)
        
        # Here is the AD-safe NaN padding:
        # We look up the match id. If it exists, we get the `key` value.
        # If it doesn't, we return NaN. 
        # This creates a perfectly flat, type-stable Vector{Float64}
        F_data[dict_key] = [
            haskey(market_map, id) ? market_map[id][key] : NaN 
            for id in ordered_ids
        ]
    end
    
    return F_data
end
