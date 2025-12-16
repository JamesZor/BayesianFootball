# src/training/methods.jl

using ..Models.PreGame: build_turing_model
using ...TypesInterfaces: AbstractFootballModel
export train

# --- Core Single-Split Train Function ---


function train(model::AbstractFootballModel, config::TrainingConfig, feature_set::FeatureSet)
    # This logic remains the same: Build -> Run Sampler
    turing_model = build_turing_model(model, feature_set) 
    return run_sampler(turing_model, config.sampler)
end

# --- Main Driver Function ---

"""
    train(model, config, feature_sets)

Main entry point that dispatches to the correct strategy implementation.
"""
function train(
    model::AbstractFootballModel, 
    config::TrainingConfig, 
    feature_sets::Vector{<:Tuple{FeatureSet, M}}
) where M
    
    if config.strategy isa Independent
        return train_independent(model, config, feature_sets)
    
    elseif config.strategy isa SequentialPriorUpdate
        error("Sequential strategy not yet refactored.")
        # return train_sequential(model, config, feature_sets)
        
    else
        error("Unknown strategy type: $(typeof(config.strategy))")
    end
end

