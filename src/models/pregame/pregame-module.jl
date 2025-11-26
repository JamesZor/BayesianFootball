# src/models/pregame/pregame-module.jl

"""
This module organizes all pre-game models.
"""
module PreGame

# This module also only depends on the central interfaces.
# The '...' goes up two levels from PreGame -> Models -> BayesianFootball to find TypesInterfaces.
using ...TypesInterfaces

# Shared abstract types are now in the main interfaces file.
# include("interfaces.jl")
# Shared, reusable likelihood functions
include("turing_helpers.jl")

# Directory for concrete model implementations
module Implementations
    # It also only needs TypesInterfaces for its contracts.
    # '....' goes up three levels from Implementations -> PreGame -> Models -> BayesianFootball
    using ..TypesInterfaces: AbstractFootballModel, AbstractPregameModel, AbstractInGameModel, AbstractPoissonModel, AbstractNegBinModel, AbstractInflatedDiagonalPoissonModel, FeatureSet, Vocabulary, AbstractDixonColesModel, AbstractGRWPoissonModel
    # Each model is now in its own self-contained file
    include("./models-src/static-poisson.jl")
    include("./models-src/static-simplex-poisson.jl")
    include("./models-src/hierarchical-simplex-poisson.jl")
    include("./models-src/static-dixoncoles.jl")

    include("./models-src/grw-poisson.jl")


    # --- Your "wrapper" function ---
    """
    OVERLOADED METHOD: Extracts parameters for all pre-split dataframes.

    Processes a vector of results, applying each one to a corresponding
    pre-filtered DataFrame from `dfs_to_predict`.

    This function assumes a "lagged" relationship:
    - `results_vector[1]` is applied to `dfs_to_predict[1]`
    - `results_vector[2]` is applied to `dfs_to_predict[2]`
    ...and so on.

    The `zip` function automatically handles the edge case where `results_vector`
    is one element longer than `dfs_to_predict`.
    """
    function extract_parameters(
        model::StaticPoisson,
        dfs_to_predict::AbstractVector, 
        vocabulary::Vocabulary,
        results_vector::AbstractVector
    )
        PredictionValue = NamedTuple{(:λ_h, :λ_a), Tuple{AbstractVector{Float64}, AbstractVector{Float64}}}

        # 1. Allocate memory for the *combined* output dictionary
        full_extraction_dict = Dict{Int64, PredictionValue}()

        # 2. Iterate through the results and dataframes in parallel
        #    `zip` stops at the shortest vector, which perfectly handles
        #    the case where `results` has one extra model.
        for (result_tuple, df_for_this_split) in zip(results_vector, dfs_to_predict)
            
            # 3. Get the chains for this iteration
            chains = result_tuple[1]

            # 4. Call the *original* "inner" function
            #    This dispatches to your first method
            single_split_dict = extract_parameters(model, df_for_this_split, vocabulary, chains)

            # 5. Merge the results into the main dictionary
            merge!(full_extraction_dict, single_split_dict)
        end
        
        return full_extraction_dict
    end


    function extract_parameters(
        model::StaticDixonColes,
        dfs_to_predict::AbstractVector, 
        vocabulary::Vocabulary,
        results_vector::AbstractVector
    )
        PredictionValue = NamedTuple{(:λ_h, :λ_a, :ρ), Tuple{AbstractVector{Float64}, AbstractVector{Float64}, AbstractVector{Float64}}}

        # 1. Allocate memory for the *combined* output dictionary
        full_extraction_dict = Dict{Int64, PredictionValue}()

        # 2. Iterate through the results and dataframes in parallel
        #    `zip` stops at the shortest vector, which perfectly handles
        #    the case where `results` has one extra model.
        for (result_tuple, df_for_this_split) in zip(results_vector, dfs_to_predict)
            
            # 3. Get the chains for this iteration
            chains = result_tuple[1]

            # 4. Call the *original* "inner" function
            #    This dispatches to your first method
            single_split_dict = extract_parameters(model, df_for_this_split, vocabulary, chains)

            # 5. Merge the results into the main dictionary
            merge!(full_extraction_dict, single_split_dict)
        end
        
        return full_extraction_dict
    end







end

# Export the model structs to be used in scripts
using .Implementations
using .Implementations: extract_parameters
export StaticPoisson, StaticSimplexPoisson, HierarchicalSimplexPoisson
export StaticDixonColes
export GRWPoisson
export build_turing_model, predict, extract_parameters

# This is where we define the specific methods for our contract.
# This extends the function originally defined in TypesInterfaces.
import ...TypesInterfaces: required_mapping_keys

required_mapping_keys(model::StaticPoisson) = [:team_map, :n_teams]
required_mapping_keys(model::StaticSimplexPoisson) = [:team_map, :n_teams]
# Add more specific model implementations here, e.g.:
# required_mapping_keys(model::HierarchicalPoisson) = [:team_map, :n_teams, :league_map, :n_leagues]



end
