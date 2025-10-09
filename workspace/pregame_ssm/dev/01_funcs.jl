using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions, Plots
using CSV 


function add_global_round_column!(matches_df::DataFrame)
    sort!(matches_df, :match_date)
    num_matches = nrow(matches_df)
    global_rounds = Vector{Int}(undef, num_matches)
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(matches_df))
        home_team, away_team = row.home_team, row.away_team
        if home_team in teams_in_current_round || away_team in teams_in_current_round
            global_round_counter += 1
            empty!(teams_in_current_round)
        end
        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team, away_team)
    end
    
    matches_df.global_round = global_rounds
    println("✅ Successfully added `:global_round` column. Found $(global_round_counter) unique time steps.")
    return matches_df
end




#### notes 
"""
    load_models_from_paths(model_paths::dict{string, string})

loads multiple models from a dictionary of paths.

# arguments
- `model_paths`: a dictionary mapping a descriptive model name (string) to its file path (string).

# returns
- a dictionary mapping the model name to the loaded model object.
"""
function load_models_from_paths(model_paths::dict{string, string})
    loaded_models = dict{string, any}()
    for (name, path) in model_paths
        println("loading model: '$name' from path: $path")
        # assuming you have a function `load_model` available
        loaded_models[name] = load_model(path) 
    end
    return loaded_models
end



# models
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson_ha.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA
using .AR1PoissonHA

using BayesianFootball.AR1NegativeBinomialHA
using .AR1PoissonHA



all_model_paths = Dict(
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_poisson_ha_20251004-111854",
  "ssm_neg_bin" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_neg_bin_ha_20251004-122001",
)


loaded_models_all = load_models_from_paths(all_model_paths)


module PredictionCubes

export PredictionCube, get_mean_predictions, get_posterior, get_match_outcome_probs

using DataFrames
using BayesianFootball
using Statistics

struct PredictionCube
    predictions::Array{Float64, 3}
    match_ids::Vector{Int}
    markets::Vector{Symbol}
    match_id_map::Dict{Int, Int}
    market_map::Dict{Symbol, Int}

    function PredictionCube(
        matches_df::DataFrame,
        ssm; # ssm is the entire model object, e.g., ssm_neg_m
        markets::Vector{Symbol} = [
            :home, :draw, :away, :under_05, :under_15,
            :under_25, :under_35, :btts
        ]
    )
        # --- THIS IS THE CORRECTED LINE ---
        ssm_result = ssm.result # Changed from ssm.results

        println("Generating posterior predictions with $(Threads.nthreads()) threads...")
        model_def = ssm.config.model_def
        mapping = ssm_result.mapping
        chain = ssm_result.chains_sequence[1]
        n_matches, n_markets, n_chains = nrow(matches_df), length(markets), length(chain.ft)
        match_ids = matches_df.match_id
        market_map = Dict(m => i for (i, m) in enumerate(markets))
        predictions_array = Array{Float64, 3}(undef, n_matches, n_markets, n_chains)

        Threads.@threads for i in 1:n_matches
            match_row = matches_df[i, :]
            features = BayesianFootball.create_master_features(DataFrame(match_row), mapping)
            preds_struct = BayesianFootball.predict(model_def, chain, features, mapping)
            for (market_idx, market_symbol) in enumerate(markets)
                pred_vector = getfield(preds_struct.ft, market_symbol)
                predictions_array[i, market_idx, :] = pred_vector
            end
        end
        
        match_id_map = Dict(id => i for (i, id) in enumerate(match_ids))
        println("✅ Posterior predictive cube generated successfully!")
        new(predictions_array, match_ids, markets, match_id_map, market_map)
    end
end

### ### --- Analysis Helper Functions ---

"""
    get_posterior(cube::PredictionCube, match_id::Int, market::Symbol) -> Vector{Float64}

Extracts the full posterior predictive distribution (all chain samples) for a 
single match and market.
"""
function get_posterior(cube::PredictionCube, match_id::Int, market::Symbol)
    match_idx = cube.match_id_map[match_id]
    market_idx = cube.market_map[market]
    return cube.predictions[match_idx, market_idx, :]
end

"""
    get_mean_predictions(cube::PredictionCube) -> DataFrame

Calculates the posterior mean probability for every match and market. This DataFrame is
the typical input for calculating scoring rules like RPS and ECE.
"""
function get_mean_predictions(cube::PredictionCube)
    mean_probs = mean(cube.predictions; dims=3)[:,:,1]
    df = DataFrame(match_id = cube.match_ids)
    for (i, market) in enumerate(cube.markets)
        df[!, market] = mean_probs[:, i]
    end
    return df
end

"""
    get_match_outcome_probs(cube::PredictionCube, match_id::Int) -> Vector{Float64}

Returns the posterior mean probability vector `[P(home), P(draw), P(away)]`
for a single match. This is the exact format needed for an RPS calculation.
"""
function get_match_outcome_probs(cube::PredictionCube, match_id::Int)
    match_idx = cube.match_id_map[match_id]
    
    # Get indices for the three outcomes
    h_idx = cube.market_map[:home]
    d_idx = cube.market_map[:draw]
    a_idx = cube.market_map[:away]
    
    # Slice the cube for this one match and the three markets
    outcome_chains = cube.predictions[match_idx, [h_idx, d_idx, a_idx], :]
    
    # Return the mean across the chains
    return vec(mean(outcome_chains; dims=2))
end

# Pretty printing
function Base.show(io::IO, cube::PredictionCube)
    n_matches, n_markets, n_chains = size(cube.predictions)
    print(io, "Posterior Predictive Cube with dimensions:\n")
    print(io, "  - Matches: $n_matches\n")
    print(io, "  - Markets: $n_markets\n")
    print(io, "  - Chains:  $n_chains")
end

end # module PredictionCubes
