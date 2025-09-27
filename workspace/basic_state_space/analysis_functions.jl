
module AnalysisSSM
using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions

export load_models_from_paths, generate_predictions, create_odds_dataframe, plot_odds_distributions

"""
    load_models_from_paths(model_paths::Dict{String, String})

Loads multiple models from a dictionary of paths.

# Arguments
- `model_paths`: A Dictionary mapping a descriptive model name (String) to its file path (String).

# Returns
- A Dictionary mapping the model name to the loaded model object.
"""
function load_models_from_paths(model_paths::Dict{String, String})
    loaded_models = Dict{String, Any}()
    for (name, path) in model_paths
        println("Loading model: '$name' from path: $path")
        # Assuming you have a function `load_model` available
        loaded_models[name] = load_model(path) 
    end
    return loaded_models
end




"""
    generate_predictions(
        models::Dict{String, Any}, 
        home_team::String, 
        away_team::String, 
        league_id::Int
    )

Generates match line predictions for a given match across multiple models.

# Arguments
- `models`: A dictionary of loaded model objects from `load_models_from_paths`.
- `home_team`, `away_team`, `league_id`: Details of the match to predict.

# Returns
- A Dictionary mapping the model name to its `MatchLinePredictions` object.
"""
function generate_predictions(models::Dict{String, Any}, home_team::String, away_team::String, league_id::Int)
    
    predictions = Dict{String, Any}()

    for (name, model) in models
        println("Generating predictions for model: '$name'")
        
        # Use 'local' to ensure the variable is accessible after the if/else block
        local match_to_predict::DataFrame

      # --- State Space Model: Requires 'global_round' ---
      println("-> Detected State Space model. Calculating next round.")
      
      chains = model.result.chains_sequence[1]
      mapping = model.result.mapping

      posterior_samples = BayesianFootball.extract_posterior_samples(
          model.config.model_def,
          chains.ft,
          mapping
      )
      last_training_round = posterior_samples.n_rounds
      next_round = last_training_round + 1
      println("   - Predicting for global_round: $next_round")

      match_to_predict = DataFrame(
          home_team=home_team,
          away_team=away_team,
          tournament_id=league_id,
          global_round=next_round, # The crucial addition
          home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
      )

        # --- Common steps for ALL models ---
        
        # 1. Create features from the DataFrame prepared above
        features = BayesianFootball.create_master_features(
            match_to_predict,
            model.result.mapping
        )

        # 2. Call the single, generic prediction function
        preds = BayesianFootball.predict_match_lines(
            model.config.model_def,
            model.result.chains_sequence[1],
            features,
            model.result.mapping
        )
        
        predictions[name] = preds
    end
    
    return predictions
end

"""
    create_odds_dataframe(predictions::Dict{String, Any})

Converts a dictionary of predictions into a DataFrame of mean odds.

# Arguments
- `predictions`: A dictionary of `MatchLinePredictions` from `generate_predictions`.

# Returns
- A DataFrame with rows for each model/time (FT/HT) and columns for market odds.
"""
function create_odds_dataframe(predictions::Dict{String, Any})
    
    df = DataFrame(
        Model=String[], Time=String[], Home=Float64[], Draw=Float64[], Away=Float64[],
        O05=Float64[], U05=Float64[], O15=Float64[], U15=Float64[],
        O25=Float64[], U25=Float64[], BTTS_Yes=Union{Missing, Float64}[], BTTS_No=Union{Missing, Float64}[]
    )

    for (name, pred) in predictions
        # Full Time (FT) Predictions
        ft = pred.ft
        push!(df, (
            Model=name, Time="FT",
            Home = mean(1 ./ ft.home),
            Draw = mean(1 ./ ft.draw),
            Away = mean(1 ./ ft.away),
            U05 = mean(1 ./ ft.under_05),
            O05 = mean(1 ./ (1 .- ft.under_05)),
            U15 = mean(1 ./ ft.under_15),
            O15 = mean(1 ./ (1 .- ft.under_15)),
            U25 = mean(1 ./ ft.under_25),
            O25 = mean(1 ./ (1 .- ft.under_25)),
            BTTS_Yes = mean(1 ./ ft.btts),
            BTTS_No = mean(1 ./ (1 .- ft.btts))
        ))
        
        # Half Time (HT) Predictions
        ht = pred.ht
        push!(df, (
            Model=name, Time="HT",
            Home = mean(1 ./ ht.home),
            Draw = mean(1 ./ ht.draw),
            Away = mean(1 ./ ht.away),
            U05 = mean(1 ./ ht.under_05),
            O05 = mean(1 ./ (1 .- ht.under_05)),
            U15 = mean(1 ./ ht.under_15),
            O15 = mean(1 ./ (1 .- ht.under_15)),
            U25 = mean(1 ./ ht.under_25),
            O25 = mean(1 ./ (1 .- ht.under_25)),
            BTTS_Yes = missing, # BTTS not typically calculated for HT
            BTTS_No = missing
        ))
    end
    
    return df
end


"""
    plot_odds_distributions(
        predictions::Dict{String, Any}, 
        time::Symbol, 
        market::Symbol; 
        title_suffix=""
    )

Plots the density of odds for a specific market from multiple models.

# Arguments
- `predictions`: Dictionary of `MatchLinePredictions`.
- `time`: A symbol, either `:ft` or `:ht`.
- `market`: A symbol for the market (e.g., `:home`, `:draw`, `:btts`, `:under_25`).
- `title_suffix`: Optional string to add to the plot title.
"""
function plot_odds_distributions(predictions::Dict{String, Any}, time::Symbol, market::Symbol; title_suffix="")
    
    title = "Odds Distribution for $(uppercase(string(time))) $(uppercase(string(market))) $(title_suffix)"
    p = plot(title=title, xlabel="Odds", ylabel="Density", legend=:outertopright)
    
    for (name, pred) in predictions
        # Access the correct struct (ft or ht) and then the market vector
        prob_vector = getfield(getfield(pred, time), market)
        odds_vector = 1 ./ prob_vector
        
        density!(p, odds_vector, label=name)
    end
    
    # display(p)
    return p
end


  
end
