using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions

include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/prediction.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/analysis_funcs.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/match_day_utils.jl")
using .BivariateMaher
using .BivariatePrediction
using .Analysis
using .MatchDayUtils 



function generate_prediction_matrix(model, home_team, away_team, league_id, market_list)
    # Generate predictions using your existing function
    preds_dict = generate_predictions(
        Dict{String, Any}("temp" => model), 
        home_team, 
        away_team, 
        league_id
    )
    preds = preds_dict["temp"]
    
    # Get MCMC sample size from one of the prediction vectors
    num_samples = length(preds.ft.home)
    market_map = Dict(m => i for (i, m) in enumerate(market_list))
    num_markets = length(market_list)
    
    # Initialize the probability matrix
    prob_matrix = Matrix{Float64}(undef, num_samples, num_markets)

    # --- Expanded Logic to Populate All Prediction Lines ---
    for (market, idx) in market_map
        market_str = String(market)
        prob_vector = nothing # Default to nothing
        
        try
            if startswith(market_str, "ft_")
                time_preds = preds.ft
                # --- FT 1x2 ---
                if market in (:ft_1x2_home, :ft_1x2_draw, :ft_1x2_away)
                    field = Symbol(split(market_str, '_')[end])
                    prob_vector = getfield(time_preds, field)
                # --- FT Over/Under ---
                elseif startswith(market_str, "ft_ou_")
                    parts = split(market_str, '_') # ft, ou, 05, over
                    field = Symbol(parts[4], "_", parts[3]) # :over_05 or :under_05
                    base_prob = getfield(time_preds, Symbol("under_", parts[3])) # always get the 'under' prob
                    prob_vector = (parts[4] == "over") ? (1.0 .- base_prob) : base_prob
                # --- FT BTTS ---
                elseif startswith(market_str, "ft_btts_")
                    side = split(market_str, '_')[end] # yes or no
                    base_prob = time_preds.btts
                    prob_vector = (side == "yes") ? base_prob : (1.0 .- base_prob)
                # --- FT Correct Score ---
                elseif startswith(market_str, "ft_cs_")
                    score_key_str = market_str[7:end]
                    score_key = if occursin(r"\d_\d", score_key_str)
                        Tuple(parse(Int, i) for i in split(score_key_str, '_'))
                    elseif score_key_str == "other_home"
                        "other_home_win"
                    elseif score_key_str == "other_away"
                        "other_away_win"
                    else "other_draw" end
                    prob_vector = get(time_preds.correct_score, score_key, fill(NaN, num_samples))
                end
            elseif startswith(market_str, "ht_")
                time_preds = preds.ht
                # --- HT 1x2 ---
                if market in (:ht_1x2_home, :ht_1x2_draw, :ht_1x2_away)
                    field = Symbol(split(market_str, '_')[end])
                    prob_vector = getfield(time_preds, field)
                # --- HT Over/Under ---
                elseif startswith(market_str, "ht_ou_")
                    parts = split(market_str, '_')
                    field = Symbol(parts[4], "_", parts[3])
                    base_prob = getfield(time_preds, Symbol("under_", parts[3]))
                    prob_vector = (parts[4] == "over") ? (1.0 .- base_prob) : base_prob
                # --- HT Correct Score ---
                elseif startswith(market_str, "ht_cs_")
                    score_key_str = market_str[7:end]
                    score_key = if occursin(r"\d_\d", score_key_str)
                        Tuple(parse(Int, i) for i in split(score_key_str, '_'))
                    else "any_unquoted" end
                    prob_vector = get(time_preds.correct_score, score_key, fill(NaN, num_samples))
                end
            end

            # Assign the found vector or NaN if nothing was found
            prob_matrix[:, idx] = isnothing(prob_vector) ? fill(NaN, num_samples) : prob_vector

        catch e
            # If any parsing or field access fails, mark as NaN
            prob_matrix[:, idx] .= NaN
        end
    end
    
    return PredictionMatrix(market_list, market_map, prob_matrix)
end


## --- 1. Define Models and Match ---
all_model_paths = Dict(
    "maher_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2526_20250919-200800",
    "bivar_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2526_20250919-200835",
    "maher_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2425_2526_20250919-202508",
    "bivar_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2425_2526_20250919-204350"
)

model_2526_paths = Dict(
    "maher_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2526_20250919-200800",
    "bivar_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2526_20250919-200835",
)

model_24_26_paths = Dict(
    "maher_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2425_2526_20250919-202508",
    "bivar_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2425_2526_20250919-204350"
)

loaded_models_all = load_models_from_paths(all_model_paths)
loaded_models_2526 = load_models_from_paths(model_2526_paths)
loaded_models_24_26 = load_models_from_paths(model_24_26_paths)

# --- 2. Get today's matches ---
todays_matches = get_todays_matches(["scotland", "england"]; cli_path=CLI_PATH)


# --- 3. Fetch all odds using the new function ---
odds_df = fetch_all_market_odds(
    todays_matches,
    MARKET_LIST;
    cli_path=CLI_PATH
)


# --- 3. Run the Consolidated Analysis ---
match_to_analyze = first(todays_matches) # Analyze the first match

(comparison_df, prediction_matrices, market_book) = generate_match_analysis(
    match_to_analyze,
    odds_df, # Your wide DataFrame of all market odds
    loaded_models_all,
    MARKET_LIST,
    generate_prediction_matrix # Pass the existing helper function
);
