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


#### testing 
# --- Step 1: Setup Paths and Define Market Structure ---
# Path to your python CLI tool
CLI_PATH = "/home/james/bet_project/whatstheodds"

# Define the master list of markets we want to analyze. 
# The order here is important as it defines the structure of our matrices.
MARKET_LIST = [
    :ft_1x2_home,
    :ft_1x2_draw,
    :ft_1x2_away
]

COMPREHENSIVE_MARKET_LIST = [
    # Full Time 1x2
    :ft_1x2_home, :ft_1x2_draw, :ft_1x2_away,
    
    # Full Time Over/Under
    :ft_ou_05_under, :ft_ou_05_over,
    :ft_ou_15_under, :ft_ou_15_over,
    :ft_ou_25_under, :ft_ou_25_over,
    :ft_ou_35_under, :ft_ou_35_over,
    
    # Full Time BTTS
    :ft_btts_yes, :ft_btts_no,
    
    # Full Time Correct Score (add as many as you want to track)
    :ft_cs_0_0, :ft_cs_1_0, :ft_cs_0_1, :ft_cs_1_1, :ft_cs_2_0, :ft_cs_0_2,
    :ft_cs_2_1, :ft_cs_1_2, :ft_cs_2_2, :ft_cs_3_0, :ft_cs_0_3, :ft_cs_3_1,
    :ft_cs_1_3, :ft_cs_3_2, :ft_cs_2_3, :ft_cs_3_3,
     # :ft_cs_other_home, :ft_cs_other_draw, :ft_cs_other_away,

    # Half Time 1x2
    :ht_1x2_home, :ht_1x2_draw, :ht_1x2_away,
    
    # Half Time Over/Under
    :ht_ou_05_under, :ht_ou_05_over,
    :ht_ou_15_under, :ht_ou_15_over,
    :ht_ou_25_under, :ht_ou_25_over,
    
    # Half Time Correct Score
    :ht_cs_0_0, :ht_cs_1_0, :ht_cs_0_1, :ht_cs_1_1,
    # :ht_cs_other
]

println("✅ Setup complete.")
# --- Step 2: Get Today's Matches ---
println("\nFetching today's matches for England...")
todays_matches = get_todays_matches(["england", "scotland"]; cli_path=CLI_PATH)

# Let's assume the fixture list contains "Liverpool v Everton"
display(todays_matches)
MATCH_OF_INTEREST = "Liverpool v Everton"
MATCH_OF_INTEREST = "Cardiff v Bradford"
MATCH_OF_INTEREST = "Brighton v Tottenham"
MATCH_OF_INTEREST = "Bournemouth v Newcastle"
LEAGUE_ID = 1 # Assuming Premier League ID for your model

# --- Step 3: Get Live Market Odds for the Selected Match ---
println("\nFetching live odds for: $MATCH_OF_INTEREST")
market_book = get_live_market_odds(MATCH_OF_INTEREST, MARKET_LIST; cli_path=CLI_PATH)

market_book = get_live_market_odds(MATCH_OF_INTEREST, COMPREHENSIVE_MARKET_LIST; cli_path=CLI_PATH)


println("MarketBook created:")
println("Markets: ", market_book.markets)
println("Back Odds: ", market_book.back_odds)
println("Lay Odds: ", market_book.lay_odds)

# --- Step 4: Adapt Model Predictions to the PredictionMatrix Format ---
# This helper function bridges your existing prediction logic with our new tensor format.
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

    # Populate the matrix based on the market list
    for (market, idx) in market_map
        if market == :ft_1x2_home
            prob_matrix[:, idx] = preds.ft.home
        elseif market == :ft_1x2_draw
            prob_matrix[:, idx] = preds.ft.draw
        elseif market == :ft_1x2_away
            prob_matrix[:, idx] = preds.ft.away
        else
            prob_matrix[:, idx] .= NaN # Mark unsupported markets as NaN
        end
    end
    
    return PredictionMatrix(market_list, market_map, prob_matrix)
end

# v2 
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

println("\n✅ PredictionMatrix helper function defined.")

# --- Step 5: Generate Predictions and Calculate EV for a Set of Models ---
println("\nGenerating predictions and EV distributions for models trained on 24/25-25/26 seasons...")

# Use one of your loaded model sets
models_to_run = loaded_models_24_26 

all_ev_dists = Dict{String, EVDistribution}()
all_pred_matrices = Dict{String, PredictionMatrix}()

for (model_name, model) in models_to_run
    println("Processing model: $model_name...")
    
    # Generate the prediction matrix for the match of interest
    pred_matrix = generate_prediction_matrix(
        model, 
        "liverpool", # Assuming model uses 'liverpool'
        "everton",   # Assuming model uses 'everton'
        LEAGUE_ID,
        COMPREHENSIVE_MARKET_LIST
    )
    all_pred_matrices[model_name] = pred_matrix

    # Calculate the EV distribution against the live market book
    ev_dist = calculate_ev_distributions(pred_matrix, market_book)
    all_ev_dists[model_name] = ev_dist
end

println("\n✅ EV calculation complete for all models.")

# --- Step 6: Visualize the Results ---
println("\nGenerating visualizations...")

# Example 1: Plot a single model's odds distribution vs. the market spread
# Let's inspect the bivariate model
model_to_plot = "bivar_2526"
p1 = plot_market_distribution_vs_odds(
    all_pred_matrices[model_to_plot], 
    market_book, 
    :ft_1x2_home # We are interested in the home win market
)
title!(p1, "Odds Dist. ($model_to_plot) vs Market for Home Win")
display(p1)

model_to_plot = "maher_24_26"
p1 = plot_market_distribution_vs_odds(
    all_pred_matrices[model_to_plot], 
    market_book, 
    :ft_1x2_home # We are interested in the home win market
)
title!(p1, "Odds Dist. ($model_to_plot) vs Market for Home Win")
display(p1)

# --- Plotting Multiple Odds Distributions on One Graph ---

# 1. Define the models you want to compare on this plot
model1_name = "bivar_24_26"
model2_name = "maher_24_26"
market_to_plot = :ft_1x2_home

# 2. Create the initial plot using the first model
# This sets up the axes, title, and market lines
p1 = plot_market_distribution_vs_odds(
    all_pred_matrices[model1_name], 
    market_book, 
    market_to_plot
)

# We need to manually rename the first model's label for clarity in the legend
p1[1][1][:label] = model1_name # Update the label of the first series

# 3. Extract the odds distribution for the second model
pred_matrix_2 = all_pred_matrices[model2_name]
idx_2 = pred_matrix_2.market_map[market_to_plot]
model2_odds_dist = 1 ./ pred_matrix_2.probabilities[:, idx_2]

# 4. Add the second model's distribution to the existing plot `p1` using `density!`
density!(p1, model2_odds_dist, label=model2_name)

# 5. Finalize and display the combined plot
title!(p1, "Odds Dist. Comparison vs Market for Home Win")
display(p1)

###
# Example 2: Plot and compare the EV distributions for all models
p2 = plot_ev_distributions(
    all_ev_dists, 
    :ft_1x2_home # Compare EV for the home win market across models
)
title!(p2, "EV Comparison for Home Win $MATCH_OF_INTEREST")
display(p2)



###### other match 
#
todays_matches = get_todays_matches(["scotland"]; cli_path=CLI_PATH)
tm = filter(row -> row.time=="14:00", todays_matches)

m1_event = first(tm[1, [:event_name]])

market_book = get_live_market_odds(m1_event, MARKET_LIST; cli_path=CLI_PATH)
println("MarketBook created:")
println("Markets: ", market_book.markets)
println("Back Odds: ", market_book.back_odds)
println("Lay Odds: ", market_book.lay_odds)
"""
julia> println("Markets: ", market_book.markets)
Markets: [:ft_1x2_home, :ft_1x2_draw, :ft_1x2_away]

julia> println("Back Odds: ", market_book.back_odds)
Back Odds: [4.6, 3.95, 1.72]

julia> println("Lay Odds: ", market_book.lay_odds)
Lay Odds: [5.3, 4.5, 1.84]

"""

all_ev_dists = Dict{String, EVDistribution}()
all_pred_matrices = Dict{String, PredictionMatrix}()

for (model_name, model) in models_to_run
    println("Processing model: $model_name...")
    
    # Generate the prediction matrix for the match of interest
    pred_matrix = generate_prediction_matrix(
        model, 
        "liverpool", # Assuming model uses 'liverpool'
        "everton",   # Assuming model uses 'everton'
        LEAGUE_ID,
        MARKET_LIST
    )
    all_pred_matrices[model_name] = pred_matrix

    # Calculate the EV distribution against the live market book
    ev_dist = calculate_ev_distributions(pred_matrix, market_book)
    all_ev_dists[model_name] = ev_dist
end

m1_event

p2 = plot_ev_distributions(
    all_ev_dists, 
    :ft_1x2_home # Compare EV for the home win market across models
)
title!(p2, "EV Comparison for Home Win $MATCH_OF_INTEREST")
display(p2)







# --- Script to Create a Summary EV DataFrame ---

todays_matches = get_todays_matches(["scotland", "england"]; cli_path=CLI_PATH)
tm = todays_matches
tm = filter(row -> row.time=="14:00", todays_matches)

# 1. Select the specific model you want to use for this analysis
MODEL_TO_ANALYZE_NAME = "bivar_24_26"
model_to_run = models_to_run[MODEL_TO_ANALYZE_NAME] # models_to_run is from your previous setup

# 2. Initialize an empty array to store the results for each market
results_list = []

println("Starting analysis for $(nrow(tm)) matches...")

# 3. Loop through each match in the filtered DataFrame `tm`
for row in eachrow(tm)
    event = row.event_name
    home_team_model = row.home_team
    away_team_model = row.away_team

    println("Processing: $event")
    
    # --- Get Live Market Odds ---
    # Note: A try-catch block is good practice in case the market for a match isn't available
    try
        market_book = get_live_market_odds(event, COMPREHENSIVE_MARKET_LIST; cli_path=CLI_PATH)
        
        # --- Generate Model Predictions ---
        pred_matrix = generate_prediction_matrix(
            model_to_run,
            home_team_model,
            away_team_model,
            # NOTE: Assuming a single league ID. This would need to be dynamic
            # if your DataFrame contains matches from multiple leagues.
            LEAGUE_ID, 
            COMPREHENSIVE_MARKET_LIST
        )
        
        # --- Calculate Mean Model Odds and Mean EV ---
        # Mean of probabilities across all MCMC samples (dim 1)
        mean_probs = mean(pred_matrix.probabilities, dims=1)
        # Convert mean probabilities to mean odds
        mean_model_odds = 1 ./ mean_probs' # Transpose to make it a column vector
        
        # Calculate the full EV distribution
        ev_dist = calculate_ev_distributions(pred_matrix, market_book)
        # Calculate the mean of the EV distribution
        mean_evs = mean(ev_dist.ev, dims=1)' # Transpose
        
        # --- Append results for each market to our list ---
        for i in 1:length(MARKET_LIST)
            market_name = MARKET_LIST[i]
            
            # Skip if market odds weren't found
            isnan(market_book.back_odds[i]) && continue
            
            push!(results_list, (
                event_name = event,
                market = market_name,
                market_odds = market_book.back_odds[i],
                model_odds = round(mean_model_odds[i], digits=2),
                mean_ev = round(mean_evs[i] * 100, digits=2) # As a percentage
            ))
        end
        
    catch e
        @error "Could not process match: $event. Error: $e"
    end
end

# 4. Convert the list of results into a DataFrame and sort it
summary_df = DataFrame(results_list)
sort!(summary_df, :mean_ev, rev=true)

println("\n✅ Analysis complete.")
display(summary_df)

home = filter(row -> row.market==:ft_1x2_home, summary_df)

""" 
julia> home = filter(row -> row.market==:ft_1x2_home, summary_df)
35×5 DataFrame
 Row │ event_name                     market       market_odds  model_odds  mean_ev 
     │ String                         Symbol       Float64      Float64     Float64 
─────┼──────────────────────────────────────────────────────────────────────────────
   1 │ Blackburn v Ipswich            ft_1x2_home         4.1         1.85   121.14
   2 │ Colchester v Bristol Rovers    ft_1x2_home         3.4         1.87    82.3
   3 │ Burnley v Nottm Forest         ft_1x2_home         3.45        1.92    79.88
   4 │ Queen of South v Inverness CT  ft_1x2_home         4.0         2.3     73.82
   5 │ Hull v Southampton             ft_1x2_home         3.1         1.83    69.23
   6 │ Newport County v Gillingham    ft_1x2_home         4.4         3.05    44.22
   7 │ Notts Co v Crawley Town        ft_1x2_home         2.12        1.57    35.37
   8 │ Airdrieonians v Raith          ft_1x2_home         4.2         3.21    30.94
   9 │ Barrow v Crewe                 ft_1x2_home         3.3         2.58    28.13
  10 │ Cheltenham v Oldham            ft_1x2_home         3.6         2.82    27.56
  11 │ Bromley v Chesterfield         ft_1x2_home         2.84        2.35    20.62
  12 │ Rotherham v Stockport          ft_1x2_home         4.0         3.34    19.63
  13 │ Walsall v Tranmere             ft_1x2_home         2.0         1.73    15.56
  14 │ Wycombe v Northampton          ft_1x2_home         1.97        1.73    14.12
  15 │ Brighton v Tottenham           ft_1x2_home         2.28        2.03    12.13
  16 │ Port Vale v Mansfield          ft_1x2_home         2.18        1.96    11.49
  17 │ Salford City v Swindon         ft_1x2_home         2.42        2.26     6.85
  18 │ Derby v Preston                ft_1x2_home         2.48        2.36     4.96
  19 │ Reading v Leyton Orient        ft_1x2_home         2.52        2.44     3.43
  20 │ Arbroath v Morton              ft_1x2_home         2.6         2.53     2.64
  21 │ West Ham v Crystal Palace      ft_1x2_home         3.2         3.28    -2.54
  22 │ Aberdeen v Motherwell          ft_1x2_home         1.81        1.91    -5.3
  23 │ Huddersfield v Burton Albion   ft_1x2_home         1.75        1.88    -7.14
  24 │ Alloa v Cove Rangers           ft_1x2_home         2.16        2.43   -11.28
  25 │ Dundee v Livingston            ft_1x2_home         3.1         3.57   -13.21
  26 │ Doncaster v AFC Wimbledon      ft_1x2_home         1.99        2.4    -16.97
  27 │ Portsmouth v Sheff Wed         ft_1x2_home         1.7         2.12   -19.85
  28 │ Plymouth v Peterborough        ft_1x2_home         1.97        2.47   -20.16
  29 │ Stevenage v Exeter             ft_1x2_home         1.75        2.31   -24.33
  30 │ Wolves v Leeds                 ft_1x2_home         3.1         4.19   -25.93
  31 │ Sheff Utd v Charlton           ft_1x2_home         1.84        2.61   -29.62
  32 │ Ross Co v Queens Park          ft_1x2_home         1.63        2.32   -29.85
  33 │ MK Dons v Accrington           ft_1x2_home         1.47        2.13   -31.0
  34 │ Norwich v Wrexham              ft_1x2_home         2.04        2.99   -31.75
  35 │ Cardiff v Bradford             ft_1x2_home         1.8         3.25   -44.54
"""


#### get todays_matches odds 

using CSV
using Dates # To create a date-stamped filename

# --- 1. Setup the Output Path ---
# Define where you want to save the final CSV file.
# The folder will be created if it doesn't exist.
OUTPUT_FOLDER = "/home/james/bet_project/models_julia/workspace/bivariate_maher_model/data/"
mkpath(OUTPUT_FOLDER) # Ensures the directory exists

todays_matches = get_todays_matches(["scotland", "england"]; cli_path=CLI_PATH)
todays_matches = filter(row -> row.time !="14:00", todays_matches)
# --- 2. Initialize a list to hold the results ---
# Each element will be a dictionary representing one row (one match).
all_match_odds_list = []

println("Starting to fetch market odds for $(nrow(todays_matches)) matches...")

# --- 3. Iterate through each match ---
for row in eachrow(todays_matches)
    event = row.event_name
    println("Fetching odds for: $event")
    
    try
        # Fetch the comprehensive odds using the list defined previously
        market_book = get_live_market_odds(event, COMPREHENSIVE_MARKET_LIST; cli_path=CLI_PATH)
        
        # Create a dictionary for this match's row
        match_row = Dict{Symbol, Any}()
        match_row[:event_name] = event
        match_row[:home_team] = row.home_team
        match_row[:away_team] = row.away_team

        for i in 1:length(market_book.markets)
            market_symbol = market_book.markets[i]
            back_col_name = Symbol(market_symbol, "_back")
            lay_col_name = Symbol(market_symbol, "_lay")
            
            # --- MODIFIED SECTION ---
            # Get the odds from the market book
            back_odd = market_book.back_odds[i]
            lay_odd = market_book.lay_odds[i]
            
            # Check if the value is NaN. If it is, use `missing`; otherwise, use the value.
            match_row[back_col_name] = isnan(back_odd) ? missing : back_odd
            match_row[lay_col_name] = isnan(lay_odd) ? missing : lay_odd
        end
        
        # Add the completed row to our list
        push!(all_match_odds_list, match_row)
        
    catch e
        @error "Failed to process match: $event. Error: $e"
        continue # Move to the next match
    end
end

println("\n✅ Finished fetching odds.")

# --- 4. Convert the list of dictionaries into a DataFrame ---
if !isempty(all_match_odds_list)
    odds_df = DataFrame(all_match_odds_list)

    # --- 5. Save the DataFrame to a CSV file ---
    # Create a filename with today's date
    filename = "market_odds_$(today()).csv"
    full_path = joinpath(OUTPUT_FOLDER, filename)
    
    CSV.write(full_path, odds_df)
    
    println("Successfully saved market odds to:")
    println(full_path)
    
    # Display the first few rows and columns of the result
    display(first(odds_df, 5))
else
    println("No odds were successfully fetched. CSV file not created.")
end



####
todays_matches
match_number = 1 
home_team_model = only(todays_matches[match_number, [:home_team]])
away_team_model = only(todays_matches[match_number, [:away_team]])
LEAGUE_ID = 1

models_to_run = loaded_models_24_26 
model_to_run = models_to_run["bivar_24_26"]
pred_matrix = generate_prediction_matrix(
    model_to_run,
    home_team_model,
    away_team_model,
    # NOTE: Assuming a single league ID. This would need to be dynamic
    # if your DataFrame contains matches from multiple leagues.
    LEAGUE_ID, 
    COMPREHENSIVE_MARKET_LIST
)



odds_df 

# --- 1. Select the Match to Analyze ---
match_to_analyze_row = todays_matches[3, :]
event_to_find = match_to_analyze_row.event_name

println("Analyzing Match: $event_to_find")

# --- 2. Extract this Match's Market Odds from the main `odds_df` ---
match_market_odds = first(filter(row -> row.event_name == event_to_find, odds_df))

# --- 3. Reconstruct a MarketBook and Calculate EV Distributions ---
# The `calculate_ev_distributions` function needs a `MarketBook` struct, 
# so we'll quickly rebuild one for our selected match.
market_map = Dict(m => i for (i, m) in enumerate(COMPREHENSIVE_MARKET_LIST))
back_odds_vec = [match_market_odds[Symbol(m, "_back")] for m in COMPREHENSIVE_MARKET_LIST]
lay_odds_vec = [match_market_odds[Symbol(m, "_lay")] for m in COMPREHENSIVE_MARKET_LIST]
single_match_book = MarketBook(COMPREHENSIVE_MARKET_LIST, market_map, back_odds_vec, lay_odds_vec)

# Now, calculate the full EV distribution using the prediction matrix and the new market book
ev_dist = calculate_ev_distributions(pred_matrix, single_match_book)

# --- 4. Calculate All Required Statistics ---
# Convert probability matrix to odds matrix
odds_matrix = 1 ./ pred_matrix.probabilities

# Calculate stats for the model's odds predictions
model_mean_odds = mean(odds_matrix, dims=1)'
model_std_odds = std(odds_matrix, dims=1)'

# Calculate stats for the EV distribution
mean_evs = mean(ev_dist.ev, dims=1)'
std_evs = std(ev_dist.ev, dims=1)'

# --- 5. Build the Final Pivoted DataFrame ---
analysis_list = []

for i in 1:length(COMPREHENSIVE_MARKET_LIST)
    market_symbol = COMPREHENSIVE_MARKET_LIST[i]
    
    market_back = single_match_book.back_odds[i]
    market_lay = single_match_book.lay_odds[i]
    
    # Skip if odds are missing
    ismissing(market_back) && continue
    
    push!(analysis_list, (
        market = market_symbol,
        market_back = market_back,
        market_lay = market_lay,
        model_mean_odds = round(model_mean_odds[i], digits=2),
        model_std = round(model_std_odds[i], digits=2),
        mean_ev = round(mean_evs[i] * 100, digits=2), # As a percentage
        std_ev = round(std_evs[i] * 100, digits=2)   # As a percentage
    ))
end

single_match_df = DataFrame(analysis_list)

println("\n✅ Detailed analysis with EV stats complete for $event_to_find")
display(single_match_df)

# as a functions 
####################
# As a functions part 
####################

"""
    create_match_analysis_df(
        match_info::DataFrameRow, 
        all_market_odds::DataFrame, 
        model_predictions::PredictionMatrix
    )

Generates a detailed analysis DataFrame for a single match.

# Arguments
- `match_info`: A row from the `todays_matches` DataFrame containing event details.
- `all_market_odds`: The "wide" DataFrame containing market odds for all matches.
- `model_predictions`: The `PredictionMatrix` object for the specific match.

# Returns
- A `DataFrame` with rows for each market and columns for market odds, 
  model predictions (mean/std), and EV stats (mean/std).
"""
function create_match_analysis_df(
    match_info::DataFrameRow, 
    all_market_odds::DataFrame, 
    model_predictions::PredictionMatrix
)
    event_to_find = match_info.event_name

    # 1. Find the corresponding row in the main odds table
    match_market_odds_row = first(filter(row -> row.event_name == event_to_find, all_market_odds))

    # 2. Reconstruct a MarketBook to use our helper functions
    market_list = model_predictions.markets
    market_map = model_predictions.market_map
    back_odds_vec = [match_market_odds_row[Symbol(m, "_back")] for m in market_list]
    lay_odds_vec = [match_market_odds_row[Symbol(m, "_lay")] for m in market_list]
    single_match_book = MarketBook(market_list, market_map, back_odds_vec, lay_odds_vec)

    # 3. Calculate EV and other statistics
    ev_dist = calculate_ev_distributions(model_predictions, single_match_book)
    odds_matrix = 1 ./ model_predictions.probabilities
    
    model_mean_odds = mean(odds_matrix, dims=1)'
    model_std_odds = std(odds_matrix, dims=1)'
    mean_evs = mean(ev_dist.ev, dims=1)'
    std_evs = std(ev_dist.ev, dims=1)'

    # 4. Build the final analysis list
    analysis_list = []
    for i in 1:length(market_list)
        market_symbol = market_list[i]
        market_back = single_match_book.back_odds[i]
        
        ismissing(market_back) && continue
        
        push!(analysis_list, (
            market = market_symbol,
            market_back = market_back,
            market_lay = single_match_book.lay_odds[i],
            model_mean_odds = round(model_mean_odds[i], digits=2),
            model_std = round(model_std_odds[i], digits=2),
            mean_ev = round(mean_evs[i] * 100, digits=2),
            std_ev = round(std_evs[i] * 100, digits=2)
        ))
    end
    
    return DataFrame(analysis_list)
end


# 1. Select a match and generate its prediction matrix (as you did before)
todays_matches
match_row_to_analyze = todays_matches[match_number, :]
home_team = match_row_to_analyze.home_team
away_team = match_row_to_analyze.away_team
league = 1 # Example League ID

model_to_run = models_to_run["bivar_24_26"]
pred_matrix = generate_prediction_matrix(
    model_to_run,
    home_team,
    away_team,
    league,
    COMPREHENSIVE_MARKET_LIST
)

# 2. Call the new function with the required inputs
#    (assuming `odds_df` is your wide DataFrame of market odds)
analysis_df = create_match_analysis_df(
    match_row_to_analyze, 
    odds_df, 
    pred_matrix
)

# 3. Display the result
show(analysis_df; vlines = :all )
match_row_to_analyze.event_name

"""
julia> match_row_to_analyze.event_name
"Bournemouth v Newcastle"

bivar_24_26 n
julia> show(analysis_df; vlines = :all )
42×7 DataFrame
│ Row │ market         │ market_back │ market_lay │ model_mean_odds │ model_std │ mean_ev │ std_ev  │
│     │ Symbol         │ Float64     │ Float64    │ Float64         │ Float64   │ Float64 │ Float64 │
├─────┼────────────────┼─────────────┼────────────┼─────────────────┼───────────┼─────────┼─────────┤
│   1 │ ft_1x2_home    │        2.56 │       2.58 │            2.41 │      0.41 │    8.94 │   17.52 │
│   2 │ ft_1x2_draw    │        3.55 │       3.6  │            3.85 │      0.29 │   -7.3  │    6.96 │
│   3 │ ft_1x2_away    │        3.0  │       3.05 │            3.32 │      0.68 │   -6.0  │   18.16 │
│   4 │ ft_ou_05_under │       14.5  │      15.5  │           13.88 │      4.2  │   13.56 │   32.32 │
│   5 │ ft_ou_05_over  │        1.07 │       1.08 │            1.09 │      0.03 │   -1.38 │    2.39 │
│   6 │ ft_ou_15_under │        4.3  │       4.5  │            3.8  │      0.83 │   18.1  │   24.14 │
│   7 │ ft_ou_15_over  │        1.29 │       1.3  │            1.39 │      0.11 │   -6.43 │    7.24 │
│   8 │ ft_ou_25_under │        2.1  │       2.12 │            1.95 │      0.28 │   10.0  │   15.08 │
│   9 │ ft_ou_25_over  │        1.89 │       1.9  │            2.15 │      0.34 │  -10.0  │   13.58 │
│  10 │ ft_ou_35_under │        1.43 │       1.46 │            1.37 │      0.12 │    5.42 │    8.9  │
│  11 │ ft_ou_35_over  │        3.2  │       3.3  │            4.04 │      1.02 │  -15.91 │   19.92 │
│  12 │ ft_btts_yes    │        1.7  │       1.73 │            1.97 │      0.22 │  -12.44 │    9.58 │
│  13 │ ft_btts_no     │        2.36 │       2.42 │            2.09 │      0.25 │   14.44 │   13.31 │
│  14 │ ft_cs_0_0      │       15.0  │      15.5  │           13.88 │      4.2  │   17.48 │   33.44 │
│  15 │ ft_cs_1_0      │       10.5  │      11.5  │            9.69 │      1.94 │   12.41 │   21.22 │
│  16 │ ft_cs_0_1      │       12.5  │      13.0  │           11.78 │      2.77 │   11.6  │   24.6  │
│  17 │ ft_cs_1_1      │        7.6  │       7.8  │            8.23 │      0.64 │   -7.18 │    6.69 │
│  18 │ ft_cs_2_0      │       16.0  │      16.5  │           13.89 │      2.93 │   19.96 │   23.42 │
│  19 │ ft_cs_0_2      │       19.0  │      20.0  │           20.48 │      5.44 │   -1.09 │   24.67 │
│  20 │ ft_cs_2_1      │       10.5  │      11.5  │           11.8  │      1.26 │  -10.12 │    8.72 │
│  21 │ ft_cs_1_2      │       12.0  │      13.0  │           14.32 │      2.15 │  -14.49 │   11.82 │
│  22 │ ft_cs_2_2      │       14.5  │      15.0  │           20.55 │      3.54 │  -27.53 │   11.43 │
│  23 │ ft_cs_3_0      │       34.0  │      36.0  │           30.63 │     10.14 │   21.88 │   35.96 │
│  24 │ ft_cs_0_3      │       42.0  │      44.0  │           54.7  │     20.58 │  -13.09 │   30.21 │
│  25 │ ft_cs_3_1      │       23.0  │      24.0  │           26.05 │      7.15 │   -5.78 │   22.93 │
│  26 │ ft_cs_1_3      │       27.0  │      28.0  │           38.29 │     11.76 │  -23.47 │   21.18 │
│  27 │ ft_cs_3_2      │       34.0  │      36.0  │           45.39 │     13.97 │  -18.61 │   22.55 │
│  28 │ ft_cs_2_3      │       36.0  │      38.0  │           54.99 │     17.77 │  -28.38 │   20.75 │
│  29 │ ft_cs_3_3      │       70.0  │      75.0  │          121.51 │     50.69 │  -33.25 │   24.91 │
│  30 │ ht_1x2_home    │        3.2  │       3.3  │            3.55 │      0.83 │   -5.22 │   20.61 │
│  31 │ ht_1x2_draw    │        2.38 │       2.4  │            2.56 │      0.27 │   -5.92 │    9.77 │
│  32 │ ht_1x2_away    │        3.6  │       3.7  │            3.42 │      0.85 │   11.07 │   25.4  │
│  33 │ ht_ou_05_under │        3.35 │       3.45 │            3.9  │      0.97 │   -9.61 │   19.63 │
│  34 │ ht_ou_05_over  │        1.4  │       1.42 │            1.38 │      0.11 │    2.23 │    8.2  │
│  35 │ ht_ou_15_under │        1.52 │       1.55 │            1.65 │      0.23 │   -6.24 │   11.81 │
│  36 │ ht_ou_15_over  │        2.84 │       2.94 │            2.72 │      0.59 │    8.82 │   22.06 │
│  37 │ ht_ou_25_under │        1.13 │       1.15 │            1.19 │      0.08 │   -4.38 │    6.07 │
│  38 │ ht_ou_25_over  │        7.8  │       8.6  │            7.35 │      2.79 │   19.97 │   41.91 │
│  39 │ ht_cs_0_0      │        3.4  │       3.45 │            3.9  │      0.97 │   -8.27 │   19.92 │
│  40 │ ht_cs_1_0      │        5.1  │       5.5  │            6.03 │      1.21 │  -12.55 │   15.4  │
│  41 │ ht_cs_0_1      │        5.7  │       6.2  │            5.85 │      1.02 │    0.06 │   15.92 │
│  42 │ ht_cs_1_1      │        8.4  │       9.2  │            9.07 │      1.15 │   -6.09 │   10.69 │
"""


"""
julia> display(analysis_df)
42×7 DataFrame
 Row │ market          market_back  market_lay  model_mean_odds  model_std  mean_ev  std_ev  
     │ Symbol          Float64      Float64     Float64          Float64    Float64  Float64 
─────┼───────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home            2.8         2.82             3.54       0.78   -17.39    17.02
   2 │ ft_1x2_draw            3.75        3.8              3.71       0.3      1.59     8.04
   3 │ ft_1x2_away            2.64        2.66             2.36       0.39    14.59    18.08
   4 │ ft_ou_05_under        20.0        21.0             11.66       3.52    85.75    51.22
   5 │ ft_ou_05_over          1.05        1.06             1.1        0.03    -4.75     2.69
   6 │ ft_ou_15_under         5.7         5.8              3.36       0.71    76.75    34.38
   7 │ ft_ou_15_over          1.21        1.22             1.46       0.13   -16.52     7.3
   8 │ ft_ou_25_under         2.56        2.6              1.79       0.25    45.28    18.53
   9 │ ft_ou_25_over          1.63        1.64             2.38       0.42   -29.5     11.8
  10 │ ft_ou_35_under         1.62        1.65             1.3        0.11    25.32     9.56
  11 │ ft_ou_35_over          2.52        2.6              4.74       1.32   -42.94    14.87
  12 │ ft_btts_yes            1.55        1.57             2.13       0.27   -26.0      9.09
  13 │ ft_btts_no             2.78        2.84             1.94       0.23    45.27    16.3
  14 │ ft_cs_0_0             20.0        21.0             11.66       3.52    85.75    51.22
  15 │ ft_cs_1_0             16.0        16.5             10.95       2.45    52.79    31.29
  16 │ ft_cs_0_1             16.0        16.5              8.53       1.75    94.75    36.65
  17 │ ft_cs_1_1              8.4         8.6              8.02       0.54     5.21     6.43
  18 │ ft_cs_2_0             20.0        22.0             21.2        5.97     1.06    25.64
  19 │ ft_cs_0_2             19.0        19.5             12.79       2.81    55.09    31.26
  20 │ ft_cs_2_1             12.5        13.0             15.54       2.88   -17.08    13.73
  21 │ ft_cs_1_2             11.5        12.0             12.03       1.34    -3.33     9.76
  22 │ ft_cs_2_2             13.0        13.5             23.33       4.9    -42.07    10.93
  23 │ ft_cs_3_0             36.0        40.0             63.55      27.79   -33.97    25.31
  24 │ ft_cs_0_3             34.0        36.0             29.48       9.84    26.94    38.34
  25 │ ft_cs_3_1             23.0        24.0             46.61      17.69   -44.57    18.31
  26 │ ft_cs_1_3             22.0        23.0             27.75       7.66   -15.34    20.93
  27 │ ft_cs_3_2             27.0        28.0             70.05      27.55   -56.26    15.08
  28 │ ft_cs_2_3             28.0        29.0             53.88      17.98   -42.84    17.07
  29 │ ft_cs_3_3             46.0        50.0            161.85      77.03   -65.83    14.36
  30 │ ht_1x2_home            3.25        3.35             5.07       1.43   -31.2     17.9
  31 │ ht_1x2_draw            2.56        2.6              2.41       0.27     7.68    11.83
  32 │ ht_1x2_away            3.15        3.3              2.83       0.58    15.82    22.54
  33 │ ht_ou_05_under         4.0         4.2              3.32       0.7     25.7     24.86
  34 │ ht_ou_05_over          1.32        1.34             1.47       0.14    -9.48     8.2
  35 │ ht_ou_15_under         1.67        1.69             1.51       0.17    12.2     12.02
  36 │ ht_ou_15_over          2.44        2.5              3.21       0.78   -19.93    17.57
  37 │ ht_ou_25_under         1.19        1.2              1.14       0.06     4.88     5.19
  38 │ ht_ou_25_over          6.0         6.4              9.74       4.21   -28.81    26.14
  39 │ ht_cs_0_0              4.0         4.1              3.32       0.7     25.7     24.86
  40 │ ht_cs_1_0              5.6         6.0              7.39       1.67   -20.79    16.07
  41 │ ht_cs_0_1              5.5         6.2              4.71       0.66    18.89    15.6
  42 │ ht_cs_1_1              7.6         8.4             10.53       1.85   -25.89    11.41

julia> match_row_to_analyze.event_name
"Man Utd v Chelsea"
"""


"""

julia> display(analysis_df)
42×7 DataFrame
 Row │ market          market_back  market_lay  model_mean_odds  model_std  mean_ev  std_ev  
     │ Symbol          Float64      Float64     Float64          Float64    Float64  Float64 
─────┼───────────────────────────────────────────────────────────────────────────────────────
   1 │ ft_1x2_home            2.16        2.2              2.34       0.4     -5.24    15.19
   2 │ ft_1x2_draw            3.5         3.55             4.02       0.33   -12.37     7.06
   3 │ ft_1x2_away            3.9         3.95             3.35       0.7     21.27    23.65
   4 │ ft_ou_05_under        12.5        13.5             16.87       5.99   -17.51    26.2
   5 │ ft_ou_05_over          1.08        1.09             1.07       0.02     0.87     2.26
   6 │ ft_ou_15_under         3.7         3.8              4.38       1.13   -10.42    20.83
   7 │ ft_ou_15_over          1.35        1.37             1.33       0.1      2.32     7.6
   8 │ ft_ou_25_under         1.91        1.93             2.14       0.38    -8.28    14.71
   9 │ ft_ou_25_over          2.06        2.1              1.97       0.31     7.08    15.87
  10 │ ft_ou_35_under         1.37        1.38             1.45       0.16    -4.41     9.81
  11 │ ft_ou_35_over          3.65        3.7              3.51       0.89    10.32    26.14
  12 │ ft_btts_yes            1.86        1.88             1.85       0.2      2.02    11.02
  13 │ ft_btts_no             2.12        2.16             2.26       0.31    -4.28    12.56
  14 │ ft_cs_0_0             12.5        13.0             16.87       5.99   -17.51    26.2
  15 │ ft_cs_1_0              8.8         9.2             10.83       2.46   -14.93    17.63
  16 │ ft_cs_0_1             13.0        13.5             13.46       3.72     3.26    25.77
  17 │ ft_cs_1_1              7.4         7.6              8.67       0.94   -13.72     8.33
  18 │ ft_cs_2_0             12.0        12.5             14.31       2.96   -12.8     16.83
  19 │ ft_cs_0_2             25.0        26.0             21.97       6.23    22.13    31.73
  20 │ ft_cs_2_1             10.0        11.0             11.48       1.11   -12.16     7.58
  21 │ ft_cs_1_2             15.5        16.0             14.19       2.07    11.4     14.95
  22 │ ft_cs_2_2             17.0        17.5             18.84       2.95    -7.74    13.2
  23 │ ft_cs_3_0             26.0        28.0             29.18       9.33    -2.86    27.54
  24 │ ft_cs_0_3             70.0        75.0             55.07      20.58    43.29    49.01
  25 │ ft_cs_3_1             22.0        23.0             23.47       6.33    -0.41    23.06
  26 │ ft_cs_1_3             44.0        46.0             35.65      10.23    32.81    35.05
  27 │ ft_cs_3_2             36.0        38.0             38.6       11.84     0.93    27.13
  28 │ ft_cs_2_3             50.0        55.0             47.45      14.26    14.24    31.71
  29 │ ft_cs_3_3             90.0        95.0             97.46      39.97     6.5     39.42
  30 │ ht_1x2_home            2.86        2.92             3.94       0.98   -23.09    18.16
  31 │ ht_1x2_draw            2.3         2.38             2.64       0.28   -11.95     9.32
  32 │ ht_1x2_away            4.4         4.6              3.0        0.66    53.25    31.64
  33 │ ht_ou_05_under         3.1         3.25             4.17       1.02   -21.48    17.79
  34 │ ht_ou_05_over          1.45        1.47             1.35       0.11     8.27     8.32
  35 │ ht_ou_15_under         1.44        1.48             1.71       0.24   -14.38    11.32
  36 │ ht_ou_15_over          3.1         3.25             2.57       0.54    25.67    24.37
  37 │ ht_ou_25_under         1.11        1.13             1.21       0.09    -7.77     6.21
  38 │ ht_ou_25_over          9.0         9.8              6.65       2.5     52.17    50.34
  39 │ ht_cs_0_0              3.05        3.2              4.17       1.02   -22.75    17.51
  40 │ ht_cs_1_0              4.4         4.8              6.71       1.34   -31.98    12.6
  41 │ ht_cs_0_1              6.4         7.0              5.49       0.9     19.51    18.1
  42 │ ht_cs_1_1              9.0         9.8              8.86       1.05     2.84    10.95

julia> match_row_to_analyze.event_name
"Fulham v Brentford"
"""


"""
I need a util function in MatchDayUtils to help with the following: 

i have a python package i made to get the odds and matchs from betfair market api service. 
here: 
(webscraper) ⚡➜ whatstheodds (! main) pwd                                             
/home/james/bet_project/whatstheodds

note we need to be in the correct conda envirmoent: webscrape 

need a functions to get the matchs for today via the cli tool i made 
(webscraper) ⚡➜ whatstheodds (! main) python live_odds_cli.py list -f england scotland 
Loaded 99 teams from '/home/james/bet_project/whatstheodds/mappings/england.json'
Loaded 27 teams from '/home/james/bet_project/whatstheodds/mappings/scotland.json'

Finding today's soccer matches...
Found 56 matches:
  - 11:30 | Blackpool v Barnsley
  - 11:30 | Bolton v Wigan
  - 11:30 | Birmingham v Swansea
  - 11:30 | Leicester v Coventry
  - 11:30 | QPR v Stoke
  - 11:30 | Lincoln v Luton
  - 11:30 | Brackley Town v Sutton Utd
  - 11:30 | Cambridge Utd v Fleetwood Town
  - 11:30 | Harrogate Town v Shrewsbury
  - 11:30 | Liverpool v Everton
  - 14:00 | Rotherham v Stockport
  - 14:00 | Huddersfield v Burton Albion
  - 14:00 | Cardiff v Bradford
  - 14:00 | Reading v Leyton Orient
  - 14:00 | Hull v Southampton
  - 14:00 | Plymouth v Peterborough
  - 14:00 | Port Vale v Mansfield

if possible can run this python script via julia functions and create a DataFrames of the results, 
columns: time | event name | home_team | away_team | 

where we need to do a reverse json look up for the home_team, away_team, since our model use a diffeerent name structure, we can use the json files 
at 
Loaded 99 teams from '/home/james/bet_project/whatstheodds/mappings/england.json'
Loaded 27 teams from '/home/james/bet_project/whatstheodds/mappings/scotland.json'
example of the files 
{
  "celtic": "Celtic",
  "rangers": "Rangers", 
  "heart-of-midlothian": "Hearts",
  "falkirk-fc": "Falkirk",
  "hibernian": "Hibernian",
.. 

{
  "accrington-stanley": "Accrington",
  "afc-wimbledon": "AFC Wimbledon",
  "arsenal": "Arsenal",
  "aston-villa": "Aston Villa",
  "barnsley": "Barnsley",
  "barrow": "Barrow",
  "birmingham-city": "Birmingham",
  "blackburn-rovers": "Blackburn",
...


Following this, i want to compare the models to that of the odds, in order to do this we can call the python functions 
to get the live odds in a dict format, thus we need a julia funcitons in MatchDayUtils to call this python 
(webscraper) ⚡➜ whatstheodds (! main) python live_odds_cli.py odds "Rangers v Hibernian" -d 
Searching for event: 'Rangers v Hibernian'
Found Event: Rangers v Hibernian, ID: 34684751
Found 12 markets. Fetching odds individually...
{
  "ft": {
    "Correct Score": {
      "0 - 0": {
        "back": {
          "price": 26.0,
          "size": 29.77
        },
        "lay": {
          "price": 32.0,
          "size": 15.45
        }
      },
      "0 - 1": {
        "back": {
          "price": 21.0,
          "size": 13.06
        },
        "lay": {
          "price": 30.0,
          "size": 10.86
        }
      },
      "0 - 2": {
        "back": {
          "price": 29.0,
          "size": 10.74
        },
        "lay": {
          "price": 36.0,
          "size": 10.0
        }
      },
      "0 - 3": {
        "back": {
          "price": 14.5,
          "size": 12.73
        },
        "lay": {
          "price": 85.0,
          "size": 12.12
        }
      },
      "1 - 0": {
        "back": {
          "price": 14.0,
          "size": 12.09
        },
        "lay": {
          "price": 15.5,
          "size": 13.07
        }
      },
      "1 - 1": {
        "back": {
          "price": 10.5,
          "size": 20.42
        },
        "lay": {
          "price": 12.5,
          "size": 39.92
        }
      },
      "1 - 2": {
        "back": {
          "price": 15.5,
          "size": 20.91
        },
        "lay": {
          "price": 17.0,
          "size": 18.21
        }
      },
      "1 - 3": {
        "back": {
          "price": 20.0,
          "size": 11.99
        },
        "lay": {
          "price": 38.0,
          "size": 33.23
        }
      },
      "2 - 0": {
        "back": {
          "price": 14.5,
          "size": 19.11
        },
        "lay": {
          "price": 16.0,
          "size": 31.86
        }
      },
      "2 - 1": {
        "back": {
          "price": 10.0,
          "size": 60.66
        },
        "lay": {
          "price": 11.0,
          "size": 56.14
        }
      },
      "2 - 2": {
        "back": {
          "price": 14.0,
          "size": 29.89
        },
        "lay": {
          "price": 16.0,
          "size": 59.15
        }
      },
      "2 - 3": {
        "back": {
          "price": 26.0,
          "size": 10.78
        },
        "lay": {
          "price": 38.0,
          "size": 10.04
        }
      },
      "3 - 0": {
        "back": {
          "price": 20.0,
          "size": 15.63
        },
        "lay": {
          "price": 26.0,
          "size": 38.36
        }
      },
      "3 - 1": {
        "back": {
          "price": 14.5,
          "size": 18.89
        },
        "lay": {
          "price": 16.5,
          "size": 17.14
        }
      },
      "3 - 2": {
        "back": {
          "price": 15.0,
          "size": 12.13
        },
        "lay": {
          "price": 23.0,
          "size": 30.44
        }
      },
      "3 - 3": {
        "back": {
          "price": 32.0,
          "size": 13.21
        },
        "lay": {
          "price": 55.0,
          "size": 24.57
        }
      },
      "Any Other Home Win": {
        "back": {
          "price": 6.6,
          "size": 75.86
        },
        "lay": {
          "price": 7.4,
          "size": 24.36
        }
      },
      "Any Other Away Win": {
        "back": {
          "price": 20.0,
          "size": 17.61
        },
        "lay": {
          "price": 24.0,
          "size": 27.59
        }
      },
      "Any Other Draw": {
        "back": {
          "price": 15.0,
          "size": 14.6
        },
        "lay": {
          "price": 1000.0,
          "size": 7.44
        }
      }
    },
    "Over/Under 0.5 Goals": {
      "Under 0.5 Goals": {
        "back": {
          "price": 8.2,
          "size": 12.18
        },
        "lay": {
          "price": 36.0,
          "size": 58.06
        }
      },
      "Over 0.5 Goals": {
        "back": {
          "price": 1.03,
          "size": 1697.29
        },
        "lay": {
          "price": 1.04,
          "size": 61.0
        }
      }
    },
    "Over/Under 1.5 Goals": {
      "Under 1.5 Goals": {
        "back": {
          "price": 6.8,
          "size": 11.33
        },
        "lay": {
          "price": 7.2,
          "size": 24.78
        }
      },
      "Over 1.5 Goals": {
        "back": {
          "price": 1.16,
          "size": 315.84
        },
        "lay": {
          "price": 1.17,
          "size": 31.0
        }
      }
    },
    "Over/Under 2.5 Goals": {
      "Under 2.5 Goals": {
        "back": {
          "price": 2.96,
          "size": 16.14
        },
        "lay": {
          "price": 3.1,
          "size": 14.61
        }
      },
      "Over 2.5 Goals": {
        "back": {
          "price": 1.47,
          "size": 172.93
        },
        "lay": {
          "price": 1.51,
          "size": 23.87
        }
      }
    },
    "Match Odds": {
      "Rangers": {
        "back": {
          "price": 1.86,
          "size": 23.0
        },
        "lay": {
          "price": 1.89,
          "size": 12.76
        }
      },
      "Hibernian": {
        "back": {
          "price": 3.8,
          "size": 10.79
        },
        "lay": {
          "price": 4.1,
          "size": 101.46
        }
      },
      "The Draw": {
        "back": {
          "price": 4.6,
          "size": 12.66
        },
        "lay": {
          "price": 4.8,
          "size": 10.0
        }
      }
    },
    "Both teams to Score?": {
      "Yes": {
        "back": {
          "price": 1.53,
          "size": 90.36
        },
        "lay": {
          "price": 1.61,
          "size": 69.03
        }
      },
      "No": {
        "back": {
          "price": 2.66,
          "size": 41.78
        },
        "lay": {
          "price": 2.86,
          "size": 47.81
        }
      }
    },
    "Over/Under 3.5 Goals": {
      "Under 3.5 Goals": {
        "back": {
          "price": 1.79,
          "size": 13.21
        },
        "lay": {
          "price": 1.85,
          "size": 36.21
        }
      },
      "Over 3.5 Goals": {
        "back": {
          "price": 2.18,
          "size": 18.0
        },
        "lay": {
          "price": 2.28,
          "size": 19.39
        }
      }
    }
  },
  "ht": {
    "First Half Goals 1.5": {
      "Under 1.5 Goals": {
        "back": {
          "price": 1.79,
          "size": 42.91
        },
        "lay": {
          "price": 1.84,
          "size": 20.0
        }
      },
      "Over 1.5 Goals": {
        "back": {
          "price": 2.18,
          "size": 40.88
        },
        "lay": {
          "price": 2.28,
          "size": 55.83
        }
      }
    },
    "First Half Goals 0.5": {
      "Under 0.5 Goals": {
        "back": {
          "price": 4.5,
          "size": 49.82
        },
        "lay": {
          "price": 4.9,
          "size": 29.51
        }
      },
      "Over 0.5 Goals": {
        "back": {
          "price": 1.26,
          "size": 72.0
        },
        "lay": {
          "price": 1.28,
          "size": 111.85
        }
      }
    },
    "First Half Goals 2.5": {
      "Under 2.5 Goals": {
        "back": {
          "price": 1.22,
          "size": 475.59
        },
        "lay": {
          "price": 1.25,
          "size": 154.8
        }
      },
      "Over 2.5 Goals": {
        "back": {
          "price": 5.1,
          "size": 10.0
        },
        "lay": {
          "price": 5.4,
          "size": 37.84
        }
      }
    },
    "Half Time": {
      "Rangers": {
        "back": {
          "price": 2.34,
          "size": 10.0
        },
        "lay": {
          "price": 2.46,
          "size": 17.0
        }
      },
      "Hibernian": {
        "back": {
          "price": 4.2,
          "size": 36.02
        },
        "lay": {
          "price": 4.6,
          "size": 41.21
        }
      },
      "The Draw": {
        "back": {
          "price": 2.74,
          "size": 11.0
        },
        "lay": {
          "price": 2.88,
          "size": 12.0
        }
      }
    },
    "Half Time Score": {
      "0 - 0": {
        "back": {
          "price": 4.4,
          "size": 37.96
        },
        "lay": {
          "price": 4.9,
          "size": 55.32
        }
      },
      "1 - 1": {
        "back": {
          "price": 7.6,
          "size": 14.94
        },
        "lay": {
          "price": 8.6,
          "size": 18.66
        }
      },
      "2 - 2": {
        "back": {
          "price": 13.5,
          "size": 16.31
        },
        "lay": {
          "price": 65.0,
          "size": 11.33
        }
      },
      "1 - 0": {
        "back": {
          "price": 4.7,
          "size": 38.93
        },
        "lay": {
          "price": 5.2,
          "size": 37.4
        }
      },
      "2 - 0": {
        "back": {
          "price": 10.5,
          "size": 16.84
        },
        "lay": {
          "price": 11.5,
          "size": 15.79
        }
      },
      "2 - 1": {
        "back": {
          "price": 17.0,
          "size": 10.0
        },
        "lay": {
          "price": 19.0,
          "size": 12.44
        }
      },
      "0 - 1": {
        "back": {
          "price": 7.4,
          "size": 18.28
        },
        "lay": {
          "price": 8.2,
          "size": 27.34
        }
      },
      "0 - 2": {
        "back": {
          "price": 11.0,
          "size": 10.18
        },
        "lay": {
          "price": 28.0,
          "size": 24.67
        }
      },
      "1 - 2": {
        "back": {
          "price": 25.0,
          "size": 10.83
        },
        "lay": {
          "price": 29.0,
          "size": 10.98
        }
      },
      "Any unquoted": {
        "back": {
          "price": 11.5,
          "size": 15.23
        },
        "lay": {
          "price": 12.0,
          "size": 13.5
        }
      }
    }
  }
}
Logged out.

then it would be great to have a function to generate the models predictions like 
predictions = generate_predictions(loaded_models, team_name_home, team_name_away, match_league_id)
odds_df = create_odds_dataframe(predictions)
but extend this so we can have the model predict displayed to the market odds. 
I assume in a datafram,
and if possible display the EV compared to the models and the market per lines of bets .


alos since we are using mcmc chains for or bayesain model, it would be great to have a plot function of the densisty of the market lines, 
with the market back and lay odds marked with a vertical line ( and leageu mention the quantile of the back and lay odds to the model chain ) 



"""
