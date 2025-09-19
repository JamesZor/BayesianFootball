using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions


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
    
    predictions = Dict{String, Any}() # Using Any to hold MatchLinePredictions struct
    
    match_to_predict = DataFrame(
        home_team=home_team,
        away_team=away_team,
        tournament_id=league_id,
        home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
    )

    for (name, model) in models
        println("Generating predictions for model: '$name'")
        
        # We'll use the first chain sequence, as in your example
        chains_for_round = model.result.chains_sequence[1]

        features = BayesianFootball.create_master_features(
            match_to_predict,
            model.result.mapping
        )

        preds = BayesianFootball.predict_match_lines(
            model.config.model_def,
            chains_for_round,
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


# Helper to combine two vectors with weights
_combine_vectors(v1, v2, w1, w2) = w1 .* v1 .+ w2 .* v2

"""
    combine_predictions(pred1, pred2, w1::Float64, w2::Float64)

Combines two `MatchLinePredictions` objects using a weighted linear pool.

Note: The `correct_score` dictionary is not combined as it requires re-computation 
from the combined goal expectancies (λ). The combined λ values are calculated.

# Arguments
- `pred1`, `pred2`: The two prediction objects to combine.
- `w1`, `w2`: The weights for pred1 and pred2, respectively. Should sum to 1.0.

# Returns
- A new `MatchLinePredictions` object representing the combined model.
"""
function combine_predictions(pred1, pred2, w1::Float64, w2::Float64)
    if !isapprox(w1 + w2, 1.0)
        @warn "Weights do not sum to 1.0 (sum = $(w1+w2))"
    end

    # Combine Full Time predictions
    ft1, ft2 = pred1.ft, pred2.ft
    combined_ft = Predictions.MatchFTPredictions(
        _combine_vectors(ft1.λ_h, ft2.λ_h, w1, w2),
        _combine_vectors(ft1.λ_a, ft2.λ_a, w1, w2),
        _combine_vectors(ft1.home, ft2.home, w1, w2),
        _combine_vectors(ft1.draw, ft2.draw, w1, w2),
        _combine_vectors(ft1.away, ft2.away, w1, w2),
        Dict(), # Intentionally left empty - see docstring
        _combine_vectors(ft1.under_05, ft2.under_05, w1, w2),
        _combine_vectors(ft1.under_15, ft2.under_15, w1, w2),
        _combine_vectors(ft1.under_25, ft2.under_25, w1, w2),
        _combine_vectors(ft1.under_35, ft2.under_35, w1, w2),
        _combine_vectors(ft1.btts, ft2.btts, w1, w2)
    )

    # Combine Half Time predictions
    ht1, ht2 = pred1.ht, pred2.ht
    combined_ht = Predictions.MatchHTPredictions(
        _combine_vectors(ht1.λ_h, ht2.λ_h, w1, w2),
        _combine_vectors(ht1.λ_a, ht2.λ_a, w1, w2),
        _combine_vectors(ht1.home, ht2.home, w1, w2),
        _combine_vectors(ht1.draw, ht2.draw, w1, w2),
        _combine_vectors(ht1.away, ht2.away, w1, w2),
        Dict(), # Intentionally left empty
        _combine_vectors(ht1.under_05, ht2.under_05, w1, w2),
        _combine_vectors(ht1.under_15, ht2.under_15, w1, w2),
        _combine_vectors(ht1.under_25, ht2.under_25, w1, w2)
    )

    return Predictions.MatchLinePredictions(combined_ht, combined_ft)
end



##############################
# main script 
##############################
## --- 1. Define Models and Match ---
model_paths = Dict(
    "long_term" => "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_verification_20250913-173119",
    "short_term" => "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_short_20250916-151954",
    "now_term" => "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_25_20250916-180541"
)
loaded_models = load_models_from_paths(model_paths)

const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26" # Your data path
println("Loading data...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)



match_home_team = "sheffield-wednesday"
match_away_team = "grimsby-town"
match_league_id = 2


# --- 2. Load Models and Generate Predictions ---
predictions = generate_predictions(loaded_models, match_home_team, match_away_team, match_league_id)

# --- 3. Create and View Odds DataFrame ---
odds_df = create_odds_dataframe(predictions)
println("\n--- Initial Odds Comparison ---")
display(odds_df)

# --- 4. Plot and Compare Distributions ---
# Compare Home win odds
plot_odds_distributions(predictions, :ft, :home, title_suffix="for Sheffield Wednesday vs Grimsby Town")

# Compare Over/Under 2.5 odds
plot_odds_distributions(predictions, :ft, :under_25, title_suffix="for Sheffield Wednesday vs Grimsby Town")

# --- 5. Create and Analyze a Combined Model ---
# Let's create an ensemble giving 60% weight to the short-term model
w_short = 0.6
w_long = 0.4

combined_preds = combine_predictions(
    predictions["short_term"], 
    predictions["long_term"], 
    w_short, 
    w_long
)

# Add the combined model to our predictions dictionary
predictions["ensemble_60_short"] = combined_preds;

# --- 6. View Updated DataFrame and Plots with the Ensemble Model ---
full_odds_df = create_odds_dataframe(predictions)
println("\n--- Odds Comparison with Ensemble Model ---")
display(full_odds_df)

# Re-plot to see how the ensemble distribution compares
plot_odds_distributions(predictions, :ft, :home, title_suffix="with Ensemble")



#  ==============================    
# football league trophy 
#  ==============================    

#++++++++++++++++++++++++++++++ 
# match 1  blackpool v barrow 
#++++++++++++++++++++++++++++++ 

match_home_team = "blackpool"
match_away_team = "barrow"
match_league_id = 3

predictions = generate_predictions(loaded_models, match_home_team, match_away_team, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)

"""
 Under 1.5 Goals - Over/Under 1.5 Goals
Betfair Bet ID 1:401928951484 | Placed: 16-Sep-25 18:32:53 	Back 	5.10 	1.00

 Blackpool v Barrow
Under 2.5 Goals - Over/Under 2.5 Goals
Betfair Bet ID 1:401928904674 | Matched: 16-Sep-25 18:32:27 	Back 	2.40 	1.00 

"""

#++++++++++++++++++++++++++++++ 
# match 2  Exeter v Cardiff
#++++++++++++++++++++++++++++++ 
#
match_home_team = "exeter-city"
match_away_team = "cardiff-city"
match_league_id = 3

predictions = generate_predictions(loaded_models, match_home_team, match_away_team, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)


"""
 16-Sep-25
18:43:17 	Exeter v Cardiff
Under 2.5 Goals - Over/Under 2.5 Goals
Betfair Bet ID 1:401930122216 | Matched: 16-Sep-25 18:43:17 	Back 	2.20 	1.00 	-- 	--
	1.20
	Matched
16-Sep-25
18:43:11 	Exeter v Cardiff
Under 1.5 Goals - Over/Under 1.5 Goals
Betfair Bet ID 1:401930111011 | Matched: 16-Sep-25 18:43:11 	Back 	4.50 	1.00 	-- 	--
	3.50
	Matched
16-Sep-25
18:41:21 	Exeter v Cardiff
Exeter - Match Odds
Betfair Bet ID 1:401929921368 | Matched: 16-Sep-25 18:41:21 	Back 	4.20 	1.00 	-- 	--
	3.20
	Matched 

"""

#  ==============================    
#  English football league cup  
#  ==============================    

#++++++++++++++++++++++++++++++ 
# match 3  sheffield-wednesday v grimsby-town
#++++++++++++++++++++++++++++++ 


match_home_team = "sheffield-wednesday"
match_away_team = "grimsby-town"
filter(row -> (row.season=="25/26") && ( (row.home_team==match_home_team) || (row.home_team==match_away_team)), data_store.matches)
match_league_id = 2

predictions = generate_predictions(loaded_models, match_home_team, match_away_team, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)


"""
 16-Sep-25
19:00:38 	Sheff Wed v Grimsby
Over 2.5 Goals - Over/Under 2.5 Goals	Back 	2.10 	1.00 1.10

16-Sep-25
18:53:15 	Sheff Wed v Grimsby
Sheff Wed - Match Odds
Back 	3.45 	1.00  2.45
"""

#++++++++++++++++++++++++++++++ 
# match 4  brentford v Aston villa 
#++++++++++++++++++++++++++++++ 

match_home_team = "brentford"
match_away_team = "aston-villa"
filter(row -> (row.season=="25/26") && ( (row.home_team==match_home_team) || (row.home_team==match_away_team)), data_store.matches)
match_league_id = 1

predictions = generate_predictions(loaded_models, match_home_team, match_away_team, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)

# back ft brentford @ 3.55
# back ft under 25 @ 2.28

#++++++++++++++++++++++++++++++ 
# match 5  crystal palace v millwall
#++++++++++++++++++++++++++++++ 
match_home_team = "crystal-palace"
match_away_team = "millwall"
filter(row -> (row.season=="25/26") && ( (row.home_team==match_home_team) || (row.home_team==match_away_team)), data_store.matches)
match_league_id = 1

predictions = generate_predictions(loaded_models, match_home_team, match_away_team, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)

# back ft under 05 @ 19.5
# back ft under 15 @ 4.8
# back ft under 25 @ 2.5


#++++++++++++++++++++++++++++++ 
# match 6  swansea v nottm forrest 
#++++++++++++++++++++++++++++++ 
match_home_team = "swansea-city"
match_away_team = "nottingham-forest"
filter(row -> (row.season=="25/26") && ( (row.home_team==match_home_team) || (row.home_team==match_away_team)), data_store.matches)
match_league_id = 1

predictions = generate_predictions(loaded_models, match_home_team, match_away_team, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)

plot_odds_distributions(predictions, :ft, :home, title_suffix="with Ensemble")
plot_odds_distributions(predictions, :ft, :draw, title_suffix="with Ensemble")
plot_odds_distributions(predictions, :ft, :under_25, title_suffix="with Ensemble")
plot_odds_distributions(predictions, :ft, :btts, title_suffix="with Ensemble")
"""
3×13 DataFrame
 Row │ Model       Time    Home     Draw     Away     O05      U05       O15      U15      O25      U25      BTTS_Yes  BTTS_No  
     │ String      String  Float64  Float64  Float64  Float64  Float64   Float64  Float64  Float64  Float64  Float64?  Float64? 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ long_term   FT      3.11972  3.49535  2.58975  1.11975   9.55295  1.52932  2.92981  2.58967  1.6444    2.22243   1.83038
   2 │ short_term  FT      2.80011  3.43874  3.08343  1.13509   9.01294  1.59451  2.80693  2.81156  1.59975   2.36545   1.77003
   3 │ now_term    FT      2.25249  3.85563  4.34852  1.11427  12.178    1.50814  3.42848  2.54849  1.81178   2.29143   1.86568

julia> ht_odds = filter(row->row.Time=="HT", odds_df)
3×13 DataFrame
 Row │ Model       Time    Home     Draw     Away     O05      U05      O15      U15      O25      U25      BTTS_Yes  BTTS_No  
     │ String      String  Float64  Float64  Float64  Float64  Float64  Float64  Float64  Float64  Float64  Float64?  Float64? 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ long_term   HT      4.81041  2.72077  2.40558  1.31831  4.21212  2.41112  1.72688  5.8689   1.21318   missing   missing 
   2 │ short_term  HT      4.63933  2.54875  2.80752  1.3946   3.74091  2.80172  1.61084  7.69933  1.17319   missing   missing 
   3 │ now_term    HT      5.37654  3.0343   2.27044  1.2563   5.35121  2.13154  1.99094  4.79871  1.30781   missing   missing


Match Odds	Swansea to Win	now_term	2.25	5.1	+126.7%
Match Odds	Swansea to Win	short_term	2.80	5.1	+82.1%
Match Odds	Swansea to Win	long_term	3.12	5.1	+63.5%
Under/Over 2.5	Under 2.5 Goals	short_term	1.60	2.2	+37.5%
Under/Over 2.5	Under 2.5 Goals	long_term	1.64	2.2	+34.1%
First Half Goals	Over 1.5 Goals	now_term	2.13	2.88	+35.2%
Match Odds	The Draw	short_term	3.44	4.2	+22.1%
Match Odds	The Draw	long_term	3.50	4.2	+20.0%
Under/Over 2.5	Under 2.5 Goals	now_term	1.81	2.2	+21.5%
"""

#####

"""
Extracted the new odds for uk 20 25, and just wanted to see them 
"""
data_path = "/home/james/bet_project/football/england_20_25/"

data_files = DataFiles(data_path)
data_store = DataStore(data_files)
ds = data_store

model_paths = Dict(
    "long_term" => "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_verification_20250913-173119",
    "short_term" => "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_short_20250916-151954",
#    "now_term" => "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_25_20250916-180541"
)
loaded_models = load_models_from_paths(model_paths)

ds.matches

m_id = 12436442
m_id = 12436449
m_id = 13379969
m_row = filter(row -> row.match_id==m_id, ds.matches)
m_p = generate_predictions(loaded_models, String(first(m_row.home_team)), String(first(m_row.away_team)), first(m_row.tournament_id))


odds_df = create_odds_dataframe(m_p)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)

m_odd = filter(row -> row.match_id==m_id, ds.odds)
game_line = filter(row -> row.minutes==0, m_odd)

game_line[:, [:home, :away, :draw]]
game_line[:, [:home, :away, :draw, :ht_home, :ht_away, :ht_draw, :ht_under_0_5, :ht_under_1_5, :under_0_5, :under_1_5, :btts_yes, :btts_no]]
ft_odds
