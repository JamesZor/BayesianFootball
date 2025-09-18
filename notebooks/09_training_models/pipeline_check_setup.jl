using BayesianFootball
using DataFrames
using Dates

# --- 1. General Setup ---
const EXPERIMENT_NAME = "pipeline_verification_test"
const SAVE_PATH = "./experiments"
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26" # Your data path

println("Loading data...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# --- 2. Configuration for a Single, Full-Data Training Run ---
# We train on all past data as a single block.
# The target season isn't used for training in this simplified run.
cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    ["20/21", "21/22", "22/23", "23/24", "24/25", "25/26"],
    [],
    :round
)

# Use a small number of samples for a quick test run.
# Increase this to ~1000 for more stable estimates later.
sample_config = BayesianFootball.ModelSampleConfig(2000, true)
mapping_funcs = BayesianFootball.MappingFunctions(BayesianFootball.create_list_mapping)

# Define the model configuration we want to test
config = create_experiment_config(
    "maher_basic_verification", :maher, :basic,
    cv_config, sample_config, mapping_funcs
)

# --- 3. Manually Create a Single Training Set from Base Seasons ---
println("Preparing a single training set...")
train_df = filter(row -> row.season in cv_config.base_seasons, data_store.matches)

# This is the core logic from the pipeline, run for a single split
println("Composing the training function...")
mapping = BayesianFootball.MappedData(data_store, config.mapping_funcs)
training_morphism = BayesianFootball.compose_training_morphism(
    config.model_def,
    config.sample_config,
    mapping
)

# --- 4. Run the Training ---
println("🚀 Starting training on $(nrow(train_df)) matches...")
# We pass "Full History" as the info string for this training block
trained_chains = training_morphism(train_df, "Full History")
println("✅ Training complete.")

# --- 5. Package, Save, and Load the Model ---
# Manually create the ExperimentResult and TrainedModel
result = ExperimentResult(
    [trained_chains], # The result is a sequence of chains, here with just one entry
    mapping,
    hash(config),
    0.0 # Total time isn't important here
)

# This is the final object you'll use for predictions
model = TrainedModel(config, result)

# Save and load to verify persistence
run_manager = prepare_run(EXPERIMENT_NAME, config, SAVE_PATH)
save(run_manager, result)
println("💾 Model saved.")

loaded_model = load_model(run_manager.run_path)
println("👍 Model loaded successfully. Verification complete.")


using StatsPlots, Distributions

# --- 6. Inspect Model Parameters ---
# Let's check a well-known Scottish team, e.g., "Celtic"
team_name = "crystal-palace"

# Find the integer ID for the team
team_id = loaded_model.result.mapping.team[team_name]
println("'$team_name' has team ID: $team_id")

# Extract the single set of chains from our model
chains = loaded_model.result.chains_sequence[1]

# Extract the posterior samples for attack and defense
# The 'extract_posterior_samples' function handles the identifiability constraint
posterior_samples = BayesianFootball.extract_posterior_samples(loaded_model.config.model_def, chains.ft)

# Get the samples for our specific team
attack_samples = posterior_samples.α[:, team_id]
defense_samples = posterior_samples.β[:, team_id]
home_advantage_samples = posterior_samples.γ



# Let's check the mean values
println("Posterior Mean for '$team_name' Attack (α): ", round(mean(attack_samples), digits=3))
println("Posterior Mean for '$team_name' Defense (β): ", round(mean(defense_samples), digits=3))
println("Posterior Mean for Home Advantage (γ): ", round(mean(home_advantage_samples), digits=3))

# Plot the distributions for a visual check
density(attack_samples, label="$team_name Attack (α)", title="Parameter Distributions")
density!(defense_samples, label="$team_name Defense (β)")


# --- 7. Predict a Match ---
# Let's predict a hypothetical match for today
# We need to provide the team names and a tournament_id from your data
celtic_id = 84 # Example ID for Scottish Premiership, check your data
rangers_id = 84

team_name_home = "west-ham-united"
team_name_away = "tottenham-hotspur"

# Find the integer ID for the team
team_id_home = loaded_model.result.mapping.team[team_name_home]
team_id_away = loaded_model.result.mapping.team[team_name_away]

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=1,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)

# --- Deconstructed Prediction Pipeline ---

# 1. Get the correct set of chains (we only have one)
# The round number is 1 because it's the first round of the target season
chains_for_round = loaded_model.result.chains_sequence[1]

# 2. Create the features for this single match
features = BayesianFootball.create_master_features(
    match_to_predict,
    loaded_model.result.mapping
)

# 3. Call the prediction function
println("\npredicting match: Celtic vs Rangers...")
match_predictions = BayesianFootball.predict_match_lines(
    loaded_model.config.model_def,
    chains_for_round,
    features,
    loaded_model.result.mapping
)

# --- 8. Interpret the Predictions ---
# Let's look at the Full-Time (FT) 1X2 market
ft_preds = match_predictions.ht

home_win_prob = mean(ft_preds.home)
draw_prob = mean(ft_preds.draw)
away_win_prob = mean(ft_preds.away)

# Convert probabilities to decimal odds
home_odds = 1 / home_win_prob
draw_odds = 1 / draw_prob
away_odds = 1 / away_win_prob

println("\n--- Predicted FT Odds ---")
println("$team_name_home Win: ", round(home_odds, digits=2), " (", round(home_win_prob*100, digits=1), "%)")
println("Draw:       ", round(draw_odds, digits=2), " (", round(draw_prob*100, digits=1), "%)")
println("$team_name_away Win:   ", round(away_odds, digits=2), " (", round(away_win_prob*100, digits=1), "%)")

# Compare these odds to what you see on a betting exchange or bookmaker website!
#


# --- Today's Matches with Market Odds ---
# Structure: (Home Team, Away Team, Tournament ID, (Market Home, Market Draw, Market Away))
# Note: Team names are standardized. You may need to adjust them to match your data.

todays_matches_with_odds = [
    # English Premier League (Tournament ID: 1)
    ("crystal-palace", "sunderland", 1, (1.75, 3.85, 6.2)),
    ("everton", "aston-villa", 1, (2.72, 3.35, 2.98)),
    ("newcastle-united", "wolverhampton", 1, (1.41, 5.2, 9.8)),
    ("bournemouth", "brighton-and-hove-albion", 1, (2.52, 3.65, 3.05)),
    ("fulham", "leeds-united", 1, (2.18, 3.5, 3.95)),
    ("west-ham-united", "tottenham-hotspur", 1, (3.55, 3.65, 2.24)),
    ("brentford", "chelsea", 1, (4.3, 4.4, 1.93)),

    # English Championship (Tournament ID: 2)
    ("west-bromwich-albion", "derby-county", 2, (1.87, 3.65, 5.2)),
    ("watford", "blackburn-rovers", 2, (2.58, 3.4, 3.05)),
    ("sheffield-wednesday", "bristol-city", 2, (4.2, 3.75, 2.02)),
    ("wrexham", "queens-park-rangers", 2, (2.12, 3.5, 3.95)),
    ("swansea-city", "hull-city", 2, (1.86, 3.7, 4.8)),
    ("stoke-city", "birmingham-city", 2, (3.05, 3.3, 2.62)),
    ("coventry-city", "norwich-city", 2, (1.78, 4.4, 4.7)),

    # English League One (Tournament ID: 3)
    ("luton-town", "plymouth-argyle", 3, (1.59, 4.4, 6.2)),
    ("wigan-athletic", "doncaster-rovers", 3, (2.56, 3.4, 3.1)),
    ("barnsley", "reading", 3, (1.98, 4.1, 3.95)),
    ("leyton-orient", "bolton-wanderers", 3, (3.55, 3.8, 2.12)),
    ("mansfield-town", "stevenage", 3, (2.74, 3.2, 3.0)),
    ("afc-wimbledon", "rotherham-united", 3, (2.34, 3.3, 3.55)),
    ("exeter-city", "port-vale", 3, (2.88, 3.5, 2.56)),
    ("peterborough-united", "wycombe-wanderers", 3, (3.65, 3.7, 2.12)),

    # English League Two (Tournament ID: 4)
    ("chesterfield", "milton-keynes-dons", 84, (2.44, 3.4, 3.2)),
    ("bristol-rovers", "barrow", 84, (1.96, 3.7, 4.4)),
    ("shrewsbury-town", "salford-city", 84, (3.1, 3.4, 2.52)),
    ("crewe-alexandra", "barnet", 84, (2.6, 3.65, 2.82)),
    ("grimsby-town", "cambridge-united", 84, (2.22, 3.45, 3.55)),
    ("fleetwood-town", "walsall", 84, (2.9, 3.15, 2.78)),
    ("oldham-athletic", "bromley", 84, (2.64, 3.35, 2.98)),
    ("swindon-town", "harrogate-town", 84, (1.64, 4.3, 6.0)),

    # Scottish Championship (Tournament ID: 85)
    ("raith-rovers", "st-johnstone", 55, (3.25, 3.4, 2.38)),
    ("ayr-united", "ross-county", 55, (2.66, 3.6, 2.72)),
    ("queens-park-fc", "greenock-morton", 55, (2.7, 3.55, 2.72)),
    ("arbroath", "dunfermline-athletic", 55, (3.6, 3.35, 2.26))
]

"""
Validates team names from a match list against the trained model's mapping.
Prints a list of any names that are not found in the mapping.
"""
function validate_team_names(model::TrainedModel, matches_list::Vector)
    println("🔍 Checking team names against the model's mapping...")
    
    # Use a Set to store unique invalid names
    invalid_names = Set{String}()
    
    # The mapping dictionary from your trained model
    team_mapping = model.result.mapping.team

    for match_info in matches_list
        home_team = match_info[1]
        away_team = match_info[2]
        
        # Check both home and away teams
        for team_name in [home_team, away_team]
            if !haskey(team_mapping, team_name)
                push!(invalid_names, team_name)
            end
        end
    end
    
    # --- Report the results ---
    if isempty(invalid_names)
        println("✅ All team names are valid and exist in the model mapping!")
    else
        println("\n⚠️ Found $(length(invalid_names)) team name(s) that need to be corrected:")
        for name in invalid_names
            println("   - \"$name\"")
        end
        println("\nPlease correct these names in the `todays_matches_with_odds` list and try again.")
    end
end

# --- Run the validation ---
# Ensure your `loaded_model` and `todays_matches_with_odds` are defined
validate_team_names(loaded_model, todays_matches_with_odds)

using DataFrames
using Statistics

"""
compare_model_to_market(model::TrainedModel, matches_with_odds::Vector)

Generates model predictions for a list of matches and returns a DataFrame
comparing the model's 1x2 FT odds to the provided market odds.

# Arguments
- `model`: The `TrainedModel` object from your experiment.
- `matches_with_odds`: A vector of tuples, where each tuple is
  (home_team, away_team, league_id, (market_home, market_draw, market_away)).

# Returns
- A `DataFrame` with a side-by-side comparison of model and market odds.
"""
function compare_model_to_market(model::TrainedModel, matches_with_odds::Vector)
    results_df = DataFrame(
        home_team = String[],
        away_team = String[],
        model_home_odds = Float64[],
        model_draw_odds = Float64[],
        model_away_odds = Float64[],
        market_home_odds = Float64[],
        market_draw_odds = Float64[],
        market_away_odds = Float64[]
    )

    chains_for_round = model.result.chains_sequence[1]

    println("Comparing model odds to market for $(length(matches_with_odds)) matches...")
    for match_info in matches_with_odds
        home_team, away_team, tournament_id, market_odds = match_info

        # --- Run Prediction ---
        match_df = DataFrame(
            home_team=home_team, away_team=away_team, tournament_id=tournament_id,
            home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
        )
        features = BayesianFootball.create_master_features(match_df, model.result.mapping)
        match_predictions = BayesianFootball.predict_match_lines(
            model.config.model_def, chains_for_round, features, model.result.mapping
        )

        # --- Calculate Model Odds ---
        ft_preds = match_predictions.ft
        model_home_prob = mean(ft_preds.home)
        model_draw_prob = mean(ft_preds.draw)
        model_away_prob = mean(ft_preds.away)

        # --- Push to DataFrame ---
        push!(results_df, (
            home_team = home_team,
            away_team = away_team,
            model_home_odds = round(1 / model_home_prob, digits=2),
            model_draw_odds = round(1 / model_draw_prob, digits=2),
            model_away_odds = round(1 / model_away_prob, digits=2),
            market_home_odds = market_odds[1],
            market_draw_odds = market_odds[2],
            market_away_odds = market_odds[3]
        ))
    end
    println("✅ Comparison complete.")
    return results_df
end


# Get the comparison DataFrame
comparison_df = compare_model_to_market(loaded_model, todays_matches_with_odds)

# Display the results
println(comparison_df)

comparison_df[:, [:home_team, :away_team, :model_home_odds, :market_home_odds, :model_away_odds, :market_away_odds, :model_draw_odds, :market_draw_odds]]



##### west ham v tottenham-hotspur
# --- Odds for West Ham vs. Tottenham ---

# 1. Half-Time (HT) Correct Score Odds
# I've extracted these from the "Half Time Score" market in the screenshot
ht_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}(
    (0, 0) => 3.3,
    (1, 1) => 8.4,
    (2, 2) => 90.0,
    (1, 0) => 6.2,
    (2, 0) => 24.0,
    (2, 1) => 32.0,
    (0, 1) => 5.1,
    (0, 2) => 14.0,
    (1, 2) => 26.0,
    "any_unquoted" => 19.5 # From the "Any unquoted" row
)

# 2. Populate the MatchHTOdds struct
ht_odds = BayesianFootball.Odds.MatchHTOdds(
    3.95,        # Home (West Ham)
    2.38,        # Draw
    2.88,        # Away (Tottenham)
    ht_correct_scores,
    3.4,         # Under 0.5 Goals
    1.4,         # Over 0.5 Goals
    2.84,        # Under 1.5 Goals
    1.53,        # Over 1.5 Goals
    1.14,        # Under 2.5 Goals
    7.8          # Over 2.5 Goals
)


ft_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}(
    (0, 0) => 16.0,
    (0, 1) => 10.5,
    (0, 2) => 13.5,
    (0, 3) => 27.0,
    (1, 0) => 14.0,
    (1, 1) => 7.6,
    (1, 2) => 10.0,
    (1, 3) => 19.0,
    (2, 0) => 23.0,
    (2, 1) => 14.0,
    (2, 2) => 15.5,
    (2, 3) => 32.0,
    (3, 0) => 55.0,
    (3, 1) => 34.0,
    (3, 2) => 40.0,
    (3, 3) => 65.0,
    "other_home_win" => 34.0, # Corresponds to "Any Other Home Win"
    "other_away_win" => 14.0, # Corresponds to "Any Other Away Win"
    "other_draw" => 360.0      # Corresponds to "Any Other Draw"
)

# 2. Populate the MatchFTOdds struct
ft_odds = BayesianFootball.Odds.MatchFTOdds(
    3.6,         # Home (West Ham) [cite: 1]
    3.65,        # Draw [cite: 1]
    2.2,         # Away (Tottenham) [cite: 1]
    ft_correct_scores,
    16.5,     # Under 0.5 Goals (Not available in screenshot)
    1.06,     # Over 0.5 Goals (Not available in screenshot)
    4.5,         # Under 1.5 Goals [cite: 1]
    1.32,        # Over 1.5 Goals [cite: 1]
    2.16,        # Under 2.5 Goals [cite: 1]
    1.87,        # Over 2.5 Goals [cite: 1]
    1.46,        # Under 3.5 Goals [cite: 1]
    3.15,        # Over 3.5 Goals [cite: 1]
    1.73,        # BTTS Yes [cite: 1]
    2.38         # BTTS No [cite: 1]
)

market_odds_whu_tot = BayesianFootball.Odds.MatchLineOdds(
    ht_odds,
    ft_odds
)


team_name_home = "west-ham-united"
team_name_away = "tottenham-hotspur"

# Find the integer ID for the team
team_id_home = loaded_model.result.mapping.team[team_name_home]
team_id_away = loaded_model.result.mapping.team[team_name_away]

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=1,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)

# --- Deconstructed Prediction Pipeline ---

# 1. Get the correct set of chains (we only have one)
# The round number is 1 because it's the first round of the target season
chains_for_round = loaded_model.result.chains_sequence[1]

# 2. Create the features for this single match
features = BayesianFootball.create_master_features(
    match_to_predict,
    loaded_model.result.mapping
)

# 3. Call the prediction function
match_predictions = BayesianFootball.predict_match_lines(
    loaded_model.config.model_def,
    chains_for_round,
    features,
    loaded_model.result.mapping
)

kelly_config = BayesianFootball.Kelly.Config(0.02, 0.05)
kelly = BayesianFootball.apply_kelly_to_match(
          match_predictions,
          market_odds_whu_tot,
          kelly_config 
        )
          

kelly.ft.home

density(kelly.ft.home, label="home kelly", title="kelly Distributions")
density!(kelly.ft.away, label="away kelly")

density(kelly.ht.home, label="home ht kelly", title="kelly Distributions")
density(kelly.ft.over_05, label="home ht kelly", title="kelly Distributions")

density(kelly.ft.under_15, label="under 15", title="kelly unde overDistributions")
density(kelly.ft.over_15, label="over 15", title="kelly unde overDistributions")


###
mean( 1 ./ match_predictions.ft.home )
mean( 1 ./ match_predictions.ft.away )
mean( 1 ./ match_predictions.ft.draw )

mean( 1 ./ match_predictions.ht.home )
mean( 1 ./ match_predictions.ht.away )
mean( 1 ./ match_predictions.ht.draw )

mean( 1 ./  match_predictions.ft.under_05)
mean( 1 ./ (1 .- match_predictions.ft.under_05))

mean( 1 ./  match_predictions.ft.under_15)
mean( 1 ./ (1 .- match_predictions.ft.under_15))


mean( 1 ./  match_predictions.ht.under_05)
mean( 1 ./ (1 .- match_predictions.ht.under_05))

mean( 1 ./  match_predictions.ht.under_15)
mean( 1 ./ (1 .- match_predictions.ht.under_15))

k_cs = Dict( k => mean(v) for (k,v) in kelly.ft.correct_score)
sort(collect(k_cs), by = x -> x[2], rev=true)



p_cs = Dict( k => mean(1 ./ v) for (k,v) in match_predictions.ft.correct_score)
sort(collect(p_cs), by = x -> x[2], rev=true)


p_cs = Dict( k => mean(v) for (k,v) in match_predictions.ht.correct_score)
sort(collect(p_cs), by = x -> x[2], rev=true)

mean( match_predictions.ht.λ_h)
mean( match_predictions.ht.λ_a)

mean( match_predictions.ft.λ_h)
mean( match_predictions.ft.λ_a)


density(match_predictions.ht.λ_h, label="home", title="kelly Distributions")
density!(match_predictions.ht.λ_a, label="away", title="kelly Distributions")

density(match_predictions.ft.λ_h, label="home", title="kelly Distributions")
density!(match_predictions.ft.λ_a, label="away", title="kelly Distributions")


##############################
# model 1 
##############################

p = "/home/james/bet_project/models_julia/experiments/pipeline_verification_test"
BayesianFootball.list_runs(p)

p1 = "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_verification_20250913-173119"


loaded_model_1 = load_model(p1)

# --- 6. Inspect Model Parameters ---
# Let's check a well-known Scottish team, e.g., "Celtic"

team_name = "west-ham-united"
team_name_away = "tottenham-hotspur"


# Find the integer ID for the team
team_id = loaded_model_1.result.mapping.team[team_name]
println("'$team_name' has team ID: $team_id")

# Extract the single set of chains from our model
chains = loaded_model_1.result.chains_sequence[1]

# Extract the posterior samples for attack and defense
# The 'extract_posterior_samples' function handles the identifiability constraint
posterior_samples = BayesianFootball.extract_posterior_samples(loaded_model_1.config.model_def, chains.ft)

# Get the samples for our specific team
attack_samples = posterior_samples.α[:, team_id]
defense_samples = posterior_samples.β[:, team_id]
home_advantage_samples = posterior_samples.γ_leagues[:, 1]



# Let's check the mean values
println("Posterior Mean for '$team_name' Attack (α): ", round(mean(attack_samples), digits=3))
println("Posterior Mean for '$team_name' Defense (β): ", round(mean(defense_samples), digits=3))
println("Posterior Mean for Home Advantage (γ): ", round(mean(home_advantage_samples), digits=3))

# Plot the distributions for a visual check
density(attack_samples, label="$team_name Attack (α)", title="Parameter Distributions")
density!(defense_samples, label="$team_name Defense (β)")


mean(posterior_samples.γ_leagues[:, 1])
mean(posterior_samples.γ_leagues[:, 2])
mean(posterior_samples.γ_leagues[:, 3])
mean(posterior_samples.γ_leagues[:, 4])
mean(posterior_samples.γ_leagues[:, 5])
mean(posterior_samples.γ_leagues[:, 6])

# --- Deconstructed Prediction Pipeline --- model 2
team_name_home = "west-ham-united"
team_name_away = "tottenham-hotspur"

# Find the integer ID for the team
team_id_home = loaded_model_1.result.mapping.team[team_name_home]
team_id_away = loaded_model_1.result.mapping.team[team_name_away]

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=1,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)

# --- Deconstructed Prediction Pipeline ---

# 1. Get the correct set of chains (we only have one)
# The round number is 1 because it's the first round of the target season
chains_for_round = loaded_model_1.result.chains_sequence[1]

# 2. Create the features for this single match
features = BayesianFootball.create_master_features(
    match_to_predict,
    loaded_model_1.result.mapping
)

# 3. Call the prediction function
match_predictions_1 = BayesianFootball.predict_match_lines(
    loaded_model_1.config.model_def,
    chains_for_round,
    features,
    loaded_model_1.result.mapping
)

mean( 1 ./ match_predictions.ft.home )
mean( 1 ./ match_predictions.ft.away )
mean( 1 ./ match_predictions.ft.draw )

mean( 1 ./ match_predictions_1.ft.home )
mean( 1 ./ match_predictions_1.ft.away )
mean( 1 ./ match_predictions_1.ft.draw )

market_odds_whu_tot.ft.home 
market_odds_whu_tot.ft.away
market_odds_whu_tot.ft.draw 



### match 2 

team_name_home = "brentford"
team_name_away = "chelsea"

# Find the integer ID for the team
team_id_home = loaded_model_1.result.mapping.team[team_name_home]
team_id_away = loaded_model_1.result.mapping.team[team_name_away]

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=1,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)
chains_for_round = loaded_model_1.result.chains_sequence[1]

# 2. Create the features for this single match
features = BayesianFootball.create_master_features(
    match_to_predict,
    loaded_model_1.result.mapping
)

# 3. Call the prediction function
m2 = BayesianFootball.predict_match_lines(
    loaded_model_1.config.model_def,
    chains_for_round,
    features,
    loaded_model_1.result.mapping
)





mean( 1 ./ m2.ft.home )
mean( 1 ./ m2.ft.away )
mean( 1 ./ m2.ft.draw )

mean( 1 ./ m2.ht.home )
mean( 1 ./ m2.ht.away ) 
mean( 1 ./ m2.ht.draw )


mean( 1 ./  m2.ht.under_05)
mean( 1 ./ (1 .- m2.ht.under_05))
mean( 1 ./  m2.ht.under_15)
mean( 1 ./ (1 .- m2.ht.under_15))
mean( 1 ./  m2.ht.under_25)
mean( 1 ./ (1 .- m2.ht.under_25))

mean( 1 ./  m2.ft.under_05)
mean( 1 ./ (1 .- m2.ft.under_05))
mean( 1 ./  m2.ft.under_15)
mean( 1 ./ (1 .- m2.ft.under_15))
mean( 1 ./  m2.ft.under_25)
mean( 1 ./ (1 .- m2.ft.under_25))



p_cs = Dict( k => mean(1 ./ v) for (k,v) in m2.ht.correct_score)
sort(collect(p_cs), by = x -> x[2], rev=true)

p_cs = Dict( k => mean( v) for (k,v) in m2.ft.correct_score)
sort(collect(p_cs), by = x -> x[2], rev=true)



density(m2.ht.λ_h , label="home ht", title="xG")
density!(m2.ht.λ_a , label="away ht", title="xG")

density(m2.ht.λ_h .+ m2.ht.λ_a , label="home ht", title="xG")

density(m2.ft.λ_h , label="home ft", title="xG")
density!(m2.ft.λ_a , label="away ft", title="xG")

density(m2.ft.λ_h .+ m2.ft.λ_a , label="home ht", title="xG")


##############################
####### match day two
##############################


model_path_maher = "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_basic_verification_20250913-153903"
model_path_maher_ha = "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_verification_20250913-173119"


model_maher = load_model(model_path_maher)
model_maher_ha = load_model(model_path_maher_ha)


# match 1 

team_name_home = "southampton"
team_name_away = "portsmouth"
leauge_id=2


team_id_home1 = model_maher.result.mapping.team[team_name_home]
team_id_away1 = model_maher.result.mapping.team[team_name_away]


team_id_home2 = model_maher_ha.result.mapping.team[team_name_home]
team_id_away2 = model_maher_ha.result.mapping.team[team_name_away]



match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=leauge_id,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)



chains_for_round1 = model_maher.result.chains_sequence[1]
chains_for_round2 = model_maher_ha.result.chains_sequence[1]

# 2. Create the features for this single match
features1 = BayesianFootball.create_master_features(
    match_to_predict,
    model_maher.result.mapping
)

features2 = BayesianFootball.create_master_features(
    match_to_predict,
    model_maher_ha.result.mapping
)

# 3. Call the prediction function
m1_1 = BayesianFootball.predict_match_lines(
    model_maher.config.model_def,
    chains_for_round1,
    features1,
    model_maher.result.mapping
)

m1_2 = BayesianFootball.predict_match_lines(
    model_maher_ha.config.model_def,
    chains_for_round2,
    features2,
    model_maher_ha.result.mapping
)

using StatsPlots, Distributions, UnicodePlots
unicodeplots() 


mean( 1 ./ m1_1.ft.home )
mean( 1 ./ m1_1.ft.away )
mean( 1 ./ m1_1.ft.draw )

mean( 1 ./ m1_2.ft.home )
mean( 1 ./ m1_2.ft.away )
mean( 1 ./ m1_2.ft.draw )

mean( 1 ./ m1_2.ht.under_05 )
mean( 1 ./ ( 1 .- m1_2.ht.under_05 ))
market_odds_sou_por.ht.under_05
market_odds_sou_por.ht.over_05



mean( 1 ./ m1_2.ft.under_15 )
mean( 1 ./ ( 1 .- m1_2.ft.under_15 ))

market_odds_sou_por.ft.under_15
market_odds_sou_por.ft.over_15



mean( 1 ./ m1_2.ft.under_25 )
mean( 1 ./ ( 1 .- m1_2.ft.under_25 ))

market_odds_sou_por.ft.under_25
market_odds_sou_por.ft.over_25

mean( 1 ./ m1_2.ft.under_35 )
mean( 1 ./ ( 1 .- m1_2.ft.under_35 ))

market_odds_sou_por.ft.under_35
market_odds_sou_por.ft.over_35

mean( 1 ./ m1_2.ft.under_25 )
mean( 1 ./ ( 1 .- m1_2.ft.under_25 ))

mean( 1 ./ m1_1.ht.home )
mean( 1 ./ m1_1.ht.away )
mean( 1 ./ m1_1.ht.draw )

mean( 1 ./ m1_2.ht.home )
mean( 1 ./ m1_2.ht.away )
mean( 1 ./ m1_2.ht.draw )


mean( 1 ./ match_predictions_1.ft.home )
mean( 1 ./ match_predictions_1.ft.away )
mean( 1 ./ match_predictions_1.ft.draw )



density(m1_1.ft.home, label="maher model", title="FT home p Distributions")
density!(m1_2.ft.home, label="maherHA model", title="FT home p Distributions")

density(1 ./ m1_1.ft.home, label="maher model", title="FT home p Distributions")
density!(1 ./ m1_2.ft.home, label="maherHA model", title="FT home p Distributions")


density(m1_1.ft.λ_h, label="maher model", title="FT home p Distributions", grid=true)
density!(m1_2.ft.λ_h, label="maherHA model", title="FT home p Distributions", grid=true)

density(m1_1.ft.λ_h .+ m1_2.ft.λ_a, label="maher model", title="FT home p Distributions", grid=true)



#
# 1. Half-Time (HT) Correct Score Odds
# Extracted from the "Half Time Score" market
ht_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}(
    (0, 0) => 3.35,
    (1, 1) => 8.6,
    (2, 2) => 30.0,
    (1, 0) => 4.2,
    (2, 0) => 11.5,
    (2, 1) => 16.0,
    (0, 1) => 7.4,
    (0, 2) => 32.0,
    (1, 2) => 22.0,
    "any_unquoted" => 19.5
)

# 2. Populate the MatchHTOdds struct
ht_odds = BayesianFootball.Odds.MatchHTOdds(
    2.46,   # Home (Southampton)
    2.46,   # Draw
    4.9,    # Away (Portsmouth)
    ht_correct_scores,
    3.45,   # Under 0.5 Goals
    1.39,   # Over 0.5 Goals
    1.54,   # Under 1.5 Goals
    2.74,   # Over 1.5 Goals
    1.14,   # Under 2.5 Goals
    7.4     # Over 2.5 Goals
)


# 1. Full-Time (FT) Correct Score Odds
# Extracted from the "Correct Score" market
ft_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}(
    (0, 0) => 16.5,
    (0, 1) => 16.5,
    (0, 2) => 30.0,
    (0, 3) => 80.0,
    (1, 0) => 9.6,
    (1, 1) => 8.4,
    (1, 2) => 16.5,
    (1, 3) => 48.0,
    (2, 0) => 11.0,
    (2, 1) => 9.6,
    (2, 2) => 17.0,
    (2, 3) => 50.0,
    (3, 0) => 18.0,
    (3, 1) => 16.5,
    (3, 2) => 29.0,
    (3, 3) => 80.0,
    "other_home_win" => 9.8,
    "other_away_win" => 50.0,
    "other_draw" => 320.0
)

# 2. Populate the MatchFTOdds struct
ft_odds = BayesianFootball.Odds.MatchFTOdds(
    1.83,   # Home (Southampton)
    3.9,    # Draw
    5.4,    # Away (Portsmouth)
    ft_correct_scores,
    16.5,   # Under 0.5 Goals
    1.06,   # Over 0.5 Goals
    4.5,    # Under 1.5 Goals
    1.27,   # Over 1.5 Goals
    2.18,   # Under 2.5 Goals
    1.82,   # Over 2.5 Goals
    1.44,   # Under 3.5 Goals
    3.15,   # Over 3.5 Goals
    1.79,   # BTTS Yes
    2.2     # BTTS No
)

# 3. Combine into the final MatchLineOdds struct
market_odds_sou_por = BayesianFootball.Odds.MatchLineOdds(
    ht_odds,
    ft_odds
   )



using StatsBase

# Function to calculate expected goals from a correct score dictionary
function calculate_expected_goals(cs_odds::Dict{Union{Tuple{Int,Int}, String}, Float64})
    
    # Filter out non-tuple keys and convert odds to probabilities
    implied_probs = Dict{Tuple{Int,Int}, Float64}()
    for (score, odds) in cs_odds
        if score isa Tuple{Int,Int}
            implied_probs[score] = 1 / odds
        end
    end

    # 1. Calculate the overround (sum of all implied probabilities)
    total_implied_prob = sum(values(implied_probs))
    println("Market overround is ≈ ", round((total_implied_prob - 1) * 100, digits=2), "%")

    # 2. Normalize probabilities
    normalized_probs = Dict(score => prob / total_implied_prob for (score, prob) in implied_probs)
    
    # 3. Calculate expected goals for home and away
    expected_home_goals = 0.0
    expected_away_goals = 0.0
    
    for (score, prob) in normalized_probs
        home_score, away_score = score
        expected_home_goals += home_score * prob
        expected_away_goals += away_score * prob
    end
    
    # Note: This is an approximation as it ignores the "Any Other" scores.
    # For a more precise calculation, you would need to model or estimate
    # the average goals for these catch-all categories.
    
    return (home=expected_home_goals, away=expected_away_goals)
end

# --- FULL TIME ---
println("Calculating Full-Time Expected Goals:")
ft_expected_goals = calculate_expected_goals(market_odds_sou_por.ft.correct_score)
λ_h_market_ft = ft_expected_goals.home
λ_a_market_ft = ft_expected_goals.away
λ_total_market_ft = λ_h_market_ft + λ_a_market_ft

mean( m1_2.ft.λ_h .+ m1_2.ft.λ_a )

println("Market Implied FT Expected Goals (Southampton): ", round(λ_h_market_ft, digits=3))
println("Market Implied FT Expected Goals (Portsmouth): ", round(λ_a_market_ft, digits=3))
println("Market Implied FT Total Expected Goals: ", round(λ_total_market_ft, digits=3))

println("\n" * "-"^20 * "\n")

# --- HALF TIME ---
println("Calculating Half-Time Expected Goals:")
ht_expected_goals = calculate_expected_goals(market_odds_sou_por.ht.correct_score)
λ_h_market_ht = ht_expected_goals.home
λ_a_market_ht = ht_expected_goals.away
λ_total_market_ht = λ_h_market_ht + λ_a_market_ht

mean( m1_2.ht.λ_h .+ m1_2.ht.λ_a )

println("Market Implied HT Expected Goals (Southampton): ", round(λ_h_market_ht, digits=3))
println("Market Implied HT Expected Goals (Portsmouth): ", round(λ_a_market_ht, digits=3))
println("Market Implied HT Total Expected Goals: ", round(λ_total_market_ht, digits=3))

using Plots

# --- Full-Time Home Goals Comparison ---
density(m1_1.ft.λ_h, label="Maher Model", grid=true)
density!(m1_2.ft.λ_h, label="MaherHA Model")
vline!([λ_h_market_ft], label="Market Implied λ_h", color=:red, linestyle=:dash, linewidth=2)
title!("FT Home Expected Goals (λ_h) Distributions")


# --- Full-Time Total Goals Comparison ---
# Assuming m1_1 and m1_2 have chains for away goals (λ_a) as well
density(m1_1.ft.λ_h .+ m1_1.ft.λ_a, label="Maher Model Total Goals", grid=true)
density!(m1_2.ft.λ_h .+ m1_2.ft.λ_a, label="MaherHA Model Total Goals")
vline!([λ_total_market_ft], label="Market Implied Total λ", color=:red, linestyle=:dash, linewidth=2)
title!("FT Total Expected Goals (λ_h + λ_a) Distributions")



λ_ht = m1_2.ht.λ_h .+ m1_2.ht.λ_a

λ_rem = λ_ht .* ( 1 - 22.5 / 45 )
mean(1 ./ λ_rem)

using Distributions

# -- Your Setup --
# Let's use the market's initial half-time λ from our previous discussion
λ_ht = 1.24 
# Your calculation for remaining λ after 22.5 minutes
λ_rem = mean(λ_ht .* (1 - 22.5 / 45)) # This will be 0.62

# -- Calculation --
# 2. Create a Poisson distribution object with your remaining λ
poisson_dist = Poisson(λ_rem)

# 3. Calculate the probability of Under 0.5 goals (i.e., P(goals = 0))
# We use the pdf (probability density function) for a specific outcome.
prob_under_05 = pdf(poisson_dist, 0)

# 4. The probability of Over 0.5 goals is the complement (1 - P(0))
prob_over_05 = 1 - prob_under_05

# -- Output --
println("Remaining λ: ", round(λ_rem, digits=3))
println("Probability Under 0.5 Goals (P(0)): ", round(prob_under_05, digits=4))
println("Probability Over 0.5 Goals (P(>0)): ", round(prob_over_05, digits=4))

# You can then convert these probabilities to fair decimal odds
println("\nFair Odds for Under 0.5: ", round(1 / prob_under_05, digits=2))
println("Fair Odds for Over 0.5: ", round(1 / prob_over_05, digits=2))


###


team_name_home = "burnley"
team_name_away = "liverpool"
leauge_id=1


team_id_home1 = model_maher.result.mapping.team[team_name_home]
team_id_away1 = model_maher.result.mapping.team[team_name_away]


team_id_home2 = model_maher_ha.result.mapping.team[team_name_home]
team_id_away2 = model_maher_ha.result.mapping.team[team_name_away]



match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=leauge_id,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)



chains_for_round1 = model_maher.result.chains_sequence[1]
chains_for_round2 = model_maher_ha.result.chains_sequence[1]

# 2. Create the features for this single match
features1 = BayesianFootball.create_master_features(
    match_to_predict,
    model_maher.result.mapping
)

features2 = BayesianFootball.create_master_features(
    match_to_predict,
    model_maher_ha.result.mapping
)

# 3. Call the prediction function
m1_1 = BayesianFootball.predict_match_lines(
    model_maher.config.model_def,
    chains_for_round1,
    features1,
    model_maher.result.mapping
)

m1_2 = BayesianFootball.predict_match_lines(
    model_maher_ha.config.model_def,
    chains_for_round2,
    features2,
    model_maher_ha.result.mapping
)


mean( 1 ./ m1_2.ft.home )
mean( 1 ./ m1_2.ft.away )
mean( 1 ./ m1_2.ft.draw )

mean( 1 ./ m1_2.ht.home )
mean( 1 ./ m1_2.ht.away )
mean( 1 ./ m1_2.ht.draw )

mean( 1 ./ m1_2.ft.under_05 )
mean( 1 ./ ( 1 .- m1_2.ft.under_05 ))

mean( 1 ./ m1_2.ft.under_15 )
mean( 1 ./ ( 1 .- m1_2.ft.under_15 ))

mean( 1 ./ m1_2.ft.under_15 )
mean( 1 ./ ( 1 .- m1_2.ft.under_15 ))

market_odds_sou_por.ft.under_15
market_odds_sou_por.ft.over_15



mean( 1 ./ m1_2.ft.under_25 )
mean( 1 ./ ( 1 .- m1_2.ft.under_25 ))



##### 


team_name_home = "manchester-city"
team_name_away = "manchester-united"
leauge_id=1


team_id_home1 = model_maher.result.mapping.team[team_name_home]
team_id_away1 = model_maher.result.mapping.team[team_name_away]


team_id_home2 = model_maher_ha.result.mapping.team[team_name_home]
team_id_away2 = model_maher_ha.result.mapping.team[team_name_away]



match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=leauge_id,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)



chains_for_round1 = model_maher.result.chains_sequence[1]
chains_for_round2 = model_maher_ha.result.chains_sequence[1]

# 2. Create the features for this single match
features1 = BayesianFootball.create_master_features(
    match_to_predict,
    model_maher.result.mapping
)

features2 = BayesianFootball.create_master_features(
    match_to_predict,
    model_maher_ha.result.mapping
)

# 3. Call the prediction function
m1_1 = BayesianFootball.predict_match_lines(
    model_maher.config.model_def,
    chains_for_round1,
    features1,
    model_maher.result.mapping
)

m1_2 = BayesianFootball.predict_match_lines(
    model_maher_ha.config.model_def,
    chains_for_round2,
    features2,
    model_maher_ha.result.mapping
)

mean( 1 ./ m1_2.ft.home )
mean( 1 ./ m1_2.ft.away )
mean( 1 ./ m1_2.ft.draw )

mean( 1 ./ m1_2.ht.home )
mean( 1 ./ m1_2.ht.away )
mean( 1 ./ m1_2.ht.draw )

mean( 1 ./ m1_2.ft.under_05 )
mean( 1 ./ ( 1 .- m1_2.ft.under_05 ))

mean( 1 ./ m1_2.ft.under_15 )
mean( 1 ./ ( 1 .- m1_2.ft.under_15 ))

mean( 1 ./ m1_2.ft.under_25 )
mean( 1 ./ ( 1 .- m1_2.ft.under_25 ))



mean( 1 ./ m1_2.ht.under_05 )
mean( 1 ./ ( 1 .- m1_2.ht.under_05 ))

mean( 1 ./ m1_2.ht.under_15 )
mean( 1 ./ ( 1 .- m1_2.ht.under_15 ))

mean( 1 ./ m1_2.ht.under_25 )
mean( 1 ./ ( 1 .- m1_2.ht.under_25 ))




##############################
####### match day three EFL 
##############################

using StatsPlots, Distributions, Statistics, Plots
model_path_maher_ha = "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_verification_20250913-173119"
model_short_path = "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_short_20250916-151954"

model = load_model(model_path_maher_ha)
#### Match 1 

team_name_home = "sheffield-wednesday"
team_name_away = "grimsby-town"
leauge_id=2


team_id_home = model.result.mapping.team[team_name_home]
team_id_away = model.result.mapping.team[team_name_away]

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=leauge_id,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)

chains_for_round = model.result.chains_sequence[1]
# 2. Create the features for this single match
features = BayesianFootball.create_master_features(
    match_to_predict,
    model.result.mapping
)

# 3. Call the prediction function
m1 = BayesianFootball.predict_match_lines(
    model.config.model_def,
    chains_for_round,
    features,
    model.result.mapping
)

mean( 1 ./ m1.ft.home )
mean( 1 ./ m1.ft.away )
mean( 1 ./ m1.ft.draw )


mean( 1 ./ m1.ft.under_05 )
mean( 1 ./ ( 1 .- m1.ft.under_05 ))

mean( 1 ./ m1.ft.under_15 )
mean( 1 ./ ( 1 .- m1.ft.under_15 ))

mean( 1 ./ m1.ft.under_25 )
mean( 1 ./ ( 1 .- m1.ft.under_25 ))

mean( 1 ./ m1.ft.btts)
mean( 1 ./ (1 .- m1.ft.btts))



mean( 1 ./ m1.ht.home )
mean( 1 ./ m1.ht.away )
mean( 1 ./ m1.ht.draw )

mean( 1 ./ m1.ht.under_05 )
mean( 1 ./ ( 1 .- m1.ht.under_05 ))

mean( 1 ./ m1.ht.under_15 )
mean( 1 ./ ( 1 .- m1.ht.under_15 ))

mean( 1 ./ m1.ht.under_25 )
mean( 1 ./ ( 1 .- m1.ht.under_25 ))


density(1 ./ m1.ft.home, label="maher model", title="FT home p Distributions")


mean( m1.ft.λ_h )
mean( m1.ft.λ_a )

k_cs = Dict( k => mean(1 ./ v) for (k,v) in m1.ft.correct_score)
sort(collect(k_cs), by = x -> x[2], rev=true)


# short learn model 

model_short_path = "/home/james/bet_project/models_julia/experiments/pipeline_verification_test/maher_league_ha_short_20250916-151954"

model1 = load_model(model_short_path)


chains_for_round1= model1.result.chains_sequence[1]
# 2. Create the features for this single match
features1 = BayesianFootball.create_master_features(
    match_to_predict,
    model1.result.mapping
)

m2 = BayesianFootball.predict_match_lines(
    model1.config.model_def,
    chains_for_round1,
    features1,
    model1.result.mapping
)

mean( 1 ./ m2.ft.home )
mean( 1 ./ m2.ft.away )
mean( 1 ./ m2.ft.draw )

density(1 ./ m1.ft.home, label="maher model", title="FT home p Distributions")
density!(1 ./ m2.ft.home, label="maher model short", title="FT home p Distributions")
