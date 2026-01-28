# exp/market_scot/analysis.jl


using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

d = subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))

# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/grw_basics"
println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/market_runs"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

if isempty(loaded_results)
    error("No results loaded! Did you run runner.jl?")
end

unique(d.home_team)
m = loaded_results[1]
# ==============================================================================
# 1. REGENERATE FEATURES
# ==============================================================================
# The saved experiment 'm' contains the recipe (config) but not the FeatureSets.
# We must recreate them so the model knows that "stranraer" == ID 5 (for example).

println("Regenerating feature sets from config...")

# 1. Re-create the data splits using the saved splitter config
splits = BayesianFootball.Data.create_data_splits(ds, m.config.splitter)

# 2. Re-create the feature sets using the saved model config
# feature_collection is a Vector of tuples: (FeatureSet, SplitMetaData)
feature_collection = BayesianFootball.Features.create_features(
    splits, 
    m.config.model, 
    m.config.splitter
)

# ==============================================================================
# 2. PREPARE FOR PREDICTION
# ==============================================================================

# We usually want the LAST split (the most recent training data)
last_split_idx = length(m.training_results)

# Get the Chain from the results
chain = m.training_results[last_split_idx][1]

# Get the corresponding FeatureSet from our regenerated list
# Note: [1] gets the FeatureSet, [2] would be the Metadata
feature_set = feature_collection[last_split_idx][1]

# Define the matches for tonight
# IMPORTANT: Add 'match_week'. Use a high number (e.g., 999) to project 
# the Random Walk to the latest available time step.
match_to_predict = DataFrame(
    match_id = [1, 2],
    match_week = [999, 999], 
    home_team = ["east-kilbride", "stranraer"], 
    away_team = ["the-spartans-fc", "clyde-fc"]
)

# ==============================================================================
# 3. RUN EXTRACTION
# ==============================================================================

println("Extracting parameters...")

# Now this will work because we are passing the correct types:
# (Model, DataFrame, FeatureSet, Chain)
raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m.config.model,
    match_to_predict,
    feature_set,  # <--- This was the missing piece
    chain
)



using DataFrames
using BayesianFootball
using BayesianFootball.Experiments: LatentStates

# ==============================================================================
# 1. CONVERT RAW PREDS TO LATENT STATES
# ==============================================================================
# `raw_preds` is the Dict we created in the previous step (extract_parameters).
# We need to convert it into the LatentStates struct expected by the inference engine.

function raw_preds_to_df(raw_preds::Dict)
    match_ids = collect(keys(raw_preds))
    if isempty(match_ids); return DataFrame(); end
    
    # Get the keys from the first entry (e.g., :λ_h, :λ_a, :r)
    first_val = raw_preds[match_ids[1]]
    param_keys = keys(first_val)

    # Build columns
    cols = Dict{Symbol, Vector{Any}}(:match_id => match_ids)
    for p in param_keys
        cols[p] = [raw_preds[id][p] for id in match_ids]
    end
    return DataFrame(cols)
end

# Create the wrapper
df_latents = raw_preds_to_df(raw_preds)
latents_obj = LatentStates(df_latents, m.config.model)

# ==============================================================================
# 2. RUN MODEL INFERENCE (Get Probabilities)
# ==============================================================================
# This step simulates the matches and generates probabilities (PPD).
# It relies only on your model, not external odds.

pd = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG

# 'ppd' stands for Posterior Predictive Distribution
ppd = BayesianFootball.Predictions.model_inference(latents_obj; market_config=pd)

# Ensure Statistics is loaded for 'mean'
using Statistics 

println("\n--- Model Predictions (Fair Prices) ---")

for row in eachrow(ppd.df)
    # row.distribution is a Vector{Float64} of samples. 
    # Take the mean to get the expected probability.
    prob = mean(row.distribution) 
    
    fair_odds = 1.0 / prob
    
    println("Match $(row.match_id) | $(row.selection) | Prob: $(round(prob, digits=3)) | Fair Odds: $(round(fair_odds, digits=2))")
end

using DataFrames
using BayesianFootball
using .MatchDayUtils # Assuming you loaded the module from the previous step

# ==============================================================================
# 1. SETUP YOUR MODEL MATCHES (Correct Model Names)
# ==============================================================================
# These names MUST match what is in your model's vocabulary.
match_to_predict = DataFrame(
    match_id = [1, 2],
    match_week = [999, 999], 
    home_team = ["east-kilbride", "stranraer"], 
    away_team = ["the-spartans-fc", "clyde-fc"]
)

# ==============================================================================
# 2. FETCH LIVE ODDS (Scraper Names)
# ==============================================================================
# This gets everything available today
scraper_matches, scraper_odds = fetch_todays_data(["scotland"])

println("\n--- Available Matches from Scraper ---")
select(scraper_matches, :match_id, :event_name) |> display

# ==============================================================================
# 3. MANUAL LINKING (The "Just Make It Work" Part)
# ==============================================================================
# Look at the list above. Map the Scraper's ID to Your Model's Match ID.
# Based on your logs: 
#   "East Kilbride v Spartans" is ID 1 in Scraper -> ID 1 in your DF
#   "Stranraer v Clyde" is ID 6 in Scraper        -> ID 2 in your DF

id_map = Dict(
    1 => 1,  # Scraper ID 1 maps to Model Match 1
    6 => 2   # Scraper ID 6 maps to Model Match 2
)

# Filter the odds to just these matches and update the IDs to match your model
final_odds_df = filter(row -> haskey(id_map, row.match_id), scraper_odds)
final_odds_df.match_id = [id_map[mid] for mid in final_odds_df.match_id]

println("\n--- Odds Linked to Model ---")
display(final_odds_df)

# ==============================================================================
# 4. RUN PREDICTION PIPELINE
# ==============================================================================

# A. Extract Parameters (Using YOUR dataframe with correct names)
println("\nRunning Model Extraction...")
# (Assuming 'm', 'feature_set', and 'chain' are loaded as before)
raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m.config.model, 
    match_to_predict, 
    feature_set, 
    chain
)

# B. Convert to LatentStates & Inference
# Helper to convert Dict to DataFrame
function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds)); isempty(ids) && return DataFrame()
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    for k in keys(raw_preds[ids[1]])
        cols[k] = [raw_preds[i][k] for i in ids]
    end
    return DataFrame(cols)
end

latents_obj = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents_obj)


using BayesianFootball.Signals
flat_strat = FlatStake(0.05)

# 2. Conservative Kelly: Quarter Kelly (0.25)
kelly_strat = KellyCriterion(0.25)

# 3. Bayesian/Shrinkage Kelly: Uses the Baker-McHale analytical approximation
shrink_strat = AnalyticalShrinkageKelly()

baker = BayesianKelly()

# Combine them into a vector
my_signals = [flat_strat, kelly_strat, shrink_strat, baker]
final_odds_df.is_winner .= true

# C. Process Signals (Using the Linked Odds)
println("\nCalculating Signals...")
results = BayesianFootball.Signals.process_signals(
    ppd, 
    final_odds_df, # <--- We pass the mapped odds here
    my_signals; 
    odds_column=:odds
)

# Display bets where signal > 0
bets = filter(row -> row.signal_strength > 0, results)
display(bets[:, [:match_id, :selection, :odds, :strategy, :expected_value, :signal_strength]])

# 1. Fix the Mismatches in final_odds_df
# ==============================================================================

# A. Standardize Market Names (Must match src/data/markets/implementations/*.jl)
# "1x2" -> "1X2", "btts" -> "BTTS", "ou_25" -> "OverUnder"
name_map = Dict(
    "1x2" => "1X2",
    "btts" => "BTTS",
    "ou_25" => "OverUnder"
)

# Apply mapping (default to original if not found)
final_odds_df.market_name = [get(name_map, n, n) for n in final_odds_df.market_name]


# B. Standardize Selection Symbols (Must match outcomes() in implementations)
# :yes -> :btts_yes, :over -> :over_25, etc.
sel_map = Dict(
    # BTTS
    :yes => :btts_yes,
    :no  => :btts_no,
    
    # Over/Under (Assuming 2.5 line for these specific scraper keys)
    :over  => :over_25,
    :under => :under_25
    
    # 1X2 usually matches (:home, :draw, :away), so no change needed there
)

# Apply mapping
final_odds_df.selection = [get(sel_map, s, s) for s in final_odds_df.selection]


# 2. Verify Compatibility Before Running
# ==============================================================================
# Let's check if the keys now exist in both dataframes
ppd_keys = Set(zip(ppd.df.match_id, ppd.df.market_name, ppd.df.selection))
odds_keys = Set(zip(final_odds_df.match_id, final_odds_df.market_name, final_odds_df.selection))

common_keys = intersect(ppd_keys, odds_keys)

if isempty(common_keys)
    println("❌ Still no matches found! Let's look at a sample of each to debug:")
    println("\n--- PPD Keys (Model) ---")
    println(first(ppd_keys, 3))
    println("\n--- Odds Keys (Scraper) ---")
    println(first(odds_keys, 3))
else
    println("✅ Found $(length(common_keys)) matching betting opportunities.")
    
    # 3. Run Process Signals
    # ==========================================================================
    # Ensure is_winner is present (it can be missing for live predictions)
    if !hasproperty(final_odds_df, :is_winner)
        final_odds_df.is_winner .= missing
    end

    results = BayesianFootball.Signals.process_signals(
        ppd, 
        final_odds_df, 
        my_signals; 
        odds_column=:odds
    )

    # Display Value Bets
    bets = filter(row -> row.signal_strength > 0, results)
    if isempty(bets)
        println("No bets found (Signals returned 0 strength). Check your strategy thresholds.")
    else
        display(bets[:, [:match_id, :selection, :odds, :strategy, :stake]])
    end
end

# Filter based on 'stake' > 0
bets = filter(row -> row.stake > 0, results.df)

if isempty(bets)
    println("No bets found (Calculated stake is 0 for all).")
else
    # Display the correct columns
    display(bets[:, [:match_id, :selection, :odds, :signal_name, :stake]])
end

####
# 1. Load the new Utils
include("path/to/MatchDayUtils.jl") 
using .MatchDayUtils

# 2. Fetch Data (Replaces your manual DataFrame creation)
# This will call your Python script, get the matches, and format the odds
df_matches, df_odds = fetch_todays_data(["scotland"], save_dir="./data/today")

# 3. Setup Predictions
# We need to ensure the `match_to_predict` has the right team names and ID
# match_to_predict = df_matches (with renaming if needed for your model extract_params)
match_to_predict = select(df_matches, 
    :match_id, 
    :home_team, 
    :away_team, 
    :match_id => :match_week # Placeholder for time
)
match_to_predict.match_week .= 999 # Set future time

# 4. Run Model
# ... (Regenerate features as discussed before) ...
raw_preds = BayesianFootball.Models.PreGame.extract_parameters(m.config.model, match_to_predict, feature_set, chain)

# 5. Convert to LatentStates & Inference
df_latents = raw_preds_to_df(raw_preds) # Helper from previous answer
latents_obj = LatentStates(df_latents, m.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents_obj)

# 6. Process Signals
# Now we pass the df_odds we fetched automatically!
results = BayesianFootball.Signals.process_signals(
    ppd, 
    df_odds, # <--- The fetched odds
    my_signals; 
    odds_column=:odds
)

display(results)
