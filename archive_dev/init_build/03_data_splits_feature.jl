
# dev/03_data_splits_pipeline.jl

using Revise
using BayesianFootball
using DataFrames

# ... (Phase 1: Loading Globals D, M, G remains the same) ...
println("--- ✅ Setup Complete: Packages loaded ---")

# ============================================================================
# PHASE 1: DEFINE "GLOBAL" ATOMS (DataStore and Vocabulary)
# ============================================================================
println("\n--- PHASE 1: Loading Globals (D and G) ---")

# --- Atom 1: The DataStore (D) ---
data_store = BayesianFootball.Data.load_default_datastore(); #
println("✅ (D) DataStore loaded with $(nrow(data_store.matches)) matches.") #

# --- Atom 2: The Model (M) ---
# We need a model to define the vocabulary
model = BayesianFootball.Models.PreGame.StaticPoisson() #
println("✅ (M) Using model: $(typeof(model))") #

# --- Atom 3: The Global Vocabulary (G) ---
# MORPHISM: G_map: D x M -> G
println("⏳ Creating Global Vocabulary (G)...") #
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model) #
println("✅ (G) Global Vocabulary created. $(vocabulary.mappings[:n_teams]) teams found.") #

# ============================================================================
# PHASE 2: DEFINE SPLITTERS AND TEST THE PIPELINE
# ============================================================================
println("\n--- PHASE 2: Testing Data Split Pipeline ---") #

# --- Define Splitter Configs (Config_s) ---
splitter_static = BayesianFootball.Data.StaticSplit(train_seasons =["23/24"]) #
splitter_cv = BayesianFootball.Data.ExpandingWindowCV( #
  ["22/23"], # Base
  ["23/24"], # Target
  :round, #
  :sequential #
)

splitters_to_test = [splitter_static, splitter_cv]


# --- Run the test loop ---
for splitter_config in splitters_to_test
    
    println("\n" * "="^40)
    println("Testing Splitter: $(typeof(splitter_config))")
    println("="^40)
    
    # --- 1. MORPHISM h_1: D x Config_s -> D_1:N ---
    # Now returns Vector{Tuple{SubDataFrame, String}}
    data_splits_vector = BayesianFootball.Data.create_data_splits(data_store, splitter_config)
    
    println("✅ (h_1) Created Vector of splits. Type: $(typeof(data_splits_vector))")
    println("   Number of splits: $(length(data_splits_vector))") #
    
    # --- 2. LOOP over D_1:N ---
    for (i, (data_split_view, split_metadata)) in enumerate(data_splits_vector)
        if i > 20
            println("... (skipping remaining splits for brevity) ...")
            break
        end
        
        println("\n  --- Processing Split $i ---")
        println("  Split Metadata: $split_metadata")
        # Check the type of the data split - should be SubDataFrame
        println("  Split Type (D_i): $(typeof(data_split_view))") 
        println("  Split (D_i) has $(nrow(data_split_view)) matches.") #
        
        # --- 3. MORPHISM f_i: D_i x G x M -> F_i ---
        # Pass the SubDataFrame directly
        feature_set_i = BayesianFootball.Features.create_features( #
            data_split_view,  # D_i (Now a SubDataFrame)
            vocabulary,       # G
            model             # M
        )
        
        println("  ✅ (f_i) Created FeatureSet (F_i).") #
        # Accessing data within FeatureSet might involve internal copies depending on implementation
        println("     Matches in F_i (after potential processing): $(nrow(feature_set_i.data[:matches_df]))") 
        println("     Teams in F_i:   $(feature_set_i.data[:n_teams]) (should match G)") #
        
    end
end

println("\n--- ✅ Pipeline Test Complete ---")


splitter_config = BayesianFootball.Data.ExpandingWindowCV( #
  [], # Base
  ["23/24"], # Target
  :match_week, #
  :sequential #
)

# --- 1. MORPHISM h_1: D x Config_s -> D_1:N ---
# Now returns Vector{Tuple{SubDataFrame, String}}
data_splits_vector = BayesianFootball.Data.create_data_splits(data_store, splitter_config)


# Preprocess the entire DataFrame first for this test
matches_with_mw = BayesianFootball.Data.add_match_week_column(data_store.matches)
println("Added :match_week column to a copy of the matches DataFrame.")

# Create a temporary DataStore with the preprocessed data for splitting
temp_data_store = BayesianFootball.Data.DataStore(
    matches_with_mw,
    data_store.odds,
    data_store.incidents
)

# Now create the splits using the preprocessed data
data_splits_vector_mw = BayesianFootball.Data.create_data_splits(temp_data_store, splitter_config)
println("Created $(length(data_splits_vector_mw)) splits using :match_week.")
data_splits_static= BayesianFootball.Data.create_data_splits(temp_data_store, splitter_static)

println("\n--- Inspecting First 5 Splits ---")
for i in 1:min(5, length(data_splits_vector_mw))
    data_split_view, split_metadata = data_splits_vector_mw[i]
    println("Split $i:")
    println("  Metadata: $split_metadata") # Should show increasing week numbers
    println("  Size (D_i): $(nrow(data_split_view)) matches")
    # Check the max match_week in this split
    if :match_week in names(data_split_view)
         println("  Max Match Week: $(maximum(data_split_view.match_week))")
    else
         println("  :match_week column not found in this view (check preprocessing).")
    end
end


if length(data_splits_vector_mw) >= 3
    split3_view, meta3 = data_splits_vector_mw[3]
    println("\n--- Examining Split 3 ---")
    println("Metadata: $meta3")
    println("First 5 rows of Split 3:")
    display(first(split3_view, 5))
    println("\nLast 5 rows of Split 3:")
    display(last(split3_view, 5))
    println("\nMax match_week in Split 3: $(maximum(split3_view.match_week))")
    println("Min match_date in Split 3: $(minimum(split3_view.match_date))")
    println("Max match_date in Split 3: $(maximum(split3_view.match_date))")
end


# (Assuming data_splits_vector, vocabulary, and model are already defined)

println("\n--- PHASE 3: Applying Feature Creation (using overloaded method) ---")

# Simply call the overloaded method directly with the vector of splits
feature_sets_vector = BayesianFootball.Features.create_features(
    data_splits_vector_mw,
    vocabulary,
    model,
    splitter_config
)

feature_sets_static = BayesianFootball.Features.create_features(
    data_splits_static,
    vocabulary,
    model,
    splitter_static
)




### ---- Samping explore 

sampler_config = Sampling.NUTSMethod(100, 4, 50)

turing_model = Models.PreGame.build_turing_model(model, feature_sets_static[1][1])
chains_params = Sampling.train(turing_model, sampler_config)

describe(chains_params)


 df_to_predict = last(temp_data_store.matches, 14)


predictions = Models.PreGame.predict(model, df_to_predict, vocabulary, chains_params)

p1 = predictions

id = 3
df_to_predict[id, :]

describe(p1[Symbol("predicted_home_goals[$id]")])
describe(p1[Symbol("predicted_away_goals[$id]")])

filter( row -> row.match_id==df_to_predict[id, :match_id], data_store.odds)[:, [:market_name, :choice_name, :choice_group, :decimal_odds, :winning]]





sampler_config = Sampling.NUTSMethod(100, 4, 50)

turing_model = Models.PreGame.build_turing_model(model, feature_sets_vector[26][1])
chains_params = Sampling.train(turing_model, sampler_config)

feature_sets_vector[26]



describe(chains_params)


 df_to_predict = last(temp_data_store.matches, 14)

df_to_predict = filter( row -> row.match_week==146, temp_data_store.matches)

predictions = Models.PreGame.predict(model, df_to_predict, vocabulary, chains_params)

p1 = predictions

id =  5
df_to_predict[id, :]

df_to_predict[id, [:home_score, :away_score]]

describe(p1[Symbol("predicted_home_goals[$id]")])
describe(p1[Symbol("predicted_away_goals[$id]")])

filter( row -> row.match_id==df_to_predict[id, :match_id], data_store.odds)[:, [:market_name, :choice_name, :choice_group, :decimal_odds, :winning]]


p1[Symbol("predicted_home_goals[$id]")]


### 

"""
Extracts and flattens the home and away goal prediction samples 
from the MCMC chain results (in AxisArray/DataFrame form) 
for a specific match ID.
"""
function get_match_samples(p1, id::Int)
    # Check if the keys exist before trying to access them
    home_key = Symbol("predicted_home_goals[$id]")
    away_key = Symbol("predicted_away_goals[$id]")
    
    if !(home_key in keys(p1) && away_key in keys(p1))
        error("Could not find prediction keys for id=$id in p1.")
    end

    # vec() flattens the 100x4 array into a single 400-element vector
    home_samples = vec(p1[home_key])
    away_samples = vec(p1[away_key])
    
    return home_samples, away_samples
end

"""
Calculates the decimal odds for the 1x2 (Home, Draw, Away) market 
from simulated match scores.
"""
function calculate_1x2_odds(home_samples::AbstractVector, away_samples::AbstractVector)
    total_samples = length(home_samples)
    
    if total_samples == 0
        return (home=NaN, draw=NaN, away=NaN)
    end
    
    # Count outcomes
    home_wins = sum(home_samples .> away_samples)
    draws = sum(home_samples .== away_samples)
    away_wins = sum(home_samples .< away_samples)
    
    # Calculate probabilities
    prob_1 = home_wins / total_samples
    prob_x = draws / total_samples
    prob_2 = away_wins / total_samples
    
    # Calculate odds (1 / probability)
    # Returns Inf if probability is 0
    odds_1 = 1 / prob_1
    odds_x = 1 / prob_x
    odds_2 = 1 / prob_2
    
    return (home=odds_1, draw=odds_x, away=odds_2)
end

"""
Calculates the decimal odds for the Over/Under market for a given goal line
from simulated match scores.
"""
function calculate_over_under_odds(home_samples::AbstractVector, away_samples::AbstractVector, line::Float64)
    total_samples = length(home_samples)
    
    if total_samples == 0
        return (over=NaN, under=NaN)
    end
    
    # Calculate total goals for each simulation
    total_goals = home_samples .+ away_samples
    
    # Count over/under outcomes
    over_count = sum(total_goals .> line)
    under_count = sum(total_goals .< line) # Same as sum(total_goals .<= floor(line))
    
    # Calculate probabilities
    prob_over = over_count / total_samples
    prob_under = under_count / total_samples
    
    # Calculate odds
    odds_over = 1 / prob_over
    odds_under = 1 / prob_under
    
    return (over=odds_over, under=odds_under)
end



# Your match ID
id = 5

# --- Step 1: Get the 400 samples for this match ---
# (Assuming 'p1' is your variable holding the AxisArray results)
home_samples, away_samples = get_match_samples(p1, id)


# --- Step 2: Calculate 1x2 (Full time) odds ---
ft_odds = calculate_1x2_odds(home_samples, away_samples)

println("--- Model's Full Time 1x2 Odds ---")
println("Home Win (1): $(round(ft_odds.home, digits=2))")
println("Draw (X):     $(round(ft_odds.draw, digits=2))")
println("Away Win (2): $(round(ft_odds.away, digits=2))")


# --- Step 3: Calculate Over/Under (Match goals) odds ---
println("\n--- Model's Match Goals Over/Under Odds ---")

# You can get the lines directly from your odds DataFrame to test all of them
# (Assuming 'data_store', 'df_to_predict', and 'id' are defined)
odds_df = filter(row -> row.match_id == df_to_predict[id, :match_id] && 
                         row.market_name == "Match goals", 
                  data_store.odds)
                  
goal_lines = unique(skipmissing(odds_df.choice_group))
sort!(goal_lines)

for line in goal_lines
    ou_odds = calculate_over_under_odds(home_samples, away_samples, line)
    println("Line: $line")
    println("  Over:  $(round(ou_odds.over, digits=2))")
    println("  Under: $(round(ou_odds.under, digits=2))")
end

# Or, to test a specific line like 2.5:
# line_2_5 = 2.5
# ou_2_5_odds = calculate_over_under_odds(home_samples, away_samples, line_2_5)
# println("\nLine: $line_2_5")
# println("  Over:  $(round(ou_2_5_odds.over, digits=2))")
# println("  Under: $(round(ou_2_5_odds.under, digits=2))")


using DataFrames

"""
Safely retrieves a specific decimal odd from the market odds DataFrame.
"""
function get_market_odd(market_df::DataFrame, market::String, choice::String, group::Union{Float64, Missing}=missing)
    
    # Filter by market and choice
    filtered_df = filter(row -> row.market_name == market && row.choice_name == choice, market_df)
    
    # If a line (group) is specified, filter by that as well
    if !ismissing(group)
        # We must also filter out missing choice_groups before comparing
        filtered_df = filter(row -> !ismissing(row.choice_group) && row.choice_group == group, filtered_df)
    end
    
    if isempty(filtered_df)
        return missing
    else
        # Return the first match
        return first(filtered_df.decimal_odds)
    end
end


"""
Compares model-generated odds against market odds for 1x2 and Over/Under markets.

Assumes `get_match_samples`, `calculate_1x2_odds`, 
and `calculate_over_under_odds` are defined.
"""
function compare_model_to_market(p1, id::Int, df_to_predict::DataFrame, data_store::Any)
    
    # --- 1. Get Model Predictions ---
    home_samples, away_samples = get_match_samples(p1, id)
    
    # --- 2. Get Market Odds for this match ---
    match_id = df_to_predict[id, :match_id]
    market_odds_df = filter(row -> row.match_id == match_id, data_store.odds)
    
    # --- 3. Process 1x2 (Full time) Market ---
    model_1x2 = calculate_1x2_odds(home_samples, away_samples)
    
    market_home = get_market_odd(market_odds_df, "Full time", "1")
    market_draw = get_market_odd(market_odds_df, "Full time", "X")
    market_away = get_market_odd(market_odds_df, "Full time", "2")
    
    # Create the 1x2 DataFrame
    df_1x2 = DataFrame(
        market = "1x2",
        selection = ["Home (1)", "Draw (X)", "Away (2)"],
        line = Union{Float64, Missing}[missing, missing, missing],
        model_odds = [model_1x2.home, model_1x2.draw, model_1x2.away],
        market_odds = [market_home, market_draw, market_away]
    )
    
    # --- 4. Process Over/Under (Match goals) Market ---
    
    # Find all unique O/U lines from the market data
    ou_market_df = filter(row -> row.market_name == "Match goals", market_odds_df)
    ou_lines = sort(unique(skipmissing(ou_market_df.choice_group)))
    
    # Initialize an empty DataFrame to hold O/U results
    df_ou = DataFrame(
        market = String[],
        selection = String[],
        line = Float64[],
        model_odds = Float64[],
        market_odds = Union{Float64, Missing}[]
    )
    
    # Loop over each line, calculate model odds, get market odds, and add
    for line in ou_lines
        model_ou = calculate_over_under_odds(home_samples, away_samples, line)
        
        market_over = get_market_odd(market_odds_df, "Match goals", "Over", line)
        market_under = get_market_odd(market_odds_df, "Match goals", "Under", line)
        
        push!(df_ou, ("Over/Under", "Over", line, model_ou.over, market_over))
        push!(df_ou, ("Over/Under", "Under", line, model_ou.under, market_under))
    end
    
    # --- 5. Combine and Calculate Value ---
    
    # Stack the 1x2 and O/U DataFrames
    comparison_df = vcat(df_1x2, df_ou)
    
    # Calculate value: (Market Odds / Model Odds) - 1
    # A positive value suggests the market odds are higher than the model's,
    # indicating a potential "value bet" according to the model.
    comparison_df.value = (comparison_df.market_odds ./ comparison_df.model_odds) .- 1
    
    # Round the numeric columns for cleaner display
    for col in [:model_odds, :market_odds, :value]
        if col in names(comparison_df)
            comparison_df[!, col] = round.(comparison_df[!, col], digits=3)
        end
    end
    
    return comparison_df
end



# Your match ID
id = 7

# (Assuming p1, df_to_predict, and data_store are all loaded)

# --- Generate the Comparison DataFrame ---
comparison_table = compare_model_to_market(p1, id, df_to_predict, data_store)

# --- Display the results ---
println(comparison_table)

df_to_predict[id, [:home_score, :away_score]]

df_to_predict[id, :]


#####
using DataFrames

"""
Safely retrieves the decimal odds and winning status from the market odds DataFrame.
Returns a tuple: (decimal_odds, winning_status)
"""
function get_market_info(market_df::DataFrame, market::String, choice::String, group::Union{Float64, Missing}=missing)
    
    # Filter by market and choice
    filtered_df = filter(row -> row.market_name == market && row.choice_name == choice, market_df)
    
    # If a line (group) is specified, filter by that as well
    if !ismissing(group)
        # We must also filter out missing choice_groups before comparing
        filtered_df = filter(row -> !ismissing(row.choice_group) && row.choice_group == group, filtered_df)
    end
    
    if isempty(filtered_df)
        return (missing, missing) # Return missing for both
    else
        # Return the first match's info
        row = first(filtered_df)
        return (row.decimal_odds, row.winning)
    end
end



"""
Compares model-generated odds against market odds for 1x2 and Over/Under markets.
Includes the 'winning' status of the market bet.

Assumes `get_match_samples`, `calculate_1x2_odds`, 
`calculate_over_under_odds`, and `get_market_info` are defined.
"""
function compare_model_to_market(p1, id::Int, df_to_predict::DataFrame, data_store::Any)
    
    # --- 1. Get Model Predictions ---
    home_samples, away_samples = get_match_samples(p1, id)
    
    # --- 2. Get Market Odds for this match ---
    match_id = df_to_predict[id, :match_id]
    market_odds_df = filter(row -> row.match_id == match_id, data_store.odds)
    
    # --- 3. Process 1x2 (Full time) Market ---
    model_1x2 = calculate_1x2_odds(home_samples, away_samples)
    
    # Get market info (odds and winner)
    (market_home_odds, market_home_won) = get_market_info(market_odds_df, "Full time", "1")
    (market_draw_odds, market_draw_won) = get_market_info(market_odds_df, "Full time", "X")
    (market_away_odds, market_away_won) = get_market_info(market_odds_df, "Full time", "2")
    
    # Create the 1x2 DataFrame
    df_1x2 = DataFrame(
        market = "1x2",
        selection = ["Home (1)", "Draw (X)", "Away (2)"],
        line = Union{Float64, Missing}[missing, missing, missing],
        model_odds = [model_1x2.home, model_1x2.draw, model_1x2.away],
        market_odds = [market_home_odds, market_draw_odds, market_away_odds],
        winning = [market_home_won, market_draw_won, market_away_won]  # <-- ADDED
    )
    
    # --- 4. Process Over/Under (Match goals) Market ---
    
    # Find all unique O/U lines from the market data
    ou_market_df = filter(row -> row.market_name == "Match goals", market_odds_df)
    ou_lines = sort(unique(skipmissing(ou_market_df.choice_group)))
    
    # Initialize an empty DataFrame to hold O/U results
    df_ou = DataFrame(
        market = String[],
        selection = String[],
        line = Float64[],
        model_odds = Float64[],
        market_odds = Union{Float64, Missing}[],
        winning = Union{Bool, Missing}[]  # <-- ADDED
    )
    
    # Loop over each line, calculate model odds, get market odds, and add
    for line in ou_lines
        model_ou = calculate_over_under_odds(home_samples, away_samples, line)
        
        (market_over_odds, market_over_won) = get_market_info(market_odds_df, "Match goals", "Over", line)
        (market_under_odds, market_under_won) = get_market_info(market_odds_df, "Match goals", "Under", line)
        
        # <-- ADDED market_..._won to both push! calls
        push!(df_ou, ("Over/Under", "Over", line, model_ou.over, market_over_odds, market_over_won))
        push!(df_ou, ("Over/Under", "Under", line, model_ou.under, market_under_odds, market_under_won))
    end
    
    # --- 5. Combine and Calculate Value ---
    
    # Stack the 1x2 and O/U DataFrames
    comparison_df = vcat(df_1x2, df_ou)
    
    # Calculate value: (Market Odds / Model Odds) - 1
    comparison_df.value = (comparison_df.market_odds ./ comparison_df.model_odds) .- 1
    
    # Round the numeric columns for cleaner display
    for col in [:model_odds, :market_odds, :value]
        if col in names(comparison_df)
            comparison_df[!, col] = round.(comparison_df[!, col], digits=3)
        end
    end
    
    return comparison_df
end



using DataFrames

"""
Iterates through all matches in `df_to_predict`, generates a comparison
DataFrame for each, and concatenates them into a single master DataFrame.

Adds `match_id` and `tournament_slug` for grouping.
"""
function generate_all_comparisons(p1, df_to_predict::DataFrame, data_store::Any)
    
    all_comparisons = DataFrame[] # An array to hold all our small DataFrames
    total_matches = nrow(df_to_predict)
    
    println("Starting comparison for $total_matches matches...")
    
    # Loop over every ID from 1 to the end of the df_to_predict
    for id in 1:total_matches
        try
            # Get match info for adding to the table
            match_info = df_to_predict[id, :]
            
            # Use the function from the previous step
            comparison_table = compare_model_to_market(p1, id, df_to_predict, data_store)
            
            # Add the grouping columns we care about
            comparison_table.match_id .= match_info.match_id
            comparison_table.tournament_slug .= match_info.tournament_slug
            
            push!(all_comparisons, comparison_table)
            
        catch e
            println("WARNING: Could not process id=$id (MatchID: $(df_to_predict[id, :match_id])). Error: $e")
        end
    end
    
    println("...Done. Concatenating results.")
    
    # Stack all individual DataFrames into one big one
    master_table = vcat(all_comparisons...)
    
    return master_table
end



"""
Calculates the ROI for a simple value betting strategy,
with options to group the results.

# Arguments
- `master_comparison_table`: The DataFrame from `generate_all_comparisons`.
- `group_by_cols`: A Symbol or Vector of Symbols to group by 
  (e.g., `[:tournament_slug]`, `[:market]`, or `[:tournament_slug, :market]`).
  Set to `nothing` for a single overall ROI.
- `value_threshold`: The minimum "value" to place a bet (e.g., 0.0 for >0% value).
"""
function calculate_grouped_roi(
    master_comparison_table::DataFrame;
    group_by_cols::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
    value_threshold::Float64 = 0.0
)
    
    # 1. Apply Strategy: Filter for bets we would have placed.
    # We must filter out missing values for `value` and `winning`
    # as we can't bet on or assess them.
    bets_df = filter(
        row -> !ismissing(row.value) && 
               !ismissing(row.market_odds) &&
               !ismissing(row.winning) &&
               row.value > value_threshold,
        master_comparison_table
    )
    
    if isempty(bets_df)
        println("No bets found matching the strategy (value > $value_threshold).")
        return DataFrame()
    end

    # 2. Calculate Profit/Loss for each bet (assuming 1 unit stake)
    # Note: We already filtered out ismissing(row.winning)
    bets_df.profit = [
        if row.winning == true
            row.market_odds - 1.0  # Profit = (Odds * Stake) - Stake
        else
            -1.0 # Lost the 1 unit stake
        end
        for row in eachrow(bets_df)
    ]

    # 3. Group and Combine
    
    if isnothing(group_by_cols)
        # No grouping, just calculate overall totals
        total_profit = sum(bets_df.profit)
        total_bets = nrow(bets_df)
        roi = total_profit / total_bets
        
        return DataFrame(
            group = "Overall", 
            total_bets = total_bets, 
            total_profit = round(total_profit, digits=2), 
            roi = round(roi, digits=4)
        )
    else
        # Group by the specified columns
        gdf = groupby(bets_df, group_by_cols)
        
        # Calculate stats for each group
        roi_df = combine(gdf, 
            nrow => :total_bets, 
            :profit => sum => :total_profit
        )
        
        # Calculate ROI for each group
        roi_df.roi = roi_df.total_profit ./ roi_df.total_bets
        
        # Clean up for display
        roi_df.total_profit = round.(roi_df.total_profit, digits=2)
        roi_df.roi = round.(roi_df.roi, digits=4)
        
        # Sort by most bets
        sort!(roi_df, :total_bets, rev=true)
        
        return roi_df
    end
end




# (Assuming all helper functions and p1, df_to_predict, data_store are loaded)

# --- Step 1: Generate the master table (This is the slow part) ---
# This might take a few minutes if df_to_predict is large
master_table = generate_all_comparisons(p1, df_to_predict, data_store)

println("Master table generated with $(nrow(master_table)) potential selections.")
display(first(master_table, 5))


# --- Step 2: Analyze the results (This is the fast part) ---

# Define our strategy
# We will bet on anything our model finds with at least 5% value
strategy_threshold = 0.05 

println("\n" * "="^40)
println("           ROI Analysis (Value > $(strategy_threshold*100)%)")
println("="^40 * "\n")


# Example 1: Get the Overall ROI
println("--- Overall ROI ---")
overall_roi = calculate_grouped_roi(master_table, 
    group_by_cols = nothing, 
    value_threshold = strategy_threshold
)
println(overall_roi)


# Example 2: Get ROI broken down by market
println("\n--- ROI by Market ---")
roi_by_market = calculate_grouped_roi(master_table, 
    group_by_cols = :market, 
    value_threshold = strategy_threshold
)
println(roi_by_market)


# Example 3: Get ROI broken down by league
println("\n--- ROI by League ---")
roi_by_league = calculate_grouped_roi(master_table, 
    group_by_cols = :tournament_slug, 
    value_threshold = strategy_threshold
)
println(roi_by_league)


# Example 4: Get ROI broken down by both league AND market
println("\n--- ROI by League and Market ---")
roi_by_league_market = calculate_grouped_roi(master_table, 
    group_by_cols = [:tournament_slug, :market], 
    value_threshold = strategy_threshold
)
display(first(roi_by_league_market, 10))

master_table.profit = ifelse.(master_table.winning, master_table.market_odds .- 1.0, -1.0)



for g in groupby(master_table, [:selection, :line]) 
  market_name = only(g[1,[:selection]])
  market_line = only(g[1,[:line]])
  println(" market: $market_name + $market_line ..")




end 
gdf = groupby(master_table, [:selection, :line])


# 2. Combine the groups to calculate statistics
stats_df = combine(gdf,
    nrow => :num_bets,
    :profit => sum => :wealth,
    :profit => (p -> sum(p) / length(p)) => :roi
)


master_table.ev = (master_table.market_odds ./ master_table.model_odds) .- 1.0

# Define your minimum EV threshold
c = 0.05  # For example, only bet on 5% EV or higher

# Create the filtered DataFrame of actual bets
bets_df = filter(row -> row.ev > c, master_table)

bets_df.profit = ifelse.(bets_df.winning, bets_df.market_odds .- 1.0, -1.0)
# 1. Group the DataFrame of actual bets
gdf_strategy = groupby(bets_df, [:selection, :line])

# 2. Combine to get the stats *for your strategy*
strategy_stats_df = combine(gdf_strategy,
    nrow => :num_bets,
    :profit => sum => :wealth,
    :profit => (p -> sum(p) / length(p)) => :roi
)

sort(strategy_stats_df, :wealth, rev=true)

sum(strategy_stats_df.wealth)
sum(strategy_stats_df.num_bets)


# 1. Add an 'ev' column
master_table.ev = (master_table.market_odds ./ master_table.model_odds) .- 1.0

# 2. Define your threshold and filter for bets
c = 0.15  # Your 5% EV threshold
bets_df = filter(row -> row.ev > c, master_table)

# 3. Calculate profit for those bets
bets_df.profit = ifelse.(bets_df.winning, bets_df.market_odds .- 1.0, -1.0)

# 4. Group and combine stats, now including win_rate
gdf_strategy = groupby(bets_df, [:selection, :line])

strategy_stats_df = combine(gdf_strategy,
    nrow => :num_bets,
    :winning => mean => :win_rate,  # <-- Added this line
    :profit => sum => :wealth,
    :profit => (p -> sum(p) / length(p)) => :roi
)




#=
 Row │ selection  line       num_bets  win_rate   wealth   roi       
     │ String     Float64?   Int64     Float64    Float64  Float64   
─────┼───────────────────────────────────────────────────────────────
   1 │ Draw (X)   missing           7  0.285714      7.5    1.07143
   2 │ Away (2)   missing          12  0.0833333    -9.55  -0.795833
   3 │ Over             4.5         8  0.375         8.25   1.03125
   4 │ Over             5.5         7  0.285714     18.0    2.57143
   5 │ Over             2.5         9  0.666667      2.93   0.325556
   6 │ Over             3.5         8  0.25         -0.85  -0.10625
   7 │ Over             1.5         3  0.666667     -0.35  -0.116667
   8 │ Over             6.5         2  0.0          -2.0   -1.0
   9 │ Under            0.5         5  0.0          -5.0   -1.0
  10 │ Under            1.5         9  0.111111     -6.5   -0.722222
  11 │ Under            2.5         4  0.25         -1.3   -0.325
  12 │ Under            3.5         4  0.5          -0.89  -0.2225
  13 │ Under            4.5         3  1.0           0.9    0.3
  14 │ Under            5.5         2  1.0           0.31   0.155
  15 │ Home (1)   missing           2  0.5           0.63   0.315

=#
