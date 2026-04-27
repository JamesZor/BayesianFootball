# current_development/dev_xg_models/r00_dataset_features.jl
# include("./l01_ireland.jl")

# runner.jl
using Revise
using BayesianFootball
using DataFrames

include("loader.jl")

# -------------------------------------------------------------------------
# 1. Load DataStore
# -------------------------------------------------------------------------
println("Loading Ireland SQL DataStore...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

# -------------------------------------------------------------------------
# 2. Define the Experiment Config 
# -------------------------------------------------------------------------
# Dynamically calculate the warmup period
calc_warmup = last(unique(subset(ds.matches, :season => ByRow(isequal("2025"))).match_month))
println("Calculated warmup period: ", calc_warmup)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [79], 
    target_seasons = ["2025"],
    history_seasons = 0,   
    dynamics_col = :match_month,
    warmup_period = 0, # Using the calculated variable
    stop_early = true
)

# -------------------------------------------------------------------------
# 3. Execute New Splitter Logic
# -------------------------------------------------------------------------
println("\nGenerating ID Boundaries...")
boundaries_with_meta = create_id_boundaries(ds, cv_config)

println("Generated $(length(boundaries_with_meta)) Temporal Folds.")

# -------------------------------------------------------------------------
# 4. Inspect Output
# -------------------------------------------------------------------------
for (boundary, meta) in boundaries_with_meta
    println("\n--- Fold $(boundary.fold_id) (Month $(boundary.target_step)) ---")
    println("  History Match IDs Count: ", length(boundary.history_match_ids))
    println("  Target Match IDs Count:  ", length(boundary.target_match_ids))
    if length(boundary.target_match_ids) > 0
        println("  First 3 Target IDs:      ", first(boundary.target_match_ids, 3))
    end
end


# ------
filter(name -> contains(lowercase(name), "blocked"), names(ds.statistics))
filter(name -> contains(name, "shot"), names(ds.statistics))
filter(name -> occursin(r"expected"i, name), names(ds.statistics))

# ==============================================================================
# runner.jl - Part 2: Feature Execution
# ==============================================================================

println("\n--- Starting Feature Extraction Phase ---")

# 1. Instantiate our dummy model
test_model = DevSequentialFunnel()

# 2. Run the Multiple Dispatch Pipeline
# We pass the boundaries we just generated, the SQL ds, and the model
feature_collection = build_features(boundaries_with_meta, ds, test_model)

println("Successfully extracted features for $(length(feature_collection)) folds.")

n = 15

# 3. Inspect Fold 1 to prove Relational Integrity
if length(feature_collection) > 0
    # Dig into the Tuple -> MockFeatureSet -> Dict
    fold_1_features = feature_collection[1][1].data 
    
    println("\n--- Fold 1 Feature Dictionary ---")
    # println("Month Step:       ", fold_1_features[:dynamics_step])
    # println("Target Matches:   ", fold_1_features[:n_target_matches])
    
    # Display the first 5 elements of our perfectly aligned arrays
    println("\nHome Teams:       ", first(fold_1_features[:flat_home_teams], n))
    println("Away Teams:       ", first(fold_1_features[:flat_away_teams], n))
    println("Home Goals:       ", first(fold_1_features[:flat_home_goals], n))
    println("Away Goals:       ", first(fold_1_features[:flat_away_goals], n))
    println("Home Shots:       ", first(fold_1_features[:flat_home_shots], n))
    println("Away Shots:       ", first(fold_1_features[:flat_away_shots], n))
    println("Home xG:       ", first(fold_1_features[:flat_home_xg], n))
    println("Away xG:       ", first(fold_1_features[:flat_away_xg], n))
    
    # The ultimate safety check for Turing.jl: Are all arrays identically sized?
    @assert length(fold_1_features[:flat_home_teams]) == fold_1_features[:n_target_matches]
    @assert length(fold_1_features[:flat_home_goals]) == fold_1_features[:n_target_matches]
    @assert length(fold_1_features[:flat_home_shots]) == fold_1_features[:n_target_matches]
    
    println("\n[SUCCESS] Array distributions perfectly aligned for Turing!")
end

# ==============================================================================
# 6. Validate the Output (Updated for GRW & Integers)
# ==============================================================================
n = 10
if length(feature_collection) > 0
    fold_1_features = feature_collection[1][1].data 
    
    println("\n--- Fold 1 Feature Dictionary ---")
    println("History Steps (Seasons): ", fold_1_features[:n_history_steps])
    println("Target Steps (Months):   ", fold_1_features[:n_target_steps])
    total_matches = length(fold_1_features[:time_indices])
    println("Total Matches:           ", total_matches)
    
    println("\n[Data Vectors ($total_matches matches total)]")
    println("Home Goals:  ", first(fold_1_features[:flat_home_goals], n))
    println("Away Goals:  ", first(fold_1_features[:flat_away_goals], n))
    println("Home Shots:  ", first(fold_1_features[:flat_home_shots], n))
    println("Away Shots:  ", first(fold_1_features[:flat_away_shots], n))
    println("Home xG:     ", first(fold_1_features[:flat_home_xg], n))
    println("Time Index:  ", first(fold_1_features[:time_indices], n))
    
    # The ultimate safety check for Turing.jl: Are all arrays identically sized?
    @assert length(fold_1_features[:flat_home_ids]) == total_matches
    @assert length(fold_1_features[:flat_home_goals]) == total_matches
    @assert length(fold_1_features[:flat_home_shots]) == total_matches
    @assert length(fold_1_features[:flat_home_xg]) == total_matches
    
    println("\n[SUCCESS] GRW Dimensions and Vocabulary correctly aligned for Turing!")
end

# test it, as it has been added to the module 
using Revise
using BayesianFootball
using DataFrames

# ==============================================================================
# 1. Define a Test Model 
# ==============================================================================
# We define a dummy model right in the runner just to test the pipeline.
# By making it a subtype of AbstractFootballModel, the multiple dispatch will catch it.
struct TestRelationalModel <: BayesianFootball.AbstractFootballModel end

# Tell the Feature Builder exactly which traits to pull from the DataStore
BayesianFootball.Features.required_features(::TestRelationalModel) = [:team_ids, :goals, :shots, :xg]


# ==============================================================================
# 2. Load DataStore
# ==============================================================================
println("Loading Ireland SQL DataStore...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


# ==============================================================================
# 3. Configure Splitter
# ==============================================================================
# Dynamically get a valid warmup period for 2025
calc_warmup = last(unique(subset(ds.matches, :season => ByRow(isequal("2025"))).match_month))

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [79], 
    target_seasons = ["2025"],
    history_seasons = 1,   
    dynamics_col = :match_month,
    warmup_period = 0, 
    stop_early = false
)


# ==============================================================================
# 4. Generate Boundaries (The New Architecture)
# ==============================================================================
println("\nGenerating ID Boundaries...")
boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, cv_config)
println("Generated $(length(boundaries_with_meta)) temporal folds.")


# ==============================================================================
# 5. Build Relational Features
# ==============================================================================
println("\nExtracting Relational Features via Multiple Dispatch...")
test_model = TestRelationalModel()

# This triggers your new Vector Dispatch macro-loop!
feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, test_model)
println("Successfully extracted feature sets for all folds.")


# ==============================================================================
# 6. Validate the Output
# ==============================================================================
n = 10
if length(feature_collection) > 0
    fold_1_features = feature_collection[7][1].data 
    
    println("\n--- Fold 1 Feature Dictionary ---")
    println("History Steps (Seasons): ", fold_1_features[:n_history_steps])
    println("Target Steps (Months):   ", fold_1_features[:n_target_steps])
    total_matches = length(fold_1_features[:time_indices])
    println("Total Matches:           ", total_matches)
    
    println("\n[Data Vectors ($total_matches matches total)]")
    println("Home Goals:  ", first(fold_1_features[:flat_home_goals], n))
    println("Away Goals:  ", first(fold_1_features[:flat_away_goals], n))
    println("Home Shots:  ", first(fold_1_features[:flat_home_shots], n))
    println("Away Shots:  ", first(fold_1_features[:flat_away_shots], n))
    println("Home xG:     ", first(fold_1_features[:flat_home_xg], n))
    println("Time Index:  ", first(fold_1_features[:time_indices], n))
    
    # The ultimate safety check for Turing.jl: Are all arrays identically sized?
    @assert length(fold_1_features[:flat_home_ids]) == total_matches
    @assert length(fold_1_features[:flat_home_goals]) == total_matches
    @assert length(fold_1_features[:flat_home_shots]) == total_matches
    @assert length(fold_1_features[:flat_home_xg]) == total_matches
    
    println("\n[SUCCESS] GRW Dimensions and Vocabulary correctly aligned for Turing!")
end

