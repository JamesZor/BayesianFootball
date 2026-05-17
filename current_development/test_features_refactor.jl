# current_development/test_features_refactor.jl

"""
This script serves as a verification runner for the refactored Features module.
It demonstrates the "New Relational Architecture" for data extraction:
1. Load a DataStore (Relational DB wrapper)
2. Define a Splitter (CVConfig) to generate temporal boundaries.
3. Generate SplitBoundary objects (Match ID pointers).
4. Extract Features (Relational mapping between pointers and DataStore tables).
"""

using Revise
using BayesianFootball
using DataFrames
using Dates

# -------------------------------------------------------------------------
# 1. SETUP: Load Data & Configure Model
# -------------------------------------------------------------------------
println("--- Initializing Test Runner ---")

# Load a local SQL DataStore (using Ireland as a lightweight example)
ds = Data.load_datastore_sql(Data.Ireland())

# Instantiate a concrete model engine.
# The model tells the Features module what it needs via `required_features(model)`.
# For DynamicXGModel, it requires: [:team_ids, :goals, :xg]
inter_cfg = PreGame.GlobalInterception()
dyn_cfg   = PreGame.TimeDecayDynamics(days_half_life=180.0)
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

test_model = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)

# -------------------------------------------------------------------------
# 2. SPLITTING: Generate Temporal Boundaries
# -------------------------------------------------------------------------
# We use GroupedCVConfig to define how we want to "walk forward" through time.
# GroupedCVConfig is preferred because it supports combining multiple 
# tournaments (e.g., [[56, 57]]) into a single processed sequence.
cv_config = Data.GroupedCVConfig(
    # Pass a list of lists. Each inner list is processed as a single group.
    # To combine leagues 56 and 57, use: tournament_groups = [[56, 57]]
    tournament_groups = [[79]], 
    target_seasons = ["2024"],  # We want to test on the 2024 season
    history_seasons = 1,        # Use 2023 as history
    dynamics_col = :match_month,# Step forward month-by-month
    warmup_period = 0, 
    stop_early = false
)

println("Generating ID Boundaries...")
# This returns a Vector{Tuple{SplitBoundary, SplitMetaData}}
# A SplitBoundary is just a struct of Match IDs (History + Target pointers).
boundaries = Data.create_id_boundaries(ds, cv_config)
println("Generated $(length(boundaries)) temporal folds.\n")

# -------------------------------------------------------------------------
# 3. EXTRACTION: Relational Feature Building
# -------------------------------------------------------------------------
println("Extracting Features via Relational Pipeline...")

# This is the core call to the refactored Features module.
# It uses multiple dispatch to:
#   a) Identify the required traits from the model.
#   b) Map the SplitBoundary IDs to the DataStore tables.
#   c) Construct a continuous sequence of data (History -> Target).
feature_collection = Features.create_features(boundaries, ds, test_model)

# -------------------------------------------------------------------------
# 4. VERIFICATION: Inspecting Fold 1
# -------------------------------------------------------------------------
if !isempty(feature_collection)
    # feature_collection is a Vector of Tuples: (FeatureSet, MetaData)
    f_set, meta = feature_collection[1]
    data = f_set.data
    
    println("--- Verification [Fold 1, Month $(meta.time_step)] ---")
    println("Status: Successfully extracted $(length(data)) traits.")
    
    # 1. Dimensional Checks
    n_total = length(data[:time_indices])
    println("Total Matches in Sequence: ", n_total)
    println("History Steps (Seasons):   ", data[:n_history_steps])
    println("Target Steps (Months):     ", data[:n_target_steps])
    
    # 2. Integrity Check: Are all arrays aligned?
    # Turing.jl requires that all data vectors for a single model have the same length.
    traits_to_check = [:flat_home_ids, :flat_away_ids, :flat_home_goals, :flat_home_xg, :time_indices, :season_indices]
    
    all_aligned = true
    for trait in traits_to_check
        len = length(data[trait])
        if len != n_total
            println("  [!] ERROR: Trait :$trait has length $len (Expected $n_total)")
            global all_aligned = false
        end
    end
    
    if all_aligned
        println("Relational Integrity:      [PASSED] All vectors perfectly aligned.")
    else
        println("Relational Integrity:      [FAILED] Dimensional mismatch detected.")
    end

    # 3. Content Preview
    n_preview = 5
    println("\nData Preview (First $n_preview matches):")
    println("  Home IDs:   ", data[:flat_home_ids][1:n_preview])
    println("  Home Goals: ", data[:flat_home_goals][1:n_preview])
    println("  Home xG:    ", data[:flat_home_xg][1:n_preview])
    println("  Time Idx:   ", data[:time_indices][1:n_preview])
    println("  Season Idx: ", data[:season_indices][1:n_preview])
    
    println("\n[SUCCESS] Refactored Features module is operational.")
else
    println("[ERROR] No features were extracted. Check tournament_ids or target_seasons.")
end
