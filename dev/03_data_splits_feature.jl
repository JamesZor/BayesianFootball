
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
splitter_static = BayesianFootball.Data.StaticSplit(["23/24"]) #
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

splitter_config = splitters_to_test[2]


println("\n" * "="^40)
println("Testing Splitter: $(typeof(splitter_config))")
println("="^40)

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
