using Revise
using BayesianFootball
using DataFrames

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

# Preprocess the entire DataFrame first for this test
# split the season into two 

data_store.matches.half_cat =  ifelse.( data_store.matches.match_month .∈ Ref([7,8,9,10,11,12]), true, false)


splitter_config = BayesianFootball.Data.ExpandingWindowCV( #
  [], # Base
  ["23/24"], # Target
  :half_cat, #
  :sequential #
)

data_splits = BayesianFootball.Data.create_data_splits(data_store, splitter_config)



feature_sets = BayesianFootball.Features.create_features(
    data_splits,
    vocabulary,
    model,
    splitter_config
)


