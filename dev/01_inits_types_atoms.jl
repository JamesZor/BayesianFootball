# dev/01_inits_types_atoms.jl

using Revise

using BayesianFootball



## 1. Load Data
data_store = BayesianFootball.Data.load_default_datastore()


# 2. Define an Experiment (your "Config")


model =  Models.PreGame.StaticPoisson()
splitter = Experiments.StaticSplit(["24/25"])
sampler_config = Sampling.NUTSMethod(500, 2, 50)


exp1 = Experiments.Experiment(
    "StaticPoisson_ExpandingWindow_22-23",
    model,
    splitter,
    sampler_config,
)

# This script manually executes each step of the "Trainer" pipeline
# to test that each module and function works correctly in sequence.

using Revise
using BayesianFootball
using DataFrames
using Turing # Needed for predict()

println("--- ✅ Setup Complete: Packages loaded ---")

# ============================================================================
# PHASE 1: DEFINE THE "ATOMS" (CONFIGS AND DATA)
# ============================================================================
println("\n--- PHASE 1: Defining Atoms ---")

# --- Atom 1: The DataStore (D) ---
data_store = BayesianFootball.Data.load_default_datastore()
println("Loaded DataStore with $(nrow(data_store.matches)) matches.")

# --- Atom 2: The Master Experiment Config ---
# This defines all the parameters for our single run.

# Model (M)
model = Models.PreGame.StaticPoisson()

# Splitter (defines how we get our training data D_i from D)
splitter = Experiments.StaticSplit(["24/25"])

# Sampler Config (Config_s)
sampler_config = Sampling.NUTSMethod(500, 2, 50)

# The full Experiment object
exp1 = Experiments.Experiment(
    "StaticPoisson_TestRun",
    model,
    splitter,
    sampler_config,
)

println("Experiment configured: $(exp1.name)")


# ============================================================================
# PHASE 2: EXECUTE THE MORPHISMS (THE PIPELINE STEPS)
# ============================================================================
println("\n--- PHASE 2: Executing Morphisms ---")

# --- PRE-STEP: Get the training data for this run ---
# The Experiments.jl runner would do this automatically. Here, we do it manually.
println("\n[Pre-Step] Filtering data based on splitter...")

train_df = filter(row -> row.season in splitter.train_seasons, data_store.matches)

println("Created train_df with $(nrow(train_df)) matches for season(s): $(splitter.train_seasons)")

# --- Step 1: Morphism f: (D_i, M) -> F_i ---
# Create the FeatureSet from our training data for this specific model.
println("\n[Step 1: Morphism f] Calling Features.create_features...")
features = Features.create_features(exp1.model, train_df)

println("✅ Success! Created FeatureSet with $(features.n_teams) teams.")

# --- Step 2: Build TRAINING Model ---
# Create the Turing model instance ready for training.
println("\n[Step 2] Building TRAINING model...")
turing_model = Models.PreGame.build_turing_model(exp1.model, features)
println("✅ Success! Built training model instance.")

# --- Step 3: Morphism g: (F_i, M, Config_s) -> C_params ---
# Run the sampler to get the posterior parameter chains.
println("\n[Step 3: Morphism g] Calling Sampling.train to get C_params...")
chains_params = Sampling.train(turing_model, exp1.sampler_config)
println("✅ Success! Sampling complete. C_params (parameter chains) created.")
display(chains_params)

# --- Step 4: Build PREDICTION Model ---
# Create the model instance conditioned on the parameters we just sampled.
println("\n[Step 4] Building PREDICTION model...")
# Note: For this test, we are predicting on the same data we trained on.
turing_pred_model = Models.PreGame.build_turing_model(exp1.model, chains_params, features)
println("✅ Success! Built prediction model instance.")

# --- Step 5: Morphism h_goals: (M, C_params, F_i) -> C_goals ---
# Generate the posterior predictive samples for home and away goals.
println("\n[Step 5: Morphism h_goals] Calling Turing.predict to get C_goals...")
chains_goals = Turing.predict(turing_pred_model)
println("✅ Success! Goal prediction complete. C_goals (goal chains) created.")
display(chains_goals)

println("\n--- 🎉 PIPELINE TEST COMPLETE ---")


# ============================================================================
# PHASE 3: EXECUTE THE "ANALYZER" PIPELINE (PREDICTION)
# ============================================================================
println("\n--- PHASE 3: Executing Analyzer Morphism (h_goals) ---")

# --- PRE-STEP: Define the new data we want to predict on ---
df_to_predict = first(train_df, 100)
df_to_predict = train_df
println("\n[Pre-Step] Defined new data to predict on ($(nrow(df_to_predict)) matches).")

# --- Step 4: Prepare Prediction Data ---
println("\n[Step 4] Preparing prediction data using training team_map...")
team_map = features.team_map 
n_teams = features.n_teams
home_ids_to_predict = [team_map[name] for name in df_to_predict.home_team]
away_ids_to_predict = [team_map[name] for name in df_to_predict.away_team]
println("✅ Success! Mapped team names to integer IDs.")

# --- Step 5: Build PREDICTION Model ---
println("\n[Step 5] Building PREDICTION model...")
turing_pred_model = Models.PreGame.build_turing_model(exp1.model, n_teams, home_ids_to_predict, away_ids_to_predict)
println("✅ Success! Built prediction model instance.")

# --- Step 6: Morphism h_goals: (M, C_params, F_pred) -> C_goals ---
# This is the serial (non-parallel) version of the predict call.
println("\n[Step 6: Morphism h_goals] Calling Turing.predict to get C_goals...")
chains_goals = Turing.predict(turing_pred_model, chains_params)
println("✅ Success! Goal prediction complete. C_goals (goal chains) created.")
display(chains_goals)

id = 300

df_to_predict[id, :]

describe(chains_goals[Symbol("predicted_home_goals[$id]")])
describe(chains_goals[Symbol("predicted_away_goals[$id]")])

df_to_predict
