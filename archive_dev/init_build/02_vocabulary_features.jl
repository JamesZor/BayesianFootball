using Pkg
Pkg.activate(".")

# Load Revise for live code updates and your main package
using Revise
using BayesianFootball
using DataFrames
using Turing

println("✅ Environment ready.")
println("You can now run the following blocks one by one in your REPL.\n")

# --- 1. Load the DataStore (D) ---
# This is the complete set of all data we have.
println("--- 1. Loading DataStore (D) ---")
data_store = BayesianFootball.Data.load_default_datastore();
println("✅ DataStore loaded with $(nrow(data_store.matches)) matches.\n")




# --- 2. Define the experiment parameters ---
# We define the model (M) and how to split the data.
println("--- 2. Defining experiment parameters ---")
model = BayesianFootball.Models.PreGame.StaticPoisson();
splitter = BayesianFootball.Experiments.StaticSplit(["24/25"]); # Using a small season for speed
sampler_config = BayesianFootball.Sampling.NUTSMethod(10, 2, 1); # Short run for testing
println("✅ Experiment parameters defined for model: $(typeof(model))\n")

# --- 3. Create the Global Vocabulary (G) ---
# MORPHISM: G_map: D x M -> G
# This runs *once* on the entire DataStore.
println("--- 3. Creating Global Vocabulary (G) ---")
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model);
println("✅ Global Vocabulary (G) created.")
println("Vocabulary contains the following mappings:")
for key in keys(vocabulary.mappings)
    println("  - :$key")
end
println("Number of teams in vocabulary: ", vocabulary.mappings[:n_teams], "\n")

# --- 4. Get a single data split (D_i) ---
# In a real experiment, the ExperimentRunner would loop over these.
# Here, we manually create one for demonstration.
println("--- 4. Creating a data split (D_i) ---")
train_df = filter(row -> row.season in splitter.train_seasons, data_store.matches);
println("✅ Created a data split (D_i) with $(nrow(train_df)) matches.\n")

# --- 5. Create the FeatureSet for the split (F_i) ---
# MORPHISM: f_i: D_i x G x M -> F_i
# This uses the global vocabulary (G) to transform the specific split (D_i).
println("--- 5. Creating FeatureSet (F_i) from split ---")
feature_set = BayesianFootball.Features.create_features(train_df, vocabulary, model);
println("✅ FeatureSet (F_i) created for the data split.")
println("FeatureSet contains the following data fields:")
for key in keys(feature_set.data)
    println("  - :$key")
end
println()

# --- 6. Build the Turing Model ---
# This step uses the data in the FeatureSet (F_i) to instantiate the @model block.
println("--- 6. Building Turing model instance ---")
turing_model = BayesianFootball.Models.PreGame.build_turing_model(model, feature_set);
println("✅ Turing model instance created.\n")

# --- 7. Train the Model (Sample) ---
# MORPHISM: g: F_i x M x Config_s -> C
# This is the main MCMC sampling step.
println("--- 7. Training model (sampling) ---")
println("⏳ Starting sampling... (This might take a moment)")
chains = BayesianFootball.Sampling.train(turing_model, sampler_config);
println("✅ Sampling complete! Chains (C) created.\n")

#=
# --- 8. Inspect the Results ---
# You can now inspect the chains object.
println("--- 8. Inspecting Results ---")
println("\n--- Chain Summary ---")
display(chains)

# Example of plotting a parameter
# using StatsPlots
# plot(chains, :home_adv)
=#


# --- 9. predict ---

df_to_predict = first(train_df, 5)

predictions = Models.PreGame.predict(model, df_to_predict, vocabulary, chains)

p1 = predictions


id = 1
describe(p1[Symbol("predicted_home_goals[$id]")])
describe(p1[Symbol("predicted_away_goals[$id]")])
