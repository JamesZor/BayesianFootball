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

data_store.matches.half_cat =  ifelse.( data_store.matches.match_month .∈ Ref([7,8,9,10,11,12]), true, false);


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

# small sampling to test
sampler_config = Sampling.NUTSMethod(50, 2, 50)

turing_model = Models.PreGame.build_turing_model(model, feature_sets[1][1])
chains_params = Sampling.train(turing_model, sampler_config)


turing_model2 = Models.PreGame.build_turing_model(model, feature_sets[2][1])
chains_params2 = Sampling.train(turing_model2, sampler_config)



function train_model_f(model, sampler_config, F) 
  turing_model = Models.PreGame.build_turing_model(model, F)
  chains_params = Sampling.train(turing_model, sampler_config)
return chains_params


end 

function trian_model(model, sample_config, feature_sets) 
  chains::Vector{Tuple}(undef,length(feature_sets))

for i,ff in enumrate(feature_sets)
    chains[i] = train_model_f(model, sampler_config, feature_sets[i][1]
end
  return chains
end





using DataFrames
using Turing # Required for Chains type
using ProgressMeter # Optional: for a progress bar

# --- Method 1: Train on a single FeatureSet ---
"""
    train(model, sampler_config, feature_set)

Trains the specified model on a single FeatureSet using the given sampler configuration.
Corresponds to the morphism f: F_i x M x C_s -> X_i.

Returns:
- A `Turing.Chains` object (X_i).
"""
function train(
    model::Models.AbstractPregameModel, 
    sampler_config::Sampling.AbstractTrainingMethod, 
    feature_set::Features.FeatureSet
)::Turing.Chains
    
    println("Dispatching to train(model, sampler_config, ::FeatureSet)...")
    
    # Build the Turing model instance using the specific FeatureSet (F_i)
    turing_model = Models.PreGame.build_turing_model(model, feature_set)
    
    # Train the model using the sampling configuration (C_s)
    chains_params = Sampling.train(turing_model, sampler_config)
    
    return chains_params # This is X_i
end

# --- Method 2: Train on a Vector of (FeatureSet, Metadata) Tuples ---
"""
    train(model, sampler_config, feature_sets_with_metadata)

Trains the model on each FeatureSet provided in the input vector.
Corresponds to the morphism \\hat{f}: F_1:N x M x C_s -> X_1:N.

Arguments:
- `model`: The AbstractFootballModel (M) to train.
- `sampler_config`: The AbstractTrainingMethod (C_s) configuration.
- `feature_sets_with_metadata`: A Vector{Tuple{FeatureSet, String}} (F_1:N).

Returns:
- A `Vector{Tuple{Chains, String}}` (X_1:N) where each tuple contains the resulting
  MCMC chains (X_i) and the corresponding metadata.
"""
function train(
    model::Models.AbstractFootballModel, 
    sampler_config::Sampling.AbstractTrainingMethod, 
    feature_sets_with_metadata::Vector{Tuple{Features.FeatureSet, String}}
)::Vector{Tuple{Turing.Chains, String}}

    println("Dispatching to train(model, sampler_config, ::Vector{Tuple{FeatureSet, String}})...")
    
    num_splits = length(feature_sets_with_metadata)
    all_chains = Vector{Tuple{Turing.Chains, String}}(undef, num_splits)
    
    println("🚀 Starting training for $num_splits feature sets...")
    
    @showprogress "Training splits..." for i in 1:num_splits
        feature_set_i, metadata = feature_sets_with_metadata[i] # Unpack F_i and metadata
        
        println("\n--- Training Split $i: $metadata ---")
        
        # *** Call the single FeatureSet method via dispatch ***
        chains_i = train(model, sampler_config, feature_set_i) 
        
        all_chains[i] = (chains_i, metadata) # Store X_i and metadata
        println("✅ Finished Split $i.")
    end
    
    println("\n🎉 Training complete for all splits!")
    return all_chains # This is X_1:N
end

# --- Example Usage (using your REPL variables) ---

# Assuming 'model', 'sampler_config', and 'feature_sets' are defined
# feature_sets is Vector{Tuple{FeatureSet, String}}

# Train a single split (will call the first method)
first_feature_set, first_metadata = feature_sets[1]
single_chains = train(model, sampler_config, first_feature_set) 
display(single_chains)

# Train all splits (will call the second method)
all_resulting_chains = train(model, sampler_config, feature_sets) 

display(first(all_resulting_chains)) # Show the (Chains, Metadata) tuple for the first split


### v2 

using Revise
using BayesianFootball
using DataFrames

# --- Phase 1: Globals (D, M, G) --- (Same as before)
data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)

# --- Phase 2: Data Splits & Features (D_1:N, F_1:N) --- (Same as before)
# (Assuming preprocessing and splitter_config are defined)
data_store.matches.half_cat = ifelse.( data_store.matches.match_month .∈ Ref([7,8,9,10,11,12]), true, false);
splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["23/24"], :half_cat, :sequential) #
data_splits = BayesianFootball.Data.create_data_splits(data_store, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

# --- Phase 3: Define Training Configuration ---
# Sampler Config (Choose one)
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=100) # Use renamed struct
# sampler_conf = ADVIConfig(n_iterations=10000)
# sampler_conf = MAPConfig()

## Use default (half of threads, likely physical cores)
strategy_parallel_limited = Independent(parallel=true) 

# Explicitly set a limit (e.g., if NUTS uses 2 chains, maybe allow 4 concurrent splits on 8 threads)
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

# training_config_limited = TrainingConfig(sampler_conf, strategy_parallel_limited)
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

# Then run:

results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)

# save and load 
using JLD2
JLD2.save_object("training_results.jld2", results)
results = JLD2.load_object("training_results.jld2")

### extraction 
using Statistics

r = results[2][1]


a1 = vec(r[Symbol("log_α[1]")]);
b1 = vec(r[Symbol("log_β[1]")]);
a2 = vec(r[Symbol("log_α[2]")]);
b2 = vec(r[Symbol("log_β[2]")]);
h = vec(r[Symbol("home_adv")]);

l1 = a1 .+ b2 .+ h ;
l2 = a2 .+ b1 ;


mean(l1)

using StatsPlots

density(l1, label="home")
density!(l2, label="away")

for i in vocabulary.mappings[:team_map]
  if i[2] == 2
    println(i)
  end 
end 



