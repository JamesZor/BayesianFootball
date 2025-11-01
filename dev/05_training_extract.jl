using Revise
using BayesianFootball
using DataFrames

# --- Phase 1: Globals (D, M, G) --- (Same as before)
data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)


# filter for one season for quick training
df = filter(row -> row.season=="24/25", data_store.matches)

# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)



# --- Phase 2: Data Splits & Features (D_1:N, F_1:N) --- (Same as before)
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

