
# Load your package
using BayesianFootball
using Turing
using DataFrames 
using StatsPlots, StatsBase


data_store = BayesianFootball.Data.load_default_datastore()

# reduce data size for testing.
filter!(row-> row.season=="24/25", data_store.matches)
filter!(row -> row.tournament_id in [54, 55], data_store.matches)

# get the features 
feature_set = BayesianFootball.Features.create_features(data_store)


## Models 

# model 1 simple 
static_poisson = BayesianFootball.Models.PreGame.StaticPoisson()
turing_model_1 = BayesianFootball.Models.PreGame.build_turing_model(static_poisson, feature_set)

# model 2 simplex - 

static_simplex_poisson = BayesianFootball.Models.PreGame.StaticSimplexPoisson()
turing_model_2 = BayesianFootball.Models.PreGame.build_turing_model(static_simplex_poisson, feature_set)

## training 

training_method_nuts = BayesianFootball.Sampling.NUTSMethod(n_samples=2_000, n_chains=2, n_warmup=20)


chain_simple = BayesianFootball.Sampling.train(turing_model_1, training_method_nuts)
chain_simplex = BayesianFootball.Sampling.train(turing_model_2, training_method_nuts)

