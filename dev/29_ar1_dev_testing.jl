"""
Workspace to dev and explore the ar1 model 
  - with improved splitter - features 


"""

using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 


# data pre 
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(subset( data_store.matches, 
           :tournament_id => ByRow(isequal(55)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)


model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# here want to start the expanding window cv ( 1 -38) so 38 - 35 = 3 +1 ( since we have zero ) 4
ds.matches.split_col = max.(0, ds.matches.match_week .- 35);

splitter_config = BayesianFootball.Data.ExpandingWindowCV(
    train_seasons = [], 
    test_seasons = ["24/25"], 
    window_col = :split_col,      # 1. WINDOWING: Split chunks based on this (0, 1, 2...)
    method = :sequential,
    dynamics_col = :match_week      # 2. DYNAMICS: Inside the chunk, evolve time based on this
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

# API is now clean: no extra kwargs needed
feature_sets = BayesianFootball.Features.create_features(
    data_splits, 
    vocabulary, 
    model, 
    splitter_config 
)



# sampler 

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=200) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, feature_sets)

