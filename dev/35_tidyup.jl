using Revise 
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

# BLAS.set_num_threads(1) 


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 35,
    stop_early = false
)

splits = Data.create_data_splits(ds, cv_config)

model = BayesianFootball.Models.PreGame.StaticPoisson()


vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 


feature_sets = BayesianFootball.Features.create_features(
    splits, vocabulary, model, cv_config
)



train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 


# sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=2, n_warmup=100) # Use renamed struct

init_conf = Samplers.MapInit(50,0.001) 

sampler_conf = Samplers.NUTSConfig(
                100,
                2,
                100,
                0.65,
                10,
                init_conf 
)


training_config = Training.TrainingConfig(sampler_conf, train_cfg)



results = Training.train(model, training_config, feature_sets)



sampler_conf = Samplers.NUTSConfig(
                100,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)


training_config = Training.TrainingConfig(sampler_conf, train_cfg)

results1 = Training.train(model, training_config, feature_sets)

