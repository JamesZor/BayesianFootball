"""
Dev work regarding the adding of the AR1 model.


added file: src/models/pregame/models-src/ar1-poisson.jl
needs the following functions 
  - ar1_poisson_model
  - kdef struct model - ar1-poisson
  - build_turing_model
  - extract_paramters 
  - reconstruct / trend  -( want) 

    
= src/types-interfaces - TypesInterfaces 
  - add abstract struct type 

= src/models/pregame/pregame-module.jl 
  - export model - export AR1Poisson 
  - wrapper functions extract_paramters - could move into the model as the file is getting large 
  - add file path to model src file 

"""

using Revise
using BayesianFootball
using DataFrames
using Statistics

using JLD2

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 

model = BayesianFootball.Models.PreGame.AR1Poisson()

ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

fs_modded = feature_sets[2:end]

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=6, n_warmup=500) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, fs_modded)


JLD2.save_object("debug_ar1_poisson.jld2", results)


"""
init build 
"""

using Revise
using BayesianFootball
using DataFrames
using Statistics



model = BayesianFootball.Models.PreGame.AR1Poisson()


ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

turing_model = BayesianFootball.Models.PreGame.build_turing_model(model, feature_sets[1][1])
"""

julia> turing_model = BayesianFootball.Models.PreGame.build_turing_model(model, feature_sets[1][1])
this is a place holder for the AR1Model build_turing_model
"""

ldf_1 = Turing.LogDensityFunction(turing_model)

# testing the extract_parameters 
using JLD2

# using another models chains as dummy data 
results = JLD2.load_object("training_results.jld2")

mp = filter( row -> row.round == 10 , ds.matches)
predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, results[1][1])
"""
julia> rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, results[1][1])
this is a place holder for the AR1Model extract_parameters level 1 
"""


# testing the wrapper 
split_col_sym = :round
all_split = sort(unique(ds.matches[!, split_col_sym]))
prediction_split_keys = all_split[3:end] 
grouped_matches = groupby(ds.matches, split_col_sym)

dfs_to_predict = [
    grouped_matches[(; split_col_sym => key)] 
    for key in prediction_split_keys
]

oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict, 
    vocabulary,
    results
)
"""
julia> oos_results = BayesianFootball.Models.PreGame.extract_parameters(
           model,
           dfs_to_predict, 
           vocabulary,
           results
       )
this is a place holder for the AR1Model extract_parameters level 2 - the wrapper. 
"""
