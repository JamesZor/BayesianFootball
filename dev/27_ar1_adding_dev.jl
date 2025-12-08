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

