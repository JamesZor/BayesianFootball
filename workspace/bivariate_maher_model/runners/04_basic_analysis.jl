using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions

include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/prediction.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/analysis_funcs.jl")
using .BivariateMaher
using .BivariatePrediction
using .Analysis

##############################
# main script 
##############################
## --- 1. Define Models and Match ---
model_paths = Dict(
    "maher_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2526_20250919-200800",
    "bivar_2526" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2526_20250919-200835",
    "maher_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_basic_seasons_2425_2526_20250919-202508",
    "bivar_24_26" => "/home/james/bet_project/models_julia/experiments/model_comparison/maher_bivariate_seasons_2425_2526_20250919-204350"
)

loaded_models = load_models_from_paths(model_paths)


### match 1 
team_name_home = "middlesbrough"
team_name_away = "west-bromwich-albion"
match_league_id = 2

predictions = generate_predictions(loaded_models, team_name_home, team_name_away, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)


plot_odds_distributions(predictions, :ft, :home, title_suffix="Home for $team_name_home v $team_name_away ")
plot_odds_distributions(predictions, :ht, :home, title_suffix="Home for $team_name_home v $team_name_away ")

plot_odds_distributions(predictions, :ft, :away, title_suffix="away for $team_name_home v $team_name_away ")

plot_odds_distributions(predictions, :ft, :under_15, title_suffix="under 15")
plot_odds_distributions(predictions, :ft, :under_25, title_suffix="under 25")
### match 2
team_name_home = "st-johnstone"
team_name_away = "dunfermline-athletic"
match_league_id = 55

predictions = generate_predictions(loaded_models, team_name_home, team_name_away, match_league_id)
odds_df = create_odds_dataframe(predictions)

ft_odds = filter(row->row.Time=="FT", odds_df)
ht_odds = filter(row->row.Time=="HT", odds_df)
