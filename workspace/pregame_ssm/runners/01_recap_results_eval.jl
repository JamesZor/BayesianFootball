using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions, Plots
using CSV 

# models
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson_ha.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA
using .AR1PoissonHA

all_model_paths = Dict(
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_poisson_ha_20251004-111854",
  "ssm_neg_bin" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_neg_bin_ha_20251004-122001",
)
loaded_models_all = load_models_from_paths(all_model_paths)



m1 = loaded_models_all["ssm_neg_bin"]
mapping = m1.result.mapping;
chain = m1.result.chains_sequence[1];
posterior_samples = BayesianFootball.extract_posterior_samples(
    m1.config.model_def,
    chain.ft,
    mapping
);
last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

match_to_predict = DataFrame(
    home_team=match_to_analyze.home_team,
    away_team=match_to_analyze.away_team,
    tournament_id=league_id_to_predict,
    global_round = next_round,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)
features = BayesianFootball.create_master_features(match_to_predict, mapping);
predictions = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);

const DATA_PATH = "/home/james/bet_project/football/scotland_football"
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
add_global_round_column!(data_store.matches)
