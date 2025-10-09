"""
1. load the models - work around 

"""


### need to run a new train as the models here are old - this is a work around 
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson_ha.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA
using .AR1PoissonHA

ssm_neg_bin_path = "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_neg_bin_ha_20251004-122001"
ssm_neg_m = load_model(ssm_neg_bin_path) 




# data
const DATA_PATH = "/home/james/bet_project/football/scotland_football"
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
add_global_round_column!(data_store.matches)
target_matches = filter(row -> row.season=="24/25", data_store.matches)

tm = CSV.read(data_files.match, DataFrame, header=1)
add_global_round_column!(tm)

path_odd = "/home/james/bet_project/football/scotland_football/football_data_mixed_odds.csv"
odds = CSV.read(path_odd, DataFrame, header=1)

t_m = tm[4082, :]


mapping = ssm_neg_m.result.mapping;
chain = ssm_neg_m.result.chains_sequence[1];

features = BayesianFootball.create_master_features(DataFrame(t_m), mapping);
predictions = BayesianFootball.predict(ssm_neg_m.config.model_def, chain, features, mapping);


mean(1 ./ predictions.ft.home)
mean(1 ./ predictions.ft.draw)
mean(1 ./ predictions.ft.away)
mean(1 ./ predictions.ht.home)
mean(1 ./ predictions.ht.draw)
mean(1 ./ predictions.ft.away)
median(1 ./ predictions.ft.under_25)
median(1 ./ (1 .- predictions.ft.under_25))
filter(row -> row.match_id==t_m.match_id, odds)




using .PredictionCubes

t_m = filter(row -> row.season=="25/26", tm)
t_m = t_m[1:4, :]
# This one call does all the parallel prediction work
cube = PredictionCube(t_m, ssm_neg_m);
first_match_id = t_m.match_id[1]
predicted_probs = get_match_outcome_probs(cube, first_match_id)



