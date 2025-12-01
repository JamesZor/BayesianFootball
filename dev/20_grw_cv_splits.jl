using Revise
using BayesianFootball
using DataFrames
using Statistics


# --- HPC OPTIMIZATION START ---
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 



# --- create a subset for one league for one season 
#  To improve the speed and sampling 
#  Add match week 

data_store = BayesianFootball.Data.load_default_datastore()
# ds = BayesianFootball.load_scottish_data("24/25", split_week=14)

filtered_matches = subset(data_store.matches,
                          :season => ByRow(isequal("24/25")),
                          :tournament_id => ByRow(isequal(54))
                          )

matches_df = BayesianFootball.Data.add_match_week_column(filtered_matches)

using Dates 

time_step_summary = combine(
    groupby(matches_df, :match_week),
    nrow => :number_of_matches,
    :match_date => minimum => :start_date,
    # This stores the list of rounds as a vector [1, 2] instead of splitting rows
    :round => (x -> Ref(unique(x))) => :rounds_included 
)


"""
Here limit to one league to improve the sampling speed and reduce the compleity of the data 
"""
odds_subset = semijoin(data_store.odds, matches_df, on = :match_id)
incidents_subset = semijoin(data_store.incidents, matches_df, on = :match_id)

# Create the DataStore
ds = BayesianFootball.Data.DataStore(
    matches_df,
    odds_subset,
    incidents_subset
)




model= BayesianFootball.Models.PreGame.GRWPoisson()

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :match_week, :sequential) #

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

# here we need to remove the first week since we need 2 week for the dynamic process
fs_modded = feature_sets[2:end]

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=1, n_warmup=10) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, fs_modded)


results


all_split = sort(unique(ds.matches[!, :match_week]))

prediction_split_keys = all_split[3:end] 


grouped_matches = groupby(ds.matches, :match_week)


dfs_to_predict = [
    grouped_matches[(; :match_week => key)] 
    for key in prediction_split_keys
]

model = BayesianFootball.Models.PreGame.StaticPoisson()

all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict, 
    vocabulary,
    results
)



predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )



r = results[20][1]
mp = subset(ds.matches, :match_week => ByRow(isequal(22)))

modle 
rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, r)



match_predict_dixon = BayesianFootball.Predictions.predict_market(model, predict_config, r_dixon...);
