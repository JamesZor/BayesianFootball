
using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics



save_dir = "dev_exp/simple_dixon/"




data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticDixonColes()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)



# sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=600, n_chains=2, n_warmup=100) # Use renamed struct

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=100) # Use renamed struct
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4) 
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

# seasons_to_train = ["20/21","21/22","22/23","23/24","24/25"]
seasons_to_train = ["24/25"]


for season_str in seasons_to_train

    println("Processing season $season_str.")

    # create the data set 

    # filter for one season for quick training
    df = filter(row -> row.season==season_str, data_store.matches)
    # we want to get the last 4 weeks - so added the game weeks
    df = BayesianFootball.Data.add_match_week_column(df)
    df.split_col = max.(0, df.match_week .- 14);

    ds = BayesianFootball.Data.DataStore(
        df,
        data_store.odds,
        data_store.incidents
    )

    ## Set the sets

    splitter_config = BayesianFootball.Data.ExpandingWindowCV([], [season_str], :split_col, :sequential) #
    data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
    feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

    ## run  
    results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)

    ## save 
    save_season_name_str = save_dir * "s_" * replace(season_str, "/" => "_") * ".jld2"
    
    JLD2.save_object(save_season_name_str, results)

    
println("Finished season $season_str.")



end 

