using Revise 
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)



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
    warmup_period = 10,
    stop_early = true  # Splits go 1..5, 1..6, ..., 1..37
    # stop_early = false # Splits go 1..5, ..., 1..38 (The default)
)

splits = Data.create_data_splits(ds, cv_config)

model = BayesianFootball.Models.PreGame.AR1Poisson()


vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 


feature_sets = BayesianFootball.Features.create_features(
    splits, vocabulary, model, cv_config
)


