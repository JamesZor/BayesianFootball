using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)

using Distributions


data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


model_1 = Models.PreGame.StaticPoisson()
name_1 = "normal"

model_2 = Models.PreGame.GRWPoisson()
name_2 = "grw"

model_3 = Models.PreGame.AR1Poisson()
name_3 = "ar1"


cfg_1 = Experiments.experiment_config_models(model_1, name_1)
cfg_2 = Experiments.experiment_config_models(model_2, name_2)
cfg_3 = Experiments.experiment_config_models(model_3, name_3)


results1 = Experiments.run_experiment(ds, cfg_1)
Experiments.save_experiment(results1)

results2 = Experiments.run_experiment(ds, cfg_2)
Experiments.save_experiment(results2)


results3 = Experiments.run_experiment(ds, cfg_3)
Experiments.save_experiment(results3)

