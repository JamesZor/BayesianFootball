# ----------------------------------------------
# 1. The set up 
# ----------------------------------------------
using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


# -----
data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [56,57],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  # warmup_period = 35,
  warmup_period = 15,
    stop_early = false
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = Samplers.NUTSConfig(
                500,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


###


DC_model = Models.PreGame.DixonColesNCP(
            )


exp_conf_DC = Experiments.ExperimentConfig(
                    name = "DC",
                    model = DC_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

DC = Experiments.run_experiment(ds, exp_conf_DC)



using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]


ledger = BayesianFootball.BackTesting.run_backtest(ds, [DC, exp_results_2], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [DC], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

a = BayesianFootball.BackTesting.generate_tearsheet(ledger)

a2 = DC.training_results[1][1]
using Turing
describe(a2)

model_2 = Models.PreGame.StaticHierarchicalPoisson()

experiment_conf_2 = Experiments.ExperimentConfig(
                    name = "test_static_hierarchicalPoisson",
                    model = model_2,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results_2 = Experiments.run_experiment(ds, experiment_conf_2)



model_3 = Models.PreGame.BivariatePoissonNCP(
            )

experiment_conf_3 = Experiments.ExperimentConfig(
                    name = "BivariatePoissonNCP v3 ",
                    model = model_3,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results_3 = Experiments.run_experiment(ds, experiment_conf_3)


a = exp_results_2.training_results[1][1]
describe(a)

