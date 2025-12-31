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
  warmup_period = 20,
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
                100,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

experiment_conf = Experiments.ExperimentConfig(
                    name = "test_static_poisson",
                    model = model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results = Experiments.run_experiment(ds, experiment_conf)



using Distributions
model_2 = Models.PreGame.StaticPoisson(prior= Cauchy(0))

experiment_conf_2 = Experiments.ExperimentConfig(
                    name = "test_static_poisson_cauchy",
                    model = model_2,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results_2 = Experiments.run_experiment(ds, experiment_conf_2)



using BayesianFootball.Signals
flat_strat = FlatStake(0.05)
# 2. Conservative Kelly: Quarter Kelly (0.25)
kelly_strat = KellyCriterion(0.25)

# 3. Bayesian/Shrinkage Kelly: Uses the Baker-McHale analytical approximation
shrink_strat = AnalyticalShrinkageKelly()

baker = BayesianKelly()

my_signals = [flat_strat, kelly_strat, shrink_strat, baker]

using BayesianFootball.Backtesting
ledger = BayesianFootball.BackTesting.run_backtest(ds, [exp_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)




ledger = BayesianFootball.BackTesting.run_backtest(ds, [exp_results, exp_results_2], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)


BayesianFootball.BackTesting.generate_tearsheet(ledger)


BayesianFootball.BackTesting.summarize_models(ledger)

a = BayesianFootball.BackTesting.summarize_markets(ledger; compare_models=true)

