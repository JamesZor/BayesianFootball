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



using Turing

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


# ----------------------------------------------
# 2. Experiment Configs Set up 
# ----------------------------------------------

# --- setup 1 
cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["22/23"],
    history_seasons = 0,
    dynamics_col = :match_week,
    # warmup_period = 36,
    warmup_period = 35,
    stop_early = true
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson() # place holder
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = Samplers.NUTSConfig(
                20,
                2,
                20,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05),
                false,
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


# ----------------------------------------------
# 3. GRW models
# ----------------------------------------------

# A: ---  GRW Poisson

grw_poisson_model = Models.PreGame.GRWPoisson()

conf_poisson = Experiments.ExperimentConfig(
                    name = "grw poisson",
                    model = grw_poisson_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_poisson = Experiments.run_experiment(ds, conf_poisson)

describe(results_poisson.training_results[1][1]) 
df_trends_poisson = Models.PreGame.extract_trends(grw_poisson_model, feature_sets[end][1], results_poisson.training_results[end][1])

# B: ---  GRW Dixon Coles DC 

grw_dixoncoles_model = Models.PreGame.GRWDixonColes()

conf_dixoncoles = Experiments.ExperimentConfig(
                    name = "grw dixon coles",
                    model = grw_dixoncoles_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_dixoncoles = Experiments.run_experiment(ds, conf_dixoncoles)


describe(results_dixoncoles.training_results[1][1]) 
df_trends_dixoncoles = Models.PreGame.extract_trends(grw_dixoncoles_model, feature_sets[end][1], results_dixoncoles.training_results[end][1])


# C: ---  GRW Negative binomial 

grw_negbin_model = Models.PreGame.GRWNegativeBinomial()

conf_negbin = Experiments.ExperimentConfig(
                    name = "grw negative binomial",
                    model = grw_negbin_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_negbin = Experiments.run_experiment(ds, conf_negbin)


describe(results_negbin.training_results[1][1]) 
df_trends_negbin = Models.PreGame.extract_trends(grw_negbin_model, feature_sets[end][1], results_negbin.training_results[end][1])


# D: ---  GRW Bivariate Poisson  BP

grw_bipoisson_model = Models.PreGame.GRWBivariatePoisson()

conf_bipoisson = Experiments.ExperimentConfig(
                    name = "grw bivariate poisson",
                    model = grw_bipoisson_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_bipoisson = Experiments.run_experiment(ds, conf_bipoisson)


describe(results_bipoisson.training_results[1][1]) 
df_trends_bipoisson = Models.PreGame.extract_trends(grw_bipoisson_model, feature_sets[end][1], results_bipoisson.training_results[end][1])



####

using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(ds, [results_poisson, results_negbin, results_bipoisson, results_dixoncoles], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

ledger = BayesianFootball.BackTesting.run_backtest(ds, [results_negbin], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

a = BayesianFootball.BackTesting.generate_tearsheet(ledger)



c=unique(a.selection)

for cc in c
  show(subset(a, :selection => ByRow(isequal(cc))))
end

