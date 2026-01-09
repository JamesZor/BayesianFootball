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
    # tournament_ids = [56,57],
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 34,
  # warmup_period = 15,
    # stop_early = false
    stop_early = true
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


##
mvpln_model = Models.PreGame.StaticMVPLN()

exp_conf_mvpln = Experiments.ExperimentConfig(
                    name = "mvpln",
                    model = mvpln_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

mvpln_results = Experiments.run_experiment(ds, exp_conf_mvpln)

using Turing

r = mvpln_results.training_results[1][1]

describe(r)


shp_model = Models.PreGame.StaticHierarchicalPoisson()

experiment_conf_shp = Experiments.ExperimentConfig(
                    name = "static poisson",
                    model = shp_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

shp_results = Experiments.run_experiment(ds, experiment_conf_shp)

r_shp = shp_results.training_results[1][1]
describe(r_shp)


model_BP = Models.PreGame.BivariatePoissonNCP()

experiment_conf_BP = Experiments.ExperimentConfig(
                    name = "BivariatePoissonNCP v3 ",
                    model = model_BP,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

BP_results = Experiments.run_experiment(ds, experiment_conf_BP)

describe( BP_results.training_results[1][1])



######

using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(ds, [mvpln_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [mvpln_results, shp_results, BP_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

ledger = BayesianFootball.BackTesting.run_backtest(ds, [DC, exp_results_2], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [DC], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

a = BayesianFootball.BackTesting.generate_tearsheet(ledger)

c=unique(a.selection)

for cc in c
  show(subset(a, :selection => ByRow(isequal(cc))))
end


subset(ds.matches, :match_week => ByRow(isequal(35)), :tournament_id => ByRow(isequal(55)))


latents = Experiments.extract_oos_predictions(ds, mvpln_results)
latents_bp = Experiments.extract_oos_predictions(ds, BP_results)
latents_shp = Experiments.extract_oos_predictions(ds, shp_results)


a = Predictions.model_inference(latents)
b = Predictions.model_inference(latents_bp)
c = Predictions.model_inference(latents_shp)



daaa = Data.get_next_matches(ds, fs, cv_config)
mid = daaa[3,:match_id]
market_data = Data.prepare_market_data(ds)

subset(market_data.df, :match_id => ByRow(isequal(mid)))
subset(ds.matches, :match_id => ByRow(isequal(mid)))

using StatsPlots

sym = :draw
a1 = subset( a.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
b1 = subset( b.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
c1 = subset( c.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
mean(1 ./ a1.distribution)
mean(1 ./ b1.distribution)
mean(1 ./ c1.distribution)
mean(a1.distribution)
mean(b1.distribution)
mean(c1.distribution)
subset(market_data.df, :match_id => ByRow(isequal(mid)), :selection => ByRow(isequal(sym)))

density(a1.distribution, title="$sym",label="mvpln")
density!(b1.distribution, label="bi")
density!(c1.distribution, label="shp")




symbols = [:model_name, :model_parameters, :signal_name, :signal_params]

a = BayesianFootball.BackTesting.generate_tearsheet(ledger; groupby_cols=symbols)

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

