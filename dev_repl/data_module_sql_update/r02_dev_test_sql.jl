
# repl experiment test 


using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)




ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())

ds







cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["25/26"],
    history_seasons = 2,
    dynamics_col = :match_month,
    warmup_period = 5,
    stop_early = true
)

splits_l1 = BayesianFootball.Data.create_data_splits(ds, cv_config)

sampler_conf = Samplers.NUTSConfig(
    100,      # n_samples
    4,        # n_chains
    50,      # n_warmup
    0.65,     # accept_rate
    10,       # max_depth
    Samplers.UniformInit(-1, 1),
    false     # show_progress
)


train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)



data_splits = Data.create_data_splits(data_store, config.splitter)

model = Models.PreGame.AblationStudy_NB_baseLine()



experiment_conf = Experiments.ExperimentConfig(
                    name = "test for sql",
                    model = model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp = Experiments.run_experiment(ds, experiment_conf)





