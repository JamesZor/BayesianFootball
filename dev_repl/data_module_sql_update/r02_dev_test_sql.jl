
using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)




data_store = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())



ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.statistics,
    data_store.odds,
    data_store.lineups,
    data_store.incidents
)



transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)




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

model = Models.PreGame.AblationStudy_NB_baseLine()



experiment_conf = Experiments.ExperimentConfig(
                    name = "test for sql",
                    model = model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp = Experiments.run_experiment(ds, experiment_conf)





