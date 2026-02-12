
# exp/funnel/model.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
"""
    get_grw_basics_configs(; save_dir="./data/exp/grw_basics")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the GRW Basics experiment.
"""
function get_funnel_basics_configs(; save_dir="./data/exp/funnel_basics")
    
    # 1. Setup Data & Splits
    # ======================
    data_store = BayesianFootball.Data.load_default_datastore()
    ds = BayesianFootball.Data.DataStore( 
        Data.add_match_week_column(data_store.matches),
        data_store.odds,
        data_store.incidents
    )
    transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

    cv_config = BayesianFootball.Data.CVConfig(
        tournament_ids = [56,57],       # Premiership
        target_seasons = ["21/22", "22/23", "23/24", "24/25"],  # Target Season
        history_seasons = 0,
        dynamics_col = :match_month,
        warmup_period = 2,      
        stop_early = true
    )


    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        500,     # n_samples
        2,      # n_chains
        100,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-0.05, 0.05),
        false   # show_progress (We use the Global Logger instead)
    )

    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    configs = [
        #---- first model ----
        Experiments.ExperimentConfig(
            name = "funnel basic 1",
            model = BayesianFootball.Models.PreGame.SequentialFunnelModel(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
      ]


    return ds, configs

end
