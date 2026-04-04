# exp/grw_basics/models.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
    
function get_grw_basics_configs(; save_dir="./data/exp/market_runs/april")
    # 1. Setup Data & Splits
    # ======================
    ds = Data.load_extra_ds()

    # default 
    # transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
    # testing the grw GRWNegativeBinomialMu at every 2 weeks
    transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [[56, 57]],
        target_seasons = ["25/26"],
        history_seasons = 2,
        dynamics_col = :match_month,
        warmup_period = 9,
        stop_early = false
    )


    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        300,     # n_samples
        16,      # n_chains
        150,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-1, 1),
        :perchain   # show_progress (We use the Global Logger instead)
    )




    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)



    configs = [

        Experiments.ExperimentConfig(
            name = "06_02_ablation_month_r",
            model = Models.PreGame.AblationStudy_NB_baseline_month_r(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
          ]

        return ds, configs



end

