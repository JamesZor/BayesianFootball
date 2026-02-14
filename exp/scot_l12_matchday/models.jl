
# exp/scot_l12_matchday/models.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
"""
    get_grw_basics_configs(; save_dir="./data/exp/grw_basics")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the GRW Basics experiment.
"""
function get_funnel_basics_configs(; save_dir="./data/exp/scot_l12_matchday")
    
    # 1. Setup Data & Splits
    # ======================
    ds = Data.load_extra_ds()

    # default 
    # transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
    # testing the grw GRWNegativeBinomialMu at every 2 weeks
    transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


    # League 1 cv config
    cv_config_l1 = BayesianFootball.Data.CVConfig(
        tournament_ids = [56],       # Premiership
        target_seasons = ["25/26"],  # Target Season
        history_seasons = 0,
        dynamics_col = :match_month,
        warmup_period = 7,      
        stop_early = false
    )
    # League 2 cv config
    cv_config_l2 = BayesianFootball.Data.CVConfig(
        tournament_ids = [57],       # Premiership
        target_seasons = ["25/26"],  # Target Season
        history_seasons = 0,
        dynamics_col = :match_month,
        warmup_period = 8,      
        stop_early = false
    )


    ## ----- change this to switch the leagues -- have to do it manual
    cv_config = cv_config_l1

    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        500,     # n_samples
        16,      # n_chains
        100,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-0.5, 0.5),
        :perchain   # show_progress (We use the Global Logger instead)
    )




    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    configs = [
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  1.---- funnel model ----
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Experiments.ExperimentConfig(
            name = "funnel_basic_l1",
            model = BayesianFootball.Models.PreGame.SequentialFunnelModel(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  2.---- baseline model ----
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
        # Experiments.ExperimentConfig(
        #     name = "grw_negbin_mu_l2",
        #     model = Models.PreGame.GRWNegativeBinomialMu(
        #         μ_init = Normal(0.20, 0.1),
        #         σ_μ    = Gamma(2, 0.015), 
        #         γ      = Normal(0.12, 0.5),
        #         σ_k    = Gamma(2, 0.08),
        #         σ_0    = Gamma(2, 0.08),
        #
        #         # Keep Dispersion loose as before
        #         log_r_prior = Normal(1.5, 1.0)
        #     ),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        #

      ]


    return ds, configs

end
