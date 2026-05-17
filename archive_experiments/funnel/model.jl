
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
    ds = Data.load_extra_ds()

    # default 
    # transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
    # testing the grw GRWNegativeBinomialMu at every 2 weeks
    transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)



    cv_config = BayesianFootball.Data.CVConfig(
        tournament_ids = [56,57],       # Premiership
        target_seasons = ["20/21","21/22","22/23", "23/24", "24/25", "25/26"],  # Target Season
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
        Samplers.UniformInit(-0.5, 0.5),
        false   # show_progress (We use the Global Logger instead)
    )




    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    configs = [
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  1.---- first model ----
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #   Was an init test  run 
    #
        # Experiments.ExperimentConfig(
        #     name = "funnel basic 1",
        #     model = BayesianFootball.Models.PreGame.SequentialFunnelModel(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
    #

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  2.---- baseline model ----
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # It was found that GRWNegativeBinomialMu, had the best expected growth for over 15 and couple of 
        # other markets - so will run this a baseline check, monthly as well to see the difference.
        
        # Experiments.ExperimentConfig(
        #     name = "grw_neg_bin_mu_base_line",
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

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  4.---- Re run but only for season 24/25, 25/26  ----
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #   at two week Δt 
    #
        Experiments.ExperimentConfig(
            name = "funnel_s20_25",
            model = BayesianFootball.Models.PreGame.SequentialFunnelModel(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  2.---- baseline model ----
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # It was found that GRWNegativeBinomialMu, had the best expected growth for over 15 and couple of 
        # other markets - so will run this a baseline check, monthly as well to see the difference.
        
        Experiments.ExperimentConfig(
            name = "grw_neg_bin_mu_base_s20_s25",
            model = Models.PreGame.GRWNegativeBinomialMu(
                μ_init = Normal(0.20, 0.1),
                σ_μ    = Gamma(2, 0.015), 
                γ      = Normal(0.12, 0.5),
                σ_k    = Gamma(2, 0.08),
                σ_0    = Gamma(2, 0.08),

                # Keep Dispersion loose as before
                log_r_prior = Normal(2.5, 0.5)
            ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),


      ]


    return ds, configs

end
