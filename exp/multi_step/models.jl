
# exp/multi_step/model.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
"""
    get_grw_basics_configs(; save_dir="./data/exp/grw_basics")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the GRW Basics experiment.
"""
function get_basics_configs(; save_dir="./data/exp/multi_step")
    
    # 1. Setup Data & Splits
    # ======================
    ds = Data.load_extra_ds()

    # default 
    # transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
    # testing the grw GRWNegativeBinomialMu at every 2 weeks
    transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [[56, 57]],
        target_seasons = ["22/23", "23/24", "24/25", "25/26"],
        history_seasons = 2,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = true
    )



    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        400,     # n_samples
        2,      # n_chains
        100,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-1, 1),
        false   # show_progress (We use the Global Logger instead)
    )




    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)



    configs = [

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #  1.---- baseline model ----
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Experiments.ExperimentConfig(
            #     name = "base_line_multi",
            #     model = Models.PreGame.MSNegativeBinomial(
            #         μ = Normal(0.20, 0.2),
            #         γ = Normal(0.12, 0.5),
            #         log_r = Normal(2.5, 0.5),
            #
            #         # Boundary-Avoiding Gamma(2, θ) Priors
            #         # Mean = 2 * θ. We match these to the summary stats we just found.
            #         σ₀ = Gamma(2, 0.15),   # Mean = 0.30 (Initial spread of teams)
            #         σₛ = Gamma(2, 0.04),   # Mean = 0.08 (Macro season jump)
            #         σₖ = Gamma(2, 0.015)   # Mean = 0.03 (Micro monthly jump)
            #     ),
            #     splitter = cv_config,
            #     training_config = training_config,
            #     save_dir = save_dir
            # ),
            #
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #  2.---- model with month and midweek    ----
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Experiments.ExperimentConfig(
            #     name = "delta_multi_calendar", # Renamed so it doesn't overwrite the first!
            #     model = Models.PreGame.MSNegativeBinomialDelta(
            #         μ = Normal(0.20, 0.2),
            #         γ = Normal(0.12, 0.5),
            #         log_r = Normal(2.5, 0.5),
            #
            #         # Exact same boundary-avoiding priors for the Walk
            #         σ₀ = Gamma(2, 0.15),   
            #         σₛ = Gamma(2, 0.04),   
            #         σₖ = Gamma(2, 0.015),  
            #
            #         # Weakly informative priors for the calendar effects
            #         δₘ = Normal(0, 0.1),
            #         δₙ = Normal(0, 0.1)
            #     ),
            #     splitter = cv_config,
            #     training_config = training_config,
            #     save_dir = save_dir
            # ),
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #  2.---- Rho Pooled Model (Months, Teams, Dispersion) ----
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            Experiments.ExperimentConfig(
                name = "rho_pooled_multi",
                model = Models.PreGame.MSNegativeBinomialRho(
                    # --- Globals ---
                    μ = Normal(0.20, 0.2),
                    γ = Normal(0.12, 0.5),
                    log_r = Normal(2.5, 0.5),

                    # --- Boundary-Avoiding Priors: GRW Latent States ---
                    σ₀ = Gamma(2, 0.15),   # Mean = 0.30 (Initial spread of teams)
                    σₛ = Gamma(2, 0.04),   # Mean = 0.08 (Macro season jump)
                    σₖ = Gamma(2, 0.015),  # Mean = 0.03 (Micro monthly jump)

                    # --- Boundary-Avoiding Priors: Partial Pooling ---
                    # Team dispersion varies a bit (e.g., chaotic Leeds vs rigid Burnley)
                    σ_r_team  = Gamma(2, 0.10),   # Mean = 0.20 
                    
                    # Month dispersion is likely much tighter
                    σ_r_month = Gamma(2, 0.05),   # Mean = 0.10
                    
                    # Month expected goal effect (λ modifier). In your summary, 
                    # these hovered between -0.08 and +0.12, so a mean σ of 0.1 is perfect.
                    σ_δₘ      = Gamma(2, 0.05),   # Mean = 0.10 

                    # --- Fixed Effects ---
                    δₙ = Normal(0, 0.1), # Midweek
                    δₚ = Normal(0, 0.1)  # Plastic pitch
                ),
                splitter = cv_config,
                training_config = training_config,
                save_dir = save_dir
            ),




          ]

        return ds, configs
end

