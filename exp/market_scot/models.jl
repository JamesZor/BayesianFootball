# exp/grw_basics/models.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
    
function get_grw_basics_configs(; save_dir="./data/exp/market_runs")
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
        history_seasons = 3,
        dynamics_col = :match_month,
        warmup_period = 8,
        stop_early = true
    )


    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        500,     # n_samples
        6,      # n_chains
        200,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-1, 1),
        false   # show_progress (We use the Global Logger instead)
    )




    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)



    configs = [
            Experiments.ExperimentConfig(
                name = "gamma_ha_pooled_multi",
                model = Models.PreGame.MSNegativeBinomialGamma(
                    # --- Globals ---
                    μ = Normal(0.20, 0.2),
                    γ = Normal(0.12, 0.5),      # Global Home Advantage
                    log_r = Normal(2.5, 0.5),   # Global Dispersion

                    # --- NEW: Team-Specific Home Advantage ---
                    # Gamma(2, 0.04) -> Mean = 0.08. 
                    σ_γ_team = Gamma(2, 0.04),  

                    # --- Boundary-Avoiding Priors: GRW Latent States ---
                    σ₀ = Gamma(2, 0.15),   # Mean = 0.30 (Initial spread of teams)
                    σₛ = Gamma(2, 0.04),   # Mean = 0.08 (Macro season jump)
                    σₖ = Gamma(2, 0.015),  # Mean = 0.03 (Micro monthly jump)

                    # --- Boundary-Avoiding Priors: Partial Pooling ---
                    σ_r_team  = Gamma(2, 0.10),   # Mean = 0.20 
                    σ_r_month = Gamma(2, 0.05),   # Mean = 0.10
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

