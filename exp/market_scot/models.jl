# exp/grw_basics/models.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
"""
    get_grw_basics_configs(; save_dir="./data/exp/grw_basics")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the GRW Basics experiment.
"""
function get_grw_basics_configs(; save_dir="./data/exp/market_runs")
    
    # 1. Setup Data & Splits
    # ======================
    data_store = BayesianFootball.Data.load_default_datastore()
    ds = BayesianFootball.Data.DataStore( 
        Data.add_match_week_column(data_store.matches),
        data_store.odds,
        data_store.incidents
    )

    cv_config = BayesianFootball.Data.CVConfig(
        tournament_ids = [57],       # Premiership
        target_seasons = ["25/26"],  # Target Season
        history_seasons = 0,
        dynamics_col = :match_week,
        warmup_period = 24,          # Long warmup for GRW
        stop_early = false
    )

    # 2. Setup Shared Training Config
    # ===============================
    # Note: We use a lightweight model just to build the vocabulary quickly
    vocab_model = BayesianFootball.Models.PreGame.StaticPoisson() 
    _ = BayesianFootball.Features.create_vocabulary(ds, vocab_model) 

    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        300,     # n_samples
        8,      # n_chains
        100,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-0.05, 0.05),
        :perchain   # show_progress (We use the Global Logger instead)
    )

    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

# 3. Define Priors (Gelman's Boundary-Avoiding)
    # =============================================
    # Process Noise: Push away from 0, keep mode small (approx 0.05)
    # Meaning: "Teams change by ~5% per week, but almost never 0% and rarely >20%"

    prior_σ_k = Gamma(2, 0.05) 
    # Initial Spread: Significant variation between teams at start of season
    prior_σ_0 = Gamma(2, 0.08)

    # Globals
    prior_μ = Normal(0.32, 0.05)        # Global Baseline
    prior_γ = Normal(0.12, 0.05)   # Home Advantage

    prior_δ = Normal(0.0, 0.4)


    # 4. Define Models
    # ================
    configs = [
        # Experiments.ExperimentConfig(
        #     name = "grw_neg_bin_mu_wk25_l1",
        #     model = Models.PreGame.GRWNegativeBinomialMu(
        #         μ_init = Normal(0.20, 0.1),
        #         σ_μ    = Gamma(2, 0.015), 
        #         γ      = prior_γ,
        #         σ_k    = prior_σ_k,
        #         σ_0    = prior_σ_0,
        #
        #         # Keep Dispersion loose as before
        #         log_r_prior = Normal(1.5, 1.0)
        #     ),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        # Experiments.ExperimentConfig(
        #     name = "grw_bivariate_poisson",
        #     model = Models.PreGame.GRWBivariatePoisson(
        #         μ = prior_μ,
        #         γ = prior_γ,
        #         σ_k = prior_σ_k,
        #         σ_0 = prior_σ_0,
        #         # Using default ρ = Normal(-2, 1.0) for covariance
        #     ),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # )
    #
        Experiments.ExperimentConfig(
                    name = "grw_neg_bin_full_wk27_l1",
                    model = Models.PreGame.GRWNegativeBinomialFull(
                        # --- 1. Dynamic Global Baseline ---
                        μ_init = prior_μ,              # Normal(0.32, 0.05)
                        σ_μ    = Gamma(2, 0.015),      # Process noise for league average (Small)

                        # --- 2. Home Advantage ---
                        γ = prior_γ,                   # Normal(0.12, 0.05)

                        # --- 3. Hierarchical Dispersion (r) ---
                        log_r_global = Normal(1.5, 0.5),
                        
                        # We use a fixed prior for the team offsets now (removed the hierarchical std).
                        # Normal(0, 0.5) allows r to vary by factor of ~1.6x between teams (e^0.5).
                        δ_r = Normal(0, 1),   

                        # --- 4. Hierarchical Process Noise (Volatility) ---
                        # Baselines targeting ~0.05
                        log_σ_att_global = Normal(-3.0, 0.5),
                        log_σ_def_global = Normal(-3.0, 0.5),

                        # Team Deviations (using your defined prior_δ)
                        δ_σ_att = prior_δ,             # Normal(0.0, 0.4)
                        δ_σ_def = prior_δ,             # Normal(0.0, 0.4)
                        
                        # --- 5. Initial Spread ---
                        σ_0 = prior_σ_0                # Gamma(2, 0.08)
                    ),
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir = save_dir
                )
            ]

    return ds, configs
end
