# exp/grw_basics/models.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
"""
    get_grw_basics_configs(; save_dir="./data/exp/grw_basics")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the GRW Basics experiment.
"""
function get_grw_basics_configs(; save_dir="./data/exp/grw_basics")
    
    # 1. Setup Data & Splits
    # ======================
    data_store = BayesianFootball.Data.load_default_datastore()
    ds = BayesianFootball.Data.DataStore( 
        Data.add_match_week_column(data_store.matches),
        data_store.odds,
        data_store.incidents
    )

    cv_config = BayesianFootball.Data.CVConfig(
        tournament_ids = [56],       # Premiership
        target_seasons = ["22/23"],  # Target Season
        history_seasons = 0,
        dynamics_col = :match_week,
        warmup_period = 35,          # Long warmup for GRW
        stop_early = true
    )

    # 2. Setup Shared Training Config
    # ===============================
    # Note: We use a lightweight model just to build the vocabulary quickly
    vocab_model = BayesianFootball.Models.PreGame.StaticPoisson() 
    _ = BayesianFootball.Features.create_vocabulary(ds, vocab_model) 

    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        20,     # n_samples
        2,      # n_chains
        20,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-0.05, 0.05),
        false   # show_progress (We use the Global Logger instead)
    )

    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

# 3. Define Priors (Gelman's Boundary-Avoiding)
    # =============================================
    # Process Noise: Push away from 0, keep mode small (approx 0.05)
    # Meaning: "Teams change by ~5% per week, but almost never 0% and rarely >20%"
    prior_σ_k = Gamma(2, 0.05) 
    
    # Initial Spread: Significant variation between teams at start of season
    prior_σ_0 = Gamma(2, 0.4)

    # Globals
    prior_μ = Normal(0.2, 0.5)        # Global Baseline
    prior_γ = Normal(log(1.3), 0.2)   # Home Advantage

    # 4. Define Models
    # ================
    configs = [
        Experiments.ExperimentConfig(
            name = "grw_poisson",
            model = Models.PreGame.GRWPoisson(
                μ = prior_μ,
                γ = prior_γ,
                σ_k = prior_σ_k, 
                σ_0 = prior_σ_0
            ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
        Experiments.ExperimentConfig(
            name = "grw_dixon_coles",
            model = Models.PreGame.GRWDixonColes(
                μ = prior_μ,
                γ = prior_γ,
                σ_k = prior_σ_k,
                σ_0 = prior_σ_0
                # Using default ρ_raw = Normal(0,1)
            ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
        Experiments.ExperimentConfig(
            name = "grw_neg_bin",
            model = Models.PreGame.GRWNegativeBinomial(
                μ = prior_μ,
                γ = prior_γ,
                σ_k = prior_σ_k,
                σ_0 = prior_σ_0
                # Using default log_r_prior = Normal(1.5, 1.0)
            ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
        Experiments.ExperimentConfig(
            name = "grw_bivariate_poisson",
            model = Models.PreGame.GRWBivariatePoisson(
                μ = prior_μ,
                γ = prior_γ,
                σ_k = prior_σ_k,
                σ_0 = prior_σ_0,
                # Using default ρ = Normal(-2, 1.0) for covariance
            ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        )
    ]

    return ds, configs
end
