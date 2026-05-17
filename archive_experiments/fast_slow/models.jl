using BayesianFootball
using Distributions

function get_fast_slow_simple_configs(; save_dir="./data/exp/fast_slow_simple")
    
    # 1. Setup Data & Splits (Same as before)
    data_store = BayesianFootball.Data.load_default_datastore()
    ds = BayesianFootball.Data.DataStore( 
        Data.add_match_week_column(data_store.matches),
        data_store.odds,
        data_store.incidents
    )

    cv_config = BayesianFootball.Data.CVConfig(
        tournament_ids = [56,57],  # Premiership
        target_seasons = ["21/22", "22/23", "23/24", "24/25"], 
        history_seasons = 0,
        dynamics_col = :match_week,
        warmup_period = 15,
        stop_early = true
    )

    # Standard Shared Configs
    vocab_model = BayesianFootball.Models.PreGame.StaticPoisson() 
    _ = BayesianFootball.Features.create_vocabulary(ds, vocab_model) 

    sampler_conf = Samplers.NUTSConfig(300, 2, 100, 0.65, 10, Samplers.UniformInit(-0.05, 0.05), false)
    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    # --- 2. Define The Priors ---
    
    # Shared Priors
    prior_μ = Normal(0.32, 0.05)
    prior_γ = Normal(0.12, 0.05)
    prior_σ_0 = Gamma(2, 0.08)       # Initial spread (Class difference at start)
    prior_log_r = Normal(1.5, 0.5)   # Standard Dispersion

    # SLOW Priors (Conservative)
    # Mean ~0.005. Very tight. 
    # This prevents the model from overreacting to a few bad games.
    prior_σ_slow = Truncated(Normal(0, 0.005), 0, Inf)

    # FAST Priors (Reactive)
    # Mean ~0.08. Loose. 
    # This allows the model to chase streaks
    prior_σ_fast = Truncated(Normal(0, 0.08), 0, Inf)

    # --- 3. Define Models ---
    configs = [
        # Model A: SLOW (The Anchor)
        Experiments.ExperimentConfig(
            name = "grw_simple_slow",
            model = Models.PreGame.GRWNegativeBinomial(
                μ = prior_μ,
                γ = prior_γ,
                log_r_prior = prior_log_r,
                σ_0 = prior_σ_0,
                
                # THE KEY CHANGE: Tight Process Noise
                σ_k = prior_σ_slow
            ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),

        # Model B: FAST (The Spark)
        Experiments.ExperimentConfig(
            name = "grw_simple_fast",
            model = Models.PreGame.GRWNegativeBinomial(
                μ = prior_μ,
                γ = prior_γ,
                log_r_prior = prior_log_r,
                σ_0 = prior_σ_0,
                
                # THE KEY CHANGE: Loose Process Noise
                σ_k = prior_σ_fast
            ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        )
    ]

    return ds, configs
end
