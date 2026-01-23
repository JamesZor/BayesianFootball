# exp/grw_basics/models.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)



using Statistics, Distributions

"""
    make_league_priors(df_train)

Calculates the 'Physics' of the league from raw data to set intelligent priors.
Returns NamedTuple with Normal distributions for μ and γ.
"""
function make_league_priors(df_train)
    # 1. Calculate Average Goals (The "Energy" of the league)
    # Total goals divided by total matches
    avg_goals_per_match = (sum(df_train.home_score) + sum(df_train.away_score)) / nrow(df_train)
    
    # 2. Calculate Home Advantage (Ratio)
    avg_h = mean(df_train.home_score)
    avg_a = mean(df_train.away_score)
    # Avoid division by zero in weird edge cases
    raw_home_adv = avg_a > 0 ? avg_h / avg_a : 1.3 

    println("\n--- ⚡ Data-Driven Priors Calculated ⚡ ---")
    println("  Avg Goals/Match: $(round(avg_goals_per_match, digits=3))")
    println("  Implied μ (team): $(round(log(avg_goals_per_match/2), digits=3))")
    println("  Home Adv Ratio:  $(round(raw_home_adv, digits=3))")

    # 3. Create Priors
    # We use log() because your model uses Log-Links
    # avg_goals_per_match = exp(μ_h) + exp(μ_a) ≈ 2 * exp(μ)
    # -> μ ≈ log(avg / 2)
    
    target_mu = log(avg_goals_per_match / 2.0)
    target_gamma = log(raw_home_adv)

    # We return Normal distributions centered on the truth, 
    # but with enough variance (0.2) to let the sampler adjust slightly.
    return (;
        prior_μ = Normal(target_mu, 0.05),
        prior_γ = Normal(target_gamma, 0.05)
    )
end






"""
    get_grw_basics_configs(; save_dir="./data/exp/grw_basics")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the GRW Basics experiment.
"""
function get_grw_basics_configs(; save_dir="./data/exp/grw_basics_pl_ch")
    
    # 1. Setup Data & Splits
    # ======================
    data_store = BayesianFootball.Data.load_default_datastore()
    ds = BayesianFootball.Data.DataStore( 
        Data.add_match_week_column(data_store.matches),
        data_store.odds,
        data_store.incidents
    )

    cv_config = BayesianFootball.Data.CVConfig(
        tournament_ids = [54,55],       # Premiership
        target_seasons = ["21/22", "22/23", "23/24", "24/25"],  # Target Season
        history_seasons = 0,
        dynamics_col = :match_week,
        warmup_period = 15,          # Long warmup for GRW
        stop_early = true
    )

    # 2. Setup Shared Training Config
    # ===============================
    # Note: We use a lightweight model just to build the vocabulary quickly
    vocab_model = BayesianFootball.Models.PreGame.StaticPoisson() 
    _ = BayesianFootball.Features.create_vocabulary(ds, vocab_model) 

    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        250,     # n_samples
        2,      # n_chains
        50,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-0.05, 0.05),
        false   # show_progress (We use the Global Logger instead)
    )

    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

# 3. Define Priors (Gelman's Boundary-Avoiding)
    # =============================================
    # Process Noise: Push away from 0, keep mode small (approx 0.05)
    # Meaning: "Teams change by ~5% per week, but almost never 0% and rarely >20%"
  #
      emperical_priors = make_league_priors(
              subset( ds.matches, :tournament_id => ByRow(x -> x ∈[54,55])) 
      )


    prior_σ_k = Gamma(2, 0.05) 
    
    # Initial Spread: Significant variation between teams at start of season
    prior_σ_0 = Gamma(2, 0.08)

    # Globals
  #
    prior_μ = emperical_priors.prior_μ,  # <--- INJECTED HERE
    prior_γ = emperical_priors.prior_γ,  # <--- INJECTED HERE

     # μ ~ Normal{Float64}(μ=0.3191977392306569, σ=0.05)
     # γ ~ Normal{Float64}(μ=0.12681652075405134, σ=0.05)

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
