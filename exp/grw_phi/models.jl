# exp/grw_phi/models.jl

using BayesianFootball
using Distributions # Required for prior definitions (Normal, Gamma, etc.)
"""
    get_grw_basics_configs(; save_dir="./data/exp/grw_basics")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the GRW Basics experiment.
"""
function get_grw_basics_configs(; save_dir="./data/exp/grw_phi")
    
    # 1. Setup Data & Splits
    # ======================
    data_store = BayesianFootball.Data.load_default_datastore()
    ds = BayesianFootball.Data.DataStore( 
        Data.add_match_week_column(data_store.matches),
        data_store.odds,
        data_store.incidents
    )

    cv_config = BayesianFootball.Data.CVConfig(
        tournament_ids = [56,57],       # Premiership
        target_seasons = ["20/21","21/22", "22/23", "23/24", "24/25"],  # Target Season
        history_seasons = 0,
        dynamics_col = :match_week,
        warmup_period = 5,          # Long warmup for GRW
        stop_early = true
    )

    # 2. Setup Shared Training Config
    # ===============================
    # Note: We use a lightweight model just to build the vocabulary quickly
    vocab_model = BayesianFootball.Models.PreGame.StaticPoisson() 
    _ = BayesianFootball.Features.create_vocabulary(ds, vocab_model) 

    # Shared Sampler Configuration
    sampler_conf = Samplers.NUTSConfig(
        300,     # n_samples
        2,      # n_chains
        100,     # n_warmup
        0.65,   # accept_rate
        10,     # max_depth
        Samplers.UniformInit(-0.05, 0.05),
        false   # show_progress (We use the Global Logger instead)
    )

    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
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

    log_r_init = Normal(1.5, 0.5)
    σ_0 = Truncated(Normal(0.5, 0.2), 0, Inf)

     # μ ~ Normal{Float64}(μ=0.3191977392306569, σ=0.05)
     # γ ~ Normal{Float64}(μ=0.12681652075405134, σ=0.05)
  
    ## 1. Define the Grid Dimensions
    # Stiff (0.01), Mid (0.05), Loose (0.10)
    priors_k = [0.01, 0.05, 0.10] # Team Ability Volatility
    priors_r = [0.01, 0.05, 0.10] # Chaos (Dispersion) Volatility

    # 4. Define Models
    # ================
    configs = [
        Experiments.ExperimentConfig(
            name = "base_grw_neg_bin",
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
    ]

    for k in priors_k
        for r in priors_r
            # Create a readable name: "phi_k05_r10" means k=0.05, r=0.10
            k_str = lpad(Int(k*100), 2, "0")
            r_str = lpad(Int(r*100), 2, "0")
            
            push!(configs, Experiments.ExperimentConfig(
                name = "phi_k$(k_str)_r$(r_str)",
                model = Models.PreGame.GRWNegativeBinomialPhi(
                    # The Grid Variables
                    σ_k = Gamma(2, k),
                    σ_r = Gamma(2, r),
                    
                    # The Standard Fixed Priors
                    μ = prior_μ,
                    γ = prior_γ,
                    log_r_init = Normal(1.5, 0.5),
                    σ_0 = Truncated(Normal(0.5, 0.2), 0, Inf)
                ),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
            ))
        end
    end


    return ds, configs
end
