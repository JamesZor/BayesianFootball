# current_development/ab_test_outfield_player/r04_ab_test_double_poisson_market_vs_no_market.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

# Pin threads for maximum performance
pinthreads(:cores)

function main()
    println("=========================================================")
    println(" A/B TEST: Double Poisson (Market vs No Market)")
    println("=========================================================")
    
    # 1. Setup DataStore for Ireland (2025-2026 for rich history)
    println("[1] Loading DataStore...")
    ds = Data.load_datastore_cached(Data.Ireland())
    
    # 2. Define Shared Architecture Components
    inter_config = Models.PreGame.HierarchicalInterception()
    ha_config = Models.PreGame.HierarchicalTeamHomeAdvantage()
    kappa_config = Models.PreGame.HierarchicalTeamKappa()
    p_dyn_config = Models.PreGame.OutfieldPlayerDynamicsConfig(
        days_half_life = 180.0
    )
    
    # Use the established BayesianTracker for player ratings
    tracker = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
    ratings_feature = Features.PlayerRatingsFeature(tracker)
    
    # 3. Model A: Double Poisson WITHOUT Market Data
    println("[2] Initializing Model A (No Market)...")
    model_a_no_market = Models.PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel(
        interception_config = inter_config,
        player_dynamics_config = p_dyn_config,
        dispersion_config = Models.PreGame.HomeAwayDispersion(), # Unused but required by interface
        homeadvantage_config = ha_config,
        kappa_config = kappa_config,
        player_ratings_feature = ratings_feature,
        ν_xg = truncated(Normal(3.0, 0.5), lower=0.5)
    )

    # 4. Model B: Double Poisson WITH Market Data
    println("[3] Initializing Model B (With Market)...")
    model_b_market = Models.PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
        interception_config = inter_config,
        player_dynamics_config = p_dyn_config,
        dispersion_config = Models.PreGame.HomeAwayDispersion(), # Unused but required by interface
        homeadvantage_config = ha_config,
        kappa_config = kappa_config,
        player_ratings_feature = ratings_feature,
        market_feature_config = Features.DoublePoissonMarketFeature(),
        ν_xg = truncated(Normal(3.0, 0.5), lower=0.5),
        market_σ = truncated(Normal(0.1, 0.2), lower=0.01),
        market_weight = 0.4
    )

    # 5. Define Shared Splitter & Training Config
    splitter = Data.GroupedCVSplitter(
        ds; 
        target_seasons=[2025, 2026],
        group_by=:match_month,
        history_depth=2,
        min_warmup_matches=5
    )

    training_config = Training.TrainingConfig(
        strategy = Training.IndependentStrategy(
            max_concurrent_tasks = 8, 
            sampler = Samplers.QueuedNUTSConfig(
                samples=1000, 
                warmup=500, 
                chains=16, 
                max_cpu_threads=128
            )
        )
    )

    # 6. Create Tasks
    task_a = Experiments.create_experiment_task(
        ds, model_a_no_market, splitter,
        experiment_name = "ab_dp_no_market",
        training_config = training_config,
        save_dir = "./tmp_mcmc_checkpoints"
    )

    task_b = Experiments.create_experiment_task(
        ds, model_b_market, splitter,
        experiment_name = "ab_dp_market",
        training_config = training_config,
        save_dir = "./tmp_mcmc_checkpoints"
    )

    # 7. Execute Tasks Concurrently (Queue-based)
    println("\n[4] Starting Concurrent A/B Execution...")
    
    # We run them sequentially so we don't blow up RAM, but the sampler itself uses queues
    println("--- Running Model A (No Market) ---")
    results_a = Experiments.run_experiment(task_a)
    Experiments.save_experiment(results_a)
    
    println("--- Running Model B (Market) ---")
    results_b = Experiments.run_experiment(task_b)
    Experiments.save_experiment(results_b)

    println("\n[5] A/B Test Complete!")
    println("Results saved to ./tmp_mcmc_checkpoints/")
end

main()
