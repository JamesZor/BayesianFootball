# current_development/01_optimization_tests/l01_loaders.jl

using BayesianFootball

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Signals = BayesianFootball.Signals


function create_optim_test_task(ds::Data.DataStore; use_map=true)
    # ==========================================
    # 2. MODEL DEFINITION
    # ==========================================
    # Shared Component Configs
    inter_cfg = PreGame.GlobalInterception()
    disp_cfg  = PreGame.HomeAwayDispersion()
    ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
    kap_cfg   = PreGame.HierarchicalTeamKappa()

    # Bayesian Tracker for player ratings
    tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
    feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

    model = PreGame.DynamicMarketXGPlayerTimeDecayModel(
        interception_config  = inter_cfg,
        player_dynamics_config = PreGame.PositionalPlayerDynamics(days_half_life=180.0),
        dispersion_config    = disp_cfg,
        homeadvantage_config = ha_cfg,
        kappa_config         = kap_cfg,
        player_ratings_feature = feature_cfg_bayes,
        market_weight        = 1.0
    )

    # ==========================================
    # 2. SPLITTER DEFINITION
    # ==========================================
    # Using the modern GroupedCVConfig instead of RollingWindowSplitter
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = ["2025"], # Just testing 1 season
        history_seasons = 2,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = false
    )

    # ==========================================
    # 3. SAMPLER DEFINITION
    # ==========================================
    sampler_config = if use_map
        Samplers.MAPConfig(
            maxiters=1000, 
            show_progress=true
        )
    else
        Samplers.NUTSConfig(
            500,  # samples
            2,    # chains
            200,  # warmup
            0.65, # accept_rate
            10,   # max_depth
            Samplers.UniformInit(-2, 2),
            true  # show_progress
        )
    end
    
    # ==========================================
    # 4. TRAINING & EXPERIMENT DEFINITION
    # ==========================================
    train_cfg = Training.Independent(
        parallel = true,
        max_concurrent_splits = 8
    )
    
    training_config = Training.TrainingConfig(sampler_config, train_cfg, nothing, false)

    suffix = use_map ? "MAP" : "NUTS"
    config = Experiments.ExperimentConfig(
        name = "Optim_Test_$(suffix)",
        model = model, 
        splitter = cv_config,
        training_config = training_config,
        save_dir = "./data/experiments/tests"
    )

    return Experiments.ExperimentTask(ds, config)
end
