# exp/ablation_study/models.jl

using BayesianFootball
using Distributions

"""
    get_ablation_configs(; save_dir="./data/exp/ablation_study")

Returns a tuple of (DataStore, Vector{ExperimentConfig}) for the 7-step GRW Negative Binomial Ablation Study.
"""
function get_ablation_configs(; save_dir="./data/exp/ablation_study")
    
    # 1. Setup Data & Splits
    # ======================
    ds = Data.load_extra_ds()

    # Generate the month index required by the time-varying models
    transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [[56, 57]], # e.g., English Premier League & Championship
        # target_seasons = ["22/23", "23/24", "24/25", "25/26"],
        target_seasons = ["22/23"],
        history_seasons = 2,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = true
    )

    # Shared Sampler Configuration (NUTS)
    sampler_conf = Samplers.NUTSConfig(
        10,      # n_samples
        2,        # n_chains
        20,      # n_warmup
        0.65,     # accept_rate
        10,       # max_depth
        Samplers.UniformInit(-1, 1),
        false     # show_progress
    )

    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    # 2. Build the 7 Ablation Configurations
    # ======================================
    configs = [
        # --- Model 1: Vanilla Baseline ---
        Experiments.ExperimentConfig(
            name = "01_ablation_baseline",
            model = Models.PreGame.AblationStudy_NB_baseLine(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),

        # --- Model 2: Environment (Midweek + Plastic) ---
        Experiments.ExperimentConfig(
            name = "02_ablation_env",
            model = Models.PreGame.AblationStudy_NB_env(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),

        # --- Model 3: The Fortress (Team-Specific Home Advantage) ---
        Experiments.ExperimentConfig(
            name = "03_ablation_home_hierarchy",
            model = Models.PreGame.AblationStudy_NB_home_hierarchy(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),

        # --- Model 4: Lean Target (Team HA + Team Dispersion) ---
        Experiments.ExperimentConfig(
            name = "04_ablation_team_dispersion",
            model = Models.PreGame.AblationStudy_NB_team_dispersion(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),

        # --- Model 5: Baseline + Monthly Expectations ---
        Experiments.ExperimentConfig(
            name = "05_ablation_month_mu",
            model = Models.PreGame.AblationStudy_NB_baseline_month_mu(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),

        # --- Model 6: Baseline + Monthly Dispersion ---
        Experiments.ExperimentConfig(
            name = "06_ablation_month_r",
            model = Models.PreGame.AblationStudy_NB_baseline_month_r(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),

        # --- Model 7: The Kitchen Sink (Everything) ---
        Experiments.ExperimentConfig(
            name = "07_ablation_kitchen_sink",
            model = Models.PreGame.AblationStudy_NB_KitchenSink(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        )
    ]

    return ds, configs
end
