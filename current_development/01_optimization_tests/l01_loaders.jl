# current_development/01_optimization_tests/l01_loaders.jl

using BayesianFootball

function setup_experiment_config(; use_map=true)
    # 1. Define Model
    model = Models.PreGame.DynamicGoalsModel(
        homeadvantage_config = Models.PreGame.HierarchicalHomeAdvantage(),
        dispersion_config = Models.PreGame.TeamLevelDispersion(),
        player_dynamics_config = Models.PreGame.GRWPoisson(
            σ_att_0 = 0.5, σ_def_0 = 0.5, 
            σ_att   = 0.1, σ_def   = 0.1
        )
    )

    # 2. Define Splitter
    splitter = Data.RollingWindowSplitter(
        train_window = Month(12),
        test_window = Week(1),
        dynamics_col = :week,
        step = Week(1)
    )

    # 3. Define Training Config
    sampler_config = if use_map
        Samplers.MAPConfig(
            maxiters=1000, 
            show_progress=true
        )
    else
        Samplers.NUTSConfig(
            n_samples=500, 
            n_warmup=200, 
            n_chains=2
        )
    end
    
    training_config = Training.TrainingConfig(
        sampler = sampler_config,
        strategy = Training.Independent()
    )

    # 4. Define Experiment Config
    suffix = use_map ? "MAP" : "NUTS"
    exp_config = Experiments.ExperimentConfig(
        name = "Optim_Test_$(suffix)",
        model = model,
        splitter = splitter,
        training_config = training_config,
        save_dir = "./data/experiments/tests"
    )
    
    return exp_config
end
