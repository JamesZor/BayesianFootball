# current_development/01_optimization_tests/l03_grid_search_2d_loaders.jl

using BayesianFootball
using DataFrames

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Signals = BayesianFootball.Signals

"""
    create_grid_search_2d_task(ds::Data.DataStore; days_half_life::Float64, market_weight::Float64, use_map=true, use_mle=false)

Creates an ExperimentTask for a specific days_half_life and market_weight parameter using MAP or MLE optimization.
"""
function create_grid_search_2d_task(
    ds::Data.DataStore; 
    days_half_life::Float64, 
    market_weight::Float64, 
    use_map=true, 
    use_mle=false
)
    # ==========================================
    # 1. MODEL DEFINITION
    # ==========================================
    inter_cfg = PreGame.GlobalInterception()
    disp_cfg  = PreGame.HomeAwayDispersion()
    ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
    kap_cfg   = PreGame.HierarchicalTeamKappa()

    tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
    feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

    model = PreGame.DynamicMarketXGPlayerTimeDecayModel(
        interception_config  = inter_cfg,
        player_dynamics_config = PreGame.PositionalPlayerDynamics(days_half_life=days_half_life),
        dispersion_config    = disp_cfg,
        homeadvantage_config = ha_cfg,
        kappa_config         = kap_cfg,
        player_ratings_feature = feature_cfg_bayes,
        market_weight        = market_weight
    )

    # ==========================================
    # 2. SPLITTER DEFINITION
    # ==========================================
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = ["2025","2026"], 
        history_seasons = 3,
        dynamics_col = :match_week,
        warmup_period = 0,
        stop_early = true
    )

    # ==========================================
    # 3. SAMPLER DEFINITION
    # ==========================================
    sampler_config = if use_mle
        Samplers.MLEConfig(
            maxiters=1000, 
            show_progress=false
        )
    else
        Samplers.MAPConfig(
            maxiters=1000, 
            show_progress=false
        )
    end
    
    # ==========================================
    # 4. TRAINING & EXPERIMENT DEFINITION
    # ==========================================
    train_cfg = Training.Independent(
        parallel = true,
        max_concurrent_splits = 16
    )
    
    training_config = Training.TrainingConfig(sampler_config, train_cfg, nothing, false)

    method_str = use_mle ? "MLE" : "MAP"
    config = Experiments.ExperimentConfig(
        name = "GridSearch2D_$(method_str)_hl_$(days_half_life)_mw_$(market_weight)",
        model = model, 
        splitter = cv_config,
        training_config = training_config,
        save_dir = "./data/experiments/tests"
    )

    return Experiments.ExperimentTask(ds, config)
end

"""
    get_param_value(chains_df::DataFrame, sym_name::Symbol)

Extracts a parameter's value from the chains DataFrame.
"""
function get_param_value(chains_df::DataFrame, sym_name::Symbol)
    row = filter(r -> r.raw_symbol == sym_name, chains_df)
    if nrow(row) > 0
        return row.mean[1]
    else
        row_str = filter(r -> string(r.raw_symbol) == string(sym_name), chains_df)
        if nrow(row_str) > 0
            return row_str.mean[1]
        end
    end
    return NaN
end

"""
    run_grid_search_2d(ds::Data.DataStore, half_lives::Vector{Float64}, market_weights::Vector{Float64}; use_map=true, use_mle=false)

Runs a 2D grid search over the specified days_half_life and market_weight values.
Returns a tuple of:
1. A summary DataFrame with columns: days_half_life, market_weight, logloss_model, n_obs, ν_xg, etc.
2. A dictionary mapping (days_half_life, market_weight) => experiment results.
"""
function run_grid_search_2d(
    ds::Data.DataStore, 
    half_lives::Vector{Float64}, 
    market_weights::Vector{Float64}; 
    use_map=true, 
    use_mle=false
)
    results_dict = Dict{Tuple{Float64, Float64}, Any}()
    
    println("\n============================================================")
    println(" 🔍 Starting 2D Grid Search...")
    println("   > days_half_life: ", half_lives)
    println("   > market_weight:  ", market_weights)
    println("============================================================")
    
    total_runs = length(half_lives) * length(market_weights)
    run_counter = 1
    
    for hl in half_lives
        for mw in market_weights
            println("\n>>> [$(run_counter)/$(total_runs)] Evaluating hl=$(hl), mw=$(mw) ...")
            task = create_grid_search_2d_task(ds; days_half_life=hl, market_weight=mw, use_map=use_map, use_mle=use_mle)
            
            exp_time = @elapsed begin
                res = Experiments.run_experiment(task)
            end
            println("    Completed in $(round(exp_time, digits=2)) seconds.")
            
            results_dict[(hl, mw)] = res
            run_counter += 1
        end
    end
    
    println("\n============================================================")
    println(" 📊 Evaluating all grid points individually...")
    println("============================================================")
    
    metrics = [Evaluation.LogLoss()]
    summary_rows = []
    
    # Collect and sort keys for consistent reporting
    grid_keys = sort(collect(keys(results_dict)), by = x -> (x[1], x[2]))
    
    for (hl, mw) in grid_keys
        res = results_dict[(hl, mw)]
        
        # Run evaluation individually
        local eval_df = DataFrame()
        local n_joined_obs = 0
        
        try
            eval_df = Evaluation.evaluate_experiments(metrics, [res], ds)
        catch e
            @warn "Failed during evaluation of hl=$(hl), mw=$(mw): $e"
        end
        
        logloss_model = NaN
        if nrow(eval_df) > 0
            r = eval_df[1, :]
            logloss_model = :logloss_overall_model_ll in propertynames(eval_df) ? r.logloss_overall_model_ll : NaN
            n_joined_obs = :logloss_overall_n_obs in propertynames(eval_df) ? r.logloss_overall_n_obs : 0
        end
        
        # Extract parameter estimates
        local ν_xg_val = NaN
        local σ_market_val = NaN
        local ha_σ_val = NaN
        local kap_σ_val = NaN
        local lp_val = NaN
        
        try
            chains = Experiments.Diagnostics.extract_chains(ds, res)
            ν_xg_val = get_param_value(chains.df, :ν_xg)
            σ_market_val = get_param_value(chains.df, :σ_market)
            ha_σ_val = get_param_value(chains.df, Symbol("ha.σ_γ"))
            kap_σ_val = get_param_value(chains.df, Symbol("kap.σ_κ"))
            lp_val = get_param_value(chains.df, :lp)
        catch e
            # Silent fallback
        end
        
        push!(summary_rows, (
            days_half_life = hl,
            market_weight = mw,
            logloss_model = logloss_model,
            n_obs = n_joined_obs,
            ν_xg = ν_xg_val,
            σ_market = σ_market_val,
            ha_σ_γ = ha_σ_val,
            kap_σ_κ = kap_σ_val,
            lp = lp_val
        ))
    end
    
    summary_df = DataFrame(summary_rows)
    # Sort primarily by logloss (ascending - lower is better)
    sort!(summary_df, :logloss_model)
    
    return summary_df, results_dict
end
