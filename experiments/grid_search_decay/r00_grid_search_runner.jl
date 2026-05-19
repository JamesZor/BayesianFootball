using BayesianFootball
using Revise
using DataFrames
using ThreadPinning
pinthreads(:cores)
using Dates

include("./l00_grid_search_loader.jl")
const PreGame = BayesianFootball.Models.PreGame
const Evaluation = BayesianFootball.Evaluation

# ==============================================================================
# 0. SETTINGS & DATA
# ==============================================================================
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/experiments/grid_search_decay/"
target_seasons = ["2025"]

# ==============================================================================
# 1. SETUP THE GRID
# ==============================================================================
# Define the half-lives and market weights we want to test
half_lives_to_test = [7, 14, 30, 90, 180, 365, 400, 500]
market_weights_to_test = [0.1, 0.25, 0.5, 0.75, 1.0]

# Base component configs
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

all_tasks = ExperimentTask[]

for hl in half_lives_to_test
    for mw in market_weights_to_test
        
        # A. Create Dynamics Config for this half-life
        dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = hl)
        
        # B. Define Model 1: Market Goals Time Decay
        model_goals = PreGame.DynamicMarketGoalsTimeDecayModel(
            interception_config  = inter_cfg,
            dynamics_config      = dyn_cfg,
            dispersion_config    = disp_cfg,
            homeadvantage_config = ha_cfg,
            market_weight        = mw
        )
        
        # C. Define Model 2: Market XG Time Decay
        model_xg = PreGame.DynamicMarketXGTimeDecayModel(
            interception_config  = inter_cfg,
            dynamics_config      = dyn_cfg,
            dispersion_config    = disp_cfg,
            homeadvantage_config = ha_cfg,
            kappa_config         = kap_cfg,
            market_weight        = mw
        )
        
        # D. Generate Tasks
        label_goals = "goals_hl_$(hl)_mw_$(mw)"
        label_xg    = "xg_hl_$(hl)_mw_$(mw)"
        
        push!(all_tasks, create_experiment_tasks(ds, model_goals, label_goals, save_dir, target_seasons))
        push!(all_tasks, create_experiment_tasks(ds, model_xg, label_xg, save_dir, target_seasons))
    end
end

println("Created $(length(all_tasks)) tasks for Grid Search.")

# ==============================================================================
# 2. EXECUTE THE SEARCH
# ==============================================================================
# results_status = run_experiment_task.(all_tasks)

# ==============================================================================
# 3. EVALUATION & MASTER TABLE
# ==============================================================================
# Note: This section assumes the experiments have finished and are saved in save_dir.
# We can run this separately after the execution is done.

function summarize_grid_results(save_dir, ds)
    saved_folders = Experiments.list_experiments(save_dir; data_dir="")
    if isempty(saved_folders)
        println("No experiment folders found in $save_dir")
        return DataFrame()
    end

    loaded_results = [Experiments.load_experiment(f) for f in saved_folders]
    summary_rows = []

    for exp in loaded_results
        model_name = exp.config.name
        println("Evaluating: $model_name")
        
        # Extract Half-Life and Market Weight from the name string
        m_hl = match(r"hl_(\d+)", model_name)
        m_mw = match(r"mw_([\d\.]+)", model_name)
        
        hl_val = isnothing(m_hl) ? missing : parse(Int, m_hl.captures[1])
        mw_val = isnothing(m_mw) ? missing : parse(Float64, m_mw.captures[1])
        
        engine_type = startswith(model_name, "xg") ? "XG" : "Goals"

        try
            # 1. RQR Metric
            rqr_data = Evaluation.compute_metric(Evaluation.RQR(), exp, ds)
            rqr_row  = Evaluation.to_dataframe_row(exp, rqr_data)
            
            # 2. GLM Metric
            glm_data = Evaluation.compute_metric(Evaluation.GLMEdge(), exp, ds)
            glm_row  = Evaluation.to_dataframe_row(exp, glm_data)
            
            # 3. LogLoss Metric
            ll_data  = Evaluation.compute_metric(Evaluation.LogLoss(), exp, ds)
            ll_row   = Evaluation.to_dataframe_row(exp, ll_data)
            
            combined_row = (;
                engine = engine_type,
                half_life = hl_val,
                market_weight = mw_val,
                
                # LogLoss
                model_logloss = ll_row.logloss_overall_model_ll,
                diff_logloss  = ll_row.logloss_overall_diff_ll,
                
                # GLM Edge
                glm_coef = glm_row.glmedge_spread_fair_coef,
                glm_pval = glm_row.glmedge_spread_fair_p_value,
                
                # Goodness of Fit
                rqr_overall_p = rqr_row.rqr_all_shapiro_p,
                rqr_home_skew = rqr_row.rqr_home_skewness
            )
            
            push!(summary_rows, combined_row)
            
        catch e
            @warn "❌ Error evaluating $model_name: $e"
        end
    end

    return DataFrame(summary_rows)
end

# To run evaluation:
# grid_results_df = summarize_grid_results(save_dir, ds)
# sort!(grid_results_df, [:engine, :half_life, :market_weight])
# display(grid_results_df)
