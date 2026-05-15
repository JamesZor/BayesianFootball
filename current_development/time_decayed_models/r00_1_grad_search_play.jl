using BayesianFootball
using Revise
using DataFrames
using ThreadPinning
pinthreads(:cores)
using Dates

include("./l00_basic_time_engine.jl")
const PreGame = BayesianFootball.Models.PreGame

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_time_decay_models/grid_search/"

# ==============================================================================
# 1. SETUP THE GRID
# ==============================================================================
# Define the half-lives we want to test (e.g., Bi-weekly, Monthly, Quarterly, Half-Year, Yearly)
half_lives_to_test = [7, 14, 30, 90, 180, 365, 400]

# Base configs
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()

all_tasks = ExperimentTask[]

for hl in half_lives_to_test
    # Create a specific dynamics config for this half-life
    dyn_cfg = TimeDecayDynamics(days_half_life = hl)
    
    model = DynamicGoalsTimeDecayModel(
        interception_config  = inter_cfg,
        dynamics_config      = dyn_cfg,
        dispersion_config    = disp_cfg,
        homeadvantage_config = ha_cfg
    )
    
    # Label it clearly so we can parse it in the evaluation loop
    label = "decay_hl_$(hl)"
    
    # Generate tasks for this specific half-life
    tasks = create_experiment_tasks(ds, model, label, save_dir, ["2025"])
    append!(all_tasks, tasks)
end

println("Created $(length(all_tasks)) tasks for Grid Search.")

# ==============================================================================
# 2. EXECUTE THE SEARCH
# ==============================================================================
# This will run all the models. Go grab a coffee!
results = run_experiment_task.(all_tasks)

# ==============================================================================
# 3. EVALUATION & MASTER TABLE
# ==============================================================================
saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)

summary_rows = []

for (i, exp) in enumerate(loaded_results)
    model_name = exp.config.name
    println("\nEvaluating: $model_name")
    
    # Extract Half-Life from the name string (Assumes format "decay_hl_XXX_")
    # A bit of regex to pull the number out safely
    m = match(r"hl_(\d+)", model_name)
    hl_val = isnothing(m) ? missing : parse(Int, m.captures[1])

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
        
        # 4. Combine into a single NamedTuple for this experiment
        combined_row = (;
            model = model_name,
            half_life = hl_val,
            
            # LogLoss
            model_logloss = ll_row.logloss_overall_model_ll,
            diff_logloss  = ll_row.logloss_overall_diff_ll,
            
            # GLM Edge
            glm_coef = glm_row.glmedge_spread_fair_coef,
            glm_pval = glm_row.glmedge_spread_fair_p_value,
            
            # Goodness of Fit (RQR Overall Shapiro P-Value)
            rqr_overall_p = rqr_row.rqr_all_shapiro_p,
            rqr_home_skew = rqr_row.rqr_home_skewness
        )
        
        push!(summary_rows, combined_row)
        println("✅ Merged metrics for $model_name")
        
    catch e
        @warn "❌ Error evaluating $model_name: $e"
    end
end

# ==============================================================================
# 4. FINAL OUTPUT
# ==============================================================================
grid_results_df = DataFrame(summary_rows)

# Sort the table by half-life so you can see the progression!
sort!(grid_results_df, :half_life)

println("\n============================================================")
println(" 🎯 GRID SEARCH RESULTS: TIME DECAY HALF-LIFE")
println("============================================================")
display(grid_results_df)
