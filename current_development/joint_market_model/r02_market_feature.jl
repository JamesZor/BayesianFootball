# current_development/joint_market_model/r02_market_feature.jl
include("./l00_inverse_problem.jl") # experiment stuff 
include("./l02_market_featue.jl")

using Revise
using BayesianFootball
using DataFrames


# ---
struct TestModel <: BayesianFootball.AbstractFootballModel end

BayesianFootball.Features.required_features(::TestModel) = [:team_ids, :market_odds]

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [79], 
    target_seasons = ["2025"],
    history_seasons = 1,   
    dynamics_col = :match_month,
    warmup_period = 0, 
    stop_early = false
)

boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, cv_config)

test_model = TestModel()

feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, test_model)



# -----
# dev model 

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


# for running the models
const PreGame = BayesianFootball.Models.PreGame


# ==========================================
# 1. LOAD THE ROBUST COMPONENTS
# ==========================================
inter_cfg = PreGame.GlobalInterception()
# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
# "ha global"
ha_cfg    = PreGame.GlobalHomeAdvantage() 

dyn_cfg   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)

model = DynamicMarketGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

save_dir::String = "./data/dev_inverse_model/"



cfgs = create_CVsplit_training_config(ds, get_target_seasons_string(ds.segment))
task_base    = build_experiment_task(ds, model,    "test_MG",    save_dir, cfgs)

res    = Experiments.run_experiment(task_base.ds,    task_base.config)
Experiments.save_experiment(res)
###############

latents = Experiments.extract_oos_predictions(ds, res)



saved_folders = Experiments.list_experiments(save_dir; data_dir="")
# saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

baker = BayesianFootball.Signals.BayesianKelly()
my_signals = [baker]


market_config = Data.Markets.DEFAULT_MARKET_CONFIG

a = market_config.markets[[1,2,4,5,6,7,8,9,10]]
cfgM = Data.MarketConfig(a)

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    loaded_results, 
    my_signals; 
    market_config = market_config
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)



model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end


expr = loaded_results[1]

expr.training_results[2][1]


# ------

using DataFrames
using Statistics

function check_parameter_stability(chains::Vector, target_params::Vector{Symbol})
    # Initialize an empty DataFrame
    df = DataFrame(Fold = Int[])
    
    # FIX: Explicitly tell Julia these columns can contain missing values
    for p in target_params
        df[!, Symbol(string(p), "_mean")] = Union{Missing, Float64}[]
        df[!, Symbol(string(p), "_std")]  = Union{Missing, Float64}[]
    end
    
    # Iterate through each fold's MCMCChain
    for (fold_idx, chain) in enumerate(chains)
        row_dict = Dict{Symbol, Any}(:Fold => fold_idx)
        
        for p in target_params
            # Check if the parameter exists in the chain
            if p in keys(chain)
                samples = vec(chain[p]) 
                row_dict[Symbol(string(p), "_mean")] = mean(samples)
                row_dict[Symbol(string(p), "_std")]  = std(samples)
            else
                row_dict[Symbol(string(p), "_mean")] = missing
                row_dict[Symbol(string(p), "_std")]  = missing
            end
        end
        
        push!(df, row_dict) # This will now safely accept the missing values!
    end
    
    return df
end



params_to_track_xg = [
    Symbol("σ_market"), # NEW: Variance/spread of team conversion abilities
    Symbol("inter.μ"), 
    Symbol("disp.log_r"), 
    Symbol("ha.γ_global"),
]

all_chains = [res[1] for res in expr.training_results] 
# 3. Generate the Stability Report
stability_df_xg = check_parameter_stability(all_chains, params_to_track_xg)




# ----

println("============================================================")
println(" 🚀 Running Batch RQR Evaluation...")
println("============================================================")

# 1. Initialize an empty array to hold our NamedTuple rows
flat_rows = []

# 2. Loop through all loaded experiments
for (i, exp) in enumerate(loaded_results)
    model_name = exp.config.name
  print("[$i/$(length(loaded_results))] Evaluating: $(model_name) ... ")
    
    try
        # Compute the nested RQR struct
        rqr_data = Evaluation.compute_metric(Evaluation.RQR(), exp, ds)
        
        # Flatten it using the magic unroller
        flat_row = Evaluation.to_dataframe_row(exp, rqr_data)
        
        # Save to our list
        push!(flat_rows, flat_row)
        println("✅ Done")
    catch e
        println("❌ Failed")
        @warn "Error evaluating $model_name: $e"
    end
end

# 3. Build the Master DataFrame
master_rqr_df = DataFrame(flat_rows)

sort!(master_rqr_df, :model)

summary_df = select(master_rqr_df, 
    :model, 
    :rqr_all_mean, 
    :rqr_all_std, 
    :rqr_all_skewness, 
    :rqr_all_kurtosis, 
    :rqr_all_shapiro_w,
    :rqr_all_shapiro_p
)


# ---

flat_rows_glm = []

for (i, exp) in enumerate(loaded_results)
    print("Evaluating GLM Edge for $(exp.config.name)... ")
    
    glm_data = Evaluation.compute_metric(Evaluation.GLMEdge(), exp, ds)
    flat_row = Evaluation.to_dataframe_row(exp, glm_data)
    
    push!(flat_rows_glm, flat_row)
    println("Done")
end

master_glm_df = DataFrame(flat_rows_glm)
sort!(master_glm_df, :model)

# Let's just view the most important columns: The Spread Coef and its P-Value
display(select(master_glm_df, 
     :model, 
    :glmedge_intercept_coef,
    :glmedge_spread_fair_coef, 
    :glmedge_spread_fair_p_value,
    :glmedge_n_obs
))




flat_rows_ll = []

for (i, exp) in enumerate(loaded_results)
    model_name = exp.config.name
    print("[$i/$(length(loaded_results))] Evaluating LogLoss for: $(model_name) ... ")
    
    try
        # Compute the LogLoss struct
        ll_data = Evaluation.compute_metric(Evaluation.LogLoss(), exp, ds)
        
        # Flatten it
        flat_row = Evaluation.to_dataframe_row(exp, ll_data)
        push!(flat_rows_ll, flat_row)
        println("✅ Done")
    catch e
        println("❌ Failed")
        @warn "Error evaluating $model_name: $e"
    end
end

# Build DataFrame
master_ll_df = DataFrame(flat_rows_ll)
sort!(master_ll_df, :model)

println("\n============================================================")
println(" 📉 MASTER LOGLOSS COMPARISON (LOWER IS BETTER)")
println(" Note: A negative 'diff_ll' means your model beat the bookmaker!")
println("============================================================")

display(select(master_ll_df, 
    :model, 
    :logloss_overall_model_ll, 
    :logloss_overall_market_ll, 
    :logloss_overall_diff_ll
))

#
function evaluate_batch_crps(results_array, ds; label="CRPS Evaluation")
    println("\n============================================================")
    println(" 🚀 Running Batch CRPS Evaluation: $label")
    println("============================================================")

    flat_rows_crps = []

    # Loop through all provided experiments
    for (i, exp) in enumerate(results_array)
        model_name = exp.config.name
        print("[$i/$(length(results_array))] Evaluating: $(model_name) ... ")
        
        try
            crps_data = Evaluation.compute_metric(Evaluation.CRPS(), exp, ds)
            flat_row = Evaluation.to_dataframe_row(exp, crps_data)
            
            push!(flat_rows_crps, flat_row)
            println("✅ Done")
        catch e
            println("❌ Failed")
            @warn "Error evaluating $model_name: $e"
        end
    end

    # Build the Master DataFrame
    master_crps_df = DataFrame(flat_rows_crps)

    if nrow(master_crps_df) > 0
        # Sort by model name to keep it organized
        sort!(master_crps_df, :model)

        println("\n============================================================")
        println(" 📊 MASTER CRPS COMPARISON: $label")
        println(" Note: LOWER is BETTER")
        println("============================================================")
        display(master_crps_df)
    else
        println("⚠️ No results successfully evaluated.")
    end
    
    return master_crps_df
end


df_long_run = evaluate_batch_crps(loaded_results, ds, label="test_basic")
