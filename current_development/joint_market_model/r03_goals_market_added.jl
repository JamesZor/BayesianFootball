
# include("./l00_inverse_problem.jl") # experiment stuff 
# include("./l02_market_featue.jl")


using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


# --- 2. Running the experiment
struct DSExperimentSettings 
  ds::Data.DataStore
  label::String
  save_dir::String
  target_season::Vector{<:String}
end

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end


get_target_seasons_string(::Data.Ireland)       = ["2025"]

function create_CVsplit_training_config(ds::Data.DataStore, target_seasons::Vector{<:String})

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_biweek,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = Samplers.NUTSConfig(
    1000, # n steps
    2,    # n chains
    300,  # warm up steps
    0.65, # acceptance rate
    10,   # Max depth
    Samplers.UniformInit(-1, 1), # init step up 
    :false # show progress bar
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true, max_concurrent_splits=8
    )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


    return (; cv_cfg=cv_config, training_cfg=training_config)

end



# ==========================================
#  1: Combine Model + Cfgs into an ExperimentTask
# ==========================================
function build_experiment_task(ds::BayesianFootball.Data.DataStore, model, label, save_dir::String, cfgs::NamedTuple)
    # 1. Define where this specific model will save its chains/metrics
    
    # 2. Build the master config
    exp_config = BayesianFootball.Experiments.ExperimentConfig(
        name = label,
        model = model,
        splitter = cfgs.cv_cfg,
        training_config = cfgs.training_cfg,
        save_dir = save_dir
    )
    
    # 3. Return the task ready for the execution pipeline
    return ExperimentTask(ds, exp_config)
end


function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("Running: $(conf.name)")

    try
        # 2. Execute
        results = Experiments.run_experiment(task.ds, conf)

        # 3. Re-enable logging to save and confirm
        Experiments.save_experiment(results)
        
        return true # Success flag

    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        # If you want to see the stacktrace for debugging:
        # Base.showerror(stdout, e, catch_backtrace())
        return false # Failure flag
    end
end

############
using Distributions

# for running the models
const PreGame = BayesianFootball.Models.PreGame


inter_cfg = PreGame.GlobalInterception()
# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()

dyn_cfg   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)
kap_cfg   = PreGame.HierarchicalTeamKappa() 

## models 

model_gm = PreGame.DynamicMarketGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    market_σ = Gamma(1.8, 0.15)
)

model_g = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

model_gxg = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)



save_dir::String = "./data/dev_inverse_model_run1/"

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

cfgs = create_CVsplit_training_config(ds,get_target_seasons_string(ds.segment))

task_g = build_experiment_task(ds, model_g, "goals_biweek", save_dir, cfgs)
task_gxg = build_experiment_task(ds, model_gxg, "goals_xg_biweek", save_dir, cfgs)
task_gm = build_experiment_task(ds, model_gm, "goals_market_biweek", save_dir, cfgs)

all_task = [task_g, task_gm, task_gxg]

# run_experiment_task.(all_task)
#
#

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


# ----

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    loaded_results, 
  [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end



# -----

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

expr = loaded_results[2]
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



# ----

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


# ----

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





#=
3×5 DataFrame
 Row │ model                glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value  glmedge_n_obs 
     │ String               Float64                 Float64                   Float64                      Int64         
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ goals_biweek                       -2.89086                  0.528119                    0.071043            4006
   2 │ goals_market_biweek                -2.87652                  0.461383                    0.113686            4006
   3 │ goals_xg_biweek                    -2.89202                  0.569868                    0.0496818           4006
=#



#=
3×7 DataFrame
 Row │ model                rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p 
     │ String               Float64       Float64      Float64           Float64           Float64            Float64           
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ goals_biweek            0.0394274     0.912327        -0.0617366         0.0150531           0.995898         0.474182
   2 │ goals_market_biweek     0.0260293     0.931982        -0.26549           0.165448            0.992091         0.0526936
   3 │ goals_xg_biweek         0.0189343     0.941916        -0.277765          0.288525            0.988775         0.00717526
=#


#=
3×4 DataFrame
 Row │ model                logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String               Float64                   Float64                    Float64                 
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ goals_biweek                         0.504417                    18.8217                 -18.3173
   2 │ goals_market_biweek                  0.502398                    18.8217                 -18.3193
   3 │ goals_xg_biweek                      0.502392                    18.8217                 -18.3193
=#


#=
============================================================
 📊 MASTER CRPS COMPARISON: test_basic
 Note: LOWER is BETTER
============================================================
3×4 DataFrame
 Row │ model                crps_home_mean  crps_away_mean  crps_all_mean 
     │ String               Float64         Float64         Float64       
─────┼────────────────────────────────────────────────────────────────────
   1 │ goals_biweek               0.57145         0.51638        0.543915
   2 │ goals_market_biweek        0.558076        0.515619       0.536848
   3 │ goals_xg_biweek            0.56401         0.514412       0.539211
=#

