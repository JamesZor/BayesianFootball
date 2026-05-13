
using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Norway())

save_dir::String = "./data/dev_joint_model/norway/"

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


get_target_seasons_string(::Data.Norway)       = ["2023", "2024", "2025"]

function create_CVsplit_training_config(ds::Data.DataStore, target_seasons::Vector{<:String})

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = false
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


# ------
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

inter_season_cfg = PreGame.SeasonalInterception()

model_gm = PreGame.DynamicMarketGoalsModel(
    interception_config  = inter_season_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

model_g = PreGame.DynamicGoalsModel(
    interception_config  = inter_season_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

model_gxg = PreGame.DynamicXGModel(
    interception_config  = inter_season_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)


model_gxgm = PreGame.DynamicMarketXGModel(
    interception_config  = inter_season_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)



cfgs = create_CVsplit_training_config(ds,get_target_seasons_string(ds.segment))


task_g = build_experiment_task(ds, model_g, "mu_goals_month", save_dir, cfgs)
task_gm = build_experiment_task(ds, model_gm, "mu_goals_market_month", save_dir, cfgs)
task_gxg = build_experiment_task(ds, model_gxg, "mu_goals_xg_month", save_dir, cfgs)
task_gxgm = build_experiment_task(ds, model_gxgm, "mu_goals_xg_market_month", save_dir, cfgs)

all_task = [task_g, task_gm, task_gxg, task_gxgm]

# run_experiment_task.(all_task)




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



ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results, 
  [BayesianFootball.Signals.BayesianKelly(min_edge=0.05)]; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names[1:18]
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    # show(sub)
    show(sub[:, [:model_name, :selection, :opportunities, :bets_placed, :activity_pct, :turnover, :profit, :roi_pct, :win_rate_pct]])
end

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





#=
julia> summary_df = select(master_rqr_df, 
           :model, 
           :rqr_all_mean, 
           :rqr_all_std, 
           :rqr_all_skewness, 
           :rqr_all_kurtosis, 
           :rqr_all_shapiro_w,
           :rqr_all_shapiro_p
       )
4×7 DataFrame
 Row │ model                     rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p 
     │ String                    Float64       Float64      Float64           Float64           Float64            Float64           
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ mu_goals_market_month        0.0220288     0.985679        0.0173865         -0.0430809           0.999642           0.911862
   2 │ mu_goals_month               0.0284902     1.01678         0.00732606         0.0404176           0.999701           0.965421
   3 │ mu_goals_xg_market_month     0.0189159     0.9985         -0.0247023          0.0362143           0.99951            0.707335
   4 │ mu_goals_xg_month            0.0238634     0.99725         0.0826499         -0.083157            0.999201           0.238049
=#



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




#=
4×5 DataFrame
 Row │ model                     glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value  glmedge_n_obs 
     │ String                    Float64                 Float64                   Float64                      Int64         
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ mu_goals_market_month                   -2.89326                  0.271853                   0.00696297          34924
   2 │ mu_goals_month                          -2.88441                  0.162787                   0.0968356           34924
   3 │ mu_goals_xg_market_month                -2.89219                  0.252033                   0.0111979           34924
   4 │ mu_goals_xg_month                       -2.88824                  0.206932                   0.0362283           34924
=#

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



#=
4×4 DataFrame
 Row │ model                     logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String                    Float64                   Float64                    Float64                 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ mu_goals_market_month                     0.488367                      18.69                 -18.2016
   2 │ mu_goals_month                            0.494697                      18.69                 -18.1953
   3 │ mu_goals_xg_market_month                  0.490509                      18.69                 -18.1995
   4 │ mu_goals_xg_month                         0.492332                      18.69                 -18.1976
=#



#=
4×4 DataFrame
 Row │ model                     crps_home_mean  crps_away_mean  crps_all_mean 
     │ String                    Float64         Float64         Float64       
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ mu_goals_market_month           0.701045        0.625757       0.663401
   2 │ mu_goals_month                  0.722545        0.636422       0.679484
   3 │ mu_goals_xg_market_month        0.705294        0.62857        0.666932
   4 │ mu_goals_xg_month               0.712125        0.630391       0.671258
=#

