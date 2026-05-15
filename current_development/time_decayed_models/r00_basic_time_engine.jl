using BayesianFootball
using Revise
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates

include("./l00_basic_time_engine.jl")

const PreGame = BayesianFootball.Models.PreGame

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_time_decay_models/"

## models 
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = TimeDecayDynamics()
model = DynamicGoalsTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg
)


training_task = create_experiment_tasks(ds, model, "test_1", save_dir, ["2025"])

results = run_experiment_task.(training_task)

saved_folders = Experiments.list_experiments(save_dir; data_dir="")

loaded_results = loaded_experiment_files(saved_folders);

expr = loaded_results[1]

ch = expr.training_results[2][1]

describe(ch)

latents = BayesianFootball.Experiments.extract_oos_predictions(ds, expr)
ppd = BayesianFootball.Predictions.model_inference(ds, expr)



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
1×7 DataFrame
 Row │ model    rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p 
     │ String   Float64       Float64      Float64           Float64           Float64            Float64           
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_1_    -0.0132585     0.955473          -0.16228          0.466067           0.995221           0.334904
=#



#=
julia> # Let's just view the most important columns: The Spread Coef and its P-Value
       display(select(master_glm_df, 
           :model, 
           :glmedge_intercept_coef,
           :glmedge_spread_fair_coef, 
           :glmedge_spread_fair_p_value,
           :glmedge_n_obs
       ))
1×5 DataFrame
 Row │ model    glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value  glmedge_n_obs 
     │ String   Float64                 Float64                   Float64                      Int64         
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_1_                -2.89577                  0.546279                     0.064436           4006
=#



cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [79], 
    target_seasons = ["2025"],
    history_seasons = 2,   
    dynamics_col = :match_month,
    warmup_period = 0, # Using the calculated variable
    stop_early = true
)


const PreGame = BayesianFootball.Models.PreGame

inter_cfg = PreGame.GlobalInterception()
# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = TimeDecayDynamics()

## models 

model = DynamicGoalsTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg
)

boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, cv_config)

feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, model)


f1 = feature_collection[1][1]
f2 = feature_collection[4][1]

calculate_match_weights(f2[:dates], 180)


