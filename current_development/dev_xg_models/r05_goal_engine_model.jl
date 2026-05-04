# current_development/dev_xg_models/r05_goal_engine_model.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)



# Create an alias so we don't have to type the full path every time
const PreGame = BayesianFootball.Models.PreGame

println("Building model components...")

# ==========================================
# 1. INSTANTIATE THE COMPONENTS (The Lego Blocks)
# ==========================================
# Here you can easily swap GlobalDispersion for HomeAwayDispersion
# just by changing the struct you call!
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = PreGame.MultiScaleGRW()
kap_cfg   = PreGame.GlobalKappa()

# ==========================================
# 2. BUILD THE MASTER MODEL
# ==========================================
println("Assembling DynamicGoalsModel...")

model = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg
)

println("Success! Model instantiated:")
display(model)



model_xg = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


cfgs = create_CVsplit_training_config(ds,get_target_seasons_string(ds.segment))


# TODO:
save_dir::String = "./data/dev_xg_models/"



# 1 . function to combine model + cfgs to get an experiment_confg etc, 
# Create tasks for both models
task_goals = build_experiment_task(ds, model, "Dev_Goals_v1", cfgs)
task_xg    = build_experiment_task(ds, model_xg, "Dev_XG_v1", cfgs)


# notes on the features - used to check we no getting empty features etc, 
boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, cfgs.cv_cfg)
feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, model)





