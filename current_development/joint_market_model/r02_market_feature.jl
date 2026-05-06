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

