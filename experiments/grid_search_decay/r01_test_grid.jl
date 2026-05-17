using BayesianFootball
using Revise
using DataFrames
using ThreadPinning
pinthreads(:cores)
using Dates

include("./l00_grid_search_loader.jl")
const PreGame = BayesianFootball.Models.PreGame

# ==============================================================================
# 0. SETTINGS & DATA
# ==============================================================================
# Use a small tournament or a restricted season to speed up testing
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/experiments/grid_search_test/"
target_seasons = ["2026"]

# ==============================================================================
# 1. SETUP A SINGLE TEST TASK
# ==============================================================================
hl = 30
mw = 0.5

inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.TimeDecayDynamics(days_half_life = hl)

model_xg = PreGame.DynamicMarketXGTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    market_weight        = mw
)

label_xg = "test_xg_hl_$(hl)_mw_$(mw)"
task = create_experiment_tasks(ds, model_xg, label_xg, save_dir, target_seasons)

println("Running Test Task: $(task.config.name)")

# ==============================================================================
# 2. EXECUTE
# ==============================================================================
success = run_experiment_task(task)

if success
    println("\n✅ Test Task Succeeded!")
else
    println("\n❌ Test Task Failed!")
end
