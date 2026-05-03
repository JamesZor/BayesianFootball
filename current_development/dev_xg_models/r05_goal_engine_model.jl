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







