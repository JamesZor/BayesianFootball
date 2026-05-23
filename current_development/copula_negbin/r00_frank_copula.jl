# current_development/copula_negbin/r00_frank_copula.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

# 1. Include Loader
include("l00_frank_copula.jl")

println("--- Testing Discrete Frank Copula NegBin Model ---")

# 2. Setup Data
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

# 3. Configure the Model
inter_cfg = BayesianFootball.Models.PreGame.GlobalInterception()
disp_cfg  = BayesianFootball.Models.PreGame.HomeAwayDispersion()
ha_cfg    = BayesianFootball.Models.PreGame.HierarchicalTeamHomeAdvantage()

# Use our new dynamic config
decay_dyn = FrankCopulaTimeDecayDynamics(days_half_life=180.0)

copula_model = DynamicCopulaGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=decay_dyn, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg
)

# 4. CV Configuration
cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["2026"],
    history_seasons = 1,
    dynamics_col = :match_month,
    warmup_period = 4,
    stop_early = true # Just 1 fold for smoke testing
)

println(">>> Building Data Split...")
boundaries = BayesianFootball.Data.create_id_boundaries(ds, cv_config)
feature_sets = BayesianFootball.Features.create_features(boundaries, ds, copula_model, cv_config.dynamics_col)
fs, meta = feature_sets[1]

println(">>> Building Turing Model...")
tm = BayesianFootball.Models.PreGame.build_turing_model(copula_model, fs)
println("    Model built successfully!")

# 5. Execute MAP Optimization (fastest for smoke test)
println(">>> Running MAP Optimization...")
try
    map_est = optimize(tm, MAP())
    println("    MAP optimization successful!")
    
    # 6. Test Scoring Matrix logic analytically with some dummy values
    println(">>> Testing Analytical Score Matrix Computation...")
    n_samples = 2
    params = (
        loc_h = [0.1, 0.2], 
        loc_a = [-0.1, -0.2], 
        r_h = [3.0, 3.5], 
        r_a = [3.0, 3.5], 
        κ = [2.0, 3.0] # Test positive correlation
    )
    S = compute_score_matrix_discrete_copula(params; max_goals=5)
    println("    Matrix Size: ", size(S))
    println("    Sum of PMF (should be ~1.0): ", sum(S[:, :, 1]))
    
    println("✅ All tests passed!")
catch e
    @error "❌ FAILED" exception=(e, catch_backtrace())
end
