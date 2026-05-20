# current_development/r07_smoke_test_refactor.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

# --- Setup ---
println("--- Smoke Testing Refactored PreGame Models ---")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

# Minimal sampling for quick smoke test
sampler_conf = BayesianFootball.Samplers.NUTSConfig(
    50, # warmup
    2,  # chains 
    50, # samples
    0.65,
    10,
    BayesianFootball.Samplers.UniformInit(-1, 1),
    false # No AD for speed
)

# Common Components
inter_cfg = Models.PreGame.GlobalInterception()
disp_cfg  = Models.PreGame.HomeAwayDispersion()
ha_cfg    = Models.PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = Models.PreGame.GlobalKappa()

# Dynamics for different types
std_dyn   = Models.PreGame.MultiScaleGRW()
decay_dyn = Models.PreGame.TimeDecayDynamics(days_half_life=180.0)
player_dyn = Models.PreGame.PositionalPlayerDynamics()

# Player Rating Config
tracker_lv = Features.LastValueTracker()
feature_cfg_lv = Features.PlayerRatingsFeature(tracker_lv)

# --- Model Definitions ---
models_to_test = [
    # 1. Team Level - Standard
    ("Team Standard Goals", Models.PreGame.DynamicGoalsModel(
        interception_config=inter_cfg, dynamics_config=std_dyn, dispersion_config=disp_cfg, homeadvantage_config=ha_cfg
    )),
    ("Team Standard Market XG", Models.PreGame.DynamicMarketXGModel(
        interception_config=inter_cfg, dynamics_config=std_dyn, dispersion_config=disp_cfg, homeadvantage_config=ha_cfg, kappa_config=kap_cfg
    )),
    
    # 2. Team Level - Time Decay
    ("Team Time-Decay Goals", Models.PreGame.DynamicGoalsTimeDecayModel(
        interception_config=inter_cfg, dynamics_config=decay_dyn, dispersion_config=disp_cfg, homeadvantage_config=ha_cfg
    )),
    ("Team Time-Decay XG", Models.PreGame.DynamicXGTimeDecayModel(
        interception_config=inter_cfg, dynamics_config=decay_dyn, dispersion_config=disp_cfg, homeadvantage_config=ha_cfg, kappa_config=kap_cfg
    )),

    # 3. Player Level - Standard
    ("Player Standard XG Market", Models.PreGame.DynamicMarketXGPlayerModel(
        interception_config=inter_cfg, player_dynamics_config=player_dyn, dispersion_config=disp_cfg, homeadvantage_config=ha_cfg, kappa_config=kap_cfg, player_ratings_feature=feature_cfg_lv
    )),
    ("Player Time-Decay XG Market", Models.PreGame.DynamicMarketXGPlayerTimeDecayModel(
        interception_config=inter_cfg, player_dynamics_config=player_dyn, dispersion_config=disp_cfg, homeadvantage_config=ha_cfg, kappa_config=kap_cfg, player_ratings_feature=feature_cfg_lv
    )),
    ("Player Hierarchical XG Market", Models.PreGame.DynamicMarketXGHierarchicalPlayerTimeDecayModel(
        interception_config=inter_cfg, player_dynamics_config=Models.PreGame.HierarchicalPlayerDynamicsConfig(), dispersion_config=disp_cfg, homeadvantage_config=ha_cfg, kappa_config=kap_cfg, player_ratings_feature=feature_cfg_lv
    ))
]

# --- CV Configuration ---
cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["2026"],
    history_seasons = 1,
    dynamics_col = :match_month,
    warmup_period = 4,
    stop_early = true# IMPORTANT: Only run 1 fold per model
)

# --- Execution ---
for (label, m) in models_to_test
    println("\n>>> Testing Model: $label")
    
    try
        # 1. Create Split
        boundaries = Data.create_id_boundaries(ds, cv_config)
        
        # 2. Create Features for first split
        feature_sets = Features.create_features(boundaries, ds, m, cv_config.dynamics_col)
        fs, meta = feature_sets[1]
        
        # 3. Build Turing Model
        tm = Models.PreGame.build_turing_model(m, fs)
        println("   - Turing model built successfully.")
        
        # 4. Sample (Smoke Test)
        # Note: We use the low-level Samplers.train if available, or just sample(tm, ...)
        # Using the standard experiment runner logic for consistency
        train_cfg = BayesianFootball.Training.Independent(parallel=false)
        training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)
        
        # Just run the first split manually to save time
        chain = sample(tm, NUTS(sampler_conf.n_warmup, sampler_conf.accept_rate), sampler_conf.n_samples)
        
        println("   - Sampling complete (ESS for lp: $(ess_bulk(chain[:lp])))")
        println("✅ PASSED: $label")
    catch e
        @error "❌ FAILED: $label" exception=(e, catch_backtrace())
    end
end

println("\n--- Smoke Test Finished ---")
