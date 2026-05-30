# current_development/ab_test_outfield_player/r01_sanity_check_dixon_coles.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Turing

using ThreadPinning

pinthreads(:cores)


const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
println("[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel...")
model_dixon = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4
)

# ==========================================
# 4. CREATE EXPERIMENT TASK (For Splitting)
# ==========================================
println("[INFO] Creating Experiment Task (Target Season: 2025)...")
task = Experiments.create_experiment_task(
    ds, 
    model_dixon, 
    "sanity_check_dixon_coles", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000, # Not actually used in this manual script
    warmup=500,  
    chains=8,
    use_queue=true,  # <--- Triggers the new high-performance QueuedNUTSConfig
)


results = Experiments.run_experiment(task)



chains_df_all = Experiments.Diagnostics.extract_chains(ds, results)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Experiments.Diagnostics.check_convergence(chains_df_all)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Experiments.Diagnostics.check_stability(chains_df_all)



# Grab the very first split to test
test_split = first(task.splits)
println("[INFO] Testing on Split: $(test_split.split_id)")
println("       Train Matches: $(nrow(test_split.train_df))")
println("       Test Matches:  $(nrow(test_split.test_df))")

# ==========================================
# 5. TEST: BUILD TURING MODEL
# ==========================================
println("\n[TEST 1] Building Turing Model...")
t_model = PreGame.build_turing_model(model_dixon, test_split.train_features)
println("✅ build_turing_model successful!")

# ==========================================
# 6. TEST: SAMPLING
# ==========================================
println("\n[TEST 2] Sampling 10 iterations (NUTS)...")
# We just take 10 samples to ensure AD gradients and probability constraints work
chain = sample(t_model, NUTS(10, 0.65), 10, progress=false)
println("✅ Sampling successful! Returned chain of size: ", size(chain))

# ==========================================
# 7. TEST: EXTRACT PARAMETERS
# ==========================================
println("\n[TEST 3] Extracting Parameters for Test Set...")
ext = PreGame.extract_parameters(model_dixon, test_split.test_df, test_split.test_features, chain)
println("✅ extract_parameters successful!")

# Grab the first match in the test set to verify the structure
first_match_id = first(test_split.test_df.match_id)
println("\n[VERIFICATION] Parameters for Match $first_match_id:")
display(ext[first_match_id])

println("\n🎉 All systems go! You are ready to run the full A/B test.")
