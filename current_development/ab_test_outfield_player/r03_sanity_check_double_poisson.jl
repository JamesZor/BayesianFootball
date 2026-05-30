# current_development/ab_test_outfield_player/r03_sanity_check_double_poisson.jl

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
println("[INFO] Initializing DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel...")
model_dp = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
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
println("[INFO] Creating Experiment Task (Target Season: 2026)...")
task = Experiments.create_experiment_task(
    ds, 
    model_dp, 
    "sanity_check_double_poisson", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=16,
    use_queue=true,
)

println("[INFO] Running Experiment...")
results = Experiments.run_experiment(task)

println("[INFO] Experiment Completed. Summarizing Split 1, Chain 1...")
describe(results.training_results[1][1])

chains_df_all = Experiments.Diagnostics.extract_chains(ds, results)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Experiments.Diagnostics.check_convergence(chains_df_all)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Experiments.Diagnostics.check_stability(chains_df_all)

# Grab the very first split to test manual extraction logic
test_split = first(task.splits)
println("\n[INFO] Manual Testing on Split: $(test_split.split_id)")
println("       Train Matches: $(nrow(test_split.train_df))")
println("       Test Matches:  $(nrow(test_split.test_df))")

println("\n[TEST 1] Building Turing Model...")
t_model = PreGame.build_turing_model(model_dp, test_split.train_features)
println("✅ build_turing_model successful!")

println("\n[TEST 2] Extracting Parameters for Test Set...")
ext = PreGame.extract_parameters(model_dp, test_split.test_df, test_split.test_features, results.training_results[1][1])
println("✅ extract_parameters successful!")

first_match_id = first(test_split.test_df.match_id)
println("\n[VERIFICATION] Parameters for Match $first_match_id:")
display(ext[first_match_id])

println("\n🎉 All systems go! Double Poisson runner is working.")
