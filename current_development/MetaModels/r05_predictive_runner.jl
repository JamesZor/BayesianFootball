# current_development/MetaModels/r05_predictive_runner.jl
#
# A true out-of-sample predictive regime filter runner.
# It trains a Meta Model for EVERY weekly fold (expanding window) and 
# uses fold w-1 to predict the regime (Good/Bad) for fold w.
#
# Run with: julia --project -t 32

using Pkg; Pkg.activate(".")
using BayesianFootball
using DataFrames
using Dates
using Statistics
using LogExpFunctions: logistic

using ThreadPinning
pinthreads(:cores)

# Reload the module fresh
if isdefined(Main, :MetaModels)
    println("Reloading MetaModels module...")
end
include("src/MetaModels.jl")
using .MetaModels
include("src/staking.jl")

println("="^65)
println("  META MODEL — Predictive Weekly Regime Filter (r05)")
println("="^65)

# ===========================================================================
# 1. LOAD DATA AND LAYER 1 RESULTS
# ===========================================================================
println("\n[1] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

println("[2] Loading Layer 1 Experiment Results...")
save_dir   = "./data/meta_model_layer1/"
saved_files = BayesianFootball.Experiments.list_experiments(save_dir, data_dir="")
exp_results = BayesianFootball.Experiments.load_experiment(saved_files, 1)

# ===========================================================================
# 3. CONFIGURE META MODEL
# ===========================================================================
println("\n[3] Configuring ConvexMixtureMetaModel...")

# We use the global hierarchy to speed up the 80 weekly folds (no team biases)
meta_model = ConvexMixtureMetaModel(
    dynamics_config  = MetaGRWDynamicsConfig(σ_prior=0.1),
    hierarchy_config = GlobalMetaHierarchyConfig() 
)

# Concurrency is handled by the workflow.jl (defaults to nthreads())
sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_samples = 500,
    n_chains  = 4,
    n_warmup  = 200,
    accept_rate = 0.65,
    max_depth   = 10,
    initialisation = BayesianFootball.Samplers.UniformInit(-2, 2),
)

# ===========================================================================
# 4. RUN EXPERIMENT (WEEKLY FOLDS)
# ===========================================================================
TARGET_SELECTION = :under_25

println("[4] Creating MetaExperimentTask for selection: $TARGET_SELECTION")
meta_task = MetaExperimentTask(
    exp_results,
    meta_model,
    sampler_config,
    exp_results.config.splitter,
    TARGET_SELECTION
)

println("[5] Running Queued Fold Experiment...")
println("    $(Threads.nthreads()) threads available\n")
_raw_result = MetaModels.run_meta_experiment(meta_task; ds=ds)

meta_results = if _raw_result isa Tuple
    _raw_result[1]
else
    _raw_result
end

# ===========================================================================
# 5. PREDICTIVE STAKING (1-STEP-AHEAD)
# ===========================================================================
println("\n" * "="^65)
println("  PREDICTIVE STAKING RESULTS (OOS)")
println("="^65)

ledger = MetaModels.compute_predictive_stakes(meta_results, meta_results.all_data; min_edge=0.02)

if nrow(ledger) == 0
    println("No ledger data returned.")
else
    # Calculate baseline (all L1 bets with 2% edge without the Meta gate)
    # We re-run compute_stake to see what L1 WOULD have done unconditionally.
    all_stakes = [BayesianFootball.Signals.compute_stake(BayesianFootball.Signals.BayesianKelly(min_edge=0.02), dist, odds)
                  for (dist, odds) in zip(ledger.distribution, ledger.odds_close)]
    
    all_pnls = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
                for (s,w,o) in zip(all_stakes, ledger.is_winner, ledger.odds_close)]
    
    b_all = count(>(0), all_stakes)
    p_all = sum(all_pnls)
    r_all = b_all > 0 ? (p_all / sum(filter(>(0), all_stakes)) * 100) : 0.0
    
    # Calculate performance specifically on Good Regime bets
    good_ledger = subset(ledger, :regime => ByRow(==("GOOD")))
    b_good = count(>(0), good_ledger.stake)
    p_good = sum(good_ledger.pnl)
    r_good = b_good > 0 ? (p_good / sum(filter(>(0), good_ledger.stake)) * 100) : 0.0
    
    # Calculate performance specifically on Bad Regime bets (if we HAD bet them)
    bad_ledger = subset(ledger, :regime => ByRow(==("BAD")))
    bad_stakes = [BayesianFootball.Signals.compute_stake(BayesianFootball.Signals.BayesianKelly(min_edge=0.02), dist, odds)
                  for (dist, odds) in zip(bad_ledger.distribution, bad_ledger.odds_close)]
    bad_pnls = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
                for (s,w,o) in zip(bad_stakes, bad_ledger.is_winner, bad_ledger.odds_close)]
                
    b_bad = count(>(0), bad_stakes)
    p_bad = sum(bad_pnls)
    r_bad = b_bad > 0 ? (p_bad / sum(filter(>(0), bad_stakes)) * 100) : 0.0
    
    println("\nPerformance across $(nrow(ledger)) OOS matches:")
    println("--------------------------------------------------")
    println("L1 Raw (Unfiltered) : Bets=$(b_all)  PnL=$(round(p_all, digits=4))  ROI=$(round(r_all, digits=2))%")
    println("Good Regime (Gated) : Bets=$(b_good)  PnL=$(round(p_good, digits=4))  ROI=$(round(r_good, digits=2))%")
    println("Bad Regime (Skipped): Bets=$(b_bad)  PnL=$(round(p_bad, digits=4))  ROI=$(round(r_bad, digits=2))%")
    println("--------------------------------------------------")
    
    # Diagnostic: Print the first few weeks of expanding threshold
    println("\nFirst 10 weeks of regime threshold evolution:")
    diag_df = unique(ledger[!, [:W, :θ_pred, :threshold, :regime]], :W)
    for row in eachrow(first(diag_df, 10))
        println("  Week $(row.W): θ_pred=$(round(row.θ_pred, digits=4)) | threshold=$(round(row.threshold, digits=4)) -> $(row.regime)")
    end
end
