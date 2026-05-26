# current_development/MetaModels/r04_queued_fold_runner.jl
#
# Full Meta Model runner with per-fold Queued NUTS training.
# One Meta Model chain per fold per CPU thread. Mirrors the Layer 1
# QueuedNUTSConfig approach so all cores stay busy throughout training.
#
# Prerequisites:
#   - julia --project -t 32  (or however many threads you have)
#   - A completed Layer 1 ExperimentResults saved under ./data/

using Pkg; Pkg.activate(".")
using BayesianFootball
using DataFrames
using Dates
using Statistics
using LogExpFunctions: logistic

using ThreadPinning
pinthreads(:cores)

# --- Force fresh load of the MetaModels prototype module ---
# If MetaModels is already defined in Main, remove it first so that
# struct redefinitions (MetaExperimentResults, MetaFoldResult, etc.) take effect.
if isdefined(Main, :MetaModels)
    println("Reloading MetaModels module...")
end
include("src/MetaModels.jl")
using .MetaModels

# Also force-reload staking so it picks up the new MetaFoldResult type
include("src/staking.jl")

println("="^65)
println("  META MODEL — Queued Fold Runner (r04)")
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

println("    Loaded: $(exp_results.config.name)")
println("    Folds:  $(length(exp_results.training_results.items))")

# ===========================================================================
# 2. CONFIGURE META MODEL
# ===========================================================================
# Swap hierarchy_config to GlobalMetaHierarchyConfig() to remove team biases
# and run a faster global-only model.

println("\n[3] Configuring ConvexMixtureMetaModel...")

meta_model = ConvexMixtureMetaModel(
    dynamics_config  = MetaGRWDynamicsConfig(σ_prior=0.1),
    hierarchy_config = HierarchicalMetaTeamConfig(σ_team_prior=0.1)
    # hierarchy_config = GlobalMetaHierarchyConfig()  # ← swap for faster global-only run
)

# QueuedNUTSConfig: each chain runs on its own thread.
# max_concurrent_tasks controls how many chains run simultaneously.
# Set this to Threads.nthreads() to saturate the CPU.
sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_samples = 500,
    n_chains  = 4,
    n_warmup  = 200,
    accept_rate = 0.65,
    max_depth   = 10,
    initialisation = BayesianFootball.Samplers.UniformInit(-2, 2),
)

# ===========================================================================
# 3. CREATE TASK AND RUN
# ===========================================================================
# Change target_selection to whichever market you want to study.
# Good candidates to try in order of likely L1 edge:
#   :under_25, :home, :away, :over_25, :btts_yes

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

# Robust unpacking: handle old tuple-return (r03 cache) and new single-return
meta_results = if _raw_result isa Tuple
    @warn "Detected old tuple return from run_meta_experiment. Using first element.\n" *
          "Please restart your REPL and re-run r04 for a fully fresh load."
    _raw_result[1]
else
    _raw_result
end

n_folds = length(meta_results.fold_results)
println("\n    $(n_folds) fold(s) completed.")

# ===========================================================================
# 4. PER-FOLD CHAIN DIAGNOSTICS
# ===========================================================================

println("\n" * "="^65)
println("  PER-FOLD DIAGNOSTICS")
println("="^65)

for fr in meta_results.fold_results
    chain = fr.chain
    α_s   = vec(chain[:α_intercept])
    θ_g   = round(logistic(mean(α_s)), digits=3)
    acc   = round(mean(chain[:acceptance_rate]), digits=3)

    rhats    = MCMCChains.rhat(chain)
    n_bad_rhat = count(x -> !ismissing(x) && x > 1.05, rhats[:, :rhat])

    println("  Fold $(fr.fold_idx):  " *
            "n=$(nrow(fr.fold_data))  " *
            "θ_global=$(θ_g)  " *
            "accept=$(acc)  " *
            "rhat_issues=$(n_bad_rhat)/$(length(rhats[:, :rhat]))")
end

# ===========================================================================
# 5. STAKING ANALYSIS USING FULL Q POSTERIOR
# ===========================================================================
println("\n" * "="^65)
println("  STAKING ANALYSIS — Full Q Posterior (McHale BayesianKelly)")
println("="^65)

# Use the last fold's chain for staking (it was trained on most data)
# In production you'd use each fold's chain for its next-season predictions.
last_fold   = meta_results.fold_results[end]
chain       = last_fold.chain
joined_data = meta_results.all_data

# Filter joined_data to only the LAST fold's holdout season
# (avoid training-set contamination in the staking ledger)
last_fold_idx = last_fold.fold_idx
oos_data = subset(joined_data, :fold_idx => ByRow(==(last_fold_idx)))
println("  Evaluating on last fold holdout: $(nrow(oos_data)) matches\n")

# --- L1 Raw Baseline ---
using BayesianFootball.Signals: BayesianKelly, compute_stake

l1_stakes = [compute_stake(BayesianKelly(min_edge=0.02), dist, odds)
             for (dist, odds) in zip(oos_data.distribution, oos_data.odds_close)]
l1_pnls   = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
             for (s, w, o) in zip(l1_stakes, oos_data.is_winner, oos_data.odds_close)]

bets_l1 = count(>(0), l1_stakes)
pnl_l1  = sum(l1_pnls)
roi_l1  = bets_l1 > 0 ? pnl_l1 / sum(l1_stakes[l1_stakes .> 0]) * 100 : 0.0
println("  L1 Raw (BayesianKelly 2% edge):")
println("    Bets=$(bets_l1) | PnL=$(round(pnl_l1, digits=4)) | ROI=$(round(roi_l1, digits=2))%\n")

# --- Meta Model Q Posterior ---
meta_ledger_df = MetaModels.compute_meta_stakes(
    chain, oos_data;
    signal  = BayesianKelly(min_edge=0.02),
    verbose = true
)
MetaModels.meta_ledger_summary(meta_ledger_df; label="Meta Q Posterior (2% edge)")

meta_ledger_relaxed = MetaModels.compute_meta_stakes(
    chain, oos_data;
    signal  = BayesianKelly(min_edge=0.0),
    verbose = false
)
MetaModels.meta_ledger_summary(meta_ledger_relaxed; label="Meta Q Posterior (0% edge)")

# --- Edge Distribution Diagnostic ---
println("\n  Edge Distribution Diagnostic:")
q_edges = meta_ledger_relaxed.Q_edge
l1_edges = meta_ledger_relaxed.L1_edge
println("  " * rpad("", 30) * rpad("Q (Meta)", 15) * "L1 (Raw)")
println("  " * rpad("Mean edge", 30)    * rpad(string(round(mean(q_edges),   digits=4)), 15) * string(round(mean(l1_edges),   digits=4)))
println("  " * rpad("Max edge",  30)    * rpad(string(round(maximum(q_edges), digits=4)), 15) * string(round(maximum(l1_edges), digits=4)))
println("  " * rpad("% positive", 30)   * rpad(string(round(mean(q_edges .> 0)*100, digits=1)) * "%", 15) * string(round(mean(l1_edges .> 0)*100, digits=1)) * "%")

# ===========================================================================
# 6. FULL HEADLESS ANALYSIS (all folds combined)
# ===========================================================================
println("\n[6] Running full headless analysis on last fold chain...")
include("analysis_headless.jl")

println("\n" * "="^65)
println("  r04 COMPLETE")
println("="^65)
