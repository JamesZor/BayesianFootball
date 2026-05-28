# current_development/MetaModels/r07_multi_market_runner.jl
#
# Sequential multi-market runner for the Meta Model regime filter.
# Trains the Meta Model across 80 folds for each market in the TARGET_SELECTIONS list.
# Results and ledgers are persisted to disk via Serialization for later analysis.
#
# Run with: julia --project -t 32

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Serialization
using LogExpFunctions: logistic

using ThreadPinning
pinthreads(:cores)

include("./current_development/MetaModels/src/MetaModels.jl")
# include("src/MetaModels.jl")
using .MetaModels

include("./current_development/MetaModels/src/staking.jl")
# include("src/staking.jl")

include("./current_development/MetaModels/l06_metrics.jl")

println("="^75)
println("  META MODEL — Multi-Market Sequential Runner (r07)")
println("="^75)

# ===========================================================================
# 1. LOAD DATA AND LAYER 1 RESULTS
# ===========================================================================
println("\n[1] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

println("[2] Loading Layer 1 Experiment Results...")
save_dir = "./data/meta_model_layer1/ireland/"
saved_files = BayesianFootball.Experiments.list_experiments(save_dir, data_dir="")
exp_results = BayesianFootball.Experiments.load_experiment(saved_files, 1)

# ===========================================================================
# 3. CONFIGURE META MODEL
# ===========================================================================
println("\n[3] Configuring ConvexMixtureMetaModel...")
meta_model = MetaModels.ConvexMixtureMetaModel(
    dynamics_config  = MetaModels.MetaGRWDynamicsConfig(σ_prior=0.2),
    hierarchy_config = MetaModels.GlobalMetaHierarchyConfig() 
)

sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_samples = 500,
    n_chains  = 2,
    n_warmup  = 200,
    accept_rate = 0.65,
    max_depth   = 10,
    initialisation = BayesianFootball.Samplers.UniformInit(-2, 2),
)

# ===========================================================================
# 4. RUN EXPERIMENTS ACROSS TARGET MARKETS
# ===========================================================================
# Define the markets to evaluate
TARGET_SELECTIONS = [:under_15, :under_25, :under_35, :over_15, :over_25, :over_35]
TARGET_SELECTIONS = [:under_15, :under_25]

min_edge = 0.00
multi_market_results, multi_market_ledgers = MetaModels.run_multi_market_experiments(
    TARGET_SELECTIONS,
    exp_results,
    meta_model,
    sampler_config,
    ds;
    min_edge=min_edge
)

# ===========================================================================
# 5. PERSIST RESULTS TO DISK
# ===========================================================================
println("\n[5] Serializing all multi-market results to disk...")
serialize(joinpath(save_dir, "multi_market_results.jls"), multi_market_results)
serialize(joinpath(save_dir, "multi_market_ledgers.jls"), multi_market_ledgers)
println("Successfully saved to: $save_dir")

# ===========================================================================
# 6. STREAMLINED REPORTING
# ===========================================================================
println("\n" * "="^75)
println("  GENERATING CONSOLIDATED HURDLE REPORTS")
println("="^75)

# Evaluate and print the reports for all successfully evaluated markets
markets_evaluated = collect(keys(multi_market_ledgers))
final_metrics = MetaModels.evaluate_multiple_markets(multi_market_ledgers, markets_evaluated; min_edge=min_edge)

println("\nRunner finished successfully.")

