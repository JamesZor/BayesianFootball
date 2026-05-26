# current_development/MetaModels/r03_meta_framework_runner.jl

using Pkg; Pkg.activate(".")
using BayesianFootball
using DataFrames
using Dates
using StatsPlots

using ThreadPinning
pinthreads(:cores)


# Include our new framework
include("src/MetaModels.jl")

include("./current_development/MetaModels/src/MetaModels.jl")
println("--- Meta Model Framework MVP Runner ---")
println("Note: This script assumes you have `ds` and `exp_results` loaded in your REPL from a previous Layer 1 run.")


ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
save_dir = "./data/copula_ab_test/"
saved_files = Experiments.list_experiments(save_dir, data_dir="")
expr_results = Experiments.load_experiment(saved_files, 4)

# 1. Configure Meta Model with Parametric Composition
println("1. Configuring ConvexMixtureMetaModel...")
meta_model = MetaModels.ConvexMixtureMetaModel(
    dynamics_config = MetaModels.MetaGRWDynamicsConfig(σ_prior=0.1),
    hierarchy_config = MetaModels.HierarchicalMetaTeamConfig(σ_team_prior=0.1)
)

sampler_config = Samplers.QueuedNUTSConfig(
    n_samples=500, n_chains=4, n_warmup=200,initialisation = Samplers.UniformInit(-2, 2),
)

# 2. Create Task
println("2. Creating MetaExperimentTask...")
meta_task = MetaModels.MetaExperimentTask(
    exp_results,
    meta_model,
    sampler_config,
    exp_results.config.splitter, # Reuse the L1 splitter
    :over_15 # Focus on the Over 1.5 goals market
)

# 3. Run Experiment
println("3. Running Meta Experiment (MVP)...")
meta_results, joined_data = MetaModels.run_meta_experiment(meta_task; ds=ds)

chain = meta_results.training_results[1]

# 4. Extract and Plot Hierarchical Team Biases
println("4. Extracting Team Biases...")
team_names = unique(vcat(joined_data.home_team, joined_data.away_team))
team_map = Dict(t => i for (i, t) in enumerate(team_names))
n_teams = length(team_names)

delta_means = zeros(n_teams)
delta_stds = zeros(n_teams)

for (team, idx) in team_map
    sym = Symbol("δ_team[$idx]")
    if sym in keys(chain)
        delta_means[idx] = mean(chain[sym])
        delta_stds[idx] = std(chain[sym])
        println("Team Bias ($team): ", round(delta_means[idx], digits=3))
    end
end

p1 = bar(team_names, delta_means, yerror=delta_stds, 
         title="Hierarchical Team Bias (δ_team)\nPositive = L1 Underestimates Team", 
         rotation=45, legend=false, size=(800, 400))
display(p1)

println("\n=== Meta Model Framework MVP Complete! ===")
