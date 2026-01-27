
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

d = subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))

# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/grw_basics"
println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/market_runs"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

if isempty(loaded_results)
    error("No results loaded! Did you run runner.jl?")
end

unique(d.home_team)

match_to_predict = DataFrame(match_id=[1,2], home_team=["east-kilbride", "stranraer"], away_team=["the-spartans-fc", "clyde-fc"] )
m =loaded_results[1]

model_preds_1 = BayesianFootball.Models.PreGame.extract_parameters(
     m.config.model,
    match_to_predict,
    ,
  results[1][1]
)

# 4. turing 
using Turing

symbols =[:μ, :γ, :σ_att, :σ_def] 

for m in loaded_results 
  println("\n Model: $(m.config.name) \n")
  println( 
describe(m.training_results[1][1][symbols])
)
end 

using StatsPlots
plot(loaded_results[1].training_results[end][1][:μ])
      

describe(loaded_results[1].training_results[end][1][symbols])
describe(loaded_results[2].training_results[end][1][symbols])
describe(loaded_results[3].training_results[end][1][symbols])
describe(loaded_results[4].training_results[end][1][symbols])

