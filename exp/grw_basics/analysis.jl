# exp/grw_basics/analysis.jl

using Revise
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

# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/grw_basics"
println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/grw_basics"; data_dir="./data")

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

# 2. Run Backtest
# ===============
println("\nRunning Backtest on $(length(loaded_results)) models...")

baker = BayesianKelly()
my_signals = [baker]

# Run backtest on ALL loaded models at once
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    loaded_results, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

# 3. Analyze
# ==========
tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

println("\n=== TEARSHEET SUMMARY ===")
println(tearsheet)

# Breakdown by Model (Selection)
println("\n=== BREAKDOWN BY MODEL ===")
model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end



# 4. turing 
using Turing

describe(loaded_results[1].training_results[end][1])
describe(loaded_results[2].training_results[end][1])
describe(loaded_results[3].training_results[end][1])
describe(loaded_results[4].training_results[end][1])
