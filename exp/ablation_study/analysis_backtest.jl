using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
#
ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/ablation_study"

println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/ablation_study"; data_dir="./data")
# saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")

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

# extract the loaded files and check that they are the right ones by looking at times.
loaded_results_ = loaded_results[1:7]

for (i, lr) in enumerate(loaded_results_)
  println("file index: $i : name :$(lr.config.name) - $(lr.config.tags[1])")
end



# 2. Run Backtest
# ===============
println("\nRunning Backtest on $(length(loaded_results_)) models...")

baker = BayesianKelly()
# as = AnalyticalShrinkageKelly()
# kelly = KellyCriterion(1)
# kelly25 = KellyCriterion(1/4)
# flat_strat = FlatStake(0.05)
my_signals = [baker]

# my_signals = [baker, as, , kelly, kelly25, flat_strat]



# Run backtest on ALL loaded models at once
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results_, 
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

for m in loaded_results 
println(m.config.model)
println("\n")
end

