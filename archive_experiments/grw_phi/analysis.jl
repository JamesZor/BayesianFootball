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
saved_folders = Experiments.list_experiments("exp/grw_phi"; data_dir="./data")
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

# 2. Run Backtest
# ===============
println("\nRunning Backtest on $(length(loaded_results)) models...")

baker = BayesianKelly()

my_signals = [baker]

flat_strat = FlatStake(0.05)
my_signals = [baker, flat_strat]

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


### plots
using DataFrames, StatsPlots, Dates, Printf

# --- 1. CONFIGURATION ---
# We want to compare the "Battle" for the 2.5 Goal Line
under_sym = [:over_25]
over_sym  = [:under_25]

over_sym  = [:btts_yes, :over_15]

over_sym  = [:btts_yes, :over_15, :over_25, :over_35, :draw, :under_55]

over_sym = [:draw]
under_sym = [:over_150]
start_view = Date(2021, 6, 1)  # Start of 21/22 Season
end_view   = Date(2025, 6, 1)  # Present day

# --- 2. DATA CLEANING (Fixes the Year 0478 Bug) ---
# We filter the ledger to remove any bad dates
clean_ledger = filter(row -> row.date >= start_view, ledger.df)

println("Data Range: ", minimum(clean_ledger.date), " to ", maximum(clean_ledger.date))

# --- 3. PREPARE PLOT ---
# Get unique models to assign consistent colors
unique_models = unique(clean_ledger[:, [:model_name, :model_parameters]])
colors = palette(:tab10)

# Helper for readable labels
function make_short_label(name, params)
    short_name = replace(name, "GRWNegativeBinomial" => "NegBin", 
                               "GRWBivariatePoisson" => "BivPois",
                               "GRWPoisson" => "Pois",
                               "GRWDixonColes" => "DC")
    m = match(r"μ=([0-9\.]+)", params)
    param_str = m === nothing ? "" : " μ=$(m.captures[1])"
    return "$short_name$param_str"
end

# Initialize ONE master plot with fixed Time Axis
p = plot(title = "Regime Battle: $(over_sym) vs $(under_sym)", 
         xlabel = "Date", ylabel = "Cumulative Profit (Units)",
         legend = :outertopright, 
         size = (1000, 600), 
         margin = 5Plots.mm,
         xlims = (start_view, end_view)) # <--- Forces correct zoom

# Add Zero line (Break Even)
hline!(p, [0.0], label="Break Even", color=:black, linestyle=:dash, alpha=0.5)

# --- 4. PLOTTING LOOP ---
for (i, model_row) in enumerate(eachrow(unique_models))
# for (i, model_row) in enumerate(eachrow(unique_models))
    
    m_name = model_row.model_name
    m_params = model_row.model_parameters
    lbl_base = make_short_label(m_name, m_params)
    
    # Assign specific color
    col_idx = mod1(i, 10)
    model_color = colors[col_idx]

    # Filter for this specific model
    model_ledger = filter(row -> row.model_name == m_name && 
                                 row.model_parameters == m_params, clean_ledger)

    # Plot OVERS (Solid Line)
    overs = filter(row -> row.selection in over_sym, model_ledger)
    if !isempty(overs)
        sort!(overs, :date)
        overs.cum_pnl = cumsum(overs.pnl)
        plot!(p, overs.date, overs.cum_pnl, 
              label="$(lbl_base) | Over", 
              lw=2, color=model_color, linestyle=:solid)
    end

    # Plot UNDERS (Dashed Line)
    unders = filter(row -> row.selection in under_sym, model_ledger)
    if !isempty(unders)
        sort!(unders, :date)
        unders.cum_pnl = cumsum(unders.pnl)
        plot!(p, unders.date, unders.cum_pnl, 
              label="$(lbl_base) | Under", 
              lw=2, color=model_color, linestyle=:dash)
    end
end

display(p)

