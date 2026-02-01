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
my_signals = [flat_strat]

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

for m in loaded_results 
println(m.config.model)
println("\n")
end


#####
###
using DataFrames, StatsPlots, Dates, Printf

# --- Configuration ---
over_sym = :over_35  
under_sym = :under_35 

# Get distinct colors from the Tab10 palette
# We will cycle through these for different models
colors = palette(:tab10)

# Helper function to create short, readable labels
function make_short_label(name, params)
    # 1. Shorten Model Name
    short_name = replace(name, "GRWNegativeBinomial" => "NegBin", 
                               "GRWBivariatePoisson" => "BivPois",
                               "GRWPoisson" => "Pois",
                               "GRWDixonColes" => "DC")
    
    # 2. Extract key parameter (usually the first mu value)
    # We look for "μ=0.32" or similar patterns
    m = match(r"μ=([0-9\.]+)", params)
    param_str = m === nothing ? "" : " μ=$(m.captures[1])"
    
    return "$short_name$param_str"
end

# 1. Get list of unique model configurations
unique_models = unique(ledger.df[:, [:model_name, :model_parameters]])
println("Found $(nrow(unique_models)) unique configurations.")
start_date = Date(2021, 8, 1)
end_date   = Date(2025, 11, 1)

# Add xlims to the plot initialization
p = plot(title = "Strategy Battle: $(over_sym) vs $(under_sym)", 
         xlabel = "Date", ylabel = "Cumulative Profit",
         legend = :outertopright, size=(1000, 600), margin=5Plots.mm,
         xlims = (start_date, end_date)) # <--- FORCE THE RANGE HERE

# Initialize ONE master plot
# Add Zero line
hline!(p, [0.0], label="Break Even", color=:black, linestyle=:dash, alpha=0.5)

# 2. Iterate through each model
for (i, model_row) in enumerate(eachrow(unique_models[[1,3], :]))
    
    m_name = model_row.model_name
    m_params = model_row.model_parameters
    
    # Generate the short label for this model
    lbl_base = make_short_label(m_name, m_params)
    
    # Assign a specific color index for this model (mod 10 to cycle if needed)
    col_idx = mod1(i, 10)
    model_color = colors[col_idx]

    # --- A. Filter Ledger ---
    model_ledger = filter(row -> row.model_name == m_name && 
                                 row.model_parameters == m_params, ledger.df)

    # --- B. Process Overs ---
    overs = filter(row -> row.selection == over_sym, model_ledger)
    if !isempty(overs)
        sort!(overs, :date)
        overs.cum_pnl = cumsum(overs.pnl)
        
        # Solid line for Overs
        plot!(p, overs.date, overs.cum_pnl, 
              label="$(lbl_base) | Over", 
              lw=2, color=model_color, linestyle=:solid)
    end

    # --- C. Process Unders ---
    unders = filter(row -> row.selection == under_sym, model_ledger)
    if !isempty(unders)
        sort!(unders, :date)
        unders.cum_pnl = cumsum(unders.pnl)
        
        # Dashed line for Unders (Same color, different style)
        plot!(p, unders.date, unders.cum_pnl, 
              label="$(lbl_base) | Under", 
              lw=2, color=model_color, linestyle=:dash)
    end
end

display(p)


### 
using DataFrames, StatsPlots, Dates, Printf

# --- 1. CONFIGURATION ---
# We want to compare the "Battle" for the 2.5 Goal Line
under_sym = [:under_25]
over_sym  = [:over_25]

over_sym  = [:btts_yes, :over_15]

over_sym  = [:btts_yes, :over_15, :over_25, :over_35, :draw, :under_55]
under_sym = [:over_150]
start_view = Date(2021, 7, 1)  # Start of 21/22 Season
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
for (i, model_row) in enumerate(eachrow(unique_models[[1,3,4],:]))
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


###
using DataFrames, Statistics, Dates, Printf

function recommend_model(ledger_df; window=50)
    # 1. Sort by date to look at "Recent Form"
    sort!(ledger_df, :date)
    
    # 2. Identify all unique strategies (Model + Parameter + Market)
    # We group by these 3 things to find the specific "sub-strategy"
    groups = groupby(ledger_df, [:model_name, :model_parameters, :selection])
    
    println("=== 🛡️ MODEL SELECTION REPORT (Last $window Bets) ===\n")
    
    recommendations = []

    for gdf in groups
        # Extract strategy details
        m_name = gdf[1, :model_name]
        m_param = gdf[1, :model_parameters]
        market = gdf[1, :selection]
        
        # Get only the last N bets (The Window)
        if nrow(gdf) < window
            continue # Skip if not enough history
        end
        recent = last(gdf, window)
        
        # --- CALCULATE METRICS ---
        total_stake = sum(recent.stake)
        total_profit = sum(recent.pnl)
        roi = (total_profit / total_stake) * 100
        
        win_count = count(recent.is_winner)
        actual_win_rate = win_count / nrow(recent)
        
        # Implied prob = 1 / odds
        avg_implied_prob = mean(1.0 ./ recent.odds)
        edge = actual_win_rate - avg_implied_prob
        
        # --- THE DECISION LOGIC ---
        status = "🔴 STOP"
        if roi > 5.0 && edge > 0.0
            status = "🟢 GO (Full Stake)"
        elseif roi > 0.0
            status = "🟡 CAUTION (Half Stake)"
        end
        
        # Shorten name for display
        short_name = replace(m_name, "GRWNegativeBinomial" => "NegBin", 
                                   "GRWBivariatePoisson" => "BivPois")
        m_p_short = match(r"μ=([0-9\.]+)", m_param).captures[1]

        # Print Report
        if status != "🔴 STOP" # Only show me the good stuff
            Printf.@printf("%s | %s (μ=%s) - %s\n", status, short_name, m_p_short, string(market))
            Printf.@printf("   ├─ ROI: %.1f%%  (Profit: %.2f)\n", roi, total_profit)
            Printf.@printf("   └─ Edge: %.1f%% (WinRate: %.2f vs Odds: %.2f)\n\n", edge*100, actual_win_rate, avg_implied_prob)
            
            push!(recommendations, (model=m_name, params=m_param, market=market))
        end
    end
    
    if isempty(recommendations)
        println("⚠️ WARNING: No models are currently profitable. DO NOT BET this weekend.")
    end
    
    return recommendations
end

# RUN THE CHECK
active_strategies = recommend_model(ledger.df, window=50)

######## Equity Curve Trading" or "Meta-Labeling."

using Turing, StatsPlots, DataFrames, Statistics


# We copy the dataframe to 'analysis_df' to avoid property accessor issues
analysis_df = copy(ledger.df)

# Now we modify 'analysis_df', not 'ledger.df'
analysis_df.date_week = week(analysis_df.date)

analysis_df

using StatsPlots 


over25_df
histogram(over25_df.pnl, 
          title = "PnL Distribution", 
          xlabel = "Profit/Loss", 
          ylabel = "Frequency",
          bins = 200,
          legend = false, 
          xlims=[-0.1,0.1]
          )

roi_data = filter(row -> row.stake > 0, over25_df)
roi_values = roi_data.pnl ./ roi_data.stake


using StatsPlots

histogram(roi_values .* 100, 
    title = "Distribution of ROI per Trade",
    xlabel = "ROI (%)",
    ylabel = "Frequency",
    bins = 50,
    fillcolor = :red,
    linecolor = :white,
    alpha = 0.7,
    legend = false)


# Grouped density or histogram by model_name
@df roi_data density(:pnl ./ :stake .* 100, 
    group = :model_name, 
    title = "ROI Distribution by Model",
    xlabel = "ROI (%)",
    fill = (0, 0.3))

describe(roi_values)


using Distributions
# Fit a Normal just to see the overlap (though it will fit poorly due to the -1 spike)
d_fit = fit(Normal, roi_values)


# Separate the 'Total Loss' group from the 'Returns' group
losses = roi_values[roi_values .== -1.0]
outcomes = roi_values[roi_values .> -1.0]

# Plot density of the outcomes and note the loss percentage
density(outcomes, 
    title = "ROI Distribution (Excluding Total Losses)",
    annotation = (0.5, 1.0, "Total Loss Frequency: $(length(losses)/length(roi_values)*100)%"),
    fill = (0, 0.2, :blue))

# Add a vertical line at 0% to see winners vs losers
vline!([0], color = :black, lw = 2, linestyle = :dash)

describe(analysis_df.pnl)

# Double check it worked (Optional debug print)
println("Columns in analysis_df: ", names(analysis_df))



## 
using Distributions
using Optim # For maximum likelihood estimation

# 1. Separate the data
is_total_loss = roi_values .== -1.0
p_hat = mean(is_total_loss)  # The probability of hitting the -1.0 spike

# 2. Extract the continuous part (the "survivors")
continuous_part = roi_values[.!is_total_loss]

# 3. Fit a distribution to the survivors
# Let's try a Normal first, or a SkewNormal for better fit
dist_cont = fit(Normal, continuous_part) 

println("Spike at -1.0 probability (p): ", round(p_hat, digits=3))
println("Continuous part distribution: ", dist_cont)


using StatsPlots

# Histogram of raw data
histogram(roi_values, bins=50, norm=:pdf, label="Data", alpha=0.5)

# Overlay the continuous fit scaled by (1 - p_hat)
x_range = range(minimum(continuous_part), maximum(continuous_part), length=200)
y_fit = pdf.(dist_cont, x_range) .* (1 - p_hat);

plot!(x_range, y_fit, lw=3, label="Fitted Continuous Part", color=:red)
# Represent the spike visually as a thin high bar
vline!([-1.0], lw=4, label="Point Mass (p=$p_hat)", color=:black)

names(over25_df)

###

using DataFrames, Statistics

# 1. Filter for active bets and calculate ROI
df_active = filter(row -> row.stake > 0, over25_df)
df_active.roi = df_active.pnl ./ df_active.stake

# 2. Calculate mixed distribution parameters per model/params
model_comparison = combine(groupby(df_active, [:model_name, :model_parameters]), 
    # n: Total number of bets
    :roi => length => :n,
    
    # p: Prob of total loss (The Spike at -1.0)
    :roi => (x -> mean(x .== -1.0)) => :p_loss,
    
    # mu: Mean of the continuous part (The Wins)
    :roi => (x -> mean(x[x .> -1.0])) => :mu_win,
    
    # sigma: Std of the continuous part
    :roi => (x -> std(x[x .> -1.0])) => :sigma_win,
    
    # EV: Overall mean ROI
    :roi => mean => :expected_roi
)

# Sort by the best performing model
sort!(model_comparison, :expected_roi, rev = true)

using StatsPlots

@df model_comparison scatter(
    :p_loss, 
    :expected_roi, 
    group = :model_name,
    markersize = :n ./ 10, # Bubble size represents number of bets
    xlabel = "Probability of Total Loss (p)",
    ylabel = "Expected ROI (EV)",
    title = "Model Performance: Risk vs. Reward",
    legend = :outerright
)

#

ds.matches

# 
df_joined = innerjoin(
    df_active, 
    ds.matches[:, [:match_id, :season, :tournament_slug]], 
    on = :match_id
)

model_season_stats = combine(groupby(df_joined, [:season, :model_name, :model_parameters]), 
    :roi => length => :n,
    :roi => (x -> mean(x .== -1.0)) => :p_loss,
    :roi => (x -> mean(x[x .> -1.0])) => :mu_win,
    :roi => mean => :expected_roi
)

# Sort to see the best season-model combinations
sort!(model_season_stats, :expected_roi, rev = true)

@df df_joined boxplot(:season, :roi, 
    group = :model_name, 
    title = "ROI Distribution by Season",
    ylabel = "ROI",
    outliers = false)

using StatsPlots

# Filter for a specific model to keep the comparison clean
target_model = "GRWNegativeBinomial"
sub_df = filter(row -> row.model_name == target_model, df_joined)

# Create a ridge plot to see the ROI distribution "bleeding" into the negative over seasons
@df sub_df ridgeline(:roi, :season, 
    fillalpha = 0.6, 
    title = "ROI Distribution Shift: $target_model",
    xlabel = "ROI (Return on Stake)")

# We use density instead of a histogram for a smoother "curve" comparison
@df sub_df density(:roi, 
    group = :season, 
    fill = (0, 0.2), 
    title = "ROI Density Shift: $target_model",
    xlabel = "ROI (Return on Stake)",
    ylabel = "Probability Density",
    xlims = (-1.1, 1.5), # Focus on the relevant ROI range
    legend = :topright)


@df sub_df violin(:season, :roi, 
    fillalpha=0.5, 
    label="",
    title = "ROI Distribution by Season")

@df sub_df boxplot!(:season, :roi, 
    fillalpha=0.2, 
    width=0.1, 
    label="Quartiles")

### funciton s
function get_market_analysis(df, market::String, sel::Symbol)
    # Subset by market name, selection, and active stakes
    sub_df = subset(df, 
        :market_name => ByRow(==(market)),
        :selection => ByRow(==(sel)),
        :stake => ByRow(>(0.0))
    )
    
    # Calculate ROI immediately so it's ready for plotting
    sub_df.roi = sub_df.pnl ./ sub_df.stake
    
    return sub_df
end

# Usage Examples:
over25_df = get_market_analysis(ledger.df, "OverUnder", :over_25)
btts_df   = get_market_analysis(ledger.df, "BTTS", :btts_yes)
home_df   = get_market_analysis(ledger.df, "1X2", :home)

# A quick way to compare all major markets in your ledger
market_summary = combine(groupby(subset(ledger.df, :stake => ByRow(>(0))), [:market_name, :selection]), 
    :pnl => (p -> mean((p ./ ledger.df.stake[p .!= 0]) .== -1.0)) => :p_loss,
    :pnl => (p -> mean(p ./ ledger.df.stake[p .!= 0])) => :avg_roi,
    nrow => :bet_count
)

# Filter for markets where you've actually placed a significant number of bets
filter!(row -> row.bet_count > 50, market_summary)


function get_global_market_summary(ledger_df, matches_df)
    # 1. Join with matches to get season info (select only necessary columns)
    df_joined = innerjoin(
        subset(ledger_df, :stake => ByRow(>(0))), 
        matches_df[:, [:match_id, :season]], 
        on = :match_id
    )
    
    # 2. Add ROI column
    df_joined.roi = df_joined.pnl ./ df_joined.stake
    
    # 3. Aggregate by Market, Selection, Season, and Model
    summary = combine(groupby(df_joined, [:market_name, :selection, :season, :model_name, :model_parameters]), 
        :roi => length => :n,
        :roi => (x -> mean(x .== -1.0)) => :p_loss,
        :roi => (x -> mean(x[x .> -1.0])) => :mu_win,
        :roi => mean => :expected_roi
    )
    
    return sort(summary, :expected_roi, rev = true)
end

# Execution
global_stats = get_global_market_summary(ledger.df, ds.matches)

significant_stats = subset(global_stats, :n => ByRow(>=(30)))

####

# 1. PREPARE DATA
ledger.df

over25_df = subset(analysis_df, :selection => ByRow(isequal(:over_25)))


# Group ledger by week and calculate ROI for a SPECIFIC strategy (e.g., NegBin Over 2.5)
# weekly_roi = Profit / Stake
weekly_data = combine(groupby(ledger.df, :date_week), 
                      :pnl => sum => :profit, 
                      :stake => sum => :stake)

sort!(weekly_data, :date_week)

# Our observation: Realized ROI (e.g., 0.15, -0.05, ...)
y = weekly_data.profit ./ weekly_data.stake

# 2. THE META-MODEL (Latent Edge GRW)

###
# - subset of league 2 24/25 

dd = subset( ds.matches, :tournament_id => ByRow(in([57, 56])), :season => ByRow(isequal("24/25")))
ids_ref = collect( dd.match_id) 

ledge_1 = subset( ledger.df, :match_id => ByRow(in(ids_ref)))

tearsheet_1 = BayesianFootball.BackTesting.generate_tearsheet(ll)

ll = deepcopy(ledger)

subset!( ll.df, :match_id => ByRow(in(ids_ref)))


for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet_1, :selection => ByRow(isequal(m_name)))
    show(sub)
end

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


#= 


 Model: grw_neg_bin                                                                                                  
                                                                                                                     
Chains MCMC chain (250×4×2 Array{Float64, 3}):                                                                       
                                                                                                                     
Iterations        = 51:1:300                                                                                         
Number of chains  = 2                                                                                                
Samples per chain = 250                                                                                              
Wall duration     = 249.56 seconds                                                                                   
Compute duration  = 497.66 seconds                                                                                   
parameters        = μ, γ, σ_att, σ_def                                                                               
internals         =                                                                                                  
                                                                                                                     
Summary Statistics                                                                                                   
                                                                                                                     
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec                             
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64                             
                                                                                                                     
           μ    0.2693    0.0869    0.0042   436.3733   311.9187    1.0174        0.8769                             
           γ    0.1701    0.1108    0.0050   496.8303   317.8725    1.0023        0.9983                             
       σ_att    0.0461    0.0266    0.0012   486.8139   360.3088    1.0023        0.9782                             
       σ_def    0.0604    0.0355    0.0017   411.4484   352.9079    0.9980        0.8268                             
                                                                                                                     
                                                                                                                     
Quantiles                                                                                                            
                                                                                                                     
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%                                                       
      Symbol   Float64   Float64   Float64   Float64   Float64                                                       
                                                                                                                     
           μ    0.1074    0.2116    0.2675    0.3225    0.4525                                                       
           γ   -0.0474    0.1028    0.1709    0.2416    0.4049                                                       
       σ_att    0.0086    0.0260    0.0430    0.0601    0.1070                                                       
       σ_def    0.0081    0.0339    0.0541    0.0810    0.1442   



=# 


using Statistics, Distributions

"""
    make_league_priors(df_train)

Calculates the 'Physics' of the league from raw data to set intelligent priors.
Returns NamedTuple with Normal distributions for μ and γ.
"""
function make_league_priors(df_train)
    # 1. Calculate Average Goals (The "Energy" of the league)
    # Total goals divided by total matches
    avg_goals_per_match = (sum(df_train.home_score) + sum(df_train.away_score)) / nrow(df_train)
    
    # 2. Calculate Home Advantage (Ratio)
    avg_h = mean(df_train.home_score)
    avg_a = mean(df_train.away_score)
    # Avoid division by zero in weird edge cases
    raw_home_adv = avg_a > 0 ? avg_h / avg_a : 1.3 

    println("\n--- ⚡ Data-Driven Priors Calculated ⚡ ---")
    println("  Avg Goals/Match: $(round(avg_goals_per_match, digits=3))")
    println("  Implied μ (team): $(round(log(avg_goals_per_match/2), digits=3))")
    println("  Home Adv Ratio:  $(round(raw_home_adv, digits=3))")

    # 3. Create Priors
    # We use log() because your model uses Log-Links
    # avg_goals_per_match = exp(μ_h) + exp(μ_a) ≈ 2 * exp(μ)
    # -> μ ≈ log(avg / 2)
    
    target_mu = log(avg_goals_per_match / 2.0)
    target_gamma = log(raw_home_adv)

    # We return Normal distributions centered on the truth, 
    # but with enough variance (0.2) to let the sampler adjust slightly.
    return (;
        prior_μ = Normal(target_mu, 0.25),
        prior_γ = Normal(target_gamma, 0.25)
    )
end



using DataFrames
dd = subset( ds.matches, :tournament_id => ByRow(x -> x ∈[56,57])) 
ddd = subset( ds.matches, :tournament_id => ByRow(x -> x ∈[54,55])) 

make_league_priors(dd)
make_league_priors(ddd)

names(ds.matches)


#= 
BayesianFootball.Models.PreGame.GRWNegativeBinomial(Distributions.Normal{Float64}(μ=0.32, σ=0.05), Distributions.Normal{Float64}(μ=0.12, σ=0.05), Distributions.Normal{Float64}(μ=1.5, σ=1.0), Distributions.Gamma{Float64}(α=2.0, θ=0.05), Distributions.Gamma{Float64}(α=2.0, θ=0.08), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0))


BayesianFootball.Models.PreGame.GRWBivariatePoisson(Distributions.Normal{Float64}(μ=0.2, σ=0.5), Distributions.Normal{Float64}(μ=0.26236426446749106, σ=0.2), Distributions.Normal{Float64}(μ=-2.0, σ=1.0), Distributions.Gamma{Float64}(α=2.0, θ=0.05), Distributions.Gamma{Float64}(α=2.0, θ=0.4), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0))


BayesianFootball.Models.PreGame.GRWNegativeBinomial(Distributions.Normal{Float64}(μ=0.2, σ=0.5), Distributions.Normal{Float64}(μ=0.26236426446749106, σ=0.2), Distributions.Normal{Float64}(μ=1.5, σ=1.0), Distributions.Gamma{Float64}(α=2.0, θ=0.05), Distributions.Gamma{Float64}(α=2.0, θ=0.4), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0))


BayesianFootball.Models.PreGame.GRWDixonColes(Distributions.Normal{Float64}(μ=0.2, σ=0.5), Distributions.Normal{Float64}(μ=0.26236426446749106, σ=0.2), Distributions.Gamma{Float64}(α=2.0, θ=0.05), Distributions.Gamma{Float64}(α=2.0, θ=0.4), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0))


BayesianFootball.Models.PreGame.GRWPoisson(Distributions.Normal{Float64}(μ=0.2, σ=0.5), Distributions.Normal{Float64}(μ=0.26236426446749106, σ=0.2), Distributions.Gamma{Float64}(α=2.0, θ=0.05), Distributions.Gamma{Float64}(α=2.0, θ=0.4), Distributions.Normal{Float64}(μ=0.0, σ=1.0), Distributions.Normal{Float64}(μ=0.0, σ=1.0))

julia> for m_name in model_names
           println("\nStats for: $m_name")
           sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
           show(sub)
       end

Stats for: home
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           home                 866          248          28.6      7.41    -1.86   -25.03          28.2       -0.063            -1.856       -0.782      -0.043         -2.006        -0.091
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           home                 866          280          32.3     11.8     -1.79   -15.17          37.5       -0.046            -1.791       -0.684      -0.041         -1.717        -0.065
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           home                 866          303          35.0     10.56    -2.17   -20.57          35.3       -0.059            -2.172       -0.748      -0.042         -1.901        -0.084
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           home                 866          294          33.9     12.15    -2.09   -17.21          36.7       -0.054            -2.092       -0.725      -0.044         -1.871        -0.075
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           home                 866          308          35.6     12.4     -2.29   -18.47          37.0       -0.059            -2.291       -0.766      -0.044         -1.455        -0.082
Stats for: draw
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           draw                 866           50           5.8      0.58     0.79   136.8           18.0        0.032             0.789        3.18        0.292          6.269         0.246
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           draw                 866          111          12.8      0.96     0.15    15.85          17.1        0.01              0.152        0.44        0.042          0.781         0.04
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           draw                 866           72           8.3      0.56     0.62   110.82          19.4        0.031             0.623        3.647       0.278          6.204         0.235
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           draw                 866          105          12.1      0.87     0.28    32.27          23.8        0.026             0.279        1.8         0.114          4.75          0.073
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           draw                 866           71           8.2      0.51     0.39    76.08          19.7        0.028             0.387        2.226       0.205          4.867         0.157
Stats for: away
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           away                 866          436          50.3     12.7     -2.26   -17.78          24.5       -0.047            -2.258       -0.716      -0.041         -2.379        -0.085
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           away                 866          289          33.4      7.72    -1.87   -24.2           28.7       -0.056            -1.869       -0.849      -0.044         -2.717        -0.086
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           away                 866          319          36.8      8.0     -1.91   -23.84          25.4       -0.056            -1.907       -0.835      -0.043         -1.335        -0.089
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           away                 866          310          35.8      8.41    -1.65   -19.65          27.1       -0.05             -1.651       -0.761      -0.04          -2.655        -0.077
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           away                 866          309          35.7      8.1     -1.88   -23.19          26.5       -0.055            -1.878       -0.841      -0.043         -2.249        -0.087
Stats for: btts_yes
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           btts_yes             853          181          21.2      6.08     0.39     6.34          52.5        0.015             0.386        0.571       0.042          1.289         0.024
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_yes             853          193          22.6      7.47     0.43     5.7           54.4        0.014             0.426        0.503       0.037          1.241         0.023
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_yes             853           91          10.7      3.36     0.13     3.73          56.0        0.006             0.125        0.271       0.015          0.413         0.01
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_yes             853          168          19.7      6.85     0.04     0.52          56.5        0.001             0.035        0.035       0.003          0.079         0.002
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_yes             853          164          19.2      6.44     0.27     4.23          55.5        0.01              0.272        0.357       0.025          0.831         0.015
Stats for: btts_no
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           btts_no              853          167          19.6      3.87    -1.12   -28.93          41.3       -0.075            -1.119       -0.833      -0.041         -1.659        -0.092
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_no              853          233          27.3      6.34    -1.79   -28.22          44.2       -0.087            -1.788       -0.825      -0.041         -1.649        -0.103
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_no              853          379          44.4     14.39    -3.06   -21.28          44.6       -0.091            -3.062       -0.756      -0.039         -1.505        -0.112
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_no              853          296          34.7     10.19    -2.01   -19.72          46.6       -0.07             -2.009       -0.695      -0.038         -1.294        -0.085
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_no              853          289          33.9      9.9     -2.37   -23.88          46.0       -0.086            -2.365       -0.766      -0.04          -2.294        -0.103
Stats for: over_05
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_05              782           23           2.9      0.85     0.06     7.1           95.7        0.091             0.061       24.598      10.253         24.598         0.88
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_05              782           20           2.6      1.02     0.05     5.22          95.0        0.093             0.053       42.684      17.682         42.684         1.527
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_05              782           10           1.3      0.28     0.02     6.79         100.0        0.068             0.019        0.0         0.0            0.0         999.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_05              782           52           6.6      3.2      0.0      0.1           94.2        0.001             0.003        0.029       0.005          0.088         0.001
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_05              782           28           3.6      1.72     0.06     3.69          92.9        0.063             0.064        2.83        0.985          4.703         0.099
Stats for: under_05
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_05             782           56           7.2      0.2     -0.11   -53.5            7.1       -0.051            -0.108       -0.973      -0.047         -0.973        -0.086
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_05             782          153          19.6      1.04    -0.18   -17.72           7.2       -0.018            -0.183       -0.433      -0.026         -0.734        -0.045
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_05             782          160          20.5      1.08    -0.02    -2.11           6.9       -0.002            -0.023       -0.056      -0.003         -0.15         -0.006
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_05             782          133          17.0      1.05    -0.37   -34.98           7.5       -0.037            -0.366       -0.639      -0.038         -0.639        -0.087
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_05             782           97          12.4      0.68    -0.0     -0.48           6.2       -0.0              -0.003       -0.012      -0.001         -0.034        -0.001
Stats for: over_15
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_15              782          152          19.4      6.57     0.94    14.28          78.9        0.102             0.937        7.721       0.984         10.684         0.167
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_15              782          125          16.0      5.57     0.61    10.94          77.6        0.069             0.609        3.635       0.374          6.251         0.104
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_15              782           81          10.4      3.15     0.48    15.31          81.5        0.078             0.482        3.767       0.59          10.262         0.123
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_15              782          163          20.8      8.91     0.53     5.92          76.7        0.038             0.527        1.372       0.146          3.147         0.048
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_15              782          146          18.7      7.35     0.69     9.45          76.7        0.061             0.695        3.025       0.363          5.592         0.082
Stats for: under_15
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_15             782          178          22.8      2.39    -0.51   -21.35          21.9       -0.037            -0.51        -0.669      -0.037         -1.199        -0.07
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_15             782          318          40.7      7.43    -1.77   -23.78          22.0       -0.065            -1.767       -0.7        -0.038         -1.284        -0.103
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_15             782          350          44.8      8.19    -1.73   -21.15          21.4       -0.06             -1.732       -0.693      -0.038         -1.283        -0.098
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_15             782          284          36.3      6.83    -1.15   -16.88          23.2       -0.042            -1.152       -0.574      -0.034         -1.099        -0.068
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_15             782          285          36.4      6.38    -1.26   -19.73          22.8       -0.049            -1.259       -0.639      -0.036         -1.178        -0.08
Stats for: over_25
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_25              839          416          49.6     17.31     1.04     6.03          50.0        0.026             1.043        0.753       0.078          1.733         0.04
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_25              839          304          36.2     13.64     0.83     6.06          51.6        0.023             0.826        0.806       0.075          1.634         0.036
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_25              839          271          32.3      9.94     0.71     7.13          51.3        0.025             0.709        0.838       0.085          1.838         0.039
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_25              839          311          37.1     14.73     0.44     3.02          51.8        0.012             0.445        0.356       0.031          0.74          0.018
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_25              839          317          37.8     14.85     0.48     3.21          52.7        0.013             0.477        0.378       0.035          0.803         0.019
Stats for: under_25
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_25             839          144          17.2      3.79    -0.64   -16.93          44.4       -0.043            -0.642       -0.857      -0.042         -1.436        -0.061
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_25             839          283          33.7     11.01    -1.67   -15.19          48.1       -0.056            -1.672       -0.787      -0.039         -1.563        -0.071
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_25             839          297          35.4     11.74    -1.72   -14.61          47.8       -0.056            -1.715       -0.799      -0.039         -2.384        -0.071
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_25             839          284          33.8     12.34    -1.88   -15.25          48.9       -0.057            -1.883       -0.801      -0.04          -1.598        -0.072
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_25             839          278          33.1     11.59    -1.85   -15.97          49.3       -0.06             -1.852       -0.809      -0.041         -2.42         -0.075
Stats for: over_35
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_35              782          469          60.0     13.44     1.53    11.42          28.1        0.03              1.534        0.785       0.094          1.908         0.07
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_35              782          337          43.1      9.33     0.81     8.64          27.9        0.018             0.807        0.518       0.057          1.382         0.042
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_35              782          316          40.4      7.65     0.76     9.97          26.6        0.02              0.763        0.623       0.068          1.4           0.048
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_35              782          303          38.7      8.88     0.75     8.45          27.4        0.018             0.75         0.498       0.052          1.173         0.04
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_35              782          306          39.1      8.93     0.69     7.76          28.1        0.016             0.693        0.468       0.052          1.096         0.037
Stats for: under_35
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_35             782           79          10.1      2.3     -0.19    -8.36          62.0       -0.025            -0.193       -0.633      -0.036         -1.538        -0.03
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_35             782          163          20.8      8.17    -0.82   -10.07          71.8       -0.044            -0.823       -0.644      -0.034         -2.067        -0.049
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_35             782          172          22.0      8.27    -0.87   -10.51          70.3       -0.046            -0.869       -0.64       -0.035         -2.131        -0.051
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_35             782          188          24.0     11.77    -1.08    -9.17          70.7       -0.044            -1.079       -0.614      -0.032         -2.033        -0.048
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_35             782          185          23.7     10.99    -1.05    -9.53          71.4       -0.046            -1.047       -0.629      -0.034         -2.056        -0.051
Stats for: over_45
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_45              782          453          57.9      8.17    -2.3    -28.13          14.1       -0.063            -2.297       -0.612      -0.036         -2.847        -0.119
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_45              782          298          38.1      5.58    -2.22   -39.81          13.4       -0.078            -2.223       -0.748      -0.039         -3.513        -0.133
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_45              782          295          37.7      5.09    -2.1    -41.26          14.2       -0.081            -2.1         -0.78       -0.041         -3.659        -0.138
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_45              782          253          32.4      4.87    -2.15   -44.16          13.0       -0.082            -2.148       -0.767      -0.04          -2.898        -0.136
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_45              782          258          33.0      4.86    -2.16   -44.43          14.0       -0.083            -2.161       -0.777      -0.041         -2.929        -0.135
Stats for: under_45
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_45             782           34           4.3      0.81    -0.07    -8.09          82.4       -0.021            -0.066       -0.712      -0.034         -0.871        -0.022
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_45             782           90          11.5      4.53    -0.44    -9.65          82.2       -0.045            -0.438       -0.925      -0.054         -0.925        -0.046
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_45             782           80          10.2      4.17    -0.38    -9.08          85.0       -0.04             -0.379       -0.774      -0.048         -2.147        -0.042
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_45             782          121          15.5      9.0     -0.78    -8.7           82.6       -0.049            -0.783       -0.803      -0.048         -2.146        -0.051
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_45             782          122          15.6      8.18    -0.83   -10.15          84.4       -0.053            -0.83        -0.851      -0.052         -2.228        -0.055
Stats for: over_55
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_55              831          278          33.5      3.34    -0.66   -19.84           6.8       -0.022            -0.662       -0.412      -0.023         -0.924        -0.054
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_55              831          195          23.5      2.42    -0.57   -23.49           6.2       -0.022            -0.569       -0.514      -0.024         -1.083        -0.054
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_55              831          202          24.3      2.43    -0.62   -25.67           5.9       -0.027            -0.624       -0.548      -0.028         -1.177        -0.062
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_55              831          168          20.2      2.08    -0.4    -19.45           6.0       -0.016            -0.404       -0.388      -0.019         -0.837        -0.039
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_55              831          163          19.6      2.04    -0.45   -22.15           6.7       -0.019            -0.453       -0.454      -0.022         -0.968        -0.045
Stats for: under_55
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_55             831           11           1.3      0.13     0.01     7.46         100.0        0.057             0.01    5.80481e15  2.84947e14     5.80481e15    2.01488e14
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_55             831           41           4.9      1.49     0.06     4.07          97.6        0.076             0.061   3.262       0.351          3.262         0.113
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_55             831           33           4.0      1.15     0.05     4.69          97.0        0.116             0.054  35.122       5.527         35.122         1.219
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_55             831           70           8.4      4.92     0.16     3.18          97.1        0.076             0.156   2.943       0.339          4.116         0.094
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_55             831           64           7.7      4.37     0.06     1.44          95.3        0.02              0.063   0.714       0.066          1.343         0.021
Stats for: over_65
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_65              838          118          14.1      0.9     -0.16   -18.14           3.4       -0.012            -0.164       -0.279      -0.015         -0.405        -0.027
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_65              838           81           9.7      0.57     0.16    27.7            4.9        0.01              0.159        0.447       0.022          0.662         0.035
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_65              838           88          10.5      0.64    -0.01    -2.12           4.5       -0.001            -0.013       -0.033      -0.002         -0.049        -0.003
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_65              838           59           7.0      0.52     0.18    34.72           5.1        0.011             0.18         0.513       0.024          0.813         0.038
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_65              838           57           6.8      0.48     0.15    30.59           5.3        0.01              0.146        0.461       0.023          0.715         0.035
Stats for: under_65
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_65             838            0           0.0      0.0      0.0      0.0            0.0        0.0               0.0            0.0         0.0            0.0           0.0
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_65             838            6           0.7      0.11     0.0      1.14         100.0        0.065             0.001          0.0         0.0            0.0         999.0
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_65             838            3           0.4      0.11     0.0      1.0          100.0        0.049             0.001          0.0         0.0            0.0         999.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_65             838           27           3.2      1.49     0.02     1.29         100.0        0.113             0.019          0.0         0.0            0.0         999.0
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_65             838           25           3.0      1.07     0.02     1.41         100.0        0.129             0.015          0.0         0.0            0.0         999.0
Stats for: over_75
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_75              667           39           5.8      0.27    -0.27   -99.94           2.6       -0.105            -0.267       -0.999      -0.045         -0.999        -0.105
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_75              667           23           3.4      0.16    -0.16   -98.18           4.3       -0.084            -0.156       -0.983      -0.045         -0.983        -0.083
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_75              667           31           4.6      0.18    -0.18  -100.0            0.0       -0.099            -0.179       -1.0        -0.047         -1.0          -0.098
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_75              667           20           3.0      0.15    -0.15  -100.0            0.0       -0.08             -0.153       -1.0        -0.045         -1.0          -0.08
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_75              667           19           2.8      0.13    -0.13  -100.0            0.0       -0.08             -0.127       -1.0        -0.045         -1.0          -0.08
Stats for: under_75
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_75             667            0           0.0      0.0       0.0      0.0           0.0        0.0                 0.0          0.0         0.0            0.0           0.0
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_75             667            0           0.0      0.0       0.0      0.0           0.0        0.0                 0.0          0.0         0.0            0.0           0.0
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_75             667            0           0.0      0.0       0.0      0.0           0.0        0.0                 0.0          0.0         0.0            0.0           0.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_75             667            4           0.6      0.24      0.0      0.2         100.0        0.052               0.0          0.0         0.0            0.0         999.0
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_75             667            3           0.4      0.02      0.0      0.2         100.0        0.051               0.0          0.0         0.0            0.0         999.0
Stats for: over_85
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_85               40            3           7.5      0.02    -0.02   -100.0           0.0       -0.179            -0.024         -1.0      -0.18            -1.0        -0.179
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_85               40            2           5.0      0.02    -0.02   -100.0           0.0       -0.188            -0.017         -1.0      -0.171           -1.0        -0.187
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_85               40            3           7.5      0.02    -0.02   -100.0           0.0       -0.192            -0.018         -1.0      -0.183           -1.0        -0.191
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_85               40            3           7.5      0.02    -0.02   -100.0           0.0       -0.201            -0.015         -1.0      -0.171           -1.0        -0.2
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_85               40            2           5.0      0.01    -0.01   -100.0           0.0       -0.187            -0.012         -1.0      -0.171           -1.0        -0.186
Stats for: under_85
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_85              40            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_85              40            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_85              40            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_85              40            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_85              40            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
Stats for: over_95
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_95                6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_95                6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_95                6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_95                6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_95                6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
Stats for: under_95
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_95               6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_95               6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_95               6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_95               6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_95               6            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
Stats for: over_105
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           over_105               1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_105               1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_105               1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_105               1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_105               1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
Stats for: under_105
5×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWNegativeBinomial  μ=Normal(μ=0.32, σ=0.05), γ=Norm…  BayesianKelly  none           under_105              1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   2 │ GRWBivariatePoisson  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_105              1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   3 │ GRWNegativeBinomial  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_105              1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   4 │ GRWDixonColes        μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_105              1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   5 │ GRWPoisson           μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_105              1            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0

Scottish PL and CH

julia> for m_name in model_names
           println("\nStats for: $m_name")
           sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
           show(sub)
       end

Stats for: home
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           home                 830          342          41.2     11.93    -3.51   -29.43          27.5       -0.082            -3.511       -0.939      -0.05          -3.456        -0.142
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           home                 830          360          43.4     13.71    -4.78   -34.87          25.6       -0.103            -4.78        -0.966      -0.05          -2.751        -0.166
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           home                 830          349          42.0     13.43    -4.42   -32.89          28.7       -0.104            -4.416       -0.961      -0.049         -1.812        -0.161
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           home                 830          349          42.0     12.57    -3.99   -31.73          27.5       -0.092            -3.988       -0.951      -0.05          -3.503        -0.154
Stats for: draw
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           draw                 830          136          16.4      2.64    -0.68   -25.7           11.8       -0.036            -0.678       -0.771      -0.04          -1.494        -0.076
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           draw                 830          131          15.8      3.02    -0.92   -30.36          12.2       -0.044            -0.918       -0.788      -0.039         -1.521        -0.092
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           draw                 830          126          15.2      2.12    -0.4    -18.89          14.3       -0.026            -0.4         -0.642      -0.035         -1.821        -0.053
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           draw                 830          128          15.4      2.51    -0.66   -26.48          12.5       -0.036            -0.664       -0.785      -0.04          -1.522        -0.077
Stats for: away
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           away                 830          412          49.6     12.58    -2.46   -19.52          21.8       -0.052            -2.455       -1.005      -0.052         -1.005        -0.104
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           away                 830          421          50.7     14.25    -2.24   -15.72          20.7       -0.038            -2.241       -0.892      -0.051         -0.892        -0.087
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           away                 830          415          50.0     13.02    -2.75   -21.11          21.9       -0.057            -2.747       -1.01       -0.051         -1.01         -0.111
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           away                 830          406          48.9     12.41    -2.39   -19.29          21.4       -0.051            -2.393       -1.004      -0.054         -1.004        -0.102
Stats for: btts_yes
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_yes             823          546          66.3     30.64    -2.53    -8.26          48.4       -0.045            -2.531       -0.713      -0.054         -2.79         -0.061
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_yes             823          338          41.1     12.64    -0.55    -4.38          46.7       -0.016            -0.554       -0.379      -0.021         -1.449        -0.023
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_yes             823          449          54.6     22.14    -1.67    -7.54          46.3       -0.035            -1.669       -0.592      -0.042         -1.928        -0.049
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_yes             823          451          54.8     23.1     -1.62    -7.0           47.2       -0.033            -1.616       -0.578      -0.041         -1.763        -0.046
Stats for: btts_no
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_no              823           24           2.9      0.28     0.05    16.24          54.2        0.009             0.045        0.669       0.03           0.726         0.027
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_no              823           92          11.2      1.31    -0.1     -7.37          44.6       -0.01             -0.096       -0.312      -0.02          -0.631        -0.021
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_no              823           63           7.7      1.06    -0.05    -4.38          41.3       -0.006            -0.047       -0.194      -0.012         -0.357        -0.011
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           btts_no              823           58           7.0      0.78    -0.02    -2.05          41.4       -0.002            -0.016       -0.093      -0.005         -0.167        -0.005
Stats for: over_05
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_05              786          227          28.9     18.01     0.64     3.56          93.0        0.075             0.641        2.522       0.264          5.982         0.085
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_05              786          120          15.3      7.42     0.49     6.63          94.2        0.18              0.492       14.209       1.634         27.102         0.353
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_05              786          249          31.7     24.74     0.72     2.9           93.2        0.07              0.719        2.785       0.273          4.915         0.079
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_05              786          214          27.2     18.15     0.71     3.89          93.5        0.091             0.707        3.089       0.344          7.328         0.108
Stats for: under_05
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_05             786           32           4.1      0.1     -0.1    -98.01           3.1       -0.13             -0.1           -1.0      -0.051           -1.0        -0.13
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_05             786           72           9.2      0.27    -0.24   -89.2            4.2       -0.144            -0.237         -1.0      -0.053           -1.0        -0.15
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_05             786           37           4.7      0.13    -0.13   -98.7            5.4       -0.135            -0.126         -1.0      -0.05            -1.0        -0.134
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_05             786           36           4.6      0.13    -0.13   -97.98           2.8       -0.131            -0.129         -1.0      -0.052           -1.0        -0.13
Stats for: over_15
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_15              786          424          53.9     31.93     0.19     0.59          73.1        0.005             0.188        0.118       0.009          0.333         0.006
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_15              786          310          39.4     16.65     0.22     1.34          73.9        0.009             0.224        0.234       0.015          0.605         0.01
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_15              786          371          47.2     28.02    -0.06    -0.22          72.5       -0.002            -0.061       -0.039      -0.003         -0.118        -0.002
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_15              786          387          49.2     29.37     0.48     1.64          73.4        0.013             0.482        0.324       0.024          0.948         0.015
Stats for: under_15
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_15             786          108          13.7      1.11    -0.11   -10.25          20.4       -0.01             -0.114       -0.276      -0.017         -0.652        -0.023
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_15             786          169          21.5      2.29    -0.12    -5.41          20.1       -0.006            -0.124       -0.178      -0.011         -0.511        -0.016
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_15             786          137          17.4      1.77    -0.34   -19.34          20.4       -0.024            -0.343       -0.6        -0.037         -1.688        -0.051
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_15             786          132          16.8      1.54    -0.18   -11.88          19.7       -0.012            -0.183       -0.346      -0.022         -0.638        -0.03
Stats for: over_25
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_25              793          540          68.1     38.94    -2.48    -6.38          48.9       -0.037            -2.484       -0.601      -0.051         -2.083        -0.05
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_25              793          483          60.9     26.33    -2.13    -8.07          48.2       -0.042            -2.126       -0.641      -0.052         -2.611        -0.057
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_25              793          508          64.1     33.07    -2.05    -6.2           48.4       -0.034            -2.05        -0.584      -0.051         -2.298        -0.046
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_25              793          516          65.1     33.73    -2.08    -6.16          48.4       -0.034            -2.079       -0.563      -0.048         -2.245        -0.046
Stats for: under_25
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_25             793           89          11.2      2.04     0.11     5.61          36.0        0.008             0.114        0.317       0.023          0.594         0.016
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_25             793          131          16.5      3.58     0.06     1.58          32.8        0.003             0.057        0.089       0.007          0.203         0.005
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_25             793          116          14.6      3.24    -0.31    -9.51          35.3       -0.017            -0.308       -0.532      -0.035         -1.161        -0.029
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_25             793          114          14.4      3.08    -0.21    -6.93          36.0       -0.011            -0.214       -0.388      -0.027         -0.731        -0.02
Stats for: over_35
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_35              786          570          72.5     28.89    -1.84    -6.36          28.1       -0.024            -1.836       -0.348      -0.028         -1.283        -0.045
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_35              786          543          69.1     22.86    -1.22    -5.35          27.8       -0.019            -1.223       -0.301      -0.024         -1.09         -0.036
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_35              786          522          66.4     23.54    -1.12    -4.75          28.0       -0.017            -1.119       -0.26       -0.021         -0.952        -0.031
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_35              786          533          67.8     23.96    -1.47    -6.12          28.1       -0.021            -1.467       -0.32       -0.025         -1.18         -0.04
Stats for: under_35
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_35             786           58           7.4      1.96     0.1      5.34          56.9        0.012             0.105        0.415       0.023          0.936         0.017
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_35             786           75           9.5      2.95     0.17     5.82          53.3        0.014             0.172        0.457       0.028          0.956         0.023
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_35             786           84          10.7      3.42    -0.19    -5.62          57.1       -0.015            -0.192       -0.359      -0.021         -0.749        -0.02
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_35             786           82          10.4      3.27     0.02     0.47          53.7        0.001             0.015        0.035       0.002          0.068         0.002
Stats for: over_45
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_45              786          548          69.7     16.41    -3.0    -18.3           13.5       -0.05             -3.004       -0.73       -0.047         -2.917        -0.104
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_45              786          549          69.8     14.61    -2.49   -17.06          13.3       -0.045            -2.492       -0.689      -0.044         -2.761        -0.099
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_45              786          483          61.5     12.76    -1.64   -12.84          13.5       -0.03             -1.638       -0.614      -0.039         -2.3          -0.067
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_45              786          495          63.0     12.98    -2.02   -15.54          13.3       -0.037            -2.017       -0.692      -0.044         -2.654        -0.081
Stats for: under_45
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_45             786           35           4.5      1.36     0.17    12.82          82.9        0.04              0.174        1.782       0.177          3.713         0.056
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_45             786           38           4.8      1.78     0.29    16.06          84.2        0.053             0.286        2.534       0.339          5.066         0.078
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_45             786           55           7.0      2.68     0.16     5.84          74.5        0.021             0.157        0.959       0.054          1.34          0.026
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_45             786           48           6.1      2.62     0.26     9.98          75.0        0.037             0.261        1.691       0.153          3.12          0.049
Stats for: over_55
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_55              792          425          53.7      6.59    -3.22   -48.85           4.7       -0.101            -3.219       -1.003      -0.057         -1.003        -0.197
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_55              792          453          57.2      6.46    -3.32   -51.46           4.6       -0.11             -3.323       -1.003      -0.055         -1.003        -0.222
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_55              792          343          43.3      4.94    -2.47   -50.02           5.0       -0.096            -2.471       -1.001      -0.054         -1.001        -0.178
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_55              792          350          44.2      5.05    -2.5    -49.4            5.1       -0.091            -2.496       -1.002      -0.054         -1.002        -0.179
Stats for: under_55
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_55             792           22           2.8      0.62     0.02     3.88          90.9        0.014             0.024        0.469       0.033          0.469         0.017
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_55             792           21           2.7      0.69     0.05     7.8           90.5        0.041             0.054        1.92        0.134          3.84          0.065
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_55             792           32           4.0      1.66    -0.0     -0.19          87.5       -0.001            -0.003       -0.023      -0.002         -0.078        -0.001
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_55             792           30           3.8      1.66     0.0      0.18          90.0        0.001             0.003        0.022       0.002          0.038         0.001
Stats for: over_65
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_65              788          283          35.9      2.33    -1.59   -68.26           1.8       -0.117            -1.588       -1.001      -0.047         -1.001        -0.198
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_65              788          323          41.0      2.64    -1.81   -68.67           1.5       -0.123            -1.812       -1.001      -0.046         -1.001        -0.227
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_65              788          230          29.2      1.68    -1.08   -64.4            1.7       -0.095            -1.083       -0.95       -0.045         -0.95         -0.162
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_65              788          234          29.7      1.73    -1.14   -66.21           1.7       -0.101            -1.144       -0.94       -0.044         -0.94         -0.17
Stats for: under_65
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_65             788            4           0.5      0.14     0.01     7.29         100.0        0.051             0.01           0.0         0.0            0.0         999.0
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_65             788            3           0.4      0.13     0.01     7.18         100.0        0.044             0.01           0.0         0.0            0.0         999.0
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_65             788           13           1.6      0.54     0.03     5.65         100.0        0.062             0.03           0.0         0.0            0.0         999.0
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_65             788           13           1.6      0.55     0.03     5.6          100.0        0.058             0.031          0.0         0.0            0.0         999.0
Stats for: over_75
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_75              393          110          28.0      0.62    -0.62   -100.0           0.0       -0.226            -0.618         -1.0      -0.062           -1.0        -0.221
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_75              393          126          32.1      0.76    -0.76   -100.0           0.0       -0.273            -0.757         -1.0      -0.06            -1.0        -0.264
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_75              393           92          23.4      0.45    -0.45   -100.0           0.0       -0.2              -0.454         -1.0      -0.061           -1.0        -0.196
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_75              393           90          22.9      0.45    -0.45   -100.0           0.0       -0.206            -0.454         -1.0      -0.061           -1.0        -0.202
Stats for: under_75
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_75             393            0           0.0      0.0       0.0     0.0            0.0        0.0               0.0            0.0         0.0            0.0           0.0
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_75             393            0           0.0      0.0       0.0     0.0            0.0        0.0               0.0            0.0         0.0            0.0           0.0
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_75             393            3           0.8      0.05      0.0     1.51         100.0        0.082             0.001          0.0         0.0            0.0         999.0
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_75             393            3           0.8      0.05      0.0     1.42         100.0        0.071             0.001          0.0         0.0            0.0         999.0
Stats for: over_85
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_85               86           21          24.4      0.1     -0.1    -100.0           0.0       -0.201            -0.102       -1.0        -0.128         -1.0          -0.199
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_85               86           28          32.6      0.14    -0.14   -100.0           0.0       -0.259            -0.136       -1.001      -0.127         -1.001        -0.252
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_85               86           18          20.9      0.08    -0.08   -100.0           0.0       -0.183            -0.075       -1.0        -0.126         -1.0          -0.181
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_85               86           20          23.3      0.07    -0.07   -100.0           0.0       -0.183            -0.069       -1.0        -0.123         -1.0          -0.181
Stats for: under_85
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_85              86            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_85              86            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_85              86            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_85              86            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
Stats for: over_95
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_95               11            3          27.3       0.0     -0.0   -100.0           0.0       -0.322            -0.002       -1.8        -0.9           -1.8          -0.321
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_95               11            5          45.5       0.0     -0.0   -100.0           0.0       -0.602            -0.005       -2.097      -0.894         -2.097        -0.536
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_95               11            3          27.3       0.0     -0.0   -100.0           0.0       -0.323            -0.001       -1.937      -0.968         -1.937        -0.322
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           over_95               11            2          18.2       0.0     -0.0   -100.0           0.0       -0.316            -0.001      -10.099      -5.049        -10.099        -0.316
Stats for: under_95
4×18 DataFrame
 Row │ model_name           model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String               String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ GRWBivariatePoisson  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_95              11            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   2 │ GRWNegativeBinomial  μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_95              11            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   3 │ GRWDixonColes        μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_95              11            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0
   4 │ GRWPoisson           μ=Normal(μ=0.2863224529357414, σ…  BayesianKelly  none           under_95              11            0           0.0       0.0      0.0      0.0           0.0          0.0               0.0          0.0         0.0            0.0           0.0


=#
