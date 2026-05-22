# current_development/ab_test_outfield_player/r01_ab_test_runner.jl

# using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates
using Distributions
using Turing
using Statistics

# Short-hands
const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Signals = BayesianFootball.Signals

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
save_dir::String = "./data/ab_test_hierarchical_player/"

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
# Shared Component Configs
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

# Bayesian Tracker for player ratings
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# Model B: Simplified Outfield Time-Decay (G, Outfield)
model_outfield = PreGame.DynamicMarketXGOutfieldPlayerTimeDecayModel(
    interception_config  = inter_cfg,
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=180.0),
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight        = 1.0
)

model_all = PreGame.DynamicMarketXGPlayerTimeDecayModel(
    interception_config  = inter_cfg,
    player_dynamics_config = PreGame.PositionalPlayerDynamics(days_half_life=180.0),
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight        = 1.0
)


# ==========================================
# 3. EXPERIMENT TASK CREATION
# ==========================================
println("\n[INFO] Creating Experiment Task...")
task_outfield = Experiments.create_experiment_task(
    ds, 
    model_outfield, 
    "ab_outfield_player", 
    save_dir; 
    target_seasons=["2025", "2026"], 
    dynamics_col=:match_month,
    samples=1000,   # Increased samples
    warmup=1000,    # Increased warmup
    chains=4        # Ensure 4 chains for robust R-hat checking
)

task_all = Experiments.create_experiment_task(
    ds, 
    model_all, 
    "ab_all_player", 
    save_dir; 
    target_seasons=["2025", "2026"], 
    dynamics_col=:match_month,
    samples=1000,   # Increased samples
    warmup=1000,    # Increased warmup
    chains=4        # Ensure 4 chains for robust R-hat checking
)


display(task_outfield)

# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
println("\n" * "="^60)
println(">>> RUNNING EXPERIMENT: $(task_outfield.config.name)")
println("="^60)

results_all = Experiments.run_experiment(task_all)
results_outfield = Experiments.run_experiment(task_outfield)

println("\n[INFO] Saving Experiment...")
Experiments.save_experiment(results_outfield)
Experiments.save_experiment(results_all)

println("✅ Success: $(task_outfield.config.name)")





save_dir::String = "data/ab_test_hierarchical_player/"
saved_fiels = Experiments.list_experiments(save_dir, data_dir="")
results_all = Experiments.load_experiment(saved_fiels, 1)
results_outfield = Experiments.load_experiment(saved_fiels, 2)


# ==========================================
# 5. DIAGNOSTICS (NEW STANDARD WORKFLOW)
# ==========================================
println("\n" * "="^50)
println(">>> EXPERIMENT DIAGNOSTICS")
println("="^50)

# Extract MCMC chains into long-format dataframe
#
chains_df_all = Diagnostics.extract_chains(ds, results_all)
chains_df_outfield = Diagnostics.extract_chains(ds, results_outfield)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Diagnostics.check_convergence(chains_df_all)
conv_diag_outfield = Diagnostics.check_convergence(chains_df_outfield)
display(conv_diag)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Diagnostics.check_stability(chains_df_all)
stab_diag_outfield = Diagnostics.check_stability(chains_df_outfield)
display(stab_diag)

# ==========================================
# 6. EVALUATION
# ==========================================
println("\n" * "="^50)
println(">>> PREDICTIVE EVALUATION")
println("="^50)

metrics = [
    Evaluation.RQR(),
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]
master_eval_df = Evaluation.evaluate_experiments(metrics, [results_all,results_outfield], ds)

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)

# ==========================================
# 7. BACKTESTING
# ==========================================
println("\n" * "="^50)
println(">>> BACKTEST STAKING ANALYSIS")
println("="^50)

ledger = BackTesting.run_backtest(
    ds, 
    [results_all], 
    [Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)

println("\n\n✅ Complete pipeline executed successfully! Results saved to $save_dir")



odds =Data.summarize_betfair_market(
    ds, 
    open_window=(-100000.0, -10.0), 
    close_window=(-10.0, 0.0)
)

ds1 = Data.DataStore(
  ds.segment,
  ds.matches,
  ds.statistics,
  odds,
  ds.lineups,
  ds.incidents,
  ds.betfair_odds
  )

ledger1 = BackTesting.run_backtest(
    ds1, 
    [results_all, results_outfield], 
    [Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet1 = BackTesting.generate_tearsheet(ledger1)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet1[:, cols_to_show], allrows=true)


using DataFrames, Dates, HypothesisTests, Plots, TimeZones

# ==========================================
# 1. AGGREGATE TO DAILY RETURNS
# ==========================================
# First, create the new :day column safely handling missing ZonedDateTimes
ledger.df.day = passmissing(d -> Date(d)).(ledger.df.date)

# Now, group by the new column and aggregate
daily_df = combine(
    groupby(ledger.df, :day; skipmissing=true), # skipmissing drops rows with no date
    :pnl => sum => :daily_pnl,
    :stake => sum => :daily_stake
)

# Sort chronologically
sort!(daily_df, :day)

# Calculate Daily ROI (Handling zero-stake days to avoid NaN or Inf)
daily_df.daily_roi = ifelse.(
    daily_df.daily_stake .> 0, 
    daily_df.daily_pnl ./ daily_df.daily_stake, 
    0.0
)

# ==========================================
# 2. THE STATIONARITY TEST (ADF)
# ==========================================
# H0 (Null): The series has a unit root (drifting/non-stationary).
# H1 (Alternate): The series is stationary.
adf_result = ADFTest(daily_df.daily_roi, Symbol("constant"), 1)


#=
julia> adf_result = ADFTest(daily_df.daily_roi, Symbol("constant"), 1)
Augmented Dickey-Fuller unit root test
--------------------------------------
Population details:
    parameter of interest:   coefficient on lagged non-differenced variable
    value under h_0:         0
    point estimate:          -0.963271

Test summary:
    outcome with 95% confidence: reject h_0
    p-value:                     <1e-07

Details:
    sample size in regression:          90
    number of lags:                     1
    ADF statistic:                      -6.16787
    Critical values at 1%, 5%, and 10%: adjoint([-3.50351, -2.89351, -2.58382])
=#


println("\n--- Regime Stationarity Test (Daily ROI) ---")
println(adf_result)

# ==========================================
# 3. EQUITY CURVE & DRAWDOWN ANALYSIS
# ==========================================
daily_df.cumulative_pnl = cumsum(daily_df.daily_pnl)

# Calculate running maximum to find drawdowns
daily_df.peak_pnl = accumulate(max, daily_df.cumulative_pnl)
daily_df.drawdown = daily_df.cumulative_pnl .- daily_df.peak_pnl

# Plotting
p1 = plot(daily_df.day, daily_df.cumulative_pnl, 
    label="Cumulative PnL", lw=2, color=:blue, 
    title="Model Equity Curve", ylabel="Units", legend=:topleft)

p2 = plot(daily_df.day, daily_df.drawdown, 
    label="Drawdown", fill=(0, :red), linealpha=0, 
    title="Underwater Curve (Drawdowns)", ylabel="Units", legend=false)

display(plot(p1, p2, layout=(2,1), size=(800, 600)))

# alpha decay 
#
#
using DataFrames, Dates, HypothesisTests, Plots, TimeZones

# ==========================================
# 1. AGGREGATE TO DAILY RETURNS PER SELECTION
# ==========================================
# Ensure date format is handled safely
ledger.df.day = passmissing(d -> Date(d)).(ledger.df.date);

# Group by BOTH day and selection
daily_sel_df = combine(
    groupby(ledger.df, [:day, :selection]; skipmissing=true),
    :pnl => sum => :daily_pnl,
    :stake => sum => :daily_stake
)

# Sort chronologically within each selection
sort!(daily_sel_df, [:selection, :day])

# Calculate Daily ROI per selection
daily_sel_df.daily_roi = ifelse.(
    daily_sel_df.daily_stake .> 0, 
    daily_sel_df.daily_pnl ./ daily_sel_df.daily_stake, 
    0.0
)

# ==========================================
# 2. RUN ADF TEST FOR EACH SELECTION
# ==========================================
println("\n--- Stationarity Test (ADF) By Selection ---")
println(rpad("Selection", 15), " | ", rpad("P-Value", 10), " | Status")
println("-" ^ 50)

valid_selections = String[] # Keep track of which ones have enough data to plot

for sel in unique(daily_sel_df.selection)
    # Isolate this specific selection's timeline
    df_sub = filter(row -> row.selection == sel, daily_sel_df)
    
    # ADF needs a decent sample size (at least ~15-20 days) to run a valid regression
    if nrow(df_sub) >= 15
        try
            # Run test with 1 lag
            adf_result = ADFTest(df_sub.daily_roi, Symbol("constant"), 1)
            p_val = pvalue(adf_result)
            
            status = p_val < 0.05 ? "✅ Stationary (Safe)" : "❌ Drifting (Warning)"
            println(rpad(string(sel), 15), " | ", rpad(round(p_val, digits=6), 10), " | ", status)
            
            push!(valid_selections, string(sel))
        catch e
            # Catches errors if the ROI is perfectly flat (zero variance)
            println(rpad(string(sel), 15), " | ", rpad("ERROR", 10), " | Math Failed (Zero Variance?)")
        end
    else
        println(rpad(string(sel), 15), " | ", rpad("SKIP", 10), " | Not enough days (", nrow(df_sub), ")")
    end
end


#=
DC_12           | 2.2e-5     | ✅ Stationary (Safe)
DC_1X           | 0.0        | ✅ Stationary (Safe)
DC_X2           | 0.0        | ✅ Stationary (Safe)
away            | 0.0        | ✅ Stationary (Safe)
btts_no         | 0.0        | ✅ Stationary (Safe)
btts_yes        | 0.0        | ✅ Stationary (Safe)
draw            | 0.0        | ✅ Stationary (Safe)
home            | 0.0        | ✅ Stationary (Safe)
over_05         | 0.0        | ✅ Stationary (Safe)
over_15         | 0.0        | ✅ Stationary (Safe)
over_25         | 0.0        | ✅ Stationary (Safe)
over_35         | 0.0        | ✅ Stationary (Safe)
over_45         | 0.0        | ✅ Stationary (Safe)
over_55         | 0.0        | ✅ Stationary (Safe)
over_65         | 0.0        | ✅ Stationary (Safe)
over_75         | ERROR      | Math Failed (Zero Variance?)
under_05        | 0.0        | ✅ Stationary (Safe)
under_15        | 0.0        | ✅ Stationary (Safe)
under_25        | 0.0        | ✅ Stationary (Safe)
under_35        | 0.0        | ✅ Stationary (Safe)
under_45        | 0.0        | ✅ Stationary (Safe)
under_55        | 0.0        | ✅ Stationary (Safe)
under_65        | 0.0        | ✅ Stationary (Safe)
under_75        | 6.0e-5     | ✅ Stationary (Safe)
=#


# ==========================================
# 3. CUMULATIVE PNL & PLOTTING BY SELECTION
# ==========================================
# Calculate Cumulative PnL properly grouped by selection
transform!(groupby(daily_sel_df, :selection),
    :daily_pnl => cumsum => :cumulative_pnl
)


valid_selections_bak = valid_selections
v = [  "over_15","over_25"] 

# Filter out the markets that didn't have enough data
plot_df = filter(row -> string(row.selection) in v, daily_sel_df)

# Plot all valid equity curves on one chart
p_equity = plot(plot_df.day, plot_df.cumulative_pnl, group=plot_df.selection,
    title="Equity Curve by Selection",
    ylabel="Cumulative PnL (Units)",
    xlabel="Date",
    legend=:outertopright,  # Puts the legend outside so it doesn't cover lines
    lw=2, 
    size=(1000, 600))

display(p_equity)



using DataFrames, Dates, Statistics

println("\n--- Regime Shift Analysis (H1 vs H2) ---")
println(rpad("Selection", 15), " | ", rpad("H1 ROI", 10), " | ", rpad("H2 ROI", 10), " | Status")
println("-" ^ 65)

# Get the raw ledger and ensure we have dates
df_raw = copy(ledger.df)
df_raw.day = passmissing(d -> Date(d)).(df_raw.date)

for sel in unique(df_raw.selection)
    # Isolate this specific selection's bets
    df_sub = filter(row -> row.selection == sel, df_raw)
    sort!(df_sub, :day)
    
    n_bets = nrow(df_sub)
    
    # We only care about markets with enough volume to matter
    if n_bets >= 30
        # Slice the dataset exactly in half based on bet count
        mid_point = div(n_bets, 2)
        
        df_h1 = df_sub[1:mid_point, :]
        df_h2 = df_sub[mid_point+1:end, :]
        
        # Calculate ROIs safely
        roi_h1 = sum(df_h1.pnl) / sum(df_h1.stake)
        roi_h2 = sum(df_h2.pnl) / sum(df_h2.stake)
        
        # Format for printing (convert to percentages)
        h1_str = "$(round(roi_h1 * 100, digits=1))%"
        h2_str = "$(round(roi_h2 * 100, digits=1))%"
        
        # Determine the Status
        if roi_h1 > 0 && roi_h2 > 0
            status = "✅ Elite (Consistent Edge)"
        elseif roi_h1 > 0 && roi_h2 <= 0
            status = "❌ Decaying (Edge Vanished in H2)"
        elseif roi_h1 <= 0 && roi_h2 > 0
            status = "⚠️ Late Bloomer (Warming Up)"
        else
            status = "💀 Dead Market (Consistent Loser)"
        end
        
        println(rpad(string(sel), 15), " | ", rpad(h1_str, 10), " | ", rpad(h2_str, 10), " | ", status)
    end
end




#=
home            | 23.3%      | -26.0%     | ❌ Decaying (Edge Vanished in H2)
draw            | -3.2%      | 0.6%       | ⚠️ Late Bloomer (Warming Up)
away            | 99.2%      | -48.7%     | ❌ Decaying (Edge Vanished in H2)
btts_yes        | 95.1%      | 48.5%      | ✅ Elite (Consistent Edge)
btts_no         | -4.6%      | -10.1%     | 💀 Dead Market (Consistent Loser)
DC_1X           | -15.7%     | 16.4%      | ⚠️ Late Bloomer (Warming Up)
DC_X2           | -47.4%     | -79.3%     | 💀 Dead Market (Consistent Loser)
DC_12           | -100.0%    | -100.0%    | 💀 Dead Market (Consistent Loser)
over_05         | 11.4%      | 9.0%       | ✅ Elite (Consistent Edge)
under_05        | 0.1%       | -14.7%     | ❌ Decaying (Edge Vanished in H2)
over_15         | 37.5%      | -3.0%      | ❌ Decaying (Edge Vanished in H2)
under_15        | 1.5%       | -1.2%      | ❌ Decaying (Edge Vanished in H2)
over_25         | 30.3%      | 22.8%      | ✅ Elite (Consistent Edge)
under_25        | 3.6%       | 2.8%       | ✅ Elite (Consistent Edge)
over_35         | 111.6%     | -29.9%     | ❌ Decaying (Edge Vanished in H2)
under_35        | -2.7%      | 12.1%      | ⚠️ Late Bloomer (Warming Up)
over_45         | 52.4%      | 22.2%      | ✅ Elite (Consistent Edge)
under_45        | -5.9%      | -0.0%      | 💀 Dead Market (Consistent Loser)
over_55         | -100.0%    | -98.3%     | 💀 Dead Market (Consistent Loser)
under_55        | -3.2%      | -8.5%      | 💀 Dead Market (Consistent Loser)
over_65         | -100.0%    | -100.0%    | 💀 Dead Market (Consistent Loser)
under_65        | 1.7%       | 2.2%       | ✅ Elite (Consistent Edge)
over_75         | NaN%       | NaN%       | 💀 Dead Market (Consistent Loser)
under_75        | 0.4%       | 0.5%       | ✅ Elite (Consistent Edge)
=#

# btts_yes        | 95.1%      | 48.5%      | ✅ Elite (Consistent Edge)
# over_05         | 11.4%      | 9.0%       | ✅ Elite (Consistent Edge)
# over_25         | 30.3%      | 22.8%      | ✅ Elite (Consistent Edge)
# under_25        | 3.6%       | 2.8%       | ✅ Elite (Consistent Edge)
