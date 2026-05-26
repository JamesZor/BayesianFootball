# current_development/meta_model/r02_meta_model_real_data.jl

using DataFrames
using Dates
using Turing
using StatsPlots
using Statistics

# Ensure the loader is included
include("l01_meta_model.jl")

"""
    run_meta_experiment(ledger_df::DataFrame, ds_odds::DataFrame; prob_col=:prob_fair_close, n_samples=500)

Runs the Meta Model on real ledger data.
Assumes `ledger_df` has: `match_id`, `selection`, `is_winner`, `date`, `pnl`
Assumes `ds_odds` has: `match_id`, `selection`, and the column specified by `prob_col` (e.g., `:prob_fair_close` or `:clm_prob`).
"""
function run_meta_experiment(ledger_df::DataFrame, ds_odds::DataFrame; prob_col::Symbol=:prob_fair_close, n_samples=500)
    println("1. Preparing Data...")
    
    # 1. Join Ledger with Odds Data to get the baseline probability
    # We join on match_id and selection to ensure exact matches
    joined = innerjoin(ledger_df, ds_odds[!, [:match_id, :selection, prob_col]], on=[:match_id, :selection])
    
    # 2. Filter out missing outcomes or missing probabilities
    dropmissing!(joined, [:is_winner, prob_col, :date])
    
    # Ensure is_winner is integer 0 or 1
    Y = Int.(joined.is_winner)
    
    # Extract baseline probabilities
    P_L1 = Float64.(joined[!, prob_col])
    
    # 3. Construct the Calendar Week Index (W)
    # Sort chronologically
    sort!(joined, :date)
    start_date = Date(minimum(joined.date))
    
    W = Int[]
    for d in joined.date
        days_diff = (Date(d) - start_date).value
        week_idx = (days_diff ÷ 7) + 1
        push!(W, week_idx)
    end
    n_weeks = maximum(W)
    
    println("   Filtered to $(length(Y)) bets over $n_weeks weeks.")
    
    # 4. Build Turing Model
    println("2. Building Turing Meta Model...")
    meta_data = MetaModelData(Y, P_L1, W, n_weeks)
    model = build_meta_model(meta_data)
    
    # 5. Sample
    println("3. Sampling (NUTS)... This may take a few minutes for real data.")
    println("   (Tip: Make sure you ran ThreadPinning.pinthreads(:cores) before this!)")
    chain = sample(
      model,
      NUTS(200,0.65,max_depth=10),
      MCMCThreads(),
      n_samples,
      8,
      adtype = AutoReverseDiff(compile=true),
      )
    
    # 6. Extract and Analyze
    println("4. Analyzing Results...")
    
    # Reconstruct Theta Matrix
    theta_matrix = extract_theta(chain, n_weeks)
    theta_means = vec(mean(theta_matrix, dims=1))
    theta_std = vec(std(theta_matrix, dims=1))
    
    # 7. Visualization: Theta vs Rolling PnL
    # Calculate weekly PnL from the ledger
    weekly_pnl = zeros(Float64, n_weeks)
    for (i, w) in enumerate(W)
        if hasproperty(joined, :pnl) && !ismissing(joined.pnl[i])
            weekly_pnl[w] += joined.pnl[i]
        end
    end
    
    # Smoothing PnL for visualization (e.g. 4-week rolling average)
    smoothed_pnl = copy(weekly_pnl)
    window = 4
    for i in 1:n_weeks
        start_idx = max(1, i - window + 1)
        smoothed_pnl[i] = mean(weekly_pnl[start_idx:i])
    end
    
    # p1 = plot(1:n_weeks, theta_means, ribbon=theta_std, fillalpha=0.3, 
    #           label="Posterior Mean θ (Regime)", xlabel="Week", ylabel="Meta Model Shift", 
    #           title="Layer 1 Performance Regime", color=:blue, lw=2)
    #
    # p2 = bar(1:n_weeks, weekly_pnl, label="Weekly PnL", alpha=0.5, color=:gray)
    # plot!(p2, 1:n_weeks, smoothed_pnl, label="$(window)-Week Rolling PnL", color=:red, lw=2,
    #       xlabel="Week", ylabel="PnL")
    #
    # final_plot = plot(p1, p2, layout=(2, 1), size=(900, 700))
    # display(final_plot)
    
    println("\n=== Experiment Complete ===")
    println("Check the plot to see how well the Meta Model state (θ) correlates with actual PnL drawdowns!")
    
    return chain, joined
end

save_dir = "./data/copula_ab_test/"

saved_files = Experiments.list_experiments(save_dir, data_dir="")
result = Experiments.load_experiment(saved_files, 1)


ledger = BackTesting.run_backtest(
    ds, 
    result, 
    [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

led_over_25 = subset(ledger.df , :selection => ByRow(isequal(:under_25)))


chain, joined = run_meta_experiment(led_over_25, ds.odds)

describe(chain)

using Statistics

# 1. Reconstruct the week index (W) since it was local to the function
start_date = Date(minimum(joined.date))
W = [(Date(d) - start_date).value ÷ 7 + 1 for d in joined.date]
n_weeks = maximum(W)

# 2. Extract Theta means
theta_matrix = extract_theta(chain, n_weeks)
theta_means = vec(mean(theta_matrix, dims=1))

# 3. Calculate actual weekly PnL
weekly_pnl = zeros(Float64, n_weeks)
for (i, w) in enumerate(W)
if hasproperty(joined, :pnl) && !ismissing(joined.pnl[i])
        weekly_pnl[w] += joined.pnl[i]
end
end

# 4. Calculate Metrics
corr = cor(theta_means, weekly_pnl)
good_weeks = findall(x -> x > 0, theta_means)
bad_weeks = findall(x -> x <= 0, theta_means)

avg_pnl_good = isempty(good_weeks) ? 0.0 : mean(weekly_pnl[good_weeks])
avg_pnl_bad = isempty(bad_weeks) ? 0.0 : mean(weekly_pnl[bad_weeks])

total_pnl_good = isempty(good_weeks) ? 0.0 : sum(weekly_pnl[good_weeks])
total_pnl_bad = isempty(bad_weeks) ? 0.0 : sum(weekly_pnl[bad_weeks])

pp(corr, good_weeks, bad_weeks, total_pnl_good, total_pnl_bad, avg_pnl_good, avg_pnl_bad)

function pp(corr, good_weeks, bad_weeks, total_pnl_good, total_pnl_bad, avg_pnl_good, avg_pnl_bad)
println("\n=== Headless Server Metrics ===")
println("Correlation (θ vs Weekly PnL): ", round(corr, digits=3))
println("--------------------------------")
println("Weeks in Good Regime (θ > 0): ", length(good_weeks))
println("Weeks in Bad Regime  (θ ≤ 0): ", length(bad_weeks))
println("--------------------------------")
println("Total PnL (Good Regime): ", round(total_pnl_good, digits=4))
println("Total PnL (Bad Regime):  ", round(total_pnl_bad, digits=4))
println("Avg Weekly PnL (Good Regime): ", round(avg_pnl_good, digits=4))
println("Avg Weekly PnL (Bad Regime):  ", round(avg_pnl_bad, digits=4))
end


#=
=== Headless Server Metrics ===




#=
under 25 
julia> total_pnl_good + total_pnl_bad
0.5125551673663231
=== Headless Server Metrics ===
Correlation (θ vs Weekly PnL): -0.065
--------------------------------
Weeks in Good Regime (θ > 0): 108
Weeks in Bad Regime  (θ ≤ 0): 88
--------------------------------
Total PnL (Good Regime): -0.5777
Total PnL (Bad Regime):  1.0903
Avg Weekly PnL (Good Regime): -0.0053
Avg Weekly PnL (Bad Regime):  0.0124
=#


# Goals model
#=
=== Headless Server Metrics ===
Correlation (θ vs Weekly PnL): 0.073
--------------------------------
Weeks in Good Regime (θ > 0): 92
Weeks in Bad Regime  (θ ≤ 0): 104
--------------------------------
Total PnL (Good Regime): 0.1736
Total PnL (Bad Regime):  -0.2151
Avg Weekly PnL (Good Regime): 0.0019
Avg Weekly PnL (Bad Regime):  -0.0021
=#

#=
 hierarical copula 
julia> pp(corr, good_weeks, bad_weeks, total_pnl_good, total_pnl_bad, avg_pnl_good, avg_pnl_bad)

=== Headless Server Metrics ===
Correlation (θ vs Weekly PnL): 0.079
--------------------------------
Weeks in Good Regime (θ > 0): 91
Weeks in Bad Regime  (θ ≤ 0): 105
--------------------------------
Total PnL (Good Regime): 0.1524
Total PnL (Bad Regime):  -0.2611
Avg Weekly PnL (Good Regime): 0.0017
Avg Weekly PnL (Bad Regime):  -0.0025
=#



=#

