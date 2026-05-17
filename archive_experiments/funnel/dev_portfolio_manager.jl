
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
# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/funnel_basics"; data_dir="./data")
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



baker = BayesianKelly()
my_signals = [baker]

# Load the models - check leaded_results
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results[[1,2]], 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)



# ---------------------------------
#
using DataFrames, Dates, Statistics

# --- Helper: Portfolio Construction Logic ---
function select_weekly_portfolio(stats_df, cor_df; max_strategies=5, correlation_threshold=0.4)
    # 1. Filter for viable strategies (Positive Growth, Minimum history)
    candidates = filter(r -> r.Growth_Rate > 0.001 && r.Bet_Freq > 0.01, stats_df)
    
    # 2. Sort by Quality (Growth Rate is usually best)
    sort!(candidates, :Growth_Rate, rev=true)
    
    selected_portfolio = String[]
    
    # 3. Greedy Selection with Correlation Check
    for row in eachrow(candidates)
        # Construct ID to match correlation matrix columns
        strat_id = "$(row.market_name)_$(row.selection)" # Adjust based on your naming convention
        
        # If it's the first pick, take it
        if isempty(selected_portfolio)
            push!(selected_portfolio, strat_id)
            continue
        end
        
        # Check correlation against ALREADY selected strategies
        is_uncorrelated = true
        for existing in selected_portfolio
            # Find correlation value in the matrix
            # (Assumes cor_df has columns named by strat_id)
            if strat_id in names(cor_df) && existing in names(cor_df)
                # Safe lookup
                c_val = 0.0 # Default
                try
                    c_val = cor_df[cor_df.Strategy .== strat_id, existing][1]
                catch
                    c_val = 0.0
                end
                
                if c_val > correlation_threshold
                    is_uncorrelated = false
                    break
                end
            end
        end
        
        if is_uncorrelated
            push!(selected_portfolio, strat_id)
        end
        
        if length(selected_portfolio) >= max_strategies
            break
        end
    end
    
    return selected_portfolio
end

# --- Main Walk-Forward Engine ---
function run_walk_forward(ledger::DataFrame; burn_in_weeks=4, update_freq_weeks=1)
    # Ensure sorted by date
    sort!(ledger, :date)
    
    start_date = minimum(ledger.date)
    end_date = maximum(ledger.date)
    
    # We will record the performance of our "Manager" here
    manager_log = DataFrame(
        date = Date[], 
        strategy_count = Int[], 
        weekly_pnl = Float64[], 
        cumulative_pnl = Float64[],
        active_strategies = String[] # Store as joined string
    )
    
    current_date = start_date + Week(burn_in_weeks)
    total_pnl = 0.0
    
    println("Starting Walk-Forward Simulation...")
    
    while current_date < end_date
        next_date = current_date + Week(update_freq_weeks)
        
        # 1. THE LOOKBACK (Past)
        # strictly less than current_date to avoid look-ahead bias
        past_data = filter(row -> row.date < current_date, ledger)
        
        # 2. THE ANALYSIS
        # Recalculate global stats on past data only
        stats = rank_strategies(past_data)
        
        # Recalculate correlations on past data only
        # (You might need to tweak the correlation function to return a clean matrix)
        cor_matrix = calculate_portfolio_correlation(past_data) 
        
        # 3. THE SELECTION
        active_ids = select_weekly_portfolio(stats, cor_matrix)
        
        # 4. THE EXECUTION (Future)
        # Filter for games in the UPCOMING week
        next_week_games = filter(row -> row.date >= current_date && row.date < next_date, ledger)
        
        weekly_pnl = 0.0
        
        # If we have games and strategies...
        if !isempty(next_week_games) && !isempty(active_ids)
            for game in eachrow(next_week_games)
                # Construct the ID for this row to see if it's in our portfolio
                # Note: You need a consistent ID generator between 'rank_strategies' and here
                # Let's assume we match on Market + Selection for simplicity
                game_strat_id = "$(game.market_name)_$(game.selection)"
                
                if game_strat_id in active_ids
                    # We trade this!
                    # You can use the 'stake' from the ledger (static) 
                    # OR recalculate stake based on the 'stats' (dynamic)
                    
                    weekly_pnl += game.pnl 
                end
            end
        end
        
        total_pnl += weekly_pnl
        
        # Log results
        push!(manager_log, (
            current_date, 
            length(active_ids), 
            weekly_pnl, 
            total_pnl,
            join(active_ids, ", ")
        ))
        
        # Step forward
        current_date = next_date
    end
    
    return manager_log
end

results = run_walk_forward(ledger.df)

using UnicodePlots
lineplot(results.date, results.cumulative_pnl)

# (base) ⚡➜ ~ ssh -L 8080:localhost:8000 root@65.109.70.100 -N
# have python3 -m server running on the server 
#  # python3 -m http.server 8000
#  ssh -L 8080:localhost:8000 root@65.109.70.100 -N
using Plots
plotlyjs()  # Switch the backend to PlotlyJS

# Your plotting code
p = plot(results.date, results.cumulative_pnl, title="PnL Over Time");
# Save as a standalone HTML file
Plots.savefig(p, "my_plot.html")


# ======
# Basic Dynamic Mixed Gamma Tracker.
# ======

#=
In this code we will take the ledger struct, filter for the market selection 
and run a Bayesain model to infer the three key hidden variables for every single match. 
  1. πᵢ - True win rate - is the strategy improving or degrading 
  2. μ_win_ᵢ - Avg Win size
  3. μ_loss_ᵢ - Avg Win size

=#

# 1 Data Preparation
# ---- 
using Dates, DataFrames, Statistics

# A helper to define time resolution
abstract type TimeResolution end
struct Weekly <: TimeResolution end
struct BiWeekly <: TimeResolution end
struct Monthly <: TimeResolution end

function get_time_index(date, start_date, ::Weekly)
    return floor(Int, (date - start_date).value / 7) + 1
end

function get_time_index(date, start_date, ::BiWeekly)
    return floor(Int, (date - start_date).value / 14) + 1
end

function get_time_index(date, start_date, ::Monthly)
    return (year(date) - year(start_date)) * 12 + month(date) - month(start_date) + 1
end

function prepare_optimized_data(ledger::AbstractDataFrame, market::Symbol, resolution::TimeResolution=BiWeekly())
    # 1. Filter
    subset = DataFrames.subset(ledger, 
        :selection => ByRow(isequal(market)), 
        :stake => ByRow(>(1e-6))
    )
    sort!(subset, :date)
    
    if nrow(subset) == 0
        error("No data found for $market")
    end

    start_date = first(subset.date)
    
    # 2. Assign Time Indices
    # We map every row to a discrete time bucket
    raw_indices = map(d -> get_time_index(d, start_date, resolution), subset.date)
    
    # Re-index to ensure they are continuous 1..T (handles gaps nicely)
    unique_periods = sort(unique(raw_indices))
    period_map = Dict(p => i for (i, p) in enumerate(unique_periods))
    dense_indices = [period_map[idx] for idx in raw_indices]
    num_periods = length(unique_periods)
    
    # 3. SPLIT DATA (The SIMD Trick)
    # Instead of one vector, we create two.
    # We also need the time_index for each group to look up parameters.
    
    # Wins (PnL > 0)
    win_mask = subset.pnl .> 0
    wins = subset.pnl[win_mask]
    win_indices = dense_indices[win_mask]
    
    # Losses (PnL <= 0) - Store as positive magnitude
    loss_mask = .!win_mask
    # Add noise to true zeros for Gamma stability
    losses = map(x -> x == 0 ? 1e-6 : abs(x), subset.pnl[loss_mask])
    loss_indices = dense_indices[loss_mask]

    # Calculate initial mean (for initializing priors)
    global_mean_abs = mean(abs.(subset.pnl))

    return (
        wins = wins,
        win_idx = win_indices,
        losses = losses,
        loss_idx = loss_indices,
        num_periods = num_periods,
        global_mean = global_mean_abs
    )
end

# 2. model

using Turing, Distributions, LogExpFunctions

@model function optimized_mixed_gamma(wins, win_idx, losses, loss_idx, num_periods, global_mean)
    
    # --- 1. Global Priors ---
    # Volatility of the Random Walks
    σ_p ~ Truncated(Normal(0.05, 0.05), 0.001, 0.2) 
    σ_mu_win ~ Truncated(Normal(0.1, 0.1), 0.001, 0.5)
    σ_mu_loss ~ Truncated(Normal(0.1, 0.1), 0.001, 0.5)
    
    # --- 2. Non-Centered Random Walk ---
    # We sample standard normals (z scores) and scale them later.
    # This decouples the dependency structure for the sampler.
    
    # Initial States (t=1)
    z_p_init ~ Normal(0, 1)
    z_mw_init ~ Normal(0, 1)
    z_ml_init ~ Normal(0, 1)
    
    # Steps (t=2..T)
    # We use a single vector of length T-1 for the steps
    z_p_steps ~ filldist(Normal(0, 1), num_periods - 1)
    z_mw_steps ~ filldist(Normal(0, 1), num_periods - 1)
    z_ml_steps ~ filldist(Normal(0, 1), num_periods - 1)
    
    # Reconstruction (Deterministic transformation)
    # 1. Scale the steps by sigma
    # 2. Concatenate init and steps
    # 3. Cumulative Sum to create the walk
    
    # Helper to construct the walk cleanly
    # (Note: we scale init by 1.0 or separate prior std, here assuming std=1 for init latent)
    x_p = cumsum(vcat(z_p_init, z_p_steps .* σ_p))
    
    # For means, we offset by the global log-mean to center the prior sensibly
    log_global = log(global_mean)
    x_mu_win = cumsum(vcat(z_mw_init + log_global, z_mw_steps .* σ_mu_win))
    x_mu_loss = cumsum(vcat(z_ml_init + log_global, z_ml_steps .* σ_mu_loss))
    
    # --- 3. Parameter Transformation ---
    # Transform latent states to real parameters for all periods at once
    # logistic() is numerically stable sigmoid
    π_vec = logistic.(x_p)      
    μ_w_vec = exp.(x_mu_win)
    μ_l_vec = exp.(x_mu_loss)
    
    # Static CV for now (optimization target)
    cv = 0.5
    inv_cv2 = 1 / cv^2
    
    # --- 4. Vectorized Likelihood (SIMD Friendly) ---
    
    # BLOCK A: WINS
    if length(wins) > 0
        # 1. Gather parameters corresponding to the week of each win
        # (This is just indexing, very fast in Julia)
        p_wins = π_vec[win_idx]
        mu_wins = μ_w_vec[win_idx]
        
        # 2. Calculate Gamma Parameters
        # alpha = 1/cv^2, theta = mu * cv^2
        # We can broadcast this operation
        α_wins = fill(inv_cv2, length(wins)) 
        θ_wins = mu_wins .* (cv^2)
        
        # 3. Add Log Probabilities
        # log(P(Win)) + log(Gamma(r))
        # The dot (.) means element-wise broadcasting
        Turing.@addlogprob! sum(log.(p_wins) .+ logpdf.(Gamma.(α_wins, θ_wins), wins))
    end
    
    # BLOCK B: LOSSES
    if length(losses) > 0
        # 1. Gather parameters
        p_losses = π_vec[loss_idx]
        mu_losses = μ_l_vec[loss_idx]
        
        # 2. Gamma Params
        α_losses = fill(inv_cv2, length(losses))
        θ_losses = mu_losses .* (cv^2)
        
        # 3. Add Log Probabilities
        # log(P(Loss)) + log(Gamma(|r|))
        # Note: P(Loss) = 1 - P(Win)
        Turing.@addlogprob! sum(log1p.(-p_losses) .+ logpdf.(Gamma.(α_losses, θ_losses), losses))
    end
end



# 1. Prepare Data (e.g., Bi-Weekly to save parameters)
data = prepare_optimized_data(ledger.df, :home, Monthly())
data = prepare_optimized_data(ledger.df, :home, BiWeekly())

# 2. Run Model
# Note: passed as named arguments matching the tuple keys
model = optimized_mixed_gamma(
    data.wins, 
    data.win_idx, 
    data.losses, 
    data.loss_idx, 
    data.num_periods, 
    data.global_mean
)

# 3. Sample
# Multi-threaded, Non-centered, Vectorized = FAST
chain = sample(model, NUTS(0.65), MCMCThreads(), 1000, 4)

using ReverseDiff, Memoization

chain = sample(
    model, 
    NUTS(0.65), 
    MCMCThreads(), 
    500, 
    8,
    adtype = AutoReverseDiff(compile=true),
)

describe(chain)


using DataFrames, Statistics, LogisticFunctions, Plots, PlotlyJS

using DataFrames, Statistics, LogisticFunctions, MCMCChains

function extract_chain_parameters(chain, num_periods, global_mean)
    # 1. Extract Global Hyperparameters as FLATTENED Vectors
    # vec(chain[:param]) automatically stacks all chains into one long vector
    σ_p_samples = vec(chain[:σ_p])
    σ_mw_samples = vec(chain[:σ_mu_win])
    σ_ml_samples = vec(chain[:σ_mu_loss])
    
    log_global = log(global_mean)
    
    # 2. Prepare Storage
    df = DataFrame(
        period = 1:num_periods,
        prob_mean = zeros(num_periods), prob_low = zeros(num_periods), prob_high = zeros(num_periods),
        win_mean  = zeros(num_periods), win_low  = zeros(num_periods), win_high  = zeros(num_periods),
        edge_mean = zeros(num_periods), edge_low = zeros(num_periods), edge_high = zeros(num_periods)
    )

    # Temporary storage for the random walk state (flattened)
    # Initialize with the 'init' parameters
    lat_p = vec(chain[:z_p_init])
    lat_mw = vec(chain[:z_mw_init]) .+ log_global
    lat_ml = vec(chain[:z_ml_init]) .+ log_global

    # 3. Iterate Through Each Period 't'
    for t in 1:num_periods
        # If t > 1, we need to add the step from the previous period
        if t > 1
            # Extract the step for this specific period 't-1'
            # We access the symbol dynamically: z_p_steps[1], z_p_steps[2]...
            # And flatten it immediately with vec()
            z_p_step = vec(chain[Symbol("z_p_steps[$(t-1)]")])
            z_mw_step = vec(chain[Symbol("z_mw_steps[$(t-1)]")])
            z_ml_step = vec(chain[Symbol("z_ml_steps[$(t-1)]")])
            
            # Update the Random Walk State
            # x_t = x_{t-1} + (z_step * sigma)
            lat_p .+= (z_p_step .* σ_p_samples)
            lat_mw .+= (z_mw_step .* σ_mw_samples)
            lat_ml .+= (z_ml_step .* σ_ml_samples)
        end
        
        # 4. Transform to Real Space (Vectorized)
        real_p = logistic.(lat_p)
        real_mw = exp.(lat_mw)
        real_ml = exp.(lat_ml)
        
        # 5. Calculate Edge (Expected Value)
        real_edge = (real_p .* real_mw) .- ((1.0 .- real_p) .* real_ml)
        
        # 6. Compute Statistics
        # Note: Since real_p is now a Vector, quantile() will work!
        df.prob_mean[t] = mean(real_p)
        df.prob_low[t]  = quantile(real_p, 0.05)
        df.prob_high[t] = quantile(real_p, 0.95)
        
        df.win_mean[t]  = mean(real_mw)
        df.win_low[t]   = quantile(real_mw, 0.05)
        df.win_high[t]  = quantile(real_mw, 0.95)
        
        df.edge_mean[t] = mean(real_edge)
        df.edge_low[t]  = quantile(real_edge, 0.05)
        df.edge_high[t] = quantile(real_edge, 0.95)
    end
    
    return df
end


function plot_strategy_html(stats_df, output_filename="strategy_analysis.html")
    plotlyjs() # Enable Plotly backend
    
    # --- Plot 1: Win Probability ---
    p1 = plot(stats_df.period, stats_df.prob_mean,
        ribbon=(stats_df.prob_mean .- stats_df.prob_low, stats_df.prob_high .- stats_df.prob_mean),
        fillalpha=0.2, color=:blue, lw=2,
        label="Win Prob (90% CI)",
        ylabel="Probability",
        title="Dynamic Win Rate"
    )
    
    # --- Plot 2: Mean Win Size vs Mean Loss Size ---
    p2 = plot(stats_df.period, stats_df.win_mean,
        ribbon=(stats_df.win_mean .- stats_df.win_low, stats_df.win_high .- stats_df.win_mean),
        fillalpha=0.1, color=:green, lw=2, label="Avg Win Size"
    )
    # Note: We plot Loss Size as positive magnitude for comparison
    # But usually, losses are fixed around 1.0, so this might be a flat line
    # If using Raw PnL, this will vary.
    # We construct a theoretical "Loss Line" (e.g. 1.0) if using ROI model
    # Or plot the actual tracked parameter if using Raw PnL model
    
    # --- Plot 3: The EDGE (The Most Important Plot) ---
    p3 = plot(stats_df.period, stats_df.edge_mean,
        ribbon=(stats_df.edge_mean .- stats_df.edge_low, stats_df.edge_high .- stats_df.edge_mean),
        fillalpha=0.3, color=:purple, lw=2,
        label="Expected Edge",
        ylabel="Expected Units",
        title="True Edge (Signal)"
    )
    hline!(p3, [0.0], color=:red, linestyle=:dash, label="Break Even")
    
    # Combine into a dashboard
    final_plot = plot(p1, p3, layout=(2, 1), size=(900, 800))
    
    # Save to HTML
    Plots.savefig(final_plot, output_filename)
    println("Plot saved to $output_filename")
end

num_periods = data.num_periods
global_mean = data.global_mean

stats = extract_chain_parameters(chain, data.num_periods, data.global_mean)
plot_strategy_html(stats, "home_strategy_report.html")

function plot_model_vs_reality(stats_df, data, output_filename="edge_verification.html")
    plotlyjs() # Enable interactive backend

    num_periods = data.num_periods
    
    # 1. Reconstruct Realized Reality from the Data
    realized_mean = zeros(num_periods)
    realized_cum = zeros(num_periods)
    predicted_cum = zeros(num_periods)
    
    current_realized_sum = 0.0
    current_predicted_sum = 0.0
    
    for t in 1:num_periods
        # Extract the actual bets for this period
        # Note: data.wins are returns, data.losses are positive magnitudes (abs)
        period_wins = data.wins[data.win_idx .== t]
        period_losses = data.losses[data.loss_idx .== t]
        
        n_bets = length(period_wins) + length(period_losses)
        
        # Calculate Realized PnL (Sum of Wins - Sum of Losses)
        pnl_sum = sum(period_wins) - sum(period_losses)
        
        if n_bets > 0
            realized_mean[t] = pnl_sum / n_bets
        else
            realized_mean[t] = 0.0
        end
        
        # --- Cumulative Tracking ---
        # 1. Reality
        current_realized_sum += pnl_sum
        realized_cum[t] = current_realized_sum
        
        # 2. Model Prediction
        # Model Edge (Units/Bet) * Number of Bets
        current_predicted_sum += (stats_df.edge_mean[t] * n_bets)
        predicted_cum[t] = current_predicted_sum
    end

    # --- Plot 1: Edge Dynamics (Noisy Reality vs. Smooth Model) ---
    p1 = plot(stats_df.period, stats_df.edge_mean, 
        ribbon=(stats_df.edge_mean .- stats_df.edge_low, stats_df.edge_high .- stats_df.edge_mean),
        fillalpha=0.3, color=:purple, lw=3, label="Model Estimated Edge",
        title="Edge Dynamics: Signal vs. Noise",
        ylabel="Avg PnL / Bet",
        legend=:topleft
    )
    
    # Plot Realized as points connected by a thin line (to show the noise)
    plot!(p1, 1:num_periods, realized_mean, 
        color=:black, linestyle=:dash, lw=1, 
        marker=:circle, markersize=4, markercolor=:black,
        label="Realized Avg PnL"
    )
    hline!(p1, [0.0], color=:red, linestyle=:dot, label="Break Even")

    # --- Plot 2: Cumulative Validation (The Truth Teller) ---
    p2 = plot(1:num_periods, realized_cum, 
        color=:black, lw=3, label="Actual Cumulative PnL",
        title="Cumulative Validation: Did the Model Predict the Profit?",
        ylabel="Total Units Won/Lost",
        legend=:topleft
    )
    
    plot!(p2, 1:num_periods, predicted_cum, 
        color=:purple, lw=3, linestyle=:dash, label="Model Predicted PnL"
    )
    
    # Combine Layout
    final_plot = plot(p1, p2, layout=(2,1), size=(900, 900))
    
    # Save
    Plots.savefig(final_plot, output_filename)
    println("Comparison plot saved to $output_filename")
end


plot_model_vs_reality(stats, data, "home_model_validation.html")


function calculate_dynamic_stakes(stats_df, fractional_kelly=0.5)
    # 1. Create a copy to avoid messing up original
    df = copy(stats_df)
    
    # 2. Initialize column
    df.suggested_stake = zeros(nrow(df))
    df.kelly_full = zeros(nrow(df))
    
    for i in 1:nrow(df)
        # Extract Bayesian parameters
        p = df.prob_mean[i]
        b = df.win_mean[i] # This is (Decimal Odds - 1)
        
        # Check for Edge
        # Edge = (p * b) - (1-p)
        # We can just use the 'edge_mean' column you already have!
        edge = df.edge_mean[i]
        
        if edge <= 0
            # NEGATIVE EDGE: Do not bet
            df.kelly_full[i] = 0.0
            df.suggested_stake[i] = 0.0
        else
            # POSITIVE EDGE: Calculate Kelly
            # Formula: f = Edge / Odds_Profit (b)
            # This is mathematically identical to (p(b+1)-1)/b
            f_star = edge / b
            
            # Constraint: Kelly can't exceed 100% (or probability p)
            # Practically, caps around 0.5 are sane
            f_star = min(f_star, 0.5)
            
            df.kelly_full[i] = f_star
            
            # Apply Fractional Kelly (Safety Factor)
            # Professional standard is 0.3 to 0.5 (Half Kelly)
            df.suggested_stake[i] = f_star * fractional_kelly
        end
    end
    
    return df
end

# Usage:
final_strategy = calculate_dynamic_stakes(stats, 0.5) # Half Kelly


using Plots, PlotlyJS, DataFrames

function compare_wealth_trajectories(stats_df, data, output_filename="wealth_comparison.html")
    plotlyjs() # Interactive plots

    # 1. Setup
    num_periods = data.num_periods
    starting_bankroll = 100.0
    
    # Track wealth over time
    wealth_dynamic = zeros(num_periods + 1)
    wealth_flat    = zeros(num_periods + 1)
    wealth_full    = zeros(num_periods + 1)
    
    # Initialize
    wealth_dynamic[1] = starting_bankroll
    wealth_flat[1]    = starting_bankroll
    wealth_full[1]    = starting_bankroll
    
    # 2. Simulation Loop
    for t in 1:num_periods
        # Get outcomes for this period
        # Wins: Return > 0 (e.g., +0.95)
        period_wins = data.wins[data.win_idx .== t]
        # Losses: Magnitude > 0 (e.g., 1.0) -> PnL = -1.0 * magnitude
        period_losses = data.losses[data.loss_idx .== t]
        
        # Get Stakes for this period (from the stats dataframe)
        # Dynamic: The "Suggested" stake (Half Kelly / Shrunk)
        stake_pct_dyn = stats_df.suggested_stake[t]
        
        # Full: The "Raw" Kelly (Aggressive)
        stake_pct_full = stats_df.kelly_full[t]
        
        # Flat: Fixed 2% if Edge > 0, else 0%
        stake_pct_flat = stats_df.edge_mean[t] > 0 ? 0.02 : 0.0
        
        # --- Calculate Period PnL ---
        # We assume "Simple Interest" within the week (betting parallel)
        # PnL = Stake_Amount * Sum(Returns)
        
        # Sum of returns for the period
        # (Sum of Win Odds-1) - (Sum of Loss Magnitudes)
        net_return_multiplier = sum(period_wins) - sum(period_losses)
        
        # Update Bankrolls
        # Wealth += Wealth * Stake_Pct * Net_Return
        
        # Dynamic
        w_dyn = wealth_dynamic[t]
        pnl_dyn = w_dyn * stake_pct_dyn * net_return_multiplier
        wealth_dynamic[t+1] = max(0.0, w_dyn + pnl_dyn) # Prevent negative wealth
        
        # Flat
        w_flat = wealth_flat[t]
        pnl_flat = w_flat * stake_pct_flat * net_return_multiplier
        wealth_flat[t+1] = max(0.0, w_flat + pnl_flat)
        
        # Full Kelly
        w_full = wealth_full[t]
        pnl_full = w_full * stake_pct_full * net_return_multiplier
        wealth_full[t+1] = max(0.0, w_full + pnl_full)
    end

    # 3. Plotting
    p = plot(0:num_periods, wealth_dynamic, 
        label="Dynamic Kelly (Smart)", 
        lw=3, color=:blue,
        title="Wealth Comparison: Smart vs. Naive",
        ylabel="Bankroll (Units)",
        xlabel="Period (Weeks)",
        legend=:topleft
    )
    
    plot!(p, 0:num_periods, wealth_flat, 
        label="Naive Flat Stake (2%)", 
        lw=2, color=:red, linestyle=:dash
    )
    
    plot!(p, 0:num_periods, wealth_full, 
        label="Full Kelly (Aggressive)", 
        lw=1, color=:orange, linestyle=:dot
    )
    
    # Save
    Plots.savefig(p, output_filename)
    println("Wealth comparison saved to $output_filename")
end

# Usage:
# Assume 'final_strategy' is the df from the previous step
# Assume 'data' is the prepared data struct
compare_wealth_trajectories(final_strategy, data, "wealth_comparison.html")

# 
data = prepare_optimized_data(ledger.df, :over_15, BiWeekly())
chain = sample(
    model, 
    NUTS(0.65), 
    MCMCThreads(), 
    500, 
    8,
    adtype = AutoReverseDiff(compile=true),
)

stats = extract_chain_parameters(chain, data.num_periods, data.global_mean)
plot_strategy_html(stats, "home_strategy_report.html")
plot_model_vs_reality(stats, data, "home_model_validation.html")


function run_strict_walk_forward(ledger_df, market, burn_in_weeks=4)
    # 1. Prepare ALL data first (to get indices correct)
    full_data = prepare_optimized_data(ledger_df, market, BiWeekly())
    num_periods = full_data.num_periods
    
    # Storage for "Out-of-Sample" predictions
    oos_wealth = [100.0] # Start with 100 units
    oos_edge = Float64[]
    oos_stakes = Float64[]
    
    println("Starting Strict Walk-Forward (This will be slow)...")
    
    # 2. The Loop (Simulating Time)
    # We step through time 't'. At each step, we predict t+1
    for t in burn_in_weeks:(num_periods-1)
        print("Processing Week $t / $(num_periods-1)... ")
        
        # --- A. The "Past" (Data 1 to t) ---
        # We slice the data to pretend we are living in week 't'
        # We construct a subset of wins/losses that only happened BEFORE or AT week t
        current_wins_mask = full_data.win_idx .<= t
        current_losses_mask = full_data.loss_idx .<= t
        
        # --- B. The Model (Train on Past) ---
        # We use a smaller model for this slice
        model = optimized_mixed_gamma(
            full_data.wins[current_wins_mask], 
            full_data.win_idx[current_wins_mask], # Indices 1..t
            full_data.losses[current_losses_mask], 
            full_data.loss_idx[current_losses_mask], # Indices 1..t
            t, # Number of periods is 't'
            full_data.global_mean
        )
        
        # Run NUTS (fewer samples needed for update)
        # Using fewer samples (500) to speed it up
        chain = sample(model, NUTS(0.65), MCMCThreads(), 500, 4, progress=false)
        
        # --- C. The Prediction (Predict t+1) ---
        # Extract the FINAL latent state (at time t)
        # We project this forward to t+1 (Random Walk drift)
        
        # Extract last step's parameters
        # (Using the helper logic, simplified here for speed)
        # In the non-centered model, the state at 't' is the sum of all steps 1..t
        
        # We need to extract the "edge" at time t from the chain
        # Let's write a mini-extractor just for the final state
        
        # ... (Extraction logic similar to previous function) ...
        # For simplicity, assume we extracted 'edge_mean_t' from the chain
        
        # Since Random Walk is centered at 0 drift, 
        # Best Guess for t+1 = Estimate for t
        predicted_edge_next_week = mean_edge_at_t
        
        # --- D. The Decision (Bet on t+1) ---
        suggested_stake = 0.0
        if predicted_edge_next_week > 0
            # Kelly Logic
            # Approx: f = Edge / Odds
            # We use the mean win size at 't' as proxy for odds
            odds_proxy = mean_win_size_at_t
            suggested_stake = (predicted_edge_next_week / odds_proxy) * 0.5 # Half Kelly
        end
        
        push!(oos_edge, predicted_edge_next_week)
        push!(oos_stakes, suggested_stake)
        
        # --- E. The Reality (Observe t+1) ---
        # Now we look at what ACTUALLY happened in t+1
        next_wins = full_data.wins[full_data.win_idx .== (t+1)]
        next_losses = full_data.losses[full_data.loss_idx .== (t+1)]
        
        net_return = sum(next_wins) - sum(next_losses)
        
        # Update Wealth
        current_wealth = last(oos_wealth)
        new_wealth = current_wealth + (current_wealth * suggested_stake * net_return)
        push!(oos_wealth, max(0.0, new_wealth))
        
        println("Wealth: $(round(new_wealth, digits=2))")
    end
    
    return oos_wealth, oos_edge, oos_stakes
end
p


using Turing, Distributions, DataFrames, Statistics, MCMCChains

function run_strict_walk_forward(ledger_df, market, burn_in_weeks=4)
    # 1. Prepare ALL data first (to get consistent indices)
    # We use BiWeekly to keep the parameter count manageable for speed
    full_data = prepare_optimized_data(ledger_df, market, BiWeekly())
    num_periods = full_data.num_periods
    global_mean = full_data.global_mean
    log_global = log(global_mean)
    
    # Storage for the simulation
    oos_wealth = [100.0]  # Start with 100 units
    oos_edge   = Float64[]
    oos_stakes = Float64[]
    
    println("Starting Strict Walk-Forward Validation...")
    println("Total Periods: $num_periods. Burn-in: $burn_in_weeks.")
    
    # 2. The Time Loop
    # We stand at time 't' and try to predict/bet on 't+1'
    for t in burn_in_weeks:(num_periods - 1)
        
        # --- A. The "Fog of War" (Slice Data) ---
        # We only see wins/losses that happened BEFORE or AT week t
        current_wins_mask = full_data.win_idx .<= t
        current_losses_mask = full_data.loss_idx .<= t
        
        # --- B. The Model (Train on Past) ---
        # We tell the model there are only 't' periods in existence
        model = optimized_mixed_gamma(
            full_data.wins[current_wins_mask], 
            full_data.win_idx[current_wins_mask], 
            full_data.losses[current_losses_mask], 
            full_data.loss_idx[current_losses_mask], 
            t, 
            global_mean
        )
        
        # Fast Sampling (We don't need 1000 samples for a quick update, 400 is usually enough)
        # Using 4 chains in parallel
        chain = sample(model, NUTS(0.65), MCMCThreads(), 400, 4, progress=false)
        
        # --- C. The Prediction (Extract State at 't') ---
        # We need to reconstruct the latent state at time t to predict t+1
        # Random Walk Logic: Best guess for t+1 is the state at t
        
        # 1. Extract Hypers (Vectors of all samples)
        σ_p  = vec(chain[:σ_p])
        σ_mw = vec(chain[:σ_mu_win])
        σ_ml = vec(chain[:σ_mu_loss])
        
        # 2. Reconstruct Cumulative Sum up to t
        # Start with Init
        lat_p  = vec(chain[:z_p_init])
        lat_mw = vec(chain[:z_mw_init]) .+ log_global
        lat_ml = vec(chain[:z_ml_init]) .+ log_global
        
        # Add all steps from 1 to t-1
        if t > 1
            for step_i in 1:(t-1)
                z_p  = vec(chain[Symbol("z_p_steps[$step_i]")])
                z_mw = vec(chain[Symbol("z_mw_steps[$step_i]")])
                z_ml = vec(chain[Symbol("z_ml_steps[$step_i]")])
                
                lat_p  .+= (z_p .* σ_p)
                lat_mw .+= (z_mw .* σ_mw)
                lat_ml .+= (z_ml .* σ_ml)
            end
        end
        
        # 3. Calculate Predictive Edge
        real_p  = logistic.(lat_p)
        real_mw = exp.(lat_mw)
        real_ml = exp.(lat_ml)
        
        # This is the distribution of our edge for the NEXT week
        pred_edge_dist = (real_p .* real_mw) .- ((1.0 .- real_p) .* real_ml)
        
        # Conservative Estimate: Mean of the distribution
        predicted_edge = mean(pred_edge_dist)
        
        # Win Size Proxy (for Kelly sizing)
        predicted_win_size = mean(real_mw)

        # --- D. The Decision (Bet on t+1) ---
        suggested_stake = 0.0
        
        if predicted_edge > 0
            # Half Kelly Logic
            # f = Edge / Odds
            f_star = predicted_edge / predicted_win_size
            
            # Cap at 50% max (sanity) and apply Half Kelly multiplier
            f_star = min(f_star, 0.50)
            suggested_stake = f_star * 0.5 
        end
        
        push!(oos_edge, predicted_edge)
        push!(oos_stakes, suggested_stake)
        
        # --- E. The Reality (Observe t+1) ---
        # Unlock the data for t+1
        next_wins_mask = full_data.win_idx .== (t + 1)
        next_losses_mask = full_data.loss_idx .== (t + 1)
        
        next_wins = full_data.wins[next_wins_mask]
        next_losses = full_data.losses[next_losses_mask]
        
        # Calculate Returns for this period
        # Note: If no bets occurred in t+1, returns are 0
        pnl_sum = sum(next_wins) - sum(next_losses)
        
        # Update Bankroll
        current_wealth = last(oos_wealth)
        # We apply the stake to the *start* bankroll
        # Profit = Bankroll * Stake * (Sum of Returns) -> Simplified "Portfolio" view
        # (Assuming we bet 'stake' percent on EVERY bet in the bucket)
        
        # Note on Sum of Returns:
        # If we bet 2% on 5 games.
        # Game 1: Win (+1.0)
        # Game 2: Loss (-1.0)
        # Net: 0.0 units.
        # PnL = Bankroll * 0.02 * 0.0 = 0.
        
        pnl_amount = current_wealth * suggested_stake * pnl_sum
        new_wealth = max(0.0, current_wealth + pnl_amount)
        
        push!(oos_wealth, new_wealth)
        
        print("\rWeek $t -> Edge: $(round(predicted_edge, digits=3)) | Stake: $(round(suggested_stake*100, digits=1))% | Wealth: $(round(new_wealth, digits=2))")
    end
    
    println("\nValidation Complete.")
    return oos_wealth, oos_edge, oos_stakes
end


using Plots, PlotlyJS

function plot_reality_check(wf_wealth, smooth_wealth_vector, output_filename="reality_check.html")
    plotlyjs()
    
    # Align lengths (Walk Forward starts at burn_in)
    # Assuming smooth_wealth_vector comes from your previous compare_wealth_trajectories
    
    p = plot(wf_wealth, 
        label="Walk-Forward (Real Trading)", 
        color=:red, lw=2,
        title="The Reality Gap: Hindsight vs. Foresight",
        ylabel="Bankroll", xlabel="Weeks (Simulated)",
        legend=:topleft
    )
    
    # We might need to offset the smooth vector to match indices depending on burn-in
    # For now, just plotting the Walk-Forward is usually enough to see the "jaggedness"
    
    Plots.savefig(p, output_filename)
end

# Usage:
plot_reality_check(wf_wealth, [], "reality_check.html")


wf_wealth, wf_edge, wf_stakes = run_strict_walk_forward(ledger.df, :home, 4)

wf_wealth

wf_edge





# ------- copulas
function prepare_copula_data(ledger::AbstractDataFrame, market1::Symbol, market2::Symbol, resolution::TimeResolution=BiWeekly())
    
    # --- 1. Filter Raw Data for Both Markets ---
    # Selection 1 (e.g., :home)
    subset1 = DataFrames.subset(ledger, 
        :selection => ByRow(isequal(market1)), 
        :stake => ByRow(>(1e-6)),
        :model_name =>ByRow(isequal("SequentialFunnelModel")),
    )
    
    # Selection 2 (e.g., :btts_yes)
    subset2 = DataFrames.subset(ledger, 
        :selection => ByRow(isequal(market2)), 
        :stake => ByRow(>(1e-6)),
        :model_name =>ByRow(isequal("GRWNegativeBinomialMu")),
    )
    
    if nrow(subset1) == 0 || nrow(subset2) == 0
        error("One of the markets has no data!")
    end
    
    # --- 2. Synchronization Anchor ---
    # Find the earliest date across BOTH datasets to establish t=1
    start_date1 = minimum(subset1.date)
    start_date2 = minimum(subset2.date)
    global_start_date = min(start_date1, start_date2)
    
    # Find the latest date to determine Total Periods (T)
    end_date1 = maximum(subset1.date)
    end_date2 = maximum(subset2.date)
    global_end_date = max(end_date1, end_date2)
    
    # Calculate total periods based on resolution
    # (Just a dummy call to check the index of the last date)
    max_period = get_time_index(global_end_date, global_start_date, resolution)
    
    # --- 3. Helper to Process Single Market ---
    function process_single_market(df, global_start)
        sort!(df, :date)
        
        # Calculate indices relative to GLOBAL start
        raw_indices = map(d -> get_time_index(d, global_start, resolution), df.date)
        
        # Split Wins vs Losses (for the Mixed Gamma / NegBinomial logic)
        win_mask = df.pnl .> 0
        
        wins = df.pnl[win_mask]
        # Align indices
        win_idx = raw_indices[win_mask]
        
        # Losses (stored as positive magnitude)
        loss_mask = .!win_mask
        # Add tiny noise to true zeros
        losses = map(x -> x == 0 ? 1e-6 : abs(x), df.pnl[loss_mask])
        loss_idx = raw_indices[loss_mask]
        
        return (
            wins = wins,
            win_idx = win_idx,
            losses = losses,
            loss_idx = loss_idx,
            raw_pnl = df.pnl,       # Keep raw PnL for checks
            raw_idx = raw_indices   # Keep raw indices for checks
        )
    end
    
    # --- 4. Process Both ---
    data1 = process_single_market(subset1, global_start_date)
    data2 = process_single_market(subset2, global_start_date)
    
    # Calculate Global Means (useful for priors)
    mean1 = mean(abs.(data1.raw_pnl))
    mean2 = mean(abs.(data2.raw_pnl))
    
    println("Copula Data Prepared:")
    println("  Global Start Date: $global_start_date")
    println("  Total Periods (T): $max_period")
    println("  Market 1 ($market1): $(length(data1.wins)) wins, $(length(data1.losses)) losses.")
    println("  Market 2 ($market2): $(length(data2.wins)) wins, $(length(data2.losses)) losses.")
    
    return (
        # Market 1 Data
        m1_wins = data1.wins,
        m1_win_idx = data1.win_idx,
        m1_losses = data1.losses,
        m1_loss_idx = data1.loss_idx,
        m1_mean = mean1,
        
        # Market 2 Data
        m2_wins = data2.wins,
        m2_win_idx = data2.win_idx,
        m2_losses = data2.losses,
        m2_loss_idx = data2.loss_idx,
        m2_mean = mean2,
        
        # Shared Info
        num_periods = max_period,
        start_date = global_start_date
    )
end



copula_data = prepare_copula_data(ledger.df, :home, :btts_yes, Monthly())

using Turing, Distributions, LinearAlgebra, PDMats

@model function correlated_mixed_gamma(
    # Market 1 Data (Home)
    m1_wins, m1_win_idx, m1_losses, m1_loss_idx, m1_mean,
    # Market 2 Data (BTTS)
    m2_wins, m2_win_idx, m2_losses, m2_loss_idx, m2_mean,
    # Shared
    num_periods
)
    # --- 1. Global Priors (Independent) ---
    # Volatilities for Market 1
    σ_p1 ~ Truncated(Normal(0.05, 0.05), 0.001, 0.2)
    σ_mw1 ~ Truncated(Normal(0.1, 0.1), 0.001, 0.5)
    
    # Volatilities for Market 2
    σ_p2 ~ Truncated(Normal(0.05, 0.05), 0.001, 0.2)
    σ_mw2 ~ Truncated(Normal(0.1, 0.1), 0.001, 0.5)

    # --- 2. The Correlation Structure (The Copula) ---
    # We use LKJ Cholesky for the 2x2 correlation matrix of the Win Probs
    # eta = 2.0 implies weak preference for correlation near 0 (regularization)
    L_rho ~ LKJCholesky(2, 2.0) 
    
    # Construct the Covariance Matrix for the steps
    # We need to scale the correlation (L_rho) by the volatilities (sigma)
    # But for efficiency, we generate correlated standard normals first
    
    # --- 3. The Correlated Random Walk (Win Probabilities) ---
    # We draw correlated steps for the ENTIRE timeline at once
    # Z_steps is a (2 x num_periods-1) matrix of correlated innovations
    
    # Draw standard normal steps (2 x T-1)
    z_raw ~ filldist(Normal(0, 1), 2, num_periods - 1)
    
    # Apply Correlation: Z_corr = L * Z_raw
    # L_rho.L is the lower triangular Cholesky factor
    z_corr = L_rho.L * z_raw
    
    # Reconstruct the Walks (Non-Centered)
    # Market 1 Walk
    z_p1_steps = z_corr[1, :] # Correlated steps for M1
    x_p1_init ~ Normal(0, 1)
    x_p1 = cumsum(vcat(x_p1_init, z_p1_steps .* σ_p1))
    
    # Market 2 Walk
    z_p2_steps = z_corr[2, :] # Correlated steps for M2
    x_p2_init ~ Normal(0, 1)
    x_p2 = cumsum(vcat(x_p2_init, z_p2_steps .* σ_p2))

    # --- 4. The Independent Random Walks (Win/Loss Sizes) ---
    # (Same standard logic as before)
    
    # Market 1 Size
    z_mw1_init ~ Normal(0, 1)
    z_mw1_steps ~ filldist(Normal(0, 1), num_periods - 1)
    x_mw1 = cumsum(vcat(z_mw1_init + log(m1_mean), z_mw1_steps .* σ_mw1))
    # Assuming Loss Size ~ Win Size volatility for simplicity (or separate them if needed)
    x_ml1 = cumsum(vcat(z_mw1_init + log(m1_mean), z_mw1_steps .* σ_mw1)) 

    # Market 2 Size
    z_mw2_init ~ Normal(0, 1)
    z_mw2_steps ~ filldist(Normal(0, 1), num_periods - 1)
    x_mw2 = cumsum(vcat(z_mw2_init + log(m2_mean), z_mw2_steps .* σ_mw2))
    x_ml2 = cumsum(vcat(z_mw2_init + log(m2_mean), z_mw2_steps .* σ_mw2)) 

    # --- 5. Parameter Transformation ---
    # Market 1
    π1_vec = logistic.(x_p1)
    μ_w1_vec = exp.(x_mw1)
    μ_l1_vec = exp.(x_ml1)
    
    # Market 2
    π2_vec = logistic.(x_p2)
    μ_w2_vec = exp.(x_mw2)
    μ_l2_vec = exp.(x_ml2)
    
    # Fixed CV
    inv_cv2 = 1 / 0.5^2
    cv_fact = 0.5^2

    # --- 6. Likelihood Block (Market 1) ---
    if length(m1_wins) > 0
        p_w = π1_vec[m1_win_idx]; μ_w = μ_w1_vec[m1_win_idx]
        Turing.@addlogprob! sum(log.(p_w) .+ logpdf.(Gamma.(inv_cv2, μ_w .* cv_fact), m1_wins))
    end
    if length(m1_losses) > 0
        p_l = π1_vec[m1_loss_idx]; μ_l = μ_l1_vec[m1_loss_idx]
        Turing.@addlogprob! sum(log1p.(-p_l) .+ logpdf.(Gamma.(inv_cv2, μ_l .* cv_fact), m1_losses))
    end

    # --- 7. Likelihood Block (Market 2) ---
    if length(m2_wins) > 0
        p_w = π2_vec[m2_win_idx]; μ_w = μ_w2_vec[m2_win_idx]
        Turing.@addlogprob! sum(log.(p_w) .+ logpdf.(Gamma.(inv_cv2, μ_w .* cv_fact), m2_wins))
    end
    if length(m2_losses) > 0
        p_l = π2_vec[m2_loss_idx]; μ_l = μ_l2_vec[m2_loss_idx]
        Turing.@addlogprob! sum(log1p.(-p_l) .+ logpdf.(Gamma.(inv_cv2, μ_l .* cv_fact), m2_losses))
    end
end

# 1. Instantiate
model = correlated_mixed_gamma(
    copula_data.m1_wins, copula_data.m1_win_idx, copula_data.m1_losses, copula_data.m1_loss_idx, copula_data.m1_mean,
    copula_data.m2_wins, copula_data.m2_win_idx, copula_data.m2_losses, copula_data.m2_loss_idx, copula_data.m2_mean,
    copula_data.num_periods
)

# 2. Sample (using 4 chains)
println("Sampling Joint Model...")

chain_joint = sample(
    model, 
    NUTS(0.65), 
    MCMCThreads(), 
    500, 
    8,
    adtype = AutoReverseDiff(compile=true),
)

describe(chain_joint) 


using Plots, StatsPlots

function plot_correlation_posterior(chain)
    # 1. Extract the L[2,1] samples
    # Note: Turing might store it as :L_rho, or flattened
    # Based on your output, the symbol is Symbol("L_rho.L[2, 1]")
    
    # We try to access it dynamically
    if :L_rho in keys(chain)
        # If stored as matrix
        rho_samples = [x.L[2,1] for x in chain[:L_rho]]
    else
        # If stored flattened (most likely based on your output)
        rho_samples = vec(chain[Symbol("L_rho.L[2, 1]")])
    end
    
    # 2. Plot Density
    p = density(rho_samples, 
        title="Posterior Distribution of Correlation (Rho)",
        label="Correlation",
        xlabel="Correlation Coefficient",
        ylabel="Density",
        fill=(0, 0.3, :blue),
        xlims=(-1, 1),
        lw=2
    )
    
    # Add mean line
    vline!(p, [mean(rho_samples)], label="Mean: $(round(mean(rho_samples), digits=3))", color=:red, linestyle=:dash)
    
    # 3. Save
    savefig(p, "correlation_posterior.html")
    println("Correlation plot saved.")
end

using Plots, Statistics

function plot_correlation_histogram(chain)
    # Extract Rho
    # If the chain names are flattened, it's likely :L_rho_L_2_1 or similar
    # Use string matching to find it safely
    all_names = string.(names(chain))
    rho_name = filter(x -> contains(x, "L") && contains(x, "2") && contains(x, "1"), all_names)[1]
    
    rho_samples = vec(chain[Symbol(rho_name)])
    
    # Plot
    p = histogram(rho_samples, 
        title="Posterior Distribution of Correlation (Rho)",
        label="Correlation Frequency",
        xlabel="Correlation Coefficient",
        ylabel="Count",
        color=:blue,
        alpha=0.6,
        nbins=30, # 30 bars
        xlims=(-1, 1),
        normalize=:pdf # Make it look like a density
    )
    
    vline!(p, [mean(rho_samples)], label="Mean: $(round(mean(rho_samples), digits=3))", color=:red, lw=3)
    
    savefig(p, "correlation_histogram.html")
    println("Correlation histogram saved.")
end

plot_correlation_histogram(chain_joint)

# Usage
plot_correlation_posterior(chain_joint)


using Optim, LinearAlgebra, Statistics

function optimize_portfolio_weights(chain, num_periods, global_mean)
    println("Running Portfolio Optimization (Monte Carlo)...")
    
    # --- 1. Extract Latest State (Parameters at Time T) ---
    # We need the parameters for the *NEXT* bet, which implies projecting from the last known state.
    
    # A. Global Volatilities (Vectors of 4000 samples)
    σ_p1 = vec(chain[:σ_p1]); σ_mw1 = vec(chain[:σ_mw1])
    σ_p2 = vec(chain[:σ_p2]); σ_mw2 = vec(chain[:σ_mw2])
    
    # B. Latest Latent States (reconstructed from the end of the chain)
    # Note: Turing usually stores the full trajectory. We need the value at `num_periods`.
    # To save time, let's extract the "init" and assume the random walk is centered, 
    # or just use the summary statistics of the *last week* if you ran a Walk-Forward.
    
    # BETTER APPROACH FOR BATCH: 
    # We simulate "next week" by taking the current 'edge' estimate from the model 
    # and adding one step of random walk noise.
    
    # Let's rebuild the final state for each sample in the chain:
    n_samples = length(σ_p1)
    
    # We need L_rho for correlation
    # Find the name again
    all_names = string.(names(chain))
    rho_name = filter(x -> contains(x, "L") && contains(x, "2") && contains(x, "1"), all_names)[1]
    rho_samples = vec(chain[Symbol(rho_name)])
    
    # --- 2. Monte Carlo Simulation Loop ---
    # We will generate 10,000 synthetic "Next Weeks"
    n_sims = 10000
    
    # Storage for simulated returns of Market 1 and Market 2
    sim_r1 = zeros(n_sims)
    sim_r2 = zeros(n_sims)
    
    for i in 1:n_sims
        # A. Pick a random parameter sample (Posterior Uncertainty)
        idx = rand(1:n_samples)
        
        # B. Get Correlation for this sample
        ρ = rho_samples[idx]
        
        # C. Generate Correlated Innovations (The Copula)
        # Draw [z1, z2] from Bivariate Normal(0, Sigma)
        # Sigma = [1 rho; rho 1]
        # Cholesky Decomposition: L = [1 0; rho  sqrt(1-rho^2)]
        z1_raw = randn()
        z2_raw = randn()
        
        z1_corr = z1_raw
        z2_corr = (ρ * z1_raw) + (sqrt(1 - ρ^2) * z2_raw)
        
        # D. Project "Win Probability" for Next Week
        # We assume the current estimate is the mean of the last period
        # (Simplified: using a generic prior for "current state" to demonstrate logic)
        # In production, you pass the `last_p1` and `last_p2` from your extraction function.
        
        # Let's assume the strategies are currently at their historical average for this demo
        # (You should replace these with your actual `wf_edge` latest values!)
        current_p1_logit = 0.0 # ~50%
        current_p2_logit = 0.0 # ~50%
        
        # Add the Random Walk Step
        next_p1 = logistic(current_p1_logit + z1_corr * σ_p1[idx])
        next_p2 = logistic(current_p2_logit + z2_corr * σ_p2[idx])
        
        # E. Determine Win/Loss (Bernoulli Trial)
        is_win1 = rand() < next_p1
        is_win2 = rand() < next_p2
        
        # F. Determine Magnitude (Gamma)
        # Simulating a return. 
        # Win = +0.95 (approx odds 1.95), Loss = -1.0
        # You can make this stochastic too using Gamma, but fixed is fine for weights.
        r1 = is_win1 ? 0.95 : -1.0
        r2 = is_win2 ? 0.95 : -1.0
        
        sim_r1[i] = r1
        sim_r2[i] = r2
    end
    
    # --- 3. The Optimizer (Maximize Log Growth) ---
    # We want to find w = [w1, w2] to maximize sum(log(1 + w1*r1 + w2*r2))
    
    function kelly_objective(w)
        w1, w2 = w[1], w[2]
        
        # Constraint: No leverage > 1 (optional but safe)
        if w1 < 0 || w2 < 0 || w1 + w2 > 1.0
            return Inf # Invalid
        end
        
        # Calculate Bankroll Growth for all 10,000 sims
        # G = 1 + w1*r1 + w2*r2
        growth = 1.0 .+ (w1 .* sim_r1) .+ (w2 .* sim_r2)
        
        # Avoid log(negative) -> Ruin
        if any(growth .<= 0)
            return Inf
        end
        
        return -mean(log.(growth)) # Negative because we minimize
    end
    
    # Run Optimization
    # Start at [0.01, 0.01]
    res = optimize(kelly_objective, [0.0, 0.0], [0.5, 0.5], [0.01, 0.01], Fminbox(BFGS()))
    
    best_weights = Optim.minimizer(res)
    
    println("\n--- Optimal Portfolio Weights ---")
    println("Strategy 1 (Home): $(round(best_weights[1]*100, digits=2))%")
    println("Strategy 2 (BTTS): $(round(best_weights[2]*100, digits=2))%")
    println("Correlation Used: Copula-based (Mean Rho = $(round(mean(rho_samples), digits=3)))")
    
    return best_weights
end

# Usage:
w = optimize_portfolio_weights(chain_joint, 29, 0.1)

using Optim, LinearAlgebra, Statistics, LogisticFunctions

function optimize_live_portfolio(chain, current_edge_1, current_odds_1, current_edge_2, current_odds_2)
    println("Running Portfolio Optimization for LIVE bets...")
    println("Strategy 1: Edge $(round(current_edge_1*100, digits=2))% @ Odds $current_odds_1")
    println("Strategy 2: Edge $(round(current_edge_2*100, digits=2))% @ Odds $current_odds_2")
    
    # --- 1. Convert Edge to Logit (The Latent State) ---
    # We need to reverse-engineer the "current logit" that corresponds to your edge.
    # Edge = (P * (Odds-1)) - (1-P)
    # P = (Edge + 1) / Odds
    
    prob_1 = (current_edge_1 + 1.0) / current_odds_1
    prob_2 = (current_edge_2 + 1.0) / current_odds_2
    
    # Safety clamp (Kelly blows up if Prob=1.0)
    prob_1 = clamp(prob_1, 0.01, 0.99)
    prob_2 = clamp(prob_2, 0.01, 0.99)
    
    current_logit_1 = logit(prob_1)
    current_logit_2 = logit(prob_2)
    
    # --- 2. Extract Posterior Samples ---
    σ_p1 = vec(chain[:σ_p1]); 
    σ_p2 = vec(chain[:σ_p2]); 
    
    # Extract Rho
    all_names = string.(names(chain))
    rho_name = filter(x -> contains(x, "L") && contains(x, "2") && contains(x, "1"), all_names)[1]
    rho_samples = vec(chain[Symbol(rho_name)])
    
    n_samples = length(σ_p1)
    
    # --- 3. Monte Carlo Simulation (Next Week) ---
    n_sims = 20000
    sim_r1 = zeros(n_sims)
    sim_r2 = zeros(n_sims)
    
    for i in 1:n_sims
        idx = rand(1:n_samples)
        ρ = rho_samples[idx]
        
        # Generate Correlated Random Walk Step
        z1 = randn()
        z2_raw = randn()
        z2 = (ρ * z1) + (sqrt(1 - ρ^2) * z2_raw)
        
        # Project Future Probability
        # We start from YOUR predicted edge and add one week of uncertainty
        future_p1 = logistic(current_logit_1 + z1 * σ_p1[idx])
        future_p2 = logistic(current_logit_2 + z2 * σ_p2[idx])
        
        # Determine Outcome
        is_win1 = rand() < future_p1
        is_win2 = rand() < future_p2
        
        # Determine Return (Using your specific odds)
        sim_r1[i] = is_win1 ? (current_odds_1 - 1.0) : -1.0
        sim_r2[i] = is_win2 ? (current_odds_2 - 1.0) : -1.0
    end
    
    # --- 4. Optimizer (Maximize Log Growth) ---
    function kelly_objective(w)
        w1, w2 = w[1], w[2]
        if w1 < 0 || w2 < 0 || w1 + w2 > 0.99 return Inf end
        
        # Portfolio Return = 1 + w1*r1 + w2*r2
        growth = 1.0 .+ (w1 .* sim_r1) .+ (w2 .* sim_r2)
        
        if any(growth .<= 0) return Inf end
        return -mean(log.(growth))
    end
    
    # Constrain to max 30% per bet for sanity
    res = optimize(kelly_objective, [0.0, 0.0], [0.3, 0.3], [0.01, 0.01], Fminbox(BFGS()))
    best_w = Optim.minimizer(res)
    
    println("\n--- FINAL SUGGESTED STAKES ---")
    println("Home Strategy: $(round(best_w[1]*100, digits=2))%")
    println("BTTS Strategy: $(round(best_w[2]*100, digits=2))%")
    
    return best_w
end

# 1. Run Walk-Forward for Home
# (You already did this, stored in wf_edge)
latest_home_edge = last(wf_edge) # e.g. 0.015 (1.5% edge)
home_odds = 2.5 # Replace with current average odds for this strategy

# 2. Run Walk-Forward for BTTS
# (You need to run this function for the BTTS market now)
wf_wealth_btts, wf_edge_btts, _ = run_strict_walk_forward(ledger.df, :btts_yes, 4)
latest_btts_edge = last(wf_edge_btts) 
btts_odds = 1.8 # Replace with actual odds
