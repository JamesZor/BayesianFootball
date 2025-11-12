using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics


# --- Phase 1: Globals (D, M, G) --- (Same as before)
data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)

# filter for one season for quick training
df = filter(row -> row.season=="24/25", data_store.matches)

# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)

# 1. Define your "special cases" mapping
split_map = Dict(37 => 1, 38 => 2, 39 => 3)

# 2. Use get() with a default value of 0
#    We use Ref(split_map) to tell Julia to treat the Dict as a single object
#    and not try to broadcast over its elements.
ds.matches.split_col = get.(Ref(split_map), ds.matches.match_week, 0);

# large v2 
ds.matches.split_col = max.(0, ds.matches.match_week .- 14);






splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential) #
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

# --- Phase 3: Define Training Configuration ---
# Sampler Config (Choose one)
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=100) # Use renamed struct

# Explicitly set a limit (e.g., if NUTS uses 2 chains, maybe allow 4 concurrent splits on 8 threads)
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

# training_config_limited = TrainingConfig(sampler_conf, strategy_parallel_limited)
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

# Then run:
# results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)
# save and load 
#
# JLD2.save_object("training_results_large.jld2", results)

# results = JLD2.load_object("training_results.jld2")
results = JLD2.load_object("training_results_large.jld2")

### get out of sample data - chains 
# 1. Define the column you want to split on
#    (You can change this to :round, :week, etc. later)
split_col_name = :split_col

# 2. Get all unique split keys (e.g., [0, 1, 2, 3])
all_splits = sort(unique(ds.matches[!, split_col_name]))

# 3. Define the splits you want to *predict* (e.g., [1, 2, 3])
#    We skip the first key (0), as it was for training the first model
prediction_split_keys = all_splits[2:end] 

# 4. Group the data ONCE
grouped_matches = groupby(ds.matches, split_col_name)

# 5. Create the vector of DataFrames (as efficient SubDataFrame views)
#    This is the new argument for your function
dfs_to_predict = [
    grouped_matches[(; split_col_name => key)] 
    for key in prediction_split_keys
]


# --- 6. Call your new function ---
# It's now much cleaner and more flexible
all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results
)

#################################################
# Dev area 
#################################################

# here we want to use the open line odds


"""
Parses a fractional odds string (e.g., "19/10") into a decimal value (e.g., 2.9).
Returns 0.0 if parsing fails (e.g., for "SP", missing, or "1").
"""
function parse_fractional_to_decimal(s::AbstractString)
    parts = split(s, '/')
    
    # Must be exactly two parts (numerator and denominator)
    if length(parts) != 2
        return 0.0
    end
    
    try
        n = parse(Float64, parts[1])
        d = parse(Float64, parts[2])
        
        # Avoid division by zero
        if d == 0.0
            return 0.0
        end
        
        # Convert from fractional (e.g., 1.9) to decimal (e.g., 2.9)
        return (n / d) + 1.0
    catch e
        # This will catch errors if parts[1] or parts[2] are not valid numbers (e.g., "SP")
        return 0.0
    end
end


ds_odds_initial = deepcopy(ds.odds)

# Convert the initial fractional string to a decimal
ds_odds_initial.initial_decimal = parse_fractional_to_decimal.(ds_odds_initial.initial_fractional_value)

# Filter out rows where parsing failed or odds were 0
filter!(row -> row.initial_decimal > 1.0, ds_odds_initial)

# --- THIS IS THE "TRICK" ---
# Overwrite the `decimal_odds` column with our new initial odds.
# Your `get_market` function will now read this column,
# thinking it's the final odds.
ds_odds_initial.decimal_odds = round.(ds_odds_initial.initial_decimal, digits=2)

# create new data store.
ds = BayesianFootball.Data.DataStore(
    df,
    ds_odds_initial,
    data_store.incidents
)




# +++ Compute kelly 

"""
Calculates the optimal Kelly criterion fraction to bet.

# Arguments
- `decimal_odds`: The decimal odds offered by the bookmaker (e.g., 2.5, 3.0).
- `probability`: Your estimated true probability of the event occurring (e.g., 0.45).

# Returns
- `f`: The fraction of your bankroll to bet (from 0.0 to 1.0).
  A value of 0.0 means the bet has no value (p < 1/decimal_odds).
"""
function kelly_fraction(decimal_odds::Number, probability::Number)
  return max(0.0, probability - ( (1 - probability) / (decimal_odds - 1.0)))
end 

function kelly_fraction(decimal_odds::Number, probability::AbstractVector)
  return kelly_fraction.(decimal_odds, probability)
end 

function kelly_fraction(odds::NamedTuple, probabilities::NamedTuple) 
  common_keys = keys(odds) ∩ keys(probabilities)
  return NamedTuple(
          k => kelly_fraction(odds[k], probabilities[k])
          for k in common_keys
      )
end


# --- compute kelly dev 

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

match_id = rand(keys(all_oos_results))
r1 =  all_oos_results[match_id]
subset( ds.matches, :match_id => ByRow(isequal(match_id)))

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

closing_match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
match_odds = BayesianFootball.Predictions.get_market_opening_lines(match_id, predict_config, ds.odds)
match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)

kf = kelly_fraction(match_odds, match_predict)

# +++ Kelly signals 
function kelly_positive( kelly_dist::AbstractVector, c::Number)
  return mean(kelly_dist .> c)
end 

function kelly_decision_rule( kelly_dist::AbstractVector, c::Number, b::Number)::Bool
  return kelly_positive(kelly_dist, c) >= b 
end 



# --- kelly signal dev 

key_i = rand(keys(kf))

kelly_positive( kf[:home], 0.02)
kelly_decision_rule( kf[:home], 0.02, 0.5)

kelly_positive( kf[:over_25], 0.0)
kelly_positive( kf[:under_25], 0.0)

kelly_decision_rule( kf[:over_25], 0.06, 0.3)
kelly_decision_rule( kf[:under_25], 0.06, 0.3)

kelly_positive( kf[:over_35], 0.05)
kelly_positive( kf[:under_35], 0.05)

using StatsPlots
density( kf[:over_35], label="over")
density!( kf[:under_35], label="under")


# +++ kelly stake, 
function kellys_stake_precent(kelly_dist::AbstractVector, kellys_fraction::Number)::Float64 
  return kellys_fraction * median(kelly_dist)
end 

kellys_stake_precent( kf[:home], 1)


# +++ strategy 
function kelly_strategy(kelly_dist::AbstractVector, c::Number, b::Number, f::Number)::Number 
  return kelly_decision_rule(kelly_dist, c, b) * kellys_stake_precent(kelly_dist, f) 
end 


kelly_strategy(kf[:home], 0.05, 0.3, 1)





# +++ 

# --- 1. Load SciML Optimization Packages ---
using Optimization
using OptimizationBBO # Black-Box Optimization (derivative-free)
using SciMLBase # For NoAD()

"""
Calculates the geometric equity curve from a series of fractional stakes.
This correctly models compounding growth and risk of ruin.
"""
function calculate_equity_curve(
    stakes::AbstractVector{<:Number}, 
    odds::AbstractVector{<:Number}, 
    results::AbstractVector{Bool}
    )::Vector{Float64}
    
    # Start with a 1.0 unit bankroll
    bankroll = 1.0
    equity_curve = [bankroll]

    for (stake_frac, odd, result) in zip(stakes, odds, results)
        if stake_frac <= 0.0 || stake_frac > 1.0 # No bet or invalid stake
            push!(equity_curve, bankroll)
            continue
        end

        bet_amount = bankroll * stake_frac
        
        if result == true
            # Win
            bankroll += bet_amount * (odd - 1.0)
        else
            # Loss
            bankroll -= bet_amount
        end

        # Handle ruin
        if bankroll <= 0.0
            bankroll = 0.0
            push!(equity_curve, bankroll)
            break 
        end
        
        push!(equity_curve, bankroll)
    end
    return equity_curve
end

"""
Calculates the Calmar Ratio from a geometric equity curve.
We define it as Total Geometric Return / Max Drawdown.
"""
function calculate_calmar_ratio(equity_curve::Vector{Float64})::Float64
    if isempty(equity_curve) || equity_curve[1] <= 0.0
        return -1e9 # Penalize errors
    end

    # 1. Total Geometric Return (e.g., 1.5 -> 50% return)
    total_return = (equity_curve[end] / equity_curve[1]) - 1.0

    # 2. Maximum Drawdown
    peak = -Inf
    max_drawdown = 0.0
    
    for value in equity_curve
        if value > peak
            peak = value
        end
        
        if peak > 0.0
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown
                max_drawdown = drawdown
            end
        elseif peak == 0.0 # We hit ruin
            max_drawdown = 1.0 # 100% drawdown
            break
        end
    end

    # --- Scoring ---
    if total_return <= 0.0
        return total_return * 10.0 # Penalize
    end

    if max_drawdown == 0.0
        return total_return > 0.0 ? 1e9 : 0.0 # Huge score if profitable
    end

    return total_return / max_drawdown
end




HomeMarketData = @NamedTuple{
    k_dist::Vector{Float64}, # Kelly distribution
    odds::Float64,           # Opening odds
    result::Bool             # Did the home team win?
}

home_market_data = HomeMarketData[]

println("--- Starting Data Pre-processing ---")
predict_config = BayesianFootball.Predictions.PredictionConfig(BayesianFootball.Markets.get_standard_markets())
processed_count = 0
error_count = 0

for match_id in keys(all_oos_results)
    try
        r1 = all_oos_results[match_id]
        match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
        match_odds = BayesianFootball.Predictions.get_market_opening_lines(match_id, predict_config, ds.odds)
        match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)

        if :home ∉ keys(match_odds) || :home ∉ keys(match_predict) || match_odds.home <= 1.0
            continue
        end
        
        k_dist_home = kelly_fraction(match_odds.home, match_predict.home)

        push!(home_market_data, (
            k_dist = k_dist_home,
            odds = match_odds.home,
            result = match_results.home
        ))
        processed_count += 1
    catch e
        error_count += 1
    end
end

println("--- Pre-processing Complete ---")
println("Successfully processed $(processed_count) matches.")
println("Skipped $(error_count) matches due to errors or missing data.")


# --- 3. Pre-process Data (Modified for :under_25) ---

# Define the market we are targeting
TARGET_MARKET = :over_25 # <-- THE ONLY CHANGE IS HERE


MarketData = @NamedTuple{
    k_dist::Vector{Float64}, # Kelly distribution
    odds::Float64,           # Opening odds
    result::Bool             # Did the home team win?
}


market_data = MarketData[] # <-- Generic name

println("--- Starting Data Pre-processing for: $(TARGET_MARKET) ---")
predict_config = BayesianFootball.Predictions.PredictionConfig(BayesianFootball.Markets.get_standard_markets())
processed_count = 0
error_count = 0



for match_id in keys(all_oos_results)
    try
        r1 = all_oos_results[match_id]
        match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
        match_odds = BayesianFootball.Predictions.get_market_opening_lines(match_id, predict_config, ds.odds)
        match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)


        if TARGET_MARKET ∉ keys(match_odds) || 
           TARGET_MARKET ∉ keys(match_predict) || 
           match_odds[TARGET_MARKET] <= 1.0
            continue # Skip if market doesn't exist or odds are invalid
        end
        
      k_dist_home = kelly_fraction(match_odds[TARGET_MARKET], match_predict[TARGET_MARKET])

        push!(market_data, (
            k_dist = k_dist_home,
            odds = match_odds[TARGET_MARKET],
            result = match_results[TARGET_MARKET]
        ))
        processed_count += 1
    catch e
        error_count += 1
    end
end

println("--- Pre-processing Complete ---")
println("Successfully processed $(processed_count) matches for $(TARGET_MARKET).")
println("Skipped $(error_count) matches due to errors or missing data.")



"""
This is the function Optimization.jl will call.
It must take params `u` (our vector) and `p` (static data, which we don't use).
It runs a full backtest and returns a score (Negative Calmar).
"""
function objective_function(params, p_unused)
    c, b, f = params
    
    # We can put bounds-checking here, though Optimization.jl handles it
    if c < 0.0 || b < 0.0 || b > 1.0 || f <= 0.0 || f > 1.0
        return Inf # Return a "very bad" score (positive infinity)
    end

    # 1. Generate stakes
    # Note: We access the pre-processed data from the global scope
    # A more advanced setup would pass `home_market_data` as `p_unused`
    stakes = Vector{Float64}(undef, length(market_data))
    odds_list = Vector{Float64}(undef, length(market_data))
    results_list = Vector{Bool}(undef, length(market_data))

    for (i, match) in enumerate(market_data)
        stakes[i] = kelly_strategy(match.k_dist, c, b, f)
        odds_list[i] = match.odds
        results_list[i] = match.result
    end
    
    # 2. Calculate the equity curve
    equity_curve = calculate_equity_curve(stakes, odds_list, results_list)
    
    # 3. Calculate the final score
    score = calculate_calmar_ratio(equity_curve)

    if !isfinite(score)
        return Inf # "Very bad"
    end

    # We want to MAXIMIZE Calmar, so we return the NEGATIVE score
    return -score
end



# --- 5. Setup and Run Optimization.jl ---

println("\n--- Starting Optimization.jl ---")

# 1. Define the function for the optimizer
#    We explicitly state there is No Automatic Differentiation (NoAD)
#    because our function is a complex, non-differentiable backtest.
opt_func = OptimizationFunction(objective_function, SciMLBase.NoAD())

# 2. Define search space and initial guess
#    [c,     b,     f]
u0 = [0.01,  0.5,   0.2]  # Initial guess (c, b, f)
lower_bounds = [0.0,   0.5,   0.0001]
upper_bounds = [0.1,   0.6,   0.2]

# 3. Create the OptimizationProblem
prob = OptimizationProblem(
    opt_func, 
    u0, 
    lb = lower_bounds, 
    ub = upper_bounds
)


# 4. Solve the problem!
# We choose a solver from the OptimizationBBO package.
# `BBO_adaptive_de_rand_1_bin_radiuslimited()` is a robust global optimizer.

sol = solve(
    prob, 
    BBO_adaptive_de_rand_1_bin_radiuslimited(), # The chosen algorithm
    maxiters=1000  # Number of iterations
)

println("--- Optimization Complete ---")

# 5. Show the results
best_score = sol.objective
best_params = sol.u

println("Best (Negative) Calmar: $best_score")
println("Actual Calmar: $(-best_score)")
println("Optimal Parameters (c, b, f):")
println("  c (Kelly Thresh): $(best_params[1])")
println("  b (Conf Thresh):  $(best_params[2])")
println("  f (Kelly Frac):   $(best_params[3])")

# 6. Re-run final backtest (same as before)
println("\n--- Final Backtest with Optimal Params ---")
c_opt, b_opt, f_opt = best_params
stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];
odds_opt = [m.odds for m in market_data];
results_opt = [m.result for m in market_data];

final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);
final_calmar = calculate_calmar_ratio(final_curve)

println("Final Calmar (re-calculated): $final_calmar")
println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
println("Final Bankroll (starting from 1.0): $(final_curve[end])")

# If you have Plots.jl, you can uncomment this
using Plots
plot(final_curve, title="Optimal Equity Curve", label="Bankroll", legend=:topleft)


#### sens 

using Printf # For pretty-printing the table

# --- 1. Define Your Parameter Grid ---

# Test Kelly thresholds (c) from 0% to 5% in 11 steps
c_range = LinRange(0.0, 0.05, 11)

# Test Kelly Fractions (f) from 5% to 50% in 10 steps
f_range = LinRange(0.05, 0.5, 10)

# *** Your Fixed Parameter ***
b_fixed = 0.5

# --- 2. Create Storage for Results ---
# Rows will be c, Columns will be f
calmar_grid = zeros(length(c_range), length(f_range))

println("--- Starting Sensitivity Analysis (v1, b=0.5) ---")
println("Mapping $(length(calmar_grid)) parameter combinations...")

# --- 3. Run the Grid Search (Nested Loop) ---

for (i, c) in enumerate(c_range)
    for (j, f) in enumerate(f_range)
        
        # 1. Define the parameter set [c, b, f]
        params = [c, b_fixed, f]
        
        # 2. Call your original 3-parameter objective function
        #    (This function must be defined in your session)
        negative_calmar = objective_function(params, nothing)
        
        # 3. Store the *positive* Calmar
        actual_calmar = -negative_calmar
        if actual_calmar < -1.0 # This was a negative return
             actual_calmar /= 10.0 # Revert the penalty
        end

        calmar_grid[i, j] = actual_calmar
    end
end

println("--- Analysis Complete ---")

# --- 4. Print the Results as a Table ---

# Print header row (f values)
@printf "c_thresh |"
for f in f_range
    @printf "%8.2f" f
end
println("\n" * "-"^90) # Separator

# Print each row (c_thresh) and the Calmar scores
for (i, c) in enumerate(c_range)
    @printf "  %0.3f  |" c # Print c with 3 decimal places
    for j in 1:length(f_range)
        score = calmar_grid[i, j]
        if score <= 0.0
            @printf "%8s" "----"
        else
            @printf "%8.2f" score
        end
    end
    println() # New line for the next row
end


# --- 1. Define Your WFO Structure ---
n_total_matches = length(market_data)
is_size = 300 # In-sample: Use 300 matches to find params
oos_size = 100 # Out-of-sample: Test params on the next 100

all_oos_curves = [] # We'll store the small equity curve from each OOS chunk

u0 = [0.01,  0.5,   0.2]  # Initial guess (c, b, f)
lb = [0.0,   0.5,   0.0001]
ub = [0.1,   0.6,   0.2]


# --- 2. The Main WFO Loop ---
# We slide the window across the data
for start_index in 1:oos_size:(n_total_matches - is_size)
    
    is_end = start_index + is_size - 1
    oos_start = is_end + 1
    oos_end = min(oos_start + oos_size - 1, n_total_matches)
    
    if oos_end <= oos_start
        break # Not enough data left
    end

    # --- 3. Define the In-Sample Problem ---
    
    # Get the data for this *specific* in-sample chunk
    is_data_chunk = market_data[start_index:is_end]
    
    # Create the objective function *for this chunk*
    # This is a "closure" - it captures the `is_data_chunk`
    
    opt_func_is = OptimizationFunction(objective_function, SciMLBase.NoAD())
    prob_is = OptimizationProblem(opt_func_is, u0, lb=lb, ub=ub) # Use your bounds
    
    # --- 4. SOLVE the IS chunk ---
    # Find the best params *only for this past data*
    sol = solve(prob_is, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=100)
    best_params_for_this_chunk = sol.u
    
    # --- 5. TEST on OOS chunk ---
    # Get the unseen, out-of-sample data
    oos_data_chunk = market_data[oos_start:oos_end]
    
    # Apply the strategy *using the params from step 4*
    c_opt, b_opt, f_opt = best_params_for_this_chunk
    
    # Run the backtest on the OOS data
    stakes_oos = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in oos_data_chunk]
    odds_oos = [m.odds for m in oos_data_chunk]
    results_oos = [m.result for m in oos_data_chunk]
    
    # --- 6. Record the OOS Results ---
    # We save the *equity curve itself*
    oos_curve_chunk = calculate_equity_curve(stakes_oos, odds_oos, results_oos)
    push!(all_oos_curves, oos_curve_chunk)
end

# --- 7. Analyze the Final Result ---
# Now, we "stitch" the OOS results together.
# We take the *compounded return* of each chunk.
final_stitched_curve = [1.0]
total_bankroll = 1.0

for oos_curve in all_oos_curves
    # Get the fractional return of this chunk (e.g., 1.15 for +15%)
    chunk_return_frac = oos_curve[end] / oos_curve[1] 
    
    # Apply this to our total bankroll
    total_bankroll *= chunk_return_frac
    push!(final_stitched_curve, total_bankroll)
end

println("--- Walk-Forward Analysis Complete ---")
println("Final OOS Bankroll (starting from 1.0): $(final_stitched_curve[end])")

# You can now calculate the Calmar, Sharpe, etc. on `final_stitched_curve`
# This is the true, robust test of your strategy.

plot(final_stitched_curve)
