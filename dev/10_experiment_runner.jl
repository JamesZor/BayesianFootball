
using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics

save_dir = "dev_exp/simple_poisson/"

#####
# --- Phase 1: Globals (D, M, G) --- (Same as before)
######

data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)



# ---  Define Training Configuration ---
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=600, n_chains=2, n_warmup=100) # Use renamed struct
# Explicitly set a limit (e.g., if NUTS uses 2 chains, maybe allow 4 concurrent splits on 8 threads)
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4) 
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

seasons_to_train = ["20/21","21/22","22/23","23/24","24/25"]


for season_str in seasons_to_train

    println("Processing season $season_str.")

    # create the data set 

    # filter for one season for quick training
    df = filter(row -> row.season==season_str, data_store.matches)
    # we want to get the last 4 weeks - so added the game weeks
    df = BayesianFootball.Data.add_match_week_column(df)
    df.split_col = max.(0, df.match_week .- 14);

    ds = BayesianFootball.Data.DataStore(
        df,
        data_store.odds,
        data_store.incidents
    )

    ## Set the sets

    splitter_config = BayesianFootball.Data.ExpandingWindowCV([], [season_str], :split_col, :sequential) #
    data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
    feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

    ## run  
    results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)

    ## save 
    save_season_name_str = save_dir * "s_" * replace(season_str, "/" => "_") * ".jld2"
    
    JLD2.save_object(save_season_name_str, results)

    
println("Finished season $season_str.")



end 


#############################
# Loading 
#############################
season_to_load = seasons_to_train[1]

season_to_load_str = save_dir * "s_" * replace(season_to_load, "/" => "_") * ".jld2"

results = JLD2.load_object(season_to_load_str)


## sort the data set 

df = filter(row -> row.season==season_to_load, data_store.matches)
# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)
df.split_col = max.(0, df.match_week .- 14);

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)

# here we want to use the open line odds
BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(data_store)
ds = data_store


split_col_name = :split_col
all_splits = sort(unique(ds.matches[!, split_col_name]))
prediction_split_keys = all_splits[2:end] 
grouped_matches = groupby(ds.matches, split_col_name)

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


#### kelly functions 

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

function kelly_positive( kelly_dist::AbstractVector, c::Number)
  return mean(kelly_dist .> c)
end 

function kelly_decision_rule( kelly_dist::AbstractVector, c::Number, b::Number)::Bool
  return kelly_positive(kelly_dist, c) >= b 
end 

function kellys_stake_precent(kelly_dist::AbstractVector, kellys_fraction::Number)::Float64 
  return kellys_fraction * median(kelly_dist)
end 

function kelly_strategy(kelly_dist::AbstractVector, c::Number, b::Number, f::Number)::Number 
  return kelly_decision_rule(kelly_dist, c, b) * kellys_stake_precent(kelly_dist, f) 
end 



##### backtest 


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


### running a back test for a symbol / market 

TARGET_MARKET = :over_25 # <-- THE ONLY CHANGE IS HERE


MarketData = @NamedTuple{
    k_dist::Vector{Float64}, # Kelly distribution
    odds::Float64,           # Opening odds
    result::Bool             # Did the home team win?
}



println("--- Starting Data Pre-processing for: $(TARGET_MARKET) ---")
predict_config = BayesianFootball.Predictions.PredictionConfig(BayesianFootball.Markets.get_standard_markets())
processed_count = 0
error_count = 0
market_data = MarketData[] # <-- Generic name
for match_id in keys(all_oos_results)
    try
        r1 = all_oos_results[match_id]
        match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
        match_odds = BayesianFootball.Predictions.get_market_opening_lines(match_id, predict_config, ds.odds)
        # match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
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



### wealth analysis 
c_opt = 0.012
b_opt = 0.51
f_opt = 0.2

stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];
odds_opt = [m.odds for m in market_data];
results_opt = [m.result for m in market_data];

final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);
final_calmar = calculate_calmar_ratio(final_curve)

println("Final Calmar (re-calculated): $final_calmar")
println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
println("Final Bankroll (starting from 1.0): $(final_curve[end])")



using Plots
plot(final_curve, title="Optimal Equity Curve", label="Bankroll", legend=:topleft)



#= 
c_opt = 0.006
b_opt = 0.5
f_opt = 0.2 



20/21. 

- over_25
Final Calmar (re-calculated): 2.350439287869985
Total Bets Placed: 115 / 373
Final Bankroll (starting from 1.0): 1.5832078542587402

-under_25
Final Calmar (re-calculated): 1.4367729563870988
Total Bets Placed: 198 / 373
Final Bankroll (starting from 1.0): 1.653513710767277

21/22 

- over_25
Final Calmar (re-calculated): -5.025369684606352
Total Bets Placed: 138 / 481
Final Bankroll (starting from 1.0): 0.49746303153936494




=#


"""
something is weird, as the before the 2
"""

match_id = rand(collect(keys(all_oos_results)))
match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, all_oos_results[match_id]...);
model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

subset( ds.matches, :match_id => ByRow(isequal(match_id)))


kf = kelly_fraction(match_odds, match_predict)


using StatsPlots
density( kf[:over_25], label="over")
density( kf[:under_25], label="under")


r_pre = JLD2.load_object("training_results_large.jld2")


oos_results_pre = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    r_pre
)


match_predict_pre = BayesianFootball.Predictions.predict_market(model, predict_config, oos_results_pre[match_id]...);

model_odds_pre = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_pre));
model_odds_pre

kf_pre = kelly_fraction(match_odds, match_predict_pre)


density( kf[:over_25], label="over")
density!( kf_pre[:over_25], label="over pre")

kelly_positive(kf[:over_25], 0.06)
kelly_positive(kf_pre[:over_25], 0.06)


kelly_positive(kf[:under_25], 0.06)
kelly_positive(kf_pre[:under_25], 0.06)



kelly_decision_rule( kf[:over_25], 0.06, 0.3)
kelly_decision_rule( kf_pre[:over_25], 0.06, 0.3)




TARGET_MARKET = :over_25 # <-- THE ONLY CHANGE IS HERE


MarketData = @NamedTuple{
    k_dist::Vector{Float64}, # Kelly distribution
    odds::Float64,           # Opening odds
    result::Bool             # Did the home team win?
}



println("--- Starting Data Pre-processing for: $(TARGET_MARKET) ---")
predict_config = BayesianFootball.Predictions.PredictionConfig(BayesianFootball.Markets.get_standard_markets())
processed_count = 0
error_count = 0
market_data = MarketData[] # <-- Generic name
for match_id in keys(oos_results_pre)
    try
        r1 = oos_results_pre[match_id]
        match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
        # match_odds = BayesianFootball.Predictions.get_market_opening_lines(match_id, predict_config, ds.odds)
        match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
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



### wealth analysis 
c_opt = 0.0067
b_opt = 0.51
f_opt = 0.2

stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];
odds_opt = [m.odds for m in market_data];
results_opt = [m.result for m in market_data];

final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);
final_calmar = calculate_calmar_ratio(final_curve)

println("Final Calmar (re-calculated): $final_calmar")
println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
println("Final Bankroll (starting from 1.0): $(final_curve[end])")




