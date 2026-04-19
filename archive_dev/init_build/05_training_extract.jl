using Revise
using BayesianFootball
using DataFrames

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
using JLD2
# JLD2.save_object("training_results.jld2", results)

results = JLD2.load_object("training_results.jld2")


### create an extraction functions this is deconstructed

# inputs
r = results[1][1]
mp = filter( row -> row.split_col == 1, ds.matches)

BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, r)

# --- This logic is now OUTSIDE your function ---

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


##############################
# ---  predict 
##############################

using Distributions
using Statistics

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

function predict_market(
    model::BayesianFootball.TypesInterfaces.AbstractPoissonModel,
    market::BayesianFootball.Markets.Market1X2, 
    λ_h, λ_a
)

          """
          Helper function:
          Calculates 1X2 probabilities for a SINGLE pair of goal-rate parameters.
          """
          function _calculate_1x2_from_params(λ::Float64, μ::Float64, max_goals::Int=10)
              
              # Create distributions for this one MCMC sample
              home_dist = Poisson(λ)
              away_dist = Poisson(μ)
              
              p_home_win = 0.0
              p_draw = 0.0
              p_away_win = 0.0

              for h in 0:max_goals
                  for a in 0:max_goals
                      # P(H=h, A=a) = P(H=h) * P(A=a)
                      p_score = pdf(home_dist, h) * pdf(away_dist, a)
                      
                      if h > a
                          p_home_win += p_score
                      elseif h == a
                          p_draw += p_score
                      else # a > h
                          p_away_win += p_score
                      end
                  end
              end
              
              # Note: These will sum to < 1.0 due to max_goals truncation,
              # but we can re-normalize them for a cleaner 1X2 probability.
              total_p = p_home_win + p_draw + p_away_win
              
              return (
                  home = p_home_win / total_p,
                  draw = p_draw / total_p,
                  away = p_away_win / total_p
              )
          end

          function compute_1x2_distributions(λs::AbstractVector, μs::AbstractVector, max_goals::Int=10)
              
              n_samples = length(λs)
              
              # Pre-allocate thread-safe output vectors
              p_home_vec = zeros(n_samples)
              p_draw_vec = zeros(n_samples)
              p_away_vec = zeros(n_samples)
              
              # Use @threads to split the loop across your available cores
              for i in 1:n_samples
                  # Note: No need to index λs[i], the loop does it
                  λ_i = λs[i]
                  μ_i = μs[i]
                  
                  # Call the original (fast) inner-loop function
                  probs = _calculate_1x2_from_params(λ_i, μ_i, max_goals)
                  
                  # Write to the pre-allocated vectors
                  # This is safe because each thread writes to a different index `i`
                  p_home_vec[i] = probs.home
                  p_draw_vec[i] = probs.draw
                  p_away_vec[i] = probs.away
              end
              
              return (
                  p_home_dist = p_home_vec,
                  p_draw_dist = p_draw_vec,
                  p_away_dist = p_away_vec
              )
          end

  computed_1x2 = compute_1x2_distributions(λ_h, λ_a, 10)
  return NamedTuple{(:home, :draw, :away)}((computed_1x2.p_home_dist, computed_1x2.p_draw_dist, computed_1x2.p_away_dist))
end

function predict_market(
    model::BayesianFootball.TypesInterfaces.AbstractNegBinModel,
    market::BayesianFootball.Markets.Market1X2, 
    λ_h, λ_a
)
  println("Not implemented yet")

end


function predict_market(
    model::BayesianFootball.TypesInterfaces.AbstractPoissonModel,
    market::BayesianFootball.Markets.MarketOverUnder, 
    λ_h, λ_a
)

    total_rates_chain = λ_h .+ λ_a
    total_goal_dists = Poisson.(total_rates_chain)
    
    threshold = floor(Int, market.line) # e.g., 2 for line=2.5
    
    p_under_chain = cdf.(total_goal_dists, threshold)
    p_over_chain = 1.0 .- p_under_chain


    line_str = replace(string(market.line), "." => "")
            
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
    return NamedTuple{(over_key, under_key)}((p_over_chain, p_under_chain))
end




"""
Calculates the full posterior chain for BTTS probabilities.
"""
function predict_market(
    model::BayesianFootball.TypesInterfaces.AbstractPoissonModel,
    market::BayesianFootball.Markets.MarketBTTS, 
    λ_h, λ_a
)
    home_dists = Poisson.(λ_h)
    away_dists = Poisson.(λ_a)
    
    # P(Home > 0)
    p_home_scores_chain = 1.0 .- pdf.(home_dists, 0)
    # P(Away > 0)
    p_away_scores_chain = 1.0 .- pdf.(away_dists, 0)
    
    p_btts_yes_chain = p_home_scores_chain .* p_away_scores_chain
    p_btts_no_chain = 1.0 .- p_btts_yes_chain
    
  return NamedTuple{(:btts_yes, :btts_no)}((p_btts_yes_chain, p_btts_no_chain))
end

function predict_market(
    model::BayesianFootball.TypesInterfaces.AbstractPoissonModel,
    predict_config::BayesianFootball.Predictions.PredictionConfig,
    λ_h::AbstractVector{Float64},
    λ_a::AbstractVector{Float64}
    )

    market_results_generator = (
        predict_market(model, market, λ_h, λ_a) for market in predict_config.markets
    )
  match_predict = reduce(merge, market_results_generator; init = (;) );
  return match_predict

end 

match_id = rand(keys(all_oos_results))


filter(row -> row.match_id==match_id, ds.matches)
market_odds = filter(row -> row.match_id == match_id, ds.odds)



p_match = predict_market(model, predict_config, all_oos_results[match_id]...)


struct test_neg <: BayesianFootball.TypesInterfaces.AbstractNegBinModel end 
t_neg = test_neg()

predict_market(t_neg, a, all_oos_results[match_id]...)


# 1. Initialize your dictionary to store the odds
match_odds = Dict{Symbol, Float64}();

# 2. Iterate using `pairs()`
model_odds = Dict(key => mean(1 ./ value) for (key, value) in pairs(match_predict));
model_odds
match_odds 



#### get market odds

function get_market(
    match_id::Int64,
    market::BayesianFootball.Markets.Market1X2,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Full time")),
                       :market_group => ByRow(==("1X2"))
                      )
    odds_map = Dict(market_odds.choice_name .=> market_odds.decimal_odds)

    return (; home=odds_map["1"],
              draw=odds_map["X"],
              away=odds_map["2"])
end

function get_market(
    match_id::Int64,
    market::BayesianFootball.Markets.MarketOverUnder,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Match goals")),
                       :choice_group => ByRow(isequal(market.line))
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.decimal_odds)

    line_str = replace(string(market.line), "." => "")
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
  return NamedTuple{(over_key, under_key)}((odds_map["Over"], odds_map["Under"]))

end


function get_market(
    match_id::Int64,
    market::BayesianFootball.Markets.MarketBTTS,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Both teams to score")),
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.decimal_odds)

    yes_key = Symbol("btts_yes")
    no_key = Symbol("btts_no")
    
  return NamedTuple{(yes_key, no_key)}((odds_map["Yes"], odds_map["No"]))

end

function get_market(
  match_id::Int64,
  predict_config::BayesianFootball.Predictions.PredictionConfig,
  df_odds::AbstractDataFrame 
  )

  market_odds_generator = (
    get_market(match_id, market, df_odds) for market in predict_config.markets
  )

  match_odds = reduce(merge, market_odds_generator; init = (;) );

  return match_odds
end

match_odds = get_market(match_id, predict_config, ds.odds)

#####
# market results
#####

function get_market_results(
    match_id::Int64,
    market::BayesianFootball.Markets.Market1X2,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Full time")),
                       :market_group => ByRow(==("1X2"))
                      )
    odds_map = Dict(market_odds.choice_name .=> market_odds.winning)

    return (; home=odds_map["1"],
              draw=odds_map["X"],
              away=odds_map["2"])
end


function get_market_results(
    match_id::Int64,
    market::BayesianFootball.Markets.MarketOverUnder,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Match goals")),
                       :choice_group => ByRow(isequal(market.line))
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.winning)

    line_str = replace(string(market.line), "." => "")
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
  return NamedTuple{(over_key, under_key)}((odds_map["Over"], odds_map["Under"]))
end

function get_market_results(
    match_id::Int64,
    market::BayesianFootball.Markets.MarketBTTS,
    df_odds::AbstractDataFrame
) 
    market_odds = subset(df_odds,
                       :match_id => ByRow(==(match_id)),
                       :market_name => ByRow(==("Both teams to score")),
                      )


    odds_map = Dict(market_odds.choice_name .=> market_odds.winning)

    yes_key = Symbol("btts_yes")
    no_key = Symbol("btts_no")
    
  return NamedTuple{(yes_key, no_key)}((odds_map["Yes"], odds_map["No"]))

end


function get_market_results(
  match_id::Int64,
  predict_config::BayesianFootball.Predictions.PredictionConfig,
  df_odds::AbstractDataFrame 
  )

  market_odds_generator = (
    get_market_results(match_id, market, df_odds) for market in predict_config.markets
  )

  match_odds = reduce(merge, market_odds_generator; init = (;) );

  return match_odds
end




#####
# strategy 
#####

match_id = rand(keys(all_oos_results))

subset( ds.matches, 
       :match_id => ByRow(isequal(match_id))
       )


p_match = predict_market(model, predict_config, all_oos_results[match_id]...);
match_odds = get_market(match_id, predict_config, ds.odds)
match_results = get_market_results(match_id, predict_config, ds.odds)


b = keys(p_match)
b2 = keys(match_odds)


model_odds = NamedTuple(key => round(mean(1 ./ value), digits=2) for (key, value) in pairs(p_match))


"""
    parse_market_key(key::Symbol)

Helper function to split a market symbol like :over_25 or :btts_yes
into a (market_group, market_choice) tuple.
"""
function parse_market_key(key::Symbol)
    s_key = string(key)

    if s_key == "home"
        return ("1x2", "home")
    elseif s_key == "draw"
        return ("1x2", "draw")
    elseif s_key == "away"
        return ("1x2", "away")
    elseif startswith(s_key, "over_")
        choice = replace(s_key, "over_" => "")
        choice_str = string(parse(Int, choice) / 10) # "25" -> "2.5"
        return ("over", choice_str)
    elseif startswith(s_key, "under_")
        choice = replace(s_key, "under_" => "")
        choice_str = string(parse(Int, choice) / 10) # "25" -> "2.5"
        return ("under", choice_str)
    elseif startswith(s_key, "btts_")
        choice = replace(s_key, "btts_" => "")
        return ("btts", choice)
    else
        return ("unknown", s_key)
    end
end

function compute_ev_bets(
              match_id::Int,
              match_probs::NamedTuple,
              match_odds::NamedTuple,
              match_results::NamedTuple,
              model_name::String
              ) 

    bet_records = @NamedTuple{
        match_id::Int, 
        model_name::String,
        market_group::String, 
        market_choice::String, 
        stake::Float64,
        model_odd::Float64, 
        bookie_odd::Float64,
        ev::Float64,
        winning::Bool
    }[]

    # 3. Iterate over all markets
    for market_key in keys(match_odds)
        
        if !haskey(match_probs, market_key)
            continue
        end

        model_prob = round( mean( match_probs[market_key]), digits=2)
        bookie_odd = match_odds[market_key]

        # 4. Calculate EV (the "edge")
        ev = ( model_prob * bookie_odd) -1 

        # Parse the market key
        market_group, market_choice = parse_market_key(market_key)

        # 6. Add the bet to our records
        push!(bet_records, (
            match_id = match_id,
            model_name = model_name,
            market_group = market_group,
            market_choice = market_choice,
            stake = 1,
            model_odd = round( 1 / model_prob, digits=2) ,
            bookie_odd = bookie_odd,
            ev = round(ev, digits=4),
            winning=match_results[market_key]
        ))
    end

    return DataFrame(bet_records)
end 





compute_ev_bets( match_id, p_match, match_odds, match_results, "basic_poisson")





using ProgressMeter 

all_bets_list = DataFrame[]

@showprogress "Processing matches..." for (match_id, oos_data) in all_oos_results
        
        try
            # --- This is your logic per match ---
            
            match_probs = predict_market(model, predict_config, oos_data...)
            
            # B. Get bookmaker odds
            match_odds = get_market(match_id, predict_config, ds.odds)

            match_results = get_market_results(match_id, predict_config, ds.odds)
            # C. Find positive EV bets
            #    (Using the 'v2' version you preferred, which averages probabilities)

            bets_df = compute_ev_bets( match_id, match_probs, match_odds, match_results, "basic_poisson")
            # D. Add the results to our list (only if bets were found)
            if !isempty(bets_df)
                push!(all_bets_list, bets_df)
            end
            
            # --- End of per-match logic ---

        catch e
            # This is important! If one match fails (e.g., missing odds),
            # this will log the warning and continue the loop.
            @warn "Could not process match $match_id: $e"
        end
end 
final_bets_df = vcat(all_bets_list...)

final_bets_df


ev_bets = subset(
      final_bets_df,
:ev => ByRow(>(0.0))
)
# Create a copy so we don't modify your original ev_bets
analysis_df = copy(ev_bets)

# Add the profit column
analysis_df.profit = ifelse.(analysis_df.winning, analysis_df.bookie_odd .- 1.0, -1.0);



grouped_analysis = combine(
    groupby(analysis_df, [:market_group, :market_choice]),
    
    # --- Bet Counts ---
    nrow => :n_bets,
    :winning => sum => :n_winning,
    
    # --- ROI and Profit ---
    # ROI = Total Profit / Total Stake.
    # Since Total Stake = n_bets * 1.0, ROI = sum(profit) / n_bets,
    # which is simply the mean of the profit column.
    :profit => mean => :roi,
    :profit => sum => :total_profit,
    
    # --- Mean EV ---
    :ev => mean => :mean_ev,
    
    # --- Mean Odds ---
    :bookie_odd => mean => :mean_bookie_odd,
    
    # --- Conditional Means (Winning Bets) ---
    # This syntax takes the :ev and :winning columns
    # and finds the mean of :ev where :winning is true.
    [:ev, :winning] => ((ev, w) -> mean(ev[w])) => :mean_ev_winning,
    [:bookie_odd, :winning] => ((o, w) -> mean(o[w])) => :mean_odds_winning,

    # --- Conditional Means (Losing Bets) ---
    # We use .!w to select the non-winning rows
    [:ev, :winning] => ((ev, w) -> mean(ev[.!w])) => :mean_ev_losing,
    [:bookie_odd, :winning] => ((o, w) -> mean(o[.!w])) => :mean_odds_losing
)





"""
Computes EV bets for a *list* of quantiles.
"""
function compute_ev_bets_quantile(
              match_id::Int,
              match_probs::NamedTuple,
              match_odds::NamedTuple,
              match_results::NamedTuple,
              model_name::String,
              quantiles_to_test::AbstractVector{Float64} # NEW argument
              ) 

    # NEW: Add `quantile` to the NamedTuple definition
    bet_records = @NamedTuple{
        match_id::Int, 
        model_name::String,
        market_group::String, 
        market_choice::String, 
        quantile::Float64, # NEW column
        stake::Float64,
        model_odd::Float64, 
        bookie_odd::Float64,
        ev::Float64,
        winning::Bool
    }[]

    # 3. Iterate over all markets
    for market_key in keys(match_odds)
        
        if !haskey(match_probs, market_key)
            continue
        end

        bookie_odd = match_odds[market_key]
        posterior_samples = match_probs[market_key]
        market_group, market_choice = parse_market_key(market_key)

        # NEW: Inner loop over the quantiles
        for q in quantiles_to_test
            # 4. Calculate prob and EV based on the quantile
            model_prob = round(quantile(posterior_samples, q), digits=2)
            ev = (model_prob * bookie_odd) - 1 

            # 6. Add the bet to our records
            push!(bet_records, (
                match_id = match_id,
                model_name = model_name,
                market_group = market_group,
                market_choice = market_choice,
                quantile = q, # Add the quantile value
                stake = 1, # Stake is still 1.0 for this analysis
                model_odd = round(1 / model_prob, digits=2),
                bookie_odd = bookie_odd,
                ev = round(ev, digits=4),
                winning = match_results[market_key]
            ))
        end
    end

    return DataFrame(bet_records)
end


using ProgressMeter 

# --- NEW: Define the quantiles you want to test ---
# 0.05 = 95% confident the prob is *at least* this
# 0.10 = 90% confident
# 0.25 = 75% confident
# 0.50 = Median / 50% confident
quantiles_to_test = [0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

all_bets_list = DataFrame[]

@showprogress "Processing matches..." for (match_id, oos_data) in all_oos_results
        
        try
            # --- This is your logic per match ---
            
            match_probs = predict_market(model, predict_config, oos_data...)
            match_odds = get_market(match_id, predict_config, ds.odds)
            match_results = get_market_results(match_id, predict_config, ds.odds)
            
            # --- NEW: Call the new function ---
            bets_df = compute_ev_bets_quantile(
                match_id, match_probs, match_odds, match_results,
                "basic_poisson", quantiles_to_test
            )
            
            if !isempty(bets_df)
                push!(all_bets_list, bets_df)
           end
            
        catch e
            @warn "Could not process match $match_id: $e"
        end
end 

# This vcat works exactly the same
final_bets_df = vcat(all_bets_list...)


# This step is identical:
ev_bets = subset(
      final_bets_df,
      :ev => ByRow(>(0.0))
)

# This step is also identical:
analysis_df = copy(ev_bets)
analysis_df.profit = ifelse.(analysis_df.winning, analysis_df.bookie_odd .- 1.0, -1.0);


# --- NEW: Add :quantile to the groupby ---
grouped_analysis = combine(
    groupby(analysis_df, [:quantile, :market_group, :market_choice]), # <-- CHANGED
    
    # --- All of these aggregations are the same ---
    nrow => :n_bets,
    :winning => sum => :n_winning,
    :profit => mean => :roi,
    :profit => sum => :total_profit,
    :ev => mean => :mean_ev,
    :bookie_odd => mean => :mean_bookie_odd,
    [:ev, :winning] => ((ev, w) -> mean(ev[w])) => :mean_ev_winning,
    [:bookie_odd, :winning] => ((o, w) -> mean(o[w])) => :mean_odds_winning,
    [:ev, :winning] => ((ev, w) -> mean(ev[.!w])) => :mean_ev_losing,
    [:bookie_odd, :winning] => ((o, w) -> mean(o[.!w])) => :mean_odds_losing
)

# --- Now you can sort and see the difference! ---
# Sort by quantile (most conservative first), then by market
sort!(grouped_analysis, [:quantile, :market_group, :market_choice])

sort(grouped_analysis, :roi, rev=true)

sort(subset( grouped_analysis, :market_group => ByRow(isequal("btts"))), :roi, rev=true)




# This is our "Strategy" definition
# We will only bet if the quantile is 0.35
STRATEGY_QUANTILE = 0.4

# We will only bet on these markets
STRATEGY_MARKETS = [
    ("1x2", "home"),
    ("under", "1.5"),
    ("under", "2.5"),
    ("under", "3.5"),
    ("btts", "no")
]

# --- Step 1: Set your Kelly Fraction (0.5 = Half Kelly) ---
KELLY_FRACTION = 0.5

# --- Step 2: Start with all bets that had positive EV ---
# We use 'analysis_df' from your code
kelly_df = copy(analysis_df)

# --- Step 3: Apply our "Go-Live" Strategy Filter ---
# A. Filter by our chosen quantile
kelly_df = subset(kelly_df, :quantile => ByRow(==(STRATEGY_QUANTILE)))

# B. Filter by our chosen markets
# (This is a bit more complex, but a loop is fine here)
is_strategy_bet = map(eachrow(kelly_df)) do row
    (row.market_group, row.market_choice) in STRATEGY_MARKETS
end
kelly_df = kelly_df[is_strategy_bet, :]


# --- Step 4: Calculate the Kelly Stake ---
# 'p' is the model's probability *at our chosen quantile*
# 'b' is the profit on a 1-unit bet (bookie_odd - 1)
# 'q' is the probability of losing (1 - p)

# Get 'p' (which is just model_odd = 1/p)
kelly_df.p_model = 1.0 ./ kelly_df.model_odd

# Get 'b'
kelly_df.b_odds = kelly_df.bookie_odd .- 1.0

# Get 'q'
kelly_df.q_model = 1.0 .- kelly_df.p_model

# Calculate Full Kelly Fraction (f)
# f = (p*b - q) / b
kelly_df.f_full_kelly = ( (kelly_df.p_model .* kelly_df.b_odds) .- kelly_df.q_model ) ./ kelly_df.b_odds

# Calculate our final, fractional Kelly stake
# We also use max(0, f) to ensure stake is never negative (if p=0 or b=0)
kelly_df.stake = max.(0, kelly_df.f_full_kelly .* KELLY_FRACTION)

# Clean up (optional)
kelly_df = select(kelly_df,
    :match_id, :market_group, :market_choice, :quantile, :winning,
    :bookie_odd, :model_odd, :ev, :p_model, :b_odds, :stake
)

println("Selected $(nrow(kelly_df)) bets based on our strategy.")





# --- Step 1: Recalculate profit ---
# Profit is now a function of our variable stake
kelly_df.profit = ifelse.(
    kelly_df.winning,
    kelly_df.stake .* kelly_df.b_odds,  # Win amount
    -kelly_df.stake                     # Loss amount
);

# --- Step 2: Run the final analysis ---
# We don't group this time, we just get the grand total
# (or you could group by market_group again if you wish)

total_profit = sum(kelly_df.profit)
total_staked = sum(kelly_df.stake)
roi = total_profit / total_staked
n_bets = nrow(kelly_df)

println("--- Kelly Strategy Results (Fraction = $KELLY_FRACTION) ---")
println("Total Bets:     $n_bets")
println("Total Staked:   $(round(total_staked, digits=2)) units")
println("Total Profit:   $(round(total_profit, digits=2)) units")
println("ROI:            $(round(roi * 100, digits=2))%")





#= 

julia> println("Total Bets:     $n_bets")
Total Bets:     81

julia> println("Total Staked:   $(round(total_staked, digits=2)) units")
Total Staked:   4.05 units

julia> println("Total Profit:   $(round(total_profit, digits=2)) units")
Total Profit:   0.29 units

julia> println("ROI:            $(round(roi * 100, digits=2))%")
ROI:            7.06%

=#


"""
Runs the Kelly staking and profit calculation for a *list* of quantiles.

Arguments:
- `analysis_df`: Your DataFrame of all EV > 0 bets, with all quantiles.
- `quantiles_to_test`: A vector like [0.1, 0.25, 0.35, 0.5]
- `strategy_markets`: Your list of ("market", "choice") tuples.
- `kelly_fraction`: Your risk fraction (e.g., 0.5 for Half Kelly).

Returns:
A single DataFrame containing all *strategy-approved* bets,
with Kelly stake and profit calculated for each one.
"""
function analyze_kelly_strategy(
    analysis_df::DataFrame,
    quantiles_to_test::AbstractVector{Float64},
    strategy_markets::AbstractVector,
    kelly_fraction::Float64
)
    
    # This will hold the results from each quantile loop
    all_kelly_bets = DataFrame[]

    for q in quantiles_to_test
        # --- 1. Filter by this loop's quantile ---
        # We start with all positive EV bets
        quantile_df = subset(analysis_df, :quantile => ByRow(==(q)))

        # --- 2. Filter by our chosen markets ---
        is_strategy_bet = map(eachrow(quantile_df)) do row
            (row.market_group, row.market_choice) in strategy_markets
        end
        strategy_df = quantile_df[is_strategy_bet, :]

        # If this quantile + market combo had no bets, skip
        if isempty(strategy_df)
            continue
        end

        # We must use copy() here to add new columns
        kelly_bets_for_q = copy(strategy_df)

        # --- 3. Calculate Kelly Stake (same as your code) ---
        kelly_bets_for_q.p_model = 1.0 ./ kelly_bets_for_q.model_odd
        kelly_bets_for_q.b_odds = kelly_bets_for_q.bookie_odd .- 1.0
        kelly_bets_for_q.q_model = 1.0 .- kelly_bets_for_q.p_model
        
        # Calculate Full Kelly Fraction (f)
        f_full_kelly = (
            (kelly_bets_for_q.p_model .* kelly_bets_for_q.b_odds) .- kelly_bets_for_q.q_model
        ) ./ kelly_bets_for_q.b_odds
        
        # Calculate our final, fractional Kelly stake
        kelly_bets_for_q.stake = max.(0, f_full_kelly .* kelly_fraction)

        # --- 4. Calculate final profit ---
        kelly_bets_for_q.profit = ifelse.(
            kelly_bets_for_q.winning,
            kelly_bets_for_q.stake .* kelly_bets_for_q.b_odds, # Win amount
            -kelly_bets_for_q.stake                            # Loss amount
        )

        # --- 5. Add this quantile's results to our list ---
        push!(all_kelly_bets, kelly_bets_for_q)
    end

    if isempty(all_kelly_bets)
        @warn "No bets matched the strategy for any quantile."
        return DataFrame() # Return an empty DataFrame
    end

    # Combine all results into one big DataFrame
    return vcat(all_kelly_bets...)
end



quantiles_to_test = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Define your strategy (can be the same as before)
STRATEGY_MARKETS = [
    ("1x2", "home"),
    ("1x2", "away"),
    ("1x2", "draw"),
    ("under", "1.5"),
    ("under", "2.5"),
    ("under", "3.5"),
    ("under", "4.5"),
    ("over", "1.5"),
    ("over", "2.5"),
    ("over", "3.5"),
    ("over", "4.5"),
    ("btts", "no"),
    ("btts", "yes")
]

KELLY_FRACTION = 0.5 # Half Kelly

# --- 2. Run the Full Analysis ---
full_kelly_results_df = analyze_kelly_strategy(
    analysis_df,
    quantiles_to_test,
    STRATEGY_MARKETS,
    KELLY_FRACTION
)

# Group by quantile AND market
kelly_grouped_analysis = combine(
    groupby(full_kelly_results_df, [:quantile, :market_group, :market_choice]),
    nrow => :n_bets,
    :stake => sum => :total_staked,
    :profit => sum => :total_profit
)

# Calculate ROI (Profit / Total Staked)
kelly_grouped_analysis.roi = kelly_grouped_analysis.total_profit ./ kelly_grouped_analysis.total_staked

# Sort to see the best-performing strata
sort!(kelly_grouped_analysis, :roi, rev=true)

println("--- Kelly Strategy Breakdown by Market & Quantile ---")
println(kelly_grouped_analysis)


full_kelly_results_df

# Your full_kelly_results_df has the Kelly 'stake' and 'profit'
# We will now add the 'fixed_stake' and 'fixed_profit'
# (We need a copy to add columns)
comparison_df = copy(full_kelly_results_df)

# Add the 'fixed_profit' column
# b_odds = bookie_odd - 1.0 (already calculated)
comparison_df.fixed_profit = ifelse.(
    comparison_df.winning,
    comparison_df.b_odds, # Win amount (1.0 * b_odds)
    -1.0                  # Loss amount (-1.0)
);

strategy_comparison = combine(
    groupby(comparison_df, :quantile),
    
    # --- Counts ---
    nrow => :n_bets,
    nrow => :fixed_total_staked, # <-- THE FIX: Just get the row count again
    
    # --- Kelly Strategy Aggregates ---
    :stake => sum => :kelly_total_staked,
    :profit => sum => :kelly_total_profit,
    
    # --- Fixed Strategy Aggregates ---
    :fixed_profit => sum => :fixed_total_profit
)

# --- The rest of your code is perfect ---

# Calculate ROI for both
strategy_comparison.kelly_roi = strategy_comparison.kelly_total_profit ./ strategy_comparison.kelly_total_staked
strategy_comparison.fixed_roi = strategy_comparison.fixed_total_profit ./ strategy_comparison.fixed_total_staked

# Sort by the quantile for a clean view
sort!(strategy_comparison, :quantile)

println(strategy_comparison)

function simulate_bankroll(df, initial_bankroll, kelly_fraction)
    
    current_bankroll = initial_bankroll
    bankroll_history = [initial_bankroll]

    # Make sure bets are sorted by time!
    sorted_df = sort(df, :match_id) 
    
    for row in eachrow(sorted_df)
        # 1. Calculate the 'true' Kelly stake based on *current* bankroll
        #    f = (p*b - q) / b
        p = row.p_model
        b = row.b_odds
        q = 1 - p
        
        f_full_kelly = (p*b - q) / b
        f_fractional = max(0, f_full_kelly * kelly_fraction)
        
        stake = current_bankroll * f_fractional
        
        # 2. Calculate the profit or loss from this one bet
        if row.winning
            profit = stake * b
        else
            profit = -stake
        end
        
        # 3. Update the bankroll
        current_bankroll = current_bankroll + profit
        push!(bankroll_history, current_bankroll)
    end
    
    return bankroll_history
end

# --- How to use it ---
# 1. Filter your df for ONE strategy
strategy_df = subset(comparison_df, :quantile => ByRow(==(0.45)))

# 2. Run the simulation
initial_bankroll = 100.0
kelly_history = simulate_bankroll(strategy_df, initial_bankroll, 0.5) # 0.5 = Half Kelly

# 3. You can now plot 'kelly_history' to see your bankroll grow/shrink
using Plots
plot(kelly_history)



# 1. Select only the columns we need from ds.matches
match_dates_df = select(ds.matches, :match_id, :match_date)

# 2. Join this with your full results DataFrame
# This adds the 'match_date' column to every bet
all_bets_with_dates = leftjoin(
    full_kelly_results_df,
    match_dates_df,
    on = :match_id
)

# 3. Sort by date. This is the crucial step you identified.
# This is now our master, time-ordered DataFrame of all strategy-approved bets.
sort!(all_bets_with_dates, :match_date)



"""
Simulates bankroll growth over time, tracking a separate, independent
bankroll for each market in the DataFrame.

Assumes the input DataFrame is already sorted by date.
"""
function simulate_grouped_bankrolls(
    strategy_df::DataFrame,
    initial_bankroll::Float64,
    kelly_fraction::Float64
)
    # This Dict will store the *current* bankroll for each market
    # e.g., "1x2/home" => 105.3
    bankrolls = Dict{String, Float64}()

    # This Dict will store the *full history* for plotting
    # e.g., "1x2/home" => [100.0, 101.2, 99.8, ...]
    histories = Dict{String, Vector{Float64}}()

    # Loop through every bet in our time-sorted strategy
    for row in eachrow(strategy_df)
        
        # 1. Identify the market for this bet
        market_key = "$(row.market_group)/$(row.market_choice)"

        # 2. Initialize this market if it's the first time we've seen it
        if !haskey(bankrolls, market_key)
            bankrolls[market_key] = initial_bankroll
            histories[market_key] = [initial_bankroll] # Start with the initial value
        end

        # 3. Get the *current* bankroll for this specific market
        current_bankroll = bankrolls[market_key]

        # 4. Calculate the stake based on *this market's* current bankroll
        # (This is the same Kelly calculation as before)
        p = row.p_model
        b = row.b_odds
        q = 1.0 - p
        
        f_full_kelly = (p * b - q) / b
        f_fractional = max(0, f_full_kelly * kelly_fraction)
        
        stake = current_bankroll * f_fractional

        # 5. Calculate profit/loss
        if row.winning
            profit = stake * b
        else
            profit = -stake
        end

        # 6. Update this market's bankroll and history
        new_bankroll = current_bankroll + profit
        bankrolls[market_key] = new_bankroll
        push!(histories[market_key], new_bankroll)
    end

    return histories
end




# --- 1. Define Your Inputs ---
quantiles_to_test = [0.1, 0.25, 0.35, 0.4, 0.45, 0.5]
KELLY_FRACTION = 0.5 # Half Kelly
INITIAL_BANKROLL = 100.0

# This will store all simulation results
# Key = Quantile, Value = Dict of market histories
all_simulations = Dict{Float64, Dict{String, Vector{Float64}}}()

println("Running bankroll simulations for each quantile...")

# --- 2. The Main Loop ---
for q in quantiles_to_test
    
    # Get all strategy bets for this *one* quantile
    # (This uses the master, time-sorted DataFrame)
    strategy_df = subset(all_bets_with_dates, :quantile => ByRow(==(q)))

    if isempty(strategy_df)
        @warn "No bets found for quantile = $q. Skipping."
        continue
    end

    # Run the new grouped simulation
    simulations_for_q = simulate_grouped_bankrolls(
        strategy_df,
        INITIAL_BANKROLL,
        KELLY_FRACTION
    )
    
    # Store the results
    all_simulations[q] = simulations_for_q
end

println("Simulations complete!")



# 1. Get the simulation results for the 0.4 quantile
results_for_q_0_4 = all_simulations[0.5]

# 2. Get the history vectors for each market
pnl_1x2_home = results_for_q_0_4["1x2/home"]
pnl_under_2_5 = results_for_q_0_4["under/2.5"]
pnl_btts_no = results_for_q_0_4["btts/no"]
# ... etc.

# 3. You can now pass these vectors to any plotting library
plot(
    [pnl_1x2_home, pnl_under_2_5, pnl_btts_no],
    label = ["1x2/home" "under/2.5" "btts/no"]
)

##

# 1. GET YOUR MASTER DATASET (from previous step)
match_dates_df = select(ds.matches, :match_id, :match_date)
all_bets_with_dates = leftjoin(
    full_kelly_results_df, # Your DataFrame of all strategy-approved bets
    match_dates_df,
    on = :match_id
)
sort!(all_bets_with_dates, :match_date)

# 2. DEFINE YOUR SIMULATION PARAMETERS
quantiles_to_test = [0.1, 0.25, 0.35, 0.4, 0.45, 0.5] # Your choice
KELLY_FRACTION = 0.5  # Half Kelly
INITIAL_BANKROLL = 100.0

# 3. RUN THE SIMULATION
all_simulations = Dict{Float64, Dict{String, Vector{Float64}}}()

@showprogress "Running bankroll simulations..." for q in quantiles_to_test
    
    strategy_df = subset(all_bets_with_dates, :quantile => ByRow(==(q)))
    if isempty(strategy_df); continue; end

    simulations_for_q = simulate_grouped_bankrolls(
        strategy_df,
        INITIAL_BANKROLL,
        KELLY_FRACTION
    )
    all_simulations[q] = simulations_for_q
end

println("Simulations complete!")

using Plots

# Let's analyze the q=0.4 strategy
results_for_q_0_4 = all_simulations[0.4]

# Create a new plot
p = plot(title = "P&L for Quantile = 0.4 (Half Kelly)", xlabel = "Number of Bets", ylabel = "Bankroll (units)")

# Add a line for each market
for (market_key, bankroll_history) in results_for_q_0_4
    plot!(p, bankroll_history, label=market_key, linewidth=2)
end

# Display the plot
display(p)

# You can now save this plot
# savefig(p, "pnl_plot_q0_4.png")



##############################
# --- odds and predict 
##############################

using Base.Threads
using Distributions
using Statistics

"""
Helper function:
Calculates 1X2 probabilities for a SINGLE pair of goal-rate parameters.
"""
function _calculate_1x2_from_params(λ::Float64, μ::Float64, max_goals::Int=10)
    
    # Create distributions for this one MCMC sample
    home_dist = Poisson(λ)
    away_dist = Poisson(μ)
    
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for h in 0:max_goals
        for a in 0:max_goals
            # P(H=h, A=a) = P(H=h) * P(A=a)
            p_score = pdf(home_dist, h) * pdf(away_dist, a)
            
            if h > a
                p_home_win += p_score
            elseif h == a
                p_draw += p_score
            else # a > h
                p_away_win += p_score
            end
        end
    end
    
    # Note: These will sum to < 1.0 due to max_goals truncation,
    # but we can re-normalize them for a cleaner 1X2 probability.
    total_p = p_home_win + p_draw + p_away_win
    
    return (
        home = p_home_win / total_p,
        draw = p_draw / total_p,
        away = p_away_win / total_p
    )
end

function compute_1x2_distributions(λs::AbstractVector, μs::AbstractVector, max_goals::Int=10)
    
    n_samples = length(λs)
    
    # Pre-allocate thread-safe output vectors
    p_home_vec = zeros(n_samples)
    p_draw_vec = zeros(n_samples)
    p_away_vec = zeros(n_samples)
    
    # Use @threads to split the loop across your available cores
    @threads for i in 1:n_samples
        # Note: No need to index λs[i], the loop does it
        λ_i = λs[i]
        μ_i = μs[i]
        
        # Call the original (fast) inner-loop function
        probs = _calculate_1x2_from_params(λ_i, μ_i, max_goals)
        
        # Write to the pre-allocated vectors
        # This is safe because each thread writes to a different index `i`
        p_home_vec[i] = probs.home
        p_draw_vec[i] = probs.draw
        p_away_vec[i] = probs.away
    end
    
    return (
        p_home_dist = p_home_vec,
        p_draw_dist = p_draw_vec,
        p_away_dist = p_away_vec
    )
end





# 1. Get the parameter chains for your match (you already have this)
#    This is based on your extract_parameters function [cite: 14]
match_id = rand(keys(all_oos_results))
λ_h_chain, λ_a_chain = all_oos_results[match_id]

filter(row -> row.match_id==match_id, ds.matches)

# 2. Compute the 1X2 probability distributions
prob_dists = compute_1x2_distributions_threaded(λ_h_chain, λ_a_chain)

# 3. Convert probability distributions to odds distributions
odds_home_dist = 1.0 ./ prob_dists.p_home_dist;
odds_draw_dist = 1.0 ./ prob_dists.p_draw_dist;
odds_away_dist = 1.0 ./ prob_dists.p_away_dist;

# --- 4. Now analyze the distributions! ---

using StatsPlots

# Get the market odds for this match (example)
market_odds = filter(row -> row.match_id == match_id, ds.odds)
market_home_odds = market_odds.decimal_odds[1]
market_draw_odds = market_odds.decimal_odds[2]
market_away_odds = market_odds.decimal_odds[3]

# A. Plot your model's distribution against the market
density(odds_home_dist, label="Model Home Odds Distribution", xlims=(0, 10))
vline!([market_home_odds], label="Market Odds (Bet365)", color=:red, lw=2)

density(odds_away_dist, label="Model Home Odds Distribution", xlims=(0, 10))
vline!([market_away_odds], label="Market Odds (Bet365)", color=:red, lw=2)

density(odds_draw_dist, label="Model Draw Odds Distribution", xlims=(0, 10))
vline!([market_draw_odds], label="Market Odds (Bet365)", color=:red, lw=2)


# B. Get your 50% (or 90%) Bayesian Credible Interval
# This is what you asked for!
mean_odds = mean(odds_home_dist)
interval_50 = quantile(odds_home_dist, [0.25, 0.75])
interval_90 = quantile(odds_home_dist, [0.05, 0.95])


mean_odds = mean(odds_away_dist)
interval_50 = quantile(odds_away_dist, [0.25, 0.75])
interval_90 = quantile(odds_away_dist, [0.05, 0.95])


mean_odds = mean(odds_draw_dist)
interval_50 = quantile(odds_draw_dist, [0.25, 0.75])
interval_90 = quantile(odds_draw_dist, [0.05, 0.95])



println("--- Home Win Analysis for Match $match_id ---")
println("Market Odds: $market_home_odds")
println("Model Mean Odds: $mean_odds")
println("Model 50% Interval: $interval_50")
println("Model 90% Interval: $interval_90")

# C. Calculate a "Value" probability
# What's the probability that our model thinks the "true" odds
# are lower (i.e., higher probability) than the market's odds?
p_value = mean(odds_home_dist .< market_home_odds)

println("Probability (Model < Market): $p_value")


####
# 1. Get the probability distributions (which you already have)

# 2. Get the market-implied probabilities (example)
market_prob_home = 1.0 / 1.45
market_prob_draw = 1.0 / 4.20
market_prob_away = 1.0 / 5.75

# 3. Calculate P_value (the probability your model finds an edge)
#    This is the core of the Bayesian strategy
p_value_home = mean(prob_dists.p_home_dist .> market_prob_home)
p_value_draw = mean(prob_dists.p_draw_dist .> market_prob_draw)
p_value_away = mean(prob_dists.p_away_dist .> market_prob_away)

println("P(Value Home): $p_value_home")
println("P(Value Draw): $p_value_draw")
println("P(Value Away): $p_value_away")



# We'll do this just for the Draw bet, which we identified as value
market_odds_draw = 4.20

# 1. Calculate the distribution of the edge
edge_dist_draw = (prob_dists.p_draw_dist .* market_odds_draw) .- 1.0 ; 

# 2. Calculate the distribution of the Kelly fraction
#    (We only care about positive edge)
kelly_dist_draw = [e > 0 ? e / market_odds_draw : 0.0 for e in edge_dist_draw];

# 3. A conservative staking strategy
#    (Don't bet the mean, bet a fraction of it, e.g., 25% "Fractional Kelly")
mean_kelly_stake = mean(kelly_dist_draw)
your_stake = 0.25 * mean_kelly_stake

println("Mean Kelly Stake for Draw: $(mean_kelly_stake * 100)% of bankroll")
println("Our Conservative Stake: $(your_stake * 100)% of bankroll")



market_odds_home =market_home_odds
edge_dist_home = (prob_dists.p_home_dist .* market_odds_home) .- 1.0 ; 

# 2. Calculate the distribution of the Kelly fraction
#    (We only care about positive edge)
kelly_dist_home = [e > 0 ? e / market_odds_home : 0.0 for e in edge_dist_home];

# 3. A conservative staking strategy
#    (Don't bet the mean, bet a fraction of it, e.g., 25% "Fractional Kelly")
mean_kelly_stake_home = mean(kelly_dist_home)
your_stake_home = 0.25 * mean_kelly_stake_home

println("Mean Kelly Stake for home: $(mean_kelly_stake_home * 100)% of bankroll")
println("Our Conservative Stake: $(your_stake_home * 100)% of bankroll")




market_odds_away =market_away_odds
edge_dist_away = (prob_dists.p_away_dist .* market_odds_away) .- 1.0 ;

# 2. Calculate the distribution of the Kelly fraction
#    (We only care about positive edge)
kelly_dist_away = [e > 0 ? e / market_odds_away : 0.0 for e in edge_dist_away];

# 3. A conservative staking strategy
#    (Don't bet the mean, bet a fraction of it, e.g., 25% "Fractional Kelly")
mean_kelly_stake_away = mean(kelly_dist_away)
your_stake_away = 0.25 * mean_kelly_stake_away

println("Mean Kelly Stake for away: $(mean_kelly_stake_away * 100)% of bankroll")
println("Our Conservative Stake: $(your_stake_away * 100)% of bankroll")


# Your confidence threshold
CONFIDENCE_THRESHOLD = 0.60

p_value_home = mean(prob_dists.p_home_dist .> (1/market_odds_home))
p_value_draw = mean(prob_dists.p_draw_dist .> (1/market_odds_draw))
p_value_away = mean(prob_dists.p_away_dist .> (1/market_odds_away))


if p_value_home > CONFIDENCE_THRESHOLD
    # This code will NOT run for this match
    println("Betting on Home!")
    mean_stake = mean(kelly_dist_home) 
elseif p_value_draw > CONFIDENCE_THRESHOLD
    # This code will NOT run for this match
    println("Betting on Draw!")
elseif p_value_away > CONFIDENCE_THRESHOLD
     # This code will NOT run for this match
    println("Betting on Away!")
else
    println("No value found on this match. NO BET.")
end


using StatsPlots

density(λs, label="home")
density!(μs, label="away")

using Distributions
using Statistics

"""
Computes the mean posterior predictive probability for each score
in a matrix up to max_goals.
"""

function compute_score_matrix(λs::AbstractVector, μs::AbstractVector, max_goals::Int=6)
    
    n_samples = length(λs)
    # Create distributions for each MCMC sample
    home_dists = Poisson.(λs)
    away_dists = Poisson.(μs)
    
    # Initialize the matrix to store the mean probabilities
    prob_matrix = zeros(max_goals + 1, max_goals + 1)
    
    for h in 0:max_goals
        for a in 0:max_goals
            # 1. Calculate the probability of (h-a) for EACH sample
            # P(H=h, A=a | λ_i, μ_i) = P(H=h | λ_i) * P(A=a | μ_i)
            p_chain = pdf.(home_dists, h) .* pdf.(away_dists, a)
            
            # 2. The final probability is the mean of this chain
            prob_matrix[h + 1, a + 1] = mean(p_chain)
        end
    end
    
    # Add row/col names for clarity (optional, requires DataFrames)
    # return DataFrame(prob_matrix, Symbol.(0:max_goals), row_names=Symbol.(0:max_goals))
    return prob_matrix
end

id1 = rand(keys(all_oos_results))
filter( row -> row.match_id==id1, ds.matches)
filter( row -> row.match_id==id1, ds.odds)

# --- Get the Score Matrix ---
# This matrix now contains the final predicted probabilities for each score

score_matrix = compute_score_matrix(all_oos_results[id1]... , 6)

# Example: Probability of 2-1
# (Remember 1-based indexing: [h+1, a+1])
p_2_2 = score_matrix[2 + 1, 2 + 1]
println("Probability of 2-1: $p_2_2, $(1 / p_2_1)")

# Example: Probability of 0-0
p_0_0 = score_matrix[0 + 1, 0 + 1]
println("Probability of 0-0: $p_0_0")


# ------- 1x2 odds --------- 

# Get dimensions (handles max_goals)
n_rows, n_cols = size(score_matrix)

# 1. Home Win (H > A)
p_home_win = 0.0
for h in 1:n_rows-1 # h from 1 to max_goals
    for a in 0:h-1   # a from 0 to h-1
        if a+1 <= n_cols
            p_home_win += score_matrix[h + 1, a + 1]
        end
    end
end 

# 2. Draw (H == A)
p_draw = 0.0
for k in 0:min(n_rows-1, n_cols-1) # k from 0 to max_goals
    p_draw += score_matrix[k + 1, k + 1]
end

# 3. Away Win (A > H)
p_away_win = 0.0
for a in 1:n_cols-1 # a from 1 to max_goals
    for h in 0:a-1   # h from 0 to a-1
        if h+1 <= n_rows
            p_away_win += score_matrix[h + 1, a + 1]
        end
    end
end

# Note: p_home_win + p_draw + p_away_win might be < 1.0 if max_goals is too low.
# The remainder is the probability of a score outside your matrix (e.g., 7-0).

println("P(Home Win): $p_home_win")
println("P(Draw):     $p_draw")
println("P(Away Win): $p_away_win")

# --- Calculate Decimal Odds ---
odds_home = 1.0 / p_home_win
odds_draw = 1.0 / p_draw
odds_away = 1.0 / p_away_win

println("Odds (1X2): $odds_home / $odds_draw / $odds_away")


p_home_win + p_draw + p_away_win



# ------ under / over odds --------

# 1. Create the chain of total goal rates
total_rates_chain = λs .+ μs ;

# 2. Create a Poisson distribution for each sample
total_goal_dists = Poisson.(total_rates_chain);

# --- For O/U 2.5 ---
# P(Under 2.5) = P(Total Goals <= 2)
# We use the CDF (Cumulative Distribution Function)
p_under_2_5_chain = cdf.(total_goal_dists, 2);

# P(Over 2.5) = 1 - P(Total Goals <= 2)
p_over_2_5_chain = 1.0 .- p_under_2_5_chain;

# This is your Bayesian result: the full distribution of probabilities
# You can inspect its uncertainty:
# println(quantile(p_over_2_5_chain, [0.025, 0.5, 0.975]))

# 3. Get the final mean probability
p_under_2_5_mean = mean(p_under_2_5_chain)
p_over_2_5_mean = mean(p_over_2_5_chain) # == 1.0 - p_under_2_5_mean

# 4. Calculate Odds
odds_under_2_5 = 1.0 / p_under_2_5_mean
odds_over_2_5 = 1.0 / p_over_2_5_mean

println("P(Over 2.5): $p_over_2_5_mean")
println("Odds (O/U 2.5): $odds_over_2_5 / $odds_under_2_5")

# --- To get O/U 1.5 (just change the CDF threshold) ---
p_under_1_5_chain = cdf.(total_goal_dists, 1) # P(Total Goals <= 1)
p_under_1_5_mean = mean(p_under_1_5_chain)
odds_under_1_5 = 1.0 / p_under_1_5_mean
odds_over_1_5 = 1.0 / (1.0 - p_under_1_5_mean)
println("Odds (O/U 1.5): $odds_over_1_5 / $odds_under_1_5")


# ----- btts ----- 

# 1. Create distributions for each sample (if not already done)
home_dists = Poisson.(λs)
away_dists = Poisson.(μs)

# 2. Get chain of P(Home > 0)
# pdf(Poisson(λ), 0) gives P(Home=0)
p_home_scores_chain = 1.0 .- pdf.(home_dists, 0)

# 3. Get chain of P(Away > 0)
p_away_scores_chain = 1.0 .- pdf.(away_dists, 0)

# 4. Get the chain for P(BTTS)
p_btts_yes_chain = p_home_scores_chain .* p_away_scores_chain

# This is your full posterior distribution for the BTTS probability.
# You can plot its histogram!

# 5. Get the final mean probability
p_btts_yes_mean = mean(p_btts_yes_chain)
p_btts_no_mean = 1.0 - p_btts_yes_mean

# 6. Calculate Odds
odds_btts_yes = 1.0 / p_btts_yes_mean
odds_btts_no = 1.0 / p_btts_no_mean

println("P(BTTS=Yes): $p_btts_yes_mean")
println("Odds (BTTS): $odds_btts_yes (Yes) / $odds_btts_no (No)")




#= 
--- Version 2 testing 
=# 

using Distributions
using Statistics
using DataFrames
using Turing

"""
Extracts the posterior chains for λ (home rate) and μ (away rate)
for a specific match.
"""
function get_rate_chains(chains::Chains, h_id::Int, a_id::Int)
    # Extract parameter chains
    a_h = vec(chains[Symbol("log_α[$h_id]")])
    b_h = vec(chains[Symbol("log_β[$h_id]")])
    a_a = vec(chains[Symbol("log_α[$a_id]")])
    b_a = vec(chains[Symbol("log_β[$a_id]")])
    h = vec(chains[Symbol("home_adv")])

    # Calculate goal rate chains
    # λ = Home team's (h_id) attack + Away team's (a_id) defense + home adv
    λs = exp.(a_h .+ b_a .+ h)
    # μ = Away team's (a_id) attack + Home team's (h_id) defense
    μs = exp.(a_a .+ b_h)
    
    return λs, μs
end

"""
Calculates the full posterior chain for 1, X, and 2 probabilities.
"""
function calculate_1x2_chains(λs::Vector{Float64}, μs::Vector{Float64}; max_goals=8)
    n_samples = length(λs)
    home_dists = Poisson.(λs)
    away_dists = Poisson.(μs)
    
    # Pre-calculate all pdfs needed
    # pdfs_home[k+1] will be the chain P(H=k)
    pdfs_home = [pdf.(home_dists, k) for k in 0:max_goals]
    pdfs_away = [pdf.(away_dists, k) for k in 0:max_goals]

    # Init chains for the probabilities
    p_home_win_chain = zeros(n_samples)
    p_draw_chain = zeros(n_samples)
    p_away_win_chain = zeros(n_samples)

    for h in 0:max_goals
        for a in 0:max_goals
            # p_score_chain is a vector of P(H=h, A=a) for each MCMC sample
            p_score_chain = pdfs_home[h+1] .* pdfs_away[a+1]
            
            if h > a
                p_home_win_chain .+= p_score_chain
            elseif h == a
                p_draw_chain .+= p_score_chain
            else # a > h
                p_away_win_chain .+= p_score_chain
            end
        end
    end
    
    return (p_home = p_home_win_chain, p_draw = p_draw_chain, p_away = p_away_win_chain)
end

"""
Calculates the full posterior chain for Over/Under probabilities for a given line.
"""
function calculate_ou_chains(λs::Vector{Float64}, μs::Vector{Float64}, line::Float64)
    total_rates_chain = λs .+ μs
    total_goal_dists = Poisson.(total_rates_chain)
    
    threshold = floor(Int, line) # e.g., 2 for line=2.5
    
    p_under_chain = cdf.(total_goal_dists, threshold)
    p_over_chain = 1.0 .- p_under_chain
            
    return (p_over = p_over_chain, p_under = p_under_chain)
end

"""
Calculates the full posterior chain for BTTS probabilities.
"""
function calculate_btts_chains(λs::Vector{Float64}, μs::Vector{Float64})
    home_dists = Poisson.(λs)
    away_dists = Poisson.(μs)
    
    # P(Home > 0)
    p_home_scores_chain = 1.0 .- pdf.(home_dists, 0)
    # P(Away > 0)
    p_away_scores_chain = 1.0 .- pdf.(away_dists, 0)
    
    p_btts_yes_chain = p_home_scores_chain .* p_away_scores_chain
    p_btts_no_chain = 1.0 .- p_btts_yes_chain
    
    return (p_yes = p_btts_yes_chain, p_no = p_btts_no_chain)
end

"""
Summarizes a probability chain into its key statistics.
"""
function summarize_prob_chain(chain::Vector{Float64}, prefix::String)
    return (
        Symbol(prefix, "_prob_mean") => mean(chain),
        Symbol(prefix, "_prob_median") => median(chain),
        Symbol(prefix, "_prob_q025") => quantile(chain, 0.025),
        Symbol(prefix, "_prob_q975") => quantile(chain, 0.975),
        Symbol(prefix, "_odds_mean") => 1.0 / mean(chain) # Model's decimal odds
    )
end


## ---- prepare market odds 

function pivot_market_odds(odds_df::DataFrame)
    
    # --- 1X2 Full Time ---
    odds_1x2 = filter(row -> row.market_group == "1X2" && row.market_name == "Full time", odds_df)
    pivot_1x2 = unstack(odds_1x2, [:match_id], :choice_name, :decimal_odds)
    # Handle missing columns if a market wasn't offered (e.g., "X" is missing)
    for c in ["1", "X", "2"]
        if !hasproperty(pivot_1x2, c)
            pivot_1x2[!, c] = missing
        end
    end
    rename!(pivot_1x2, "1" => :market_odds_1, "X" => :market_odds_X, "2" => :market_odds_2)
    
    # --- O/U 2.5 ---
    odds_ou25 = filter(row -> row.market_group == "Match goals" && row.choice_group == 2.5, odds_df)
    pivot_ou25 = unstack(odds_ou25, [:match_id], :choice_name, :decimal_odds)
    for c in ["Over", "Under"]
        if !hasproperty(pivot_ou25, c)
            pivot_ou25[!, c] = missing
        end
    end
    rename!(pivot_ou25, "Over" => :market_odds_O25, "Under" => :market_odds_U25)

    # --- BTTS ---
    odds_btts = filter(row -> row.market_group == "Both teams to score", odds_df)
    pivot_btts = unstack(odds_btts, [:match_id], :choice_name, :decimal_odds)
     for c in ["Yes", "No"]
        if !hasproperty(pivot_btts, c)
            pivot_btts[!, c] = missing
        end
    end
    rename!(pivot_btts, "Yes" => :market_odds_BTTS_Y, "No" => :market_odds_BTTS_N)

    # --- Join all pivoted markets ---
    market_odds_df = leftjoin(pivot_1x2, pivot_ou25, on = :match_id)
    market_odds_df = leftjoin(market_odds_df, pivot_btts, on = :match_id)
    
    return market_odds_df
end

# --- Create this DataFrame once ---
market_odds_df = pivot_market_odds(ds.odds)




### ------ 3 - main prediction loop 

team_map = vocabulary.mappings[:team_map]
all_predictions = [] # This will hold a DataFrame from each split

# results[1] was trained to predict split_col=1 (week 37)
# results[2] was trained to predict split_col=2 (week 38)
# etc.

i = 1

chains = results[i][1]

df_to_predict = data_splits[i].test.matches

df_to_predict = filter( row -> row.split_col == i+1, ds.matches)


match_predictions = [] # Store NamedTuples

for row in eachrow(df_to_predict)
    try
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        # --- Get Bayesian posteriors ---
        λs, μs = get_rate_chains(chains, h_id, a_id)
        
        # --- Calculate prob chains ---
        p_1x2_chains = calculate_1x2_chains(λs, μs)
        p_ou25_chains = calculate_ou_chains(λs, μs, 2.5)
        # You can add more lines here
        p_ou15_chains = calculate_ou_chains(λs, μs, 1.5)
        p_btts_chains = calculate_btts_chains(λs, μs)
        
        # --- Summarize all chains ---
        summaries_1 = summarize_prob_chain(p_1x2_chains.p_home, "1")
        summaries_X = summarize_prob_chain(p_1x2_chains.p_draw, "X")
        summaries_2 = summarize_prob_chain(p_1x2_chains.p_away, "2")
        
        summaries_O25 = summarize_prob_chain(p_ou25_chains.p_over, "O25")
        summaries_U25 = summarize_prob_chain(p_ou25_chains.p_under, "U25")
        
        summaries_O15 = summarize_prob_chain(p_ou15_chains.p_over, "O15")
        summaries_U15 = summarize_prob_chain(p_ou15_chains.p_under, "U15")
        
        summaries_BTTS_Y = summarize_prob_chain(p_btts_chains.p_yes, "BTTS_Y")
        summaries_BTTS_N = summarize_prob_chain(p_btts_chains.p_no, "BTTS_N")
        
        # --- Combine into one NamedTuple ---
        # We store match info, results, and all summaries
        result_tuple = (
            match_id = row.match_id,
            split = i,
            home_team = row.home_team,
            away_team = row.away_team,
            home_score = row.home_score,
            away_score = row.away_score,
            summaries_1...,
            summaries_X...,
            summaries_2...,
            summaries_O25...,
            summaries_U25...,
            summaries_O15...,
            summaries_U15...,
            summaries_BTTS_Y...,
            summaries_BTTS_N...
        )
        
        push!(match_predictions, result_tuple)
        
    catch e
        println("Error processing match $(row.match_id): $e")
        # This could happen if a team wasn't in the vocabulary (though your split logic should prevent this)
    end
end



all_predictions = [] # This will hold a DataFrame from each split
push!(all_predictions, DataFrame(match_predictions))

predictions_df = vcat(all_predictions...)

final_analysis_df = leftjoin(predictions_df, market_odds_df, on = :match_id)




# 1. Add outcome and P&L columns for analysis
analysis_df = copy(final_analysis_df) # Work on a copy

# --- Define winners ---
analysis_df.winner_1 = analysis_df.home_score .> analysis_df.away_score
analysis_df.winner_X = analysis_df.home_score .== analysis_df.away_score
analysis_df.winner_2 = analysis_df.home_score .< analysis_df.away_score
analysis_df.winner_O25 = (analysis_df.home_score .+ analysis_df.away_score) .> 2.5
analysis_df.winner_BTTS_Y = (analysis_df.home_score .> 0) .& (analysis_df.away_score .> 0)

# --- Calculate value (our_prob * market_odds) ---
# We use the mean of the posterior as our probability estimate
analysis_df.value_1 = analysis_df[!, "1_prob_mean"] .* analysis_df.market_odds_1
analysis_df.value_X = analysis_df.X_prob_mean .* analysis_df.market_odds_X
analysis_df.value_2 = analysis_df[!, "2_prob_mean"] .* analysis_df.market_odds_2
analysis_df.value_O25 = analysis_df.O25_prob_mean .* analysis_df.market_odds_O25
analysis_df.value_BTTS_Y = analysis_df.BTTS_Y_prob_mean .* analysis_df.market_odds_BTTS_Y

# --- Simulate strategy and plot ROI vs. 'c' (value threshold) ---

thresholds_to_test = 1.0:0.01:1.20 # Test c from 1.0 (any edge) to 1.2 (20% edge)
roi_results = []

# We'll just test backing the home team for this example
for c in thresholds_to_test
    
    # Find rows where we would place a bet (and market odds exist)
    bets_df = filter(row -> !ismissing(row.value_1) && row.value_1 > c, analysis_df)
    
    n_bets = nrow(bets_df)
    
    if n_bets == 0
        push!(roi_results, (c = c, n_bets = 0, total_staked = 0, total_return = 0, roi = 0.0))
        continue
    end
    
    # Calculate returns (assuming 1 unit stake)
    # If we win, we get market_odds_1 back. If we lose, we get 0.
    bets_df.pnl = bets_df.winner_1 .* (bets_df.market_odds_1) .- 1.0
    
    total_staked = n_bets
    total_return = sum(bets_df.winner_1 .* bets_df.market_odds_1)
    
    # ROI = (Total Return - Total Staked) / Total Staked
    roi = (total_return - total_staked) / total_staked
    
    push!(roi_results, (c = c, n_bets = n_bets, total_staked = total_staked, total_return = total_return, roi = roi))
end

roi_df = DataFrame(roi_results)
# display(roi_df)

# You can now plot this!
using Plots
plot(roi_df.c, roi_df.roi, label="Home Win ROI",
     xlabel="Value Threshold (c)", ylabel="ROI",
     title="Betting Strategy Performance")















###


team_map = vocabulary.mappings[:team_map]
all_predictions = [] # This will hold a DataFrame from each split



for i in 1:length(results)
    
    # 1. Get the trained model chains for this split
    # Assuming one chain group per split, hence results[i][1]
    chains = results[i][1]
    
    # 2. Get the matches we need to predict (the test set)
    # We use data_splits to get the original DataFrame rows
    df_to_predict = filter( row -> row.split_col == i, ds.matches)
    
    if isempty(df_to_predict)
        println("Skipping split $i, no test matches.")
        continue
    end
    
    split_num = minimum(df_to_predict.split_col)
    println("Processing predictions for split $split_num...")
    
    # 3. Iterate each match, calculate probabilities, and store
    match_predictions = [] # Store NamedTuples
    
    for row in eachrow(df_to_predict)
        try
            h_id = team_map[row.home_team]
            a_id = team_map[row.away_team]
            
            # --- Get Bayesian posteriors ---
            λs, μs = get_rate_chains(chains, h_id, a_id)
            
            # --- Calculate prob chains ---
            p_1x2_chains = calculate_1x2_chains(λs, μs)
            p_ou25_chains = calculate_ou_chains(λs, μs, 2.5)
            # You can add more lines here
            p_ou15_chains = calculate_ou_chains(λs, μs, 1.5)
            p_btts_chains = calculate_btts_chains(λs, μs)
            
            # --- Summarize all chains ---
            summaries_1 = summarize_prob_chain(p_1x2_chains.p_home, "1")
            summaries_X = summarize_prob_chain(p_1x2_chains.p_draw, "X")
            summaries_2 = summarize_prob_chain(p_1x2_chains.p_away, "2")
            
            summaries_O25 = summarize_prob_chain(p_ou25_chains.p_over, "O25")
            summaries_U25 = summarize_prob_chain(p_ou25_chains.p_under, "U25")
            
            summaries_O15 = summarize_prob_chain(p_ou15_chains.p_over, "O15")
            summaries_U15 = summarize_prob_chain(p_ou15_chains.p_under, "U15")
            
            summaries_BTTS_Y = summarize_prob_chain(p_btts_chains.p_yes, "BTTS_Y")
            summaries_BTTS_N = summarize_prob_chain(p_btts_chains.p_no, "BTTS_N")
            
            # --- Combine into one NamedTuple ---
            # We store match info, results, and all summaries
            result_tuple = (
                match_id = row.match_id,
                split = split_num,
                home_team = row.home_team,
                away_team = row.away_team,
                home_score = row.home_score,
                away_score = row.away_score,
                summaries_1...,
                summaries_X...,
                summaries_2...,
                summaries_O25...,
                summaries_U25...,
                summaries_O15...,
                summaries_U15...,
                summaries_BTTS_Y...,
                summaries_BTTS_N...
            )
            
            push!(match_predictions, result_tuple)
            
        catch e
            println("Error processing match $(row.match_id): $e")
            # This could happen if a team wasn't in the vocabulary (though your split logic should prevent this)
        end
    end
    
    # 4. Convert this split's predictions to a DataFrame and add to our list
    if !isempty(match_predictions)
        push!(all_predictions, DataFrame(match_predictions))
    end
end

# 5. Combine all splits into one master DataFrame
if !isempty(all_predictions)
    predictions_df = vcat(all_predictions...)
    
    # 6. Join our predictions with the market odds
    final_analysis_df = leftjoin(predictions_df, market_odds_df, on = :match_id)
    
    println("Successfully created final_analysis_df with $(nrow(final_analysis_df)) predictions.")
else
    println("No predictions were generated.")
end


# --- You now have your master DataFrame: final_analysis_df ---
 display(first(final_analysis_df, 5))



# 1. Add outcome and P&L columns for analysis
analysis_df = copy(final_analysis_df) # Work on a copy

# --- Define winners ---
analysis_df.winner_1 = analysis_df.home_score .> analysis_df.away_score
analysis_df.winner_X = analysis_df.home_score .== analysis_df.away_score
analysis_df.winner_2 = analysis_df.home_score .< analysis_df.away_score
analysis_df.winner_O25 = (analysis_df.home_score .+ analysis_df.away_score) .> 2.5
analysis_df.winner_BTTS_Y = (analysis_df.home_score .> 0) .& (analysis_df.away_score .> 0)

# --- Calculate value (our_prob * market_odds) ---
# We use the mean of the posterior as our probability estimate
analysis_df.value_1 = analysis_df[!, "1_prob_mean"] .* analysis_df.market_odds_1
analysis_df.value_X = analysis_df.X_prob_mean .* analysis_df.market_odds_X
analysis_df.value_2 = analysis_df[!, "2_prob_mean"] .* analysis_df.market_odds_2
analysis_df.value_O25 = analysis_df.O25_prob_mean .* analysis_df.market_odds_O25
analysis_df.value_BTTS_Y = analysis_df.BTTS_Y_prob_mean .* analysis_df.market_odds_BTTS_Y

# --- Simulate strategy and plot ROI vs. 'c' (value threshold) ---

thresholds_to_test = 1.0:0.01:1.20 # Test c from 1.0 (any edge) to 1.2 (20% edge)
roi_results = []

# We'll just test backing the home team for this example
for c in thresholds_to_test
    
    # Find rows where we would place a bet (and market odds exist)
    bets_df = filter(row -> !ismissing(row.value_1) && row.value_1 > c, analysis_df)
    
    n_bets = nrow(bets_df)
    
    if n_bets == 0
        push!(roi_results, (c = c, n_bets = 0, total_staked = 0, total_return = 0, roi = 0.0))
        continue
    end
    
    # Calculate returns (assuming 1 unit stake)
    # If we win, we get market_odds_1 back. If we lose, we get 0.
    bets_df.pnl = bets_df.winner_1 .* (bets_df.market_odds_1) .- 1.0
    
    total_staked = n_bets
    total_return = sum(bets_df.winner_1 .* bets_df.market_odds_1)
    
    # ROI = (Total Return - Total Staked) / Total Staked
    roi = (total_return - total_staked) / total_staked
    
    push!(roi_results, (c = c, n_bets = n_bets, total_staked = total_staked, total_return = total_return, roi = roi))
end

roi_df = DataFrame(roi_results)
# display(roi_df)

# You can now plot this!
using Plots
plot(roi_df.c, roi_df.roi, label="Home Win ROI",
     xlabel="Value Threshold (c)", ylabel="ROI",
     title="Betting Strategy Performance")



##

# (You already ran the winner and value calculations for 1, X, 2, O25, BTTS_Y)

# --- 1. Define the missing winners ---
analysis_df.winner_U25 = .!analysis_df.winner_O25
analysis_df.winner_BTTS_N = .!analysis_df.winner_BTTS_Y

# --- 2. Calculate value for the missing markets ---
analysis_df.value_U25 = analysis_df.U25_prob_mean .* analysis_df.market_odds_U25
analysis_df.value_BTTS_N = analysis_df.BTTS_N_prob_mean .* analysis_df.market_odds_BTTS_N

# --- 3. Calculate P&L for EVERY potential bet ---
# P&L = (return if win) - (stake of 1)
# We multiply by the boolean 'winner' column (true=1, false=0).
analysis_df.pnl_1 = (analysis_df.winner_1 .* analysis_df.market_odds_1) .- 1.0
analysis_df.pnl_X = (analysis_df.winner_X .* analysis_df.market_odds_X) .- 1.0
analysis_df.pnl_2 = (analysis_df.winner_2 .* analysis_df.market_odds_2) .- 1.0

analysis_df.pnl_O25 = (analysis_df.winner_O25 .* analysis_df.market_odds_O25) .- 1.0
analysis_df.pnl_U25 = (analysis_df.winner_U25 .* analysis_df.market_odds_U25) .- 1.0

analysis_df.pnl_BTTS_Y = (analysis_df.winner_BTTS_Y .* analysis_df.market_odds_BTTS_Y) .- 1.0
analysis_df.pnl_BTTS_N = (analysis_df.winner_BTTS_N .* analysis_df.market_odds_BTTS_N) .- 1.0

# --- 4. See the results per-match ---
# This now shows the P&L for every line, for every match.
# A 'missing' P&L just means the odds were missing and no bet could be placed.
println("--- P&L Breakdown per Match (example) ---")
display(
    select(
        analysis_df, 
        :match_id, 
        :value_1, :pnl_1, 
        :value_X, :pnl_X, 
        :value_O25, :pnl_O25
    )
)


# 1. Define the columns we want to stack
value_cols = ["value_1", "value_X", "value_2", "value_O25", "value_U25", "value_BTTS_Y", "value_BTTS_N"]
pnl_cols = ["pnl_1", "pnl_X", "pnl_2", "pnl_O25", "pnl_U25", "pnl_BTTS_Y", "pnl_BTTS_N"]

# Define ID columns to keep (add :league or :tournament_id if you have it)
id_cols = [:match_id, :home_team, :away_team] 

# 2. Stack the value columns
df_value_long = stack(analysis_df, value_cols, id_cols, 
                      variable_name=:bet_type, value_name=:value)

# 3. Stack the P&L columns
df_pnl_long = stack(analysis_df, pnl_cols, id_cols, 
                    variable_name=:bet_type, value_name=:pnl)

# 4. Clean up the 'bet_type' names (e.g., "value_1" -> "1")
df_value_long.bet_type = replace.(df_value_long.bet_type, "value_" => "")
df_pnl_long.bet_type = replace.(df_pnl_long.bet_type, "pnl_" => "")

# 5. Join them into one master 'bets' DataFrame
bets_analysis_df = innerjoin(df_value_long, df_pnl_long, 
                             on = [:match_id, :home_team, :away_team, :bet_type])

println("\n--- Tidy Bets DataFrame ---")
display(first(bets_analysis_df, 14)) # Show first 2 matches (7 markets each)


# Filter for all bets with a 5% edge (c = 1.05)
value_bets = filter(row -> !ismissing(row.value) && row.value > 1.05, bets_analysis_df)

println("\n--- Value Bets (c > 1.05) ---")
display(value_bets)

# --- Breakdown by Bet Type ---
println("\n--- ROI Breakdown by Bet Type (c > 1.05) ---")
roi_by_bet_type = combine(groupby(value_bets, :bet_type)) do df
    n_bets = nrow(df)
    total_pnl = sum(skipmissing(df.pnl))
    roi = total_pnl / n_bets
    return (n_bets = n_bets, total_pnl = total_pnl, roi = roi)
end

display(roi_by_bet_type)

# --- Breakdown by Match ---
println("\n--- P&L Breakdown by Match (c > 1.05) ---")
roi_by_match = combine(groupby(value_bets, [:match_id, :home_team, :away_team])) do df
    n_bets_on_match = nrow(df) # Num value bets found for this match
    total_pnl_on_match = sum(skipmissing(df.pnl))
    return (n_bets_on_match = n_bets_on_match, total_pnl_on_match = total_pnl_on_match)
end

display(roi_by_match)
