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

# `all_oos_results` will contain the merged predictions for
# splits 1, 2, and 3.








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

# --- Get the Score Matrix ---
# This matrix now contains the final predicted probabilities for each score
score_matrix = compute_score_matrix(λs, μs, 6)

# Example: Probability of 2-1
# (Remember 1-based indexing: [h+1, a+1])
p_2_1 = score_matrix[2 + 1, 1 + 1]
println("Probability of 2-1: $p_2_1")

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
