# dev/01_inits_types_atoms.jl

using Revise

using BayesianFootball



## 1. Load Data
data_store = BayesianFootball.Data.load_default_datastore()


# 2. Define an Experiment (your "Config")


model =  Models.PreGame.StaticPoisson()
splitter = Experiments.StaticSplit(["24/25"])
sampler_config = Sampling.NUTSMethod(500, 2, 50)


exp1 = Experiments.Experiment(
    "StaticPoisson_ExpandingWindow_22-23",
    model,
    splitter,
    sampler_config,
)

# This script manually executes each step of the "Trainer" pipeline
# to test that each module and function works correctly in sequence.

using Revise
using BayesianFootball
using DataFrames
using Turing # Needed for predict()

println("--- ✅ Setup Complete: Packages loaded ---")

# ============================================================================
# PHASE 1: DEFINE THE "ATOMS" (CONFIGS AND DATA)
# ============================================================================
println("\n--- PHASE 1: Defining Atoms ---")

# --- Atom 1: The DataStore (D) ---
data_store = BayesianFootball.Data.load_default_datastore()
println("Loaded DataStore with $(nrow(data_store.matches)) matches.")

# --- Atom 2: The Master Experiment Config ---
# This defines all the parameters for our single run.

# Model (M)
model = Models.PreGame.StaticPoisson()

# Splitter (defines how we get our training data D_i from D)
splitter = Experiments.StaticSplit(["24/25"])

# Sampler Config (Config_s)
sampler_config = Sampling.NUTSMethod(1000, 2, 50)

# The full Experiment object
exp1 = Experiments.Experiment(
    "StaticPoisson_TestRun",
    model,
    splitter,
    sampler_config,
)

println("Experiment configured: $(exp1.name)")


# Model 2 (M)
model2 = Models.PreGame.StaticSimplexPoisson()



# The full Experiment object
exp2 = Experiments.Experiment(
    "StaticSimplexPoisson_TestRun",
    model2,
    splitter,
    sampler_config,
)

println("Experiment configured: $(exp2.name)")

# ============================================================================
# PHASE 2: EXECUTE THE MORPHISMS (THE PIPELINE STEPS)
# ============================================================================
println("\n--- PHASE 2: Executing Morphisms ---")

# --- PRE-STEP: Get the training data for this run ---
# The Experiments.jl runner would do this automatically. Here, we do it manually.
println("\n[Pre-Step] Filtering data based on splitter...")

train_df = filter(row -> row.season in splitter.train_seasons, data_store.matches)

println("Created train_df with $(nrow(train_df)) matches for season(s): $(splitter.train_seasons)")

# --- Step 1: Morphism f: (D_i, M) -> F_i ---
# Create the FeatureSet from our training data for this specific model.
println("\n[Step 1: Morphism f] Calling Features.create_features...")
features = Features.create_features(exp1.model, train_df)

println("✅ Success! Created FeatureSet with $(features.n_teams) teams.")

# --- Step 2: Build TRAINING Model ---
# Create the Turing model instance ready for training.
println("\n[Step 2] Building TRAINING model...")
turing_model = Models.PreGame.build_turing_model(exp1.model, features)
println("✅ Success! Built training model instance.")

# --- Step 3: Morphism g: (F_i, M, Config_s) -> C_params ---
# Run the sampler to get the posterior parameter chains.
println("\n[Step 3: Morphism g] Calling Sampling.train to get C_params...")

chains_params = Sampling.train(turing_model, exp1.sampler_config)
println("✅ Success! Sampling complete. C_params (parameter chains) created.")
display(chains_params)

### model 2
# --- Step 2: Build TRAINING Model ---
# Create the Turing model instance ready for training.
println("\n[Step 2] Building TRAINING model...")
turing_model2 = Models.PreGame.build_turing_model(exp2.model, features)
println("✅ Success! Built training model instance.")

# --- Step 3: Morphism g: (F_i, M, Config_s) -> C_params ---
# Run the sampler to get the posterior parameter chains.
println("\n[Step 3: Morphism g] Calling Sampling.train to get C_params...")

chains_params2 = Sampling.train(turing_model2, exp2.sampler_config)
println("✅ Success! Sampling complete. C_params (parameter chains) created.")
display(chains_params)




# ============================================================================
# PHASE 3: EXECUTE THE "ANALYZER" PIPELINE (PREDICTION)
# ============================================================================
println("\n--- PHASE 3: Executing Analyzer Morphism (h_goals) ---")

# --- PRE-STEP: Define the new data we want to predict on ---
# df_to_predict = first(train_df, 5)
df_to_predict = train_df
println("\n[Pre-Step] Defined new data to predict on ($(nrow(df_to_predict)) matches).")



# --- Step 4: Prepare Prediction Data ---
println("\n[Step 4] Preparing prediction data using training team_map...")
team_map = features.team_map 
n_teams = features.n_teams
home_ids_to_predict = [team_map[name] for name in df_to_predict.home_team]
away_ids_to_predict = [team_map[name] for name in df_to_predict.away_team]
println("✅ Success! Mapped team names to integer IDs.")

# --- Step 5: Build PREDICTION Model ---
predictions = Models.PreGame.predict(exp1.model, df_to_predict, features, chains_params)
p1 = predictions

p2 = Models.PreGame.predict(exp2.model, df_to_predict, features, chains_params)

# m2 = Models.PreGame.build_turing_model(exp2.model, n_teams, home_ids_to_predict, away_ids_to_predict) 
# p2 = Turing.predict(m2, chains_params2)




id = 601

df_to_predict[id, :]

describe(p1[Symbol("predicted_home_goals[$id]")])
describe(p1[Symbol("predicted_away_goals[$id]")])
describe(p2[Symbol("predicted_home_goals[$id]")])
describe(p2[Symbol("predicted_away_goals[$id]")])


p1[Symbol("predicted_home_goals[$id]")]



####

using DataFrames
using Statistics # For the count function

"""
Calculates 1X2 and Over/Under odds from Turing prediction samples for a single match.
"""
function calculate_odds_from_samples(predictions, match_row_index::Int)
    
    # 1. Extract and flatten samples. 
    # We round to Int as goals are discrete counts.
    home_samples = round.(Int, vec(predictions[Symbol("predicted_home_goals[$(match_row_index)]")]))
    away_samples = round.(Int, vec(predictions[Symbol("predicted_away_goals[$(match_row_index)]")]))
    
    n_samples = length(home_samples)
    
    # --- 2. Calculate 1X2 Probabilities ---
    prob_1 = count(home_samples .> away_samples) / n_samples
    prob_X = count(home_samples .== away_samples) / n_samples
    prob_2 = count(home_samples .< away_samples) / n_samples
    
    # --- 3. Calculate Total Goals ---
    total_goals = home_samples .+ away_samples
    
    # --- 4. Calculate Over/Under Probabilities ---
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    probs_over = Dict{Float64, Float64}()
    probs_under = Dict{Float64, Float64}()
    
    for t in thresholds
        probs_over[t] = count(total_goals .> t) / n_samples
        probs_under[t] = count(total_goals .< t) / n_samples
    end
    
    # --- 5. Convert Probabilities to Odds (1 / P) ---
    # 1.0 / 0.0 will correctly result in `Inf`, which is fine.
    return (
        # Grab the actual match_id from your dataframe for easy joining later
        match_id = df_to_predict.match_id[match_row_index], 
        home_team = df_to_predict.home_team[match_row_index],
        away_team = df_to_predict.away_team[match_row_index],
        
        # 1X2 Odds
        odds_1 = 1.0 / prob_1,
        odds_X = 1.0 / prob_X,
        odds_2 = 1.0 / prob_2,
        
        # Over/Under 0.5
        odds_O_0_5 = 1.0 / probs_over[0.5],
        odds_U_0_5 = 1.0 / probs_under[0.5],
        
        # Over/Under 1.5
        odds_O_1_5 = 1.0 / probs_over[1.5],
        odds_U_1_5 = 1.0 / probs_under[1.5],
        
        # Over/Under 2.5
        odds_O_2_5 = 1.0 / probs_over[2.5],
        odds_U_2_5 = 1.0 / probs_under[2.5],
        
        # Over/Under 3.5
        odds_O_3_5 = 1.0 / probs_over[3.5],
        odds_U_3_5 = 1.0 / probs_under[3.5],
        
        # Over/Under 4.5
        odds_O_4_5 = 1.0 / probs_over[4.5],
        odds_U_4_5 = 1.0 / probs_under[4.5]
    )
end

# --- Now, loop over all matches in your df_to_predict ---

n_matches = nrow(df_to_predict)
all_odds_p1 = [] # To store results from model 1
all_odds_p2 = [] # To store results from model 2

println("\n[Step 6] Calculating odds from posterior samples...")

for i in 1:n_matches
    # Calculate for model 1
    push!(all_odds_p1, calculate_odds_from_samples(p1, i))
    
    # Calculate for model 2
    push!(all_odds_p2, calculate_odds_from_samples(p2, i))
end

# Convert the array of NamedTuples into a DataFrame
odds_df_p1 = DataFrame(all_odds_p1)
odds_df_p2 = DataFrame(all_odds_p2)

println("✅ Success! Generated odds DataFrames.")

# --- Display your results ---
println("\n--- Odds from Model 1 (p1) ---")
println(first(odds_df_p1, 5))

println("\n--- Odds from Model 2 (p2) ---")
println(first(odds_df_p2, 5))



first(df_to_predict, 5)







using DataFrames, Statistics

# --- Step 1: Define Helper Functions ---

"""
Maps the market odds structure to a single, unique key that
matches the column names from `odds_df_p1`.
"""
function get_model_market_name(market_group, choice_name, choice_group)
    # Handle potential missing values
    mg = coalesce(market_group, "")
    cn = coalesce(choice_name, "")
    
    if mg == "1X2"
        if cn == "1" return "odds_1" end
        if cn == "X" return "odds_X" end
        if cn == "2" return "odds_2" end
    elseif mg == "Match goals"
        if ismissing(choice_group) return missing end
        
        # Format the line, e.g., 2.5 -> "2_5"
        line = replace(string(choice_group), "." => "_")
        
        if cn == "Over" return "odds_O_$(line)" end
        if cn == "Under" return "odds_U_$(line)" end
    end
    
    return missing # Ignore other markets (e.g., Double chance, Asian handicap)
end

"""
Calculates profit for a 1-unit bet based on a value strategy.
Strategy: Bet 1 unit if model_odds < market_odds.
"""
function calculate_profit(model_odds, market_odds, winning)
    # Don't bet (profit=0) if:
    # 1. Game result is unknown (winning is missing)
    # 2. No value (model odds are >= market odds)
    if ismissing(winning) || model_odds >= market_odds
        return 0.0
    end
    
    # We found value and placed a bet.
    # If winning is true, profit is (odds - 1). If false, profit is -1.
    return winning ? (market_odds - 1.0) : -1.0
end


# --- Step 2: Prepare Market Odds (Long Format) ---

# Create the new mapping column
transform!(data_store.odds, 
    AsTable([:market_group, :choice_name, :choice_group]) => 
    ByRow(row -> get_model_market_name(row...)) => 
    :model_market_name
)

# Filter to only the markets we have model odds for
market_odds_filtered = filter(row -> !ismissing(row.model_market_name) && row.market_name!="1st half", data_store.odds)

# Select just the columns we need
market_odds_to_join = select(market_odds_filtered, 
    :match_id, 
    :model_market_name, 
    :decimal_odds => :market_odds, # Rename for clarity
    :winning
)

println("✅ Prepared $(nrow(market_odds_to_join)) market odds for comparison.")

# --- Step 3: Prepare Model Odds (Stack to Long Format) ---

# Get all the odds column names
odds_cols = filter(name -> startswith(String(name), "odds_"), names(odds_df_p1))

# Stack p1
odds_long_p1 = stack(odds_df_p1, odds_cols,
    [:match_id, :home_team, :away_team],
    variable_name=:model_market_name, 
    value_name=:model_odds_p1
)
transform!(odds_long_p1, :model_market_name => ByRow(String) => :model_market_name) # Convert Symbol to String

# Stack p2
odds_long_p2 = stack(odds_df_p2, odds_cols,
    [:match_id], # Don't need team names again
    variable_name=:model_market_name, 
    value_name=:model_odds_p2
)
transform!(odds_long_p2, :model_market_name => ByRow(String) => :model_market_name)

println("✅ Stacked model odds to long format.")


# --- Step 4: Join Everything into one DataFrame ---

# Join market odds with model 1 odds
comparison_df = innerjoin(market_odds_to_join, odds_long_p1, on = [:match_id, :model_market_name])

# Join the result with model 2 odds
comparison_df = innerjoin(comparison_df, odds_long_p2, on = [:match_id, :model_market_name])

println("✅ Joined market and model odds.")

# --- Step 5: Calculate Profitability ---

# Add profit columns for each model based on our strategy
transform!(comparison_df, 
    AsTable([:model_odds_p1, :market_odds, :winning]) => 
    ByRow(row -> calculate_profit(row...)) => 
    :profit_p1
)

transform!(comparison_df, 
    AsTable([:model_odds_p2, :market_odds, :winning]) => 
    ByRow(row -> calculate_profit(row...)) => 
    :profit_p2
)

println("\n--- Comparison DataFrame (Model Odds vs. Market Odds) ---")
println(first(comparison_df, 10))


# --- Step 6: Summarize Results ---

println("\n--- Profitability Summary (1-Unit Value Bets) ---")

total_profit_p1 = sum(comparison_df.profit_p1)
total_profit_p2 = sum(comparison_df.profit_p2)
total_bets_p1 = count(comparison_df.profit_p1 .!= 0)
total_bets_p2 = count(comparison_df.profit_p2 .!= 0)

# Calculate Return on Investment (ROI)
roi_p1 = total_bets_p1 > 0 ? (total_profit_p1 / total_bets_p1) * 100 : 0
roi_p2 = total_bets_p2 > 0 ? (total_profit_p2 / total_bets_p2) * 100 : 0


println("\nModel 1 (p1) Total Profit: $(round(total_profit_p1, digits=2))")
println("Model 1 (p1) Total Bets: $(total_bets_p1)")
println("Model 1 (p1) ROI: $(round(roi_p1, digits=2))%")

println("\nModel 2 (p2) Total Profit: $(round(total_profit_p2, digits=2))")
println("Model 2 (p2) Total Bets: $(total_bets_p2)")
println("Model 2 (p2) ROI: $(round(roi_p2, digits=2))%")


println("\n--- Profit by Market (Model 1) ---")
profit_by_market_p1 = combine(groupby(comparison_df, :model_market_name), 
    :profit_p1 => sum => :total_profit,
    :profit_p1 => (p -> count(p .!= 0)) => :num_bets
)
# Filter to only markets where bets were placed
println(filter(row -> row.num_bets > 0, profit_by_market_p1))

println("\n--- Profit by Market (Model 2) ---")
profit_by_market_p2 = combine(groupby(comparison_df, :model_market_name), 
    :profit_p2 => sum => :total_profit,
    :profit_p2 => (p -> count(p .!= 0)) => :num_bets
)
# Filter to only markets where bets were placed
println(filter(row -> row.num_bets > 0, profit_by_market_p2))
profit_by_market_p2
profit_by_market_p1


# Add ROI % to your profit_by_market DataFrames
transform!(profit_by_market_p1, 
    AsTable([:total_profit, :num_bets]) => 
    ByRow(row -> (row.total_profit / row.num_bets) * 100) => 
    :roi_pct
)

transform!(profit_by_market_p2, 
    AsTable([:total_profit, :num_bets]) => 
    ByRow(row -> (row.total_profit / row.num_bets) * 100) => 
    :roi_pct
)

# Sort by the best ROI
sort!(profit_by_market_p1, :roi_pct, rev=true)
sort!(profit_by_market_p2, :roi_pct, rev=true)

println(profit_by_market_p1)
println(profit_by_market_p2)



# 1. How to Calculate Uncertainty on Your Probabilities 

using Statistics

"""
Calculates a 95% Credible Interval for a market probability 
by bootstrapping the posterior predictive samples.
"""
function get_probability_ci(home_samples, away_samples, market_condition_fn; n_bootstraps=5000)
    n_samples = length(home_samples)
    bootstrapped_probs = Vector{Float64}(undef, n_bootstraps)
    
    original_samples = collect(zip(home_samples, away_samples))
    
    for i in 1:n_bootstraps
        # Step 2: Resample with replacement
        boot_sample_indices = rand(1:n_samples, n_samples)
        
        count_true = 0
        for idx in boot_sample_indices
            # Step 3: Apply market condition
            if market_condition_fn(original_samples[idx]...)
                count_true += 1
            end
        end
        
        # Step 4: Calculate and store the probability
        bootstrapped_probs[i] = count_true / n_samples
    end
    
    # Step 5: Return the 95% Credible Interval
    return (
        mean = mean(bootstrapped_probs),
        lower_95 = quantile(bootstrapped_probs, 0.025),
        upper_95 = quantile(bootstrapped_probs, 0.975)
    )
end

# --- Example Usage for a single match ---
id = 1 # For the first match
home_s = round.(Int, vec(p1[Symbol("predicted_home_goals[$(id)]")]))
away_s = round.(Int, vec(p1[Symbol("predicted_away_goals[$(id)]")]))

# Define market conditions as functions
home_win_fn(h, a) = h > a
over_2_5_fn(h, a) = (h + a) > 2.5

# Calculate CIs
home_win_ci = get_probability_ci(home_s, away_s, home_win_fn)
over_2_5_ci = get_probability_ci(home_s, away_s, over_2_5_fn)

println("Home Win Prob CI: $(home_win_ci)")
println("Over 2.5 Prob CI: $(over_2_5_ci)")




using DataFrames
using MCMCChains
using LinearAlgebra

"""
Extracts and flattens all samples for a parameter group 
(e.g., "log_α[1]", "log_α[2]", ...) into a (samples × n_params) matrix.
"""
function extract_samples(chain::Chains, base_name::String)
    # Find all parameter names that start with the base_name
    nms = names(chain)
    
    # We need to be careful to match "log_α[1]" but not "log_α_scale"
    # So we'll check for the base name followed by an opening bracket "["
    matches = filter(n -> occursin(Regex("^$(base_name)\\["), String(n)), nms)
    
    if isempty(matches)
        error("No parameters found with base name '$(base_name)['. Check your model's variable names.")
    end
    
    # Extract the array of samples
    arr = Array(chain[matches])
    
    # Reshape to (total_samples, n_params)
    # total_samples = iterations * n_chains
    # n_params = length(matches)
    return reshape(arr, :, size(arr, 2))
end

"""
Extracts and reconstructs the lambda/mu parameter chains from the
original posterior MCMC chain.
"""
function get_parameter_chains(df_to_predict, feature_set, chains)
    
    # --- 1. Get Match Info ---
    team_map = feature_set.team_map
    home_ids = [team_map[name] for name in df_to_predict.home_team]
    away_ids = [team_map[name] for name in df_to_predict.away_team]

    # --- 2. Extract Parameter Chains (using your function's logic) ---
    
    # This now correctly creates the (samples, n_teams) matrices
    log_α_chain = extract_samples(chains, "log_α") 
    log_β_chain = extract_samples(chains, "log_β")
    
    # Get the scalar parameter (this one is simpler)
    home_adv_chain = vec(chains[:home_adv])
    
    # --- 3. Manually Reconstruct λ and μ for the matches to predict ---
    # This logic is the same as before, but now it works!
    
    log_λs_chain = home_adv_chain .+ log_α_chain[:, home_ids] .+ log_β_chain[:, away_ids]
    log_μs_chain = log_α_chain[:, away_ids] .+ log_β_chain[:, home_ids]

    # --- 4. Exponentiate ---
    lambda_chain = exp.(log_λs_chain)
    mu_chain = exp.(log_μs_chain)

    # Return the (samples, n_matches) matrices
    return Dict(:lambda => lambda_chain, :mu => mu_chain)
end

param_chains_p1 = get_parameter_chains(df_to_predict, features, chains_params)

param_chains_p2 = get_parameter_chains(df_to_predict, features, chains_params2)


lambda_chains = param_chains_p1[:lambda] # This is your (2000, n_matches) matrix
mu_chains = param_chains_p1[:mu]

using Plots
using StatsPlots

density(lambda_chains[1,:])



using Distributions

"""
Calculates 1X2 and Over/Under probabilities from Poisson goal rates.
"""
function calculate_market_probs(lambda_home::Float64, lambda_away::Float64; max_goals=10)
    # Create Poisson distributions from the given rates
    home_dist = Poisson(lambda_home)
    away_dist = Poisson(lambda_away)
    
    # Calculate the probability of each score from 0 to max_goals
    home_probs = [pdf(home_dist, i) for i in 0:max_goals]
    away_probs = [pdf(away_dist, i) for i in 0:max_goals]
    
    # Initialize probabilities
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0
    p_over_2_5 = 0.0
    # Add other markets as needed (e.g., p_over_1_5, p_under_3_5, etc.)
    
    # Sum probabilities over the score grid
    for h in 0:max_goals
        for a in 0:max_goals
            prob_score = home_probs[h+1] * away_probs[a+1]
            
            if h > a
                p_home_win += prob_score
            elseif h == a
                p_draw += prob_score
            else # h < a
                p_away_win += prob_score
            end
            
            if (h + a) > 2.5
                p_over_2_5 += prob_score
            end
        end
    end
    
    return (
        prob_1 = p_home_win,
        prob_X = p_draw,
        prob_2 = p_away_win,
        prob_O_2_5 = p_over_2_5
    )
end


# Your inputs from the previous step
lambda_chains_p1 = param_chains_p1[:lambda] # (2000, n_matches)
mu_chains_p1 = param_chains_p1[:mu]         # (2000, n_matches)

# 1. Broadcast the function over the matrices.
# This creates a (2000, n_matches) matrix of NamedTuples.
println("Calculating market probabilities for Model 1...")
market_probs_chain_p1 = calculate_market_probs.(lambda_chains_p1, mu_chains_p1)

# 2. Extract the specific probability chains you want.
# We use getproperty.() to "unpack" the NamedTuples.
home_win_prob_chains_p1 = getproperty.(market_probs_chain_p1, :prob_1)
draw_prob_chains_p1 = getproperty.(market_probs_chain_p1, :prob_X)
away_win_prob_chains_p1 = getproperty.(market_probs_chain_p1, :prob_2)
over_2_5_prob_chains_p1 = getproperty.(market_probs_chain_p1, :prob_O_2_5)

println("✅ Success! Probability chains created.")



using Statistics

# Let's check match index 2
match_id_to_check = 2

# Get the full probability distribution for this one match
prob_chain_home_win = home_win_prob_chains_p1[:, match_id_to_check]

# --- 1. Get the Probability CI ---
mean_prob = mean(prob_chain_home_win)
lower_95_prob = quantile(prob_chain_home_win, 0.025)
upper_95_prob = quantile(prob_chain_home_win, 0.975)

println("Match #$match_id_to_check (St. Johnstone vs Celtic) Home Win Probability:")
println("  Mean: $(round(mean_prob, digits=4))")
println("  95% Credible Interval: [$(round(lower_95_prob, digits=4)), $(round(upper_95_prob, digits=4))]")

# --- 2. Convert to Odds CI (THE IMPORTANT PART) ---
mean_odds = 1 / mean_prob
# Remember to INVERT THE BOUNDS for odds!
lower_95_odds = 1 / upper_95_prob 
upper_95_odds = 1 / lower_95_prob

println("\nMatch #$match_id_to_check Home Win Odds:")
println("  Mean (Fair Odds): $(round(mean_odds, digits=2))")
println("  95% Credible Interval: [$(round(lower_95_odds, digits=2)), $(round(upper_95_odds, digits=2))]")

# --- 3. Make the High-Confidence Bet Decision ---
market_odds = 19.0 # The market odds for St. Johnstone to win

println("\n--- Betting Decision ---")
println("Market Odds: $market_odds")
println("Model's 95% Upper-Bound Odds: $(round(upper_95_odds, digits=2))")

if upper_95_odds < market_odds
    println("✅ BET: HIGH-CONFIDENCE VALUE")
    println("   (Your model is >97.5% certain the true odds are lower than the market's odds)")
else
    println("❌ NO BET: Market odds are within the model's uncertainty range.")
end



using Distributions, Statistics

"""
Calculates probabilities for all 1X2 and O/U markets.
"""
function calculate_market_probs(lambda_home::Float64, lambda_away::Float64; max_goals=15)
    home_dist = Poisson(lambda_home)
    away_dist = Poisson(lambda_away)
    
    home_probs = [pdf(home_dist, i) for i in 0:max_goals]
    away_probs = [pdf(away_dist, i) for i in 0:max_goals]
    
    # Initialize probabilities
    p1 = 0.0; pX = 0.0; p2 = 0.0
    pO_0_5 = 0.0; pO_1_5 = 0.0; pO_2_5 = 0.0; pO_3_5 = 0.0; pO_4_5 = 0.0
    
    for h in 0:max_goals
        for a in 0:max_goals
            prob_score = home_probs[h+1] * away_probs[a+1]
            total_goals = h + a
            
            # 1X2
            if h > a; p1 += prob_score
            elseif h == a; pX += prob_score
            else; p2 += prob_score
            end
            
            # Overs (it's faster to sum these)
            if total_goals > 0.5; pO_0_5 += prob_score; end
            if total_goals > 1.5; pO_1_5 += prob_score; end
            if total_goals > 2.5; pO_2_5 += prob_score; end
            if total_goals > 3.5; pO_3_5 += prob_score; end
            if total_goals > 4.5; pO_4_5 += prob_score; end
        end
    end
    
    # Calculate Unders
    pU_0_5 = 1.0 - pO_0_5
    pU_1_5 = 1.0 - pO_1_5
    pU_2_5 = 1.0 - pO_2_5
    pU_3_5 = 1.0 - pO_3_5
    pU_4_5 = 1.0 - pO_4_5

    return (
        prob_1=p1, prob_X=pX, prob_2=p2,
        prob_O_0_5=pO_0_5, prob_U_0_5=pU_0_5,
        prob_O_1_5=pO_1_5, prob_U_1_5=pU_1_5,
        prob_O_2_5=pO_2_5, prob_U_2_5=pU_2_5,
        prob_O_3_5=pO_3_5, prob_U_3_5=pU_3_5,
        prob_O_4_5=pO_4_5, prob_U_4_5=pU_4_5
    )
end


using DataFrames, Statistics, MCMCChains

"""
Converts lambda/mu chains into a long DataFrame of model odds,
complete with mean, lower 95%, and upper 95% CIs.
"""
function create_model_odds_ci_df(
    lambda_chains::Matrix{Float64}, 
    mu_chains::Matrix{Float64}, 
    df_to_predict::DataFrame
)
    n_samples, n_matches = size(lambda_chains)
    
    # 1. Broadcast the function to get a (samples, matches) matrix of NamedTuples
    println("Broadcasting probability calculations...")
    market_probs_chain = calculate_market_probs.(lambda_chains, mu_chains)
    println("✅ Done.")

    # Get all the market names (e.g., :prob_1, :prob_X, ...)
    market_keys = keys(market_probs_chain[1, 1])
    
    # This will store our final (match_id, market_name, mean_odds, ...) rows
    results_list = []

    for market_key in market_keys
        # 2. Extract the probability chain for this market (e.g., prob_1)
        prob_chain_matrix = getproperty.(market_probs_chain, market_key)
        
        # 3. Loop through each match (each column)
        for match_idx in 1:n_matches
            prob_chain_for_match = prob_chain_matrix[:, match_idx]
            
            # 4. Calculate stats from the probability chain
            mean_prob = mean(prob_chain_for_match)
            lower_prob = quantile(prob_chain_for_match, 0.025)
            upper_prob = quantile(prob_chain_for_match, 0.975)
            
            # 5. Convert to odds (and invert CIs)
            mean_odds = 1.0 / mean_prob
            lower_odds_95 = 1.0 / upper_prob
            upper_odds_95 = 1.0 / lower_prob
            
            # Get the model_market_name string (e.g., "odds_1", "odds_O_2_5")
            # This converts :prob_1 to "odds_1"
            model_market_name = "odds_" * replace(string(market_key)[6:end], "_0_5" => "_0_5")
            model_market_name = replace(model_market_name, "_O_" => "_O_")
            model_market_name = replace(model_market_name, "_U_" => "_U_")
            model_market_name = replace(model_market_name, "_X" => "_X")
            model_market_name = replace(model_market_name, r"_(\d)_(\d)" => s"_\1_\2") # Fix for 0_5, 1_5 etc.

            # Re-format lines like O_1_5
            if occursin(r"_\d_\d", model_market_name)
                 model_market_name = replace(model_market_name, r"_(\d)_(\d)" => s"_\1_\2")
            end
            
            # Simple replace logic for market names
            market_map = Dict(
                :prob_1 => "odds_1", :prob_X => "odds_X", :prob_2 => "odds_2",
                :prob_O_0_5 => "odds_O_0_5", :prob_U_0_5 => "odds_U_0_5",
                :prob_O_1_5 => "odds_O_1_5", :prob_U_1_5 => "odds_U_1_5",
                :prob_O_2_5 => "odds_O_2_5", :prob_U_2_5 => "odds_U_2_5",
                :prob_O_3_5 => "odds_O_3_5", :prob_U_3_5 => "odds_U_3_5",
                :prob_O_4_5 => "odds_O_4_5", :prob_U_4_5 => "odds_U_4_5"
            )
            model_market_name = market_map[market_key]


            push!(results_list, (
                match_id = df_to_predict.match_id[match_idx],
                model_market_name = model_market_name,
                model_mean_odds = mean_odds,
                model_lower_95_odds = lower_odds_95,
                model_upper_95_odds = upper_odds_95
            ))
        end
    end
    
    return DataFrame(results_list)
end


# --- Step 1: Get Lambda/Mu Chains (from previous step) ---
param_chains_p1 = get_parameter_chains(df_to_predict, features, chains_params)
param_chains_p2 = get_parameter_chains(df_to_predict, features, chains_params2)

# --- Step 2: Create the new Model Odds DataFrames ---
println("\n[Backtest] Processing Model 1 (p1)...")
model_odds_ci_df_p1 = create_model_odds_ci_df(
    param_chains_p1[:lambda], 
    param_chains_p1[:mu], 
    df_to_predict
)
# Rename columns to be specific
rename!(model_odds_ci_df_p1, 
    :model_mean_odds => :p1_mean_odds,
    :model_lower_95_odds => :p1_lower_odds,
    :model_upper_95_odds => :p1_upper_odds
)

println("\n[Backtest] Processing Model 2 (p2)...")
model_odds_ci_df_p2 = create_model_odds_ci_df(
    param_chains_p2[:lambda], 
    param_chains_p2[:mu], 
    df_to_predict
)
rename!(model_odds_ci_df_p2, 
    :model_mean_odds => :p2_mean_odds,
    :model_lower_95_odds => :p2_lower_odds,
    :model_upper_95_odds => :p2_upper_odds
)

# --- Step 3: Prepare Market Odds (same as before) ---
# (Assuming `market_odds_to_join` DataFrame still exists from our previous work)
# If not, re-run the code that creates `market_odds_to_join` from `data_store.odds`

# --- Step 4: Join Everything ---
println("\n[Backtest] Joining all data...")
ci_comparison_df = innerjoin(
    market_odds_to_join, 
    model_odds_ci_df_p1, 
    on = [:match_id, :model_market_name]
)
ci_comparison_df = innerjoin(
    ci_comparison_df, 
    model_odds_ci_df_p2, 
    on = [:match_id, :model_market_name]
)
println("✅ Success! New comparison DataFrame created.")

# --- Step 5: Define New Profit Functions ---
"""
Profit Strategy 1: Simple Value (Old)
Bet 1 unit if model_mean_odds < market_odds.
"""
function calculate_profit_simple(model_mean_odds, market_odds, winning)
    if ismissing(winning) || model_mean_odds >= market_odds
        return 0.0
    end
    return winning ? (market_odds - 1.0) : -1.0
end

"""
Profit Strategy 2: High-Confidence Value (New)
Bet 1 unit if model_upper_95_odds < market_odds.
"""
function calculate_profit_ci(model_upper_odds, market_odds, winning)
    if ismissing(winning) || model_upper_odds >= market_odds
        return 0.0
    end
    return winning ? (market_odds - 1.0) : -1.0
end

# --- Step 6: Apply Both Strategies for Comparison ---
transform!(ci_comparison_df, 
    AsTable([:p1_mean_odds, :market_odds, :winning]) => ByRow(r -> calculate_profit_simple(r...)) => :profit_p1_simple,
    AsTable([:p1_upper_odds, :market_odds, :winning]) => ByRow(r -> calculate_profit_ci(r...)) => :profit_p1_ci,
    AsTable([:p2_mean_odds, :market_odds, :winning]) => ByRow(r -> calculate_profit_simple(r...)) => :profit_p2_simple,
    AsTable([:p2_upper_odds, :market_odds, :winning]) => ByRow(r -> calculate_profit_ci(r...)) => :profit_p2_ci
)

# --- Step 7: Final Profitability Summary ---
println("\n--- FINAL PROFITABILITY COMPARISON ---")

function print_summary(df, profit_col)
    profit = sum(df[!, profit_col])
    n_bets = count(df[!, profit_col] .!= 0)
    roi = n_bets > 0 ? (profit / n_bets) * 100 : 0
    
    println("Total Profit: $(round(profit, digits=2))")
    println("Total Bets:   $n_bets")
    println("ROI:          $(round(roi, digits=2))%")
end

println("\n--- Model 1 (Simple Value) ---")
print_summary(ci_comparison_df, :profit_p1_simple)

println("\n--- Model 1 (High-Confidence CI) ---")
print_summary(ci_comparison_df, :profit_p1_ci)

println("\n--- Model 2 (Simple Value) ---")
print_summary(ci_comparison_df, :profit_p2_simple)

println("\n--- Model 2 (High-Confidence CI) ---")
print_summary(ci_comparison_df, :profit_p2_ci)


using DataFrames, Statistics, MCMCChains

"""
Generates a wide DataFrame of model odds for a list of quantiles.
"""
function create_model_odds_quantiles_df(
    lambda_chains::Matrix{Float64}, 
    mu_chains::Matrix{Float64}, 
    df_to_predict::DataFrame,
    prob_quantiles_to_test::Vector{Float64}
)
    n_samples, n_matches = size(lambda_chains)
    
    # 1. Broadcast the probability calculation
    market_probs_chain = calculate_market_probs.(lambda_chains, mu_chains)
    market_keys = keys(market_probs_chain[1, 1])
    
    results_list = []
    
    # Simple market name mapping
    market_map = Dict(
        :prob_1 => "odds_1", :prob_X => "odds_X", :prob_2 => "odds_2",
        :prob_O_0_5 => "odds_O_0_5", :prob_U_0_5 => "odds_U_0_5",
        :prob_O_1_5 => "odds_O_1_5", :prob_U_1_5 => "odds_U_1_5",
        :prob_O_2_5 => "odds_O_2_5", :prob_U_2_5 => "odds_U_2_5",
        :prob_O_3_5 => "odds_O_3_5", :prob_U_3_5 => "odds_U_3_5",
        :prob_O_4_5 => "odds_O_4_5", :prob_U_4_5 => "odds_U_4_5"
    )

    for market_key in market_keys
        prob_chain_matrix = getproperty.(market_probs_chain, market_key)
        
        for match_idx in 1:n_matches
            prob_chain_for_match = prob_chain_matrix[:, match_idx]
            model_market_name = market_map[market_key]
            
            # Start a result row with basic info
            result_row = Dict{Symbol, Any}(
                :match_id => df_to_predict.match_id[match_idx],
                :model_market_name => model_market_name,
                :model_mean_odds => 1.0 / mean(prob_chain_for_match)
            )
            
            # 2. Calculate upper-bound odds for each quantile
            for q in prob_quantiles_to_test
                # Get the lower-bound probability at this quantile
                lower_prob = quantile(prob_chain_for_match, q)
                
                # Convert to upper-bound odds
                # 1.0 / 0.0 will correctly become Inf
                upper_odds = (lower_prob == 0.0) ? Inf : 1.0 / lower_prob
                
                # e.g., :odds_q_0_05
                quantile_col_name = Symbol("odds_q_$(replace(string(q), "." => "_"))")
                result_row[quantile_col_name] = upper_odds
            end
            
            push!(results_list, NamedTuple(result_row))
        end
    end
    
    return DataFrame(results_list)
end


# --- 1. Define Quantiles to Test ---
# These are the lower-tail quantiles of the *probability* distribution
# q=0.025 means 97.5% one-sided confidence (1.0 - 0.025)
quantiles_to_test = [0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5 , 0.6, 0.7, 0.8, 0.9]

# --- 2. Get Lambda/Mu Chains (as before) ---
param_chains_p1 = get_parameter_chains(df_to_predict, features, chains_params)
param_chains_p2 = get_parameter_chains(df_to_predict, features, chains_params2)

# --- 3. Create the new WIDE Model Odds DataFrames ---
println("\n[Quantile Backtest] Processing Model 1...")
model_odds_q_df_p1 = create_model_odds_quantiles_df(
    param_chains_p1[:lambda], param_chains_p1[:mu], df_to_predict, quantiles_to_test
)
# Rename cols to p1_...
quantile_cols_p1 = [Symbol("p1_$(col)") for col in names(model_odds_q_df_p1) if startswith(string(col), "odds_q_")]
rename!(model_odds_q_df_p1, :model_mean_odds => :p1_mean_odds)
rename!(model_odds_q_df_p1, [Symbol(n) => Symbol("p1_$(n)") for n in names(model_odds_q_df_p1) if startswith(string(n), "odds_q_")]...)

println("\n[Quantile Backtest] Processing Model 2...")
model_odds_q_df_p2 = create_model_odds_quantiles_df(
    param_chains_p2[:lambda], param_chains_p2[:mu], df_to_predict, quantiles_to_test
)
# Rename cols to p2_...
quantile_cols_p2 = [Symbol("p2_$(col)") for col in names(model_odds_q_df_p2) if startswith(string(col), "odds_q_")]
rename!(model_odds_q_df_p2, :model_mean_odds => :p2_mean_odds)
rename!(model_odds_q_df_p2, [Symbol(n) => Symbol("p2_$(n)") for n in names(model_odds_q_df_p2) if startswith(string(n), "odds_q_")]...)

# --- 4. Join Everything ---
println("\n[Quantile Backtest] Joining all data...")
# (Assuming `market_odds_to_join` DataFrame still exists)
q_comparison_df = innerjoin(
    market_odds_to_join, 
    model_odds_q_df_p1, 
    on = [:match_id, :model_market_name]
)
q_comparison_df = innerjoin(
    q_comparison_df, 
    model_odds_q_df_p2, 
    on = [:match_id, :model_market_name]
)

# --- 5. Loop, Backtest, and Summarize ---
# (Using the `calculate_profit_ci` function from the previous step)

results = []
for q in quantiles_to_test
    q_str = replace(string(q), "." => "_")
    p1_col = Symbol("p1_odds_q_$(q_str)")
    p2_col = Symbol("p2_odds_q_$(q_str)")
    
    # Calculate profit for this quantile
    profit_p1 = calculate_profit_ci.(q_comparison_df[!, p1_col], q_comparison_df.market_odds, q_comparison_df.winning)
    profit_p2 = calculate_profit_ci.(q_comparison_df[!, p2_col], q_comparison_df.market_odds, q_comparison_df.winning)
    
    # Summarize P1
    n_bets_p1 = count(profit_p1 .!= 0)
    total_profit_p1 = sum(profit_p1)
    roi_p1 = n_bets_p1 > 0 ? (total_profit_p1 / n_bets_p1) * 100 : 0
    
    # Summarize P2
    n_bets_p2 = count(profit_p2 .!= 0)
    total_profit_p2 = sum(profit_p2)
    roi_p2 = n_bets_p2 > 0 ? (total_profit_p2 / n_bets_p2) * 100 : 0
    
    # Store
    push!(results, (
        q = q,
        confidence = 1.0 - q,
        n_bets_p1 = n_bets_p1,
        p1_profit = total_profit_p1,
        roi_p1 = roi_p1,
        n_bets_p2 = n_bets_p2,
        p2_profit = total_profit_p2,
        roi_p2 = roi_p2
    ))
end

summary_df = DataFrame(results)

println("\n--- Backtest Summary vs. Confidence Level ---")
println(summary_df)



using Plots

# X-axis will be the confidence level (e.g., 97.5%)
x_axis = summary_df.confidence .* 100

# --- Plot 1: ROI vs. Confidence ---
p_roi = plot(
    x_axis, 
    [summary_df.roi_p1, summary_df.roi_p2],
    label = ["Model 1 ROI" "Model 2 ROI"],
    xlabel = "Confidence Threshold % (One-Sided)",
    ylabel = "ROI %",
    title = "ROI vs. Bet Confidence Threshold",
    legend = :topleft,
    lw = 2, # line width
    markershape = :circle
)

# --- Plot 2: Number of Bets vs. Confidence ---
p_bets = plot(
    x_axis, 
    [summary_df.n_bets_p1, summary_df.n_bets_p2],
    label = ["Model 1 # Bets" "Model 2 # Bets"],
    xlabel = "Confidence Threshold % (One-Sided)",
    ylabel = "Total Number of Bets",
    title = "Number of Bets vs. Confidence",
    legend = :topright,
    lw = 2,
    markershape = :circle
)

# --- Combine and display ---
plot(p_roi, p_bets, layout = (2, 1), size = (800, 700))


using Plots

# Ensure your DataFrame is named summary_df
# summary_df = DataFrame(results) 

# X-axis will be the confidence level (e.g., 97.5%)
x_axis = summary_df.confidence .* 100

# --- Plot 1: ROI vs. Confidence ---
p_roi = plot(
    x_axis, 
    [summary_df.roi_p1, summary_df.roi_p2],
    label = ["Model 1 ROI" "Model 2 ROI"],
    xlabel = "Confidence Threshold % (One-Sided)",
    ylabel = "ROI %",
    title = "ROI vs. Bet Confidence Threshold",
    legend = :topleft,
    lw = 2, # line width
    markershape = :circle,
    xticks = x_axis
)

# --- Plot 2: Total Profit vs. Confidence ---
p_profit = plot(
    x_axis, 
    [summary_df.p1_profit, summary_df.p2_profit],
    label = ["Model 1 Profit" "Model 2 Profit"],
    xlabel = "Confidence Threshold % (One-Sided)",
    ylabel = "Total Profit (Units)",
    title = "Total Profit vs. Confidence",
    legend = :topleft,
    lw = 2,
    markershape = :circle,
    xticks = x_axis
)

# --- Plot 3: Number of Bets vs. Confidence ---
p_bets = plot(
    x_axis, 
    [summary_df.n_bets_p1, summary_df.n_bets_p2],
    label = ["Model 1 # Bets" "Model 2 # Bets"],
    xlabel = "Confidence Threshold % (One-Sided)",
    ylabel = "Total Number of Bets",
    title = "Number of Bets vs. Confidence",
    legend = :topright,
    lw = 2,
    markershape = :circle,
    yscale = :log10,  # Use a log scale for bets, as it changes a lot
    xticks = x_axis
)

# --- Combine and display ---
plot(p_roi, p_profit, p_bets, layout = (3, 1), size = (800, 1000), dpi=150)




# 1. Define the quantile and the corresponding column name
q = 0.025
p1_col = Symbol("p1_odds_q_$(replace(string(q), "." => "_"))") # :p1_odds_q_0_025

# 2. Filter the DataFrame
# The logic is:
# 1. The result must be known (!ismissing(row.winning))
# 2. The model's upper-bound odds must be less than the market odds
bets_p1_q025 = filter(row -> 
    !ismissing(row.winning) && row[p1_col] < row.market_odds, 
    q_comparison_df
)

# 3. (Optional) Select the most important columns to view
println(select(
    bets_p1_q025, 
    :match_id, 
    :model_market_name, 
    :market_odds, 
    p1_col,  # Model's upper-bound odds
    :p1_mean_odds,
    :winning
))

println("\nTotal bets found: $(nrow(bets_p1_q025))")


# Profit Function for Laying

"""
Profit Strategy: High-Confidence Lay Bet
Bet 1 unit if model_lower_odds > market_odds.
Profit is +1 if bet wins (outcome doesn't happen, winning=false)
Profit is -(market_odds - 1) if bet loses (outcome happens, winning=true)
"""
function calculate_profit_lay(model_lower_odds, market_odds, winning)
    # 1. Check for value
    if ismissing(winning) || model_lower_odds <= market_odds
        return 0.0 # No value, no bet
    end
    
    # 2. We placed a lay bet. Check the result.
    if winning == true
        # Bet lost (the outcome happened)
        return -(market_odds - 1.0)
    else
        # Bet won (the outcome didn't happen)
        return 1.0
    end
end

using DataFrames, Statistics

# We need symmetric quantiles. Your list already has these!
# e.g., 0.1 pairs with 0.9, 0.2 pairs with 0.8, etc.
# We'll test 90%, 80%, 70%, 60%, 50% confidence.
quantiles_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]

# (Assuming `q_comparison_df` and `calculate_profit_ci` still exist)

trading_results = []
for q_back in quantiles_to_test
    # 1. Define the symmetric quantiles
    q_lay = 1.0 - q_back
    
    # 2. Get the column names for this confidence level
    # (Rounding to handle 0.7 + 0.3 = 0.999... float issues)
    q_back_str = replace(string(round(q_back, digits=3)), "." => "_")
    q_lay_str = replace(string(round(q_lay, digits=3)), "." => "_")

    p1_back_col = Symbol("p1_odds_q_$(q_back_str)")
    p1_lay_col = Symbol("p1_odds_q_$(q_lay_str)")
    p2_back_col = Symbol("p2_odds_q_$(q_back_str)")
    p2_lay_col = Symbol("p2_odds_q_$(q_lay_str)")

    # 3. Calculate profit vectors for all bets
    profit_p1_back = calculate_profit_ci.(q_comparison_df[!, p1_back_col], q_comparison_df.market_odds, q_comparison_df.winning)
    profit_p1_lay  = calculate_profit_lay.(q_comparison_df[!, p1_lay_col], q_comparison_df.market_odds, q_comparison_df.winning)
    
    profit_p2_back = calculate_profit_ci.(q_comparison_df[!, p2_back_col], q_comparison_df.market_odds, q_comparison_df.winning)
    profit_p2_lay  = calculate_profit_lay.(q_comparison_df[!, p2_lay_col], q_comparison_df.market_odds, q_comparison_df.winning)
    
    # --- 4. Summarize Model 1 ---
    n_bets_p1_back = count(profit_p1_back .!= 0)
    n_bets_p1_lay  = count(profit_p1_lay .!= 0)
    total_profit_p1_back = sum(profit_p1_back)
    total_profit_p1_lay  = sum(profit_p1_lay)
    
    # Calculate ROI (Return on *Capital*, not just stake)
    # For laying, the "staked" amount is the liability
    liability_p1 = [w ? (m - 1) : 1.0 for (w, m) in zip(q_comparison_df.winning, q_comparison_df.market_odds)]
    staked_p1_back = n_bets_p1_back # 1 unit per bet
    staked_p1_lay  = sum(liability_p1[profit_p1_lay .!= 0]) # Liability is your stake
    
    roi_p1_back = staked_p1_back > 0 ? (total_profit_p1_back / staked_p1_back) * 100 : 0
    roi_p1_lay  = staked_p1_lay > 0 ? (total_profit_p1_lay / staked_p1_lay) * 100 : 0
    
    # --- 5. Summarize Model 2 ---
    n_bets_p2_back = count(profit_p2_back .!= 0)
    n_bets_p2_lay  = count(profit_p2_lay .!= 0)
    total_profit_p2_back = sum(profit_p2_back)
    total_profit_p2_lay  = sum(profit_p2_lay)
    
    liability_p2 = [w ? (m - 1) : 1.0 for (w, m) in zip(q_comparison_df.winning, q_comparison_df.market_odds)]
    staked_p2_back = n_bets_p2_back
    staked_p2_lay  = sum(liability_p2[profit_p2_lay .!= 0])
    
    roi_p2_back = staked_p2_back > 0 ? (total_profit_p2_back / staked_p2_back) * 100 : 0
    roi_p2_lay  = staked_p2_lay > 0 ? (total_profit_p2_lay / staked_p2_lay) * 100 : 0

    # --- 6. Store Combined Results ---
    push!(trading_results, (
        confidence = (1.0 - q_back * 2) * 100, # e.g., 1.0 - 0.1*2 = 80% CI
        n_bets_p1_back = n_bets_p1_back,
        n_bets_p1_lay = n_bets_p1_lay,
        total_profit_p1 = total_profit_p1_back + total_profit_p1_lay,
        roi_p1_back = roi_p1_back,
        roi_p1_lay = roi_p1_lay,
        
        n_bets_p2_back = n_bets_p2_back,
        n_bets_p2_lay = n_bets_p2_lay,
        total_profit_p2 = total_profit_p2_back + total_profit_p2_lay,
        roi_p2_back = roi_p2_back,
        roi_p2_lay = roi_p2_lay,
    ))
end

trading_summary_df = DataFrame(trading_results)
println(trading_summary_df)
