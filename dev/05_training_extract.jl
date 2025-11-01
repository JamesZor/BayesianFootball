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

results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)

# save and load 
using JLD2
JLD2.save_object("training_results.jld2", results)
results = JLD2.load_object("training_results.jld2")



### extraction 
using Statistics

# get chain set 1 
r = results[1][1]
# get matches for 2 
mp = filter( row -> row.split_col == 2, ds.matches)


mp[1, :]

filter( row -> row.match_id==only(mp[1,[:match_id]]), ds.odds) 


h_id = vocabulary.mappings[:team_map][only(mp[1, [:home_team]])]
a_id = vocabulary.mappings[:team_map][only(mp[1, [:away_team]])]

a_h = vec(r[Symbol("log_α[$h_id]")]);
b_h = vec(r[Symbol("log_β[$h_id]")]);
a_a = vec(r[Symbol("log_α[$a_id]")]);
b_a = vec(r[Symbol("log_β[$a_id]")]);
h = vec(r[Symbol("home_adv")]);


l1 = a_h .+ b_a .+ h ;
l2 = a_a .+ b_h ;


# These are your chains from the model
l1 = vec(r[Symbol("log_α[$h_id]")]) .+ vec(r[Symbol("log_β[$a_id]")]) .+ vec(r[Symbol("home_adv")]);
l2 = vec(r[Symbol("log_α[$a_id]")]) .+ vec(r[Symbol("log_β[$h_id]")]);

# λs is a chain of 'n_samples' home goal rates
λs = exp.(l1);

# μs is a chain of 'n_samples' away goal rates
μs = exp.(l2);

mean(l1)
mean(l2)

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
