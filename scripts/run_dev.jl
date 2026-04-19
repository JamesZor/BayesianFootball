# scripts/run_dev.jl

# Activate the project environment
using Pkg
Pkg.activate(".")

# Load Revise.jl for automatic code reloading
using Revise

# Load your package
using BayesianFootball

# --- Example Usage of the Data Module ---
# Define the path to your data (adjust if necessary)
# You might need to go up one level from the scripts directory
data_path = BayesianFootball.Data.DataPaths.scotland

# Create DataFiles and DataStore objects
try
    data_files = BayesianFootball.Data.DataFiles(data_path)
    println("Successfully created DataFiles object.")
    
    data_store = BayesianFootball.Data.DataStore(data_files)
    println("Successfully created DataStore object.")
    
    # Display the first few rows of the matches DataFrame
    println("\n--- Sample of Matches Data ---")
    println(first(data_store.matches, 5))
    
catch e
    println("An error occurred: ", e)
    println("\nPlease ensure the data path is correct and the CSV files exist.")
end



# --- Example Usage of the Feature Module ---
data_store = BayesianFootball.Data.load_default_datastore()
f = BayesianFootball.Features.create_features(data_store)


# --- Example Usage of the pregame model Module ---

model = BayesianFootball.Models.PreGame.PregameModel(
    BayesianFootball.Models.PreGame.PoissonGoal(),
    BayesianFootball.Models.PreGame.AR1(),
    true
)

model_1 = BayesianFootball.Models.PreGame.PregameModel(
    BayesianFootball.Models.PreGame.PoissonGoal(),
    BayesianFootball.Models.PreGame.Static(),
    true
)


# --- 2. Define a Model ---
static_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.PoissonGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)


feature_set = f
# --- 3. Build the Turing Model ---
# This calls our API to create the actual @model block
turing_model = BayesianFootball.Models.PreGame.build_turing_model(static_model, feature_set)
println("✅ Turing model built successfully.")

# --- 4. Sample from the Model ---
using Turing
# We use the NUTS sampler, a standard choice for this kind of model.
# We'll run it for 1000 iterations: 200 for warmup and 800 for sampling.
chain = sample(turing_model, NUTS(0.65), 10)
println("✅ Sampling complete!")

# --- 5. Inspect the Results ---
println("\n--- MCMC Chain Summary ---")
# Printing the 'chain' object gives a nice summary of the posterior distributions
# for all the parameters in our model.
println(chain)


# This now works out-of-the-box
using Turing
data_store = BayesianFootball.Data.load_default_datastore()
feature_set = BayesianFootball.Features.create_features(data_store)

# --- 1. Define Models ---
static_poisson_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.PoissonGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)

# --- 2. Build and Sample ---
turing_model_1 = BayesianFootball.Models.PreGame.build_turing_model(static_poisson_model, feature_set)
chain_1 = sample(turing_model_1, NUTS(), 10)


static_nb_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.NegativeBinomialGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)

# --- 2. Build and Sample ---
turing_model_2 = BayesianFootball.Models.PreGame.build_turing_model(static_nb_model, feature_set)
chain_1 = sample(turing_model_2, NUTS(), 10)

dynamic_poisson_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.PoissonGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)
dynamic_nb_model = PregameModel(NegativeBinomial(), AR1(), true)
turing_model_2 = build_turing_model(dynamic_nb_model, feature_set)
chain_2 = sample(turing_model_2, NUTS(), 100)





#######
using Revise

# Load your package
using BayesianFootball
using Turing

# --- Prepare Data (this part is unchanged) ---
println("--- Loading data and features ---")
data_store = BayesianFootball.Data.load_default_datastore()
feature_set = BayesianFootball.Features.create_features(data_store)

# --- NEW, SIMPLER WORKFLOW ---
println("\n--- Running the Static Poisson Model ---")

# 1. Instantiate the concrete model struct
static_poisson = BayesianFootball.Models.PreGame.StaticPoisson()

# 2. Build the Turing model by calling the dispatched method
turing_model_1 = BayesianFootball.Models.PreGame.build_turing_model(static_poisson, feature_set)
println("✅ Static Poisson model built.")

# 3. Sample
chain_1 = sample(turing_model_1, NUTS(), 10)
println(chain_1)


println("\n--- Running the Static Simplex Poisson Model ---")

# 1. Instantiate the new model struct
static_simplex_poisson = BayesianFootball.Models.PreGame.StaticSimplexPoisson()

# 2. Build the model
turing_model_4 = BayesianFootball.Models.PreGame.build_turing_model(static_simplex_poisson, feature_set)
println("✅ Static Simplex Poisson model built.")

# 3. Sample
chain_4 = sample(turing_model_4, NUTS(), 10)
println(chain_4)


println("\n--- Running the Hierarchical Simplex Poisson Model ---")

# 1. Instantiate the new model struct
simplex_poisson = BayesianFootball.Models.PreGame.HierarchicalSimplexPoisson()

# 2. Build the model
turing_model_3 = BayesianFootball.Models.PreGame.build_turing_model(simplex_poisson, feature_set)
println("✅ Hierarchical Simplex Poisson model built.")

# 3. Sample
chain_3 = sample(turing_model_3, NUTS(), 10)
println(chain_3)


#= 
 ----------  sampling  ------------------
=#

# Activate the project environment

# Load Revise.jl for automatic code reloading
using Revise

# Load your package
using BayesianFootball
using Turing
using DataFrames
using StatsPlots

# --- Performance Libraries ---
# It's best practice to set these once at the start of your script
# using ReverseDiff, Memoization
# Turing.setadbackend(:reversediff)
# Turing.setrdcache(true)


# --- Prepare Data (this part is unchanged) ---
println("\n--- Loading data and features ---")
data_store = BayesianFootball.Data.load_default_datastore()

filter!(row-> row.season=="24/25", data_store.matches)
filter!(row -> row.tournament_id in [54, 55], data_store.matches)

feature_set = BayesianFootball.Features.create_features(data_store)


# --- NEW, UNIFIED TRAINING WORKFLOW ---

# 1. Choose a model definition
# model_to_train = BayesianFootball.Models.PreGame.StaticPoisson()

static_poisson = BayesianFootball.Models.PreGame.StaticPoisson()
turing_model_1 = BayesianFootball.Models.PreGame.build_turing_model(static_poisson, feature_set)

println("\n--- Training Model: $(typeof(static_poisson)) ---")

# 2. Choose a training method
# --- Option A: Full MCMC Sampling with NUTS ---
training_method_nuts = BayesianFootball.Sampling.NUTSMethod(n_samples=2_000, n_chains=2, n_warmup=20)
chain_result = BayesianFootball.Sampling.train(turing_model_1, training_method_nuts)

println("\n--- NUTS Sampling Complete ---")
println(chain_result)


plot( chain_result, [Symbol("log_α_raw[13]")])

plot( chain_result, [Symbol("log_α[13]")])
plot( chain_result, [Symbol("home_adv")])




static_simplex_poisson = BayesianFootball.Models.PreGame.StaticSimplexPoisson()
turing_model_4 = BayesianFootball.Models.PreGame.build_turing_model(static_simplex_poisson, feature_set)
training_method_nuts = BayesianFootball.Sampling.NUTSMethod(n_samples=2_000, n_chains=2, n_warmup=20)
chain_result_1 = BayesianFootball.Sampling.train(turing_model_4, training_method_nuts)

describe(chain_result_1)

plot( chain_result_1, [Symbol("log_α[20]")])
plot( chain_result_1, [Symbol("log_β[19]")])


m = data_store.matches[283, :]

ht = feature_set.team_map[m.home_team]
at = feature_set.team_map[m.away_team]

@model function static_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors ---
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- Identifiability Constraint ---
    log_α := log_α_raw .- mean(log_α_raw) # using := to added to track vars,
    log_β := log_β_raw .- mean(log_β_raw)

    # --- Calculate Goal Rates ---
    log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
    log_μs = log_α[away_ids] .+ log_β[home_ids]

    if !ismissing(home_goals)
        # TRAINING CASE: Loop over observed data to calculate log-likelihood
        for i in eachindex(home_goals)
            home_goals[i] ~ LogPoisson(log_λs[i])
            away_goals[i] ~ LogPoisson(log_μs[i])
        end
    else
        # PREDICTION CASE: Sample new goal values ⚽️
        # Since home_ids is a single-element array, we access it with [1]
        home_goals ~ LogPoisson(log_λs[1])
        away_goals ~ LogPoisson(log_μs[1])
    end
  
    return nothing
  
end

@model function static_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors ---
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- Identifiability Constraint ---
    log_α := log_α_raw .- mean(log_α_raw)
    log_β := log_β_raw .- mean(log_β_raw)

    # --- Calculate Goal Rates ---
    log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
    log_μs = log_α[away_ids] .+ log_β[home_ids]

    if !ismissing(home_goals)
        # --- TRAINING CASE ---
        for i in eachindex(home_goals)
            home_goals[i] ~ LogPoisson(log_λs[i])
            away_goals[i] ~ LogPoisson(log_μs[i])
        end
    else
        # --- PREDICTION CASE ---
        predicted_home_goals ~ LogPoisson(log_λs[1])
        predicted_away_goals ~ LogPoisson(log_μs[1])

        # 2. Calculate derived quantities from the sampled goals
        #    and track them using `:=`
        total_goals := predicted_home_goals + predicted_away_goals

        # Match outcome probabilities (will be 1 for win, 0 otherwise)
        home_win := predicted_home_goals > predicted_away_goals
        draw     := predicted_home_goals == predicted_away_goals
        away_win := predicted_home_goals < predicted_away_goals

        # Over/Under probabilities
        over_05 := total_goals > 0.5
        over_15 := total_goals > 1.5
        over_25 := total_goals > 2.5
        over_35 := total_goals > 3.5
    end

    return nothing
end






prediction_model = static_poisson_model(feature_set.n_teams, [ht], [at], missing, missing)
prediction_chain = predict(prediction_model, chain_result)

using StatsPlots
plot(prediction_chain, [:predicted_home_goals, :predicted_away_goals])


# You can now analyze the predicted goals
describe(prediction_chain)

# Extract the chains into vectors
home_goals_samples = vec(prediction_chain[:predicted_home_goals]);
away_goals_samples = vec(prediction_chain[:predicted_away_goals]);


using StatsBase # Make sure you have this package installed

# Get the total number of posterior samples
total_samples = length(home_goals_samples)

# Count occurrences of each goal value
home_goal_counts = countmap(home_goals_samples)
away_goal_counts = countmap(away_goals_samples)

# --- Calculate and display probabilities for home goals ---
println("--- Home Goal Probabilities ---")
for goals in sort(collect(keys(home_goal_counts)))
    count = home_goal_counts[goals]
    probability = count / total_samples
    println("P(Home Goals = $(Int(goals))): $(round(probability * 100, digits=2))%")
end

# --- Calculate and display probabilities for away goals ---
println("\n--- Away Goal Probabilities ---")
for goals in sort(collect(keys(away_goal_counts)))
    count = away_goal_counts[goals]
    probability = count / total_samples
    println("P(Away Goals = $(Int(goals))): $(round(probability * 100, digits=2))%")
end


goals = sort(collect(keys(home_goal_counts)))
probabilities = [home_goal_counts[g] / total_samples for g in goals]

bar(goals, probabilities,
    title="Posterior Predictive Distribution for Home Goals",
    xlabel="Number of Goals",
    ylabel="Probability",
    legend=false,
    xticks=goals
)


# This is the full posterior predictive distribution for home goals
home_goals_dist = vec(prediction_chain[:predicted_home_goals])

# And for away goals
away_goals_dist = vec(prediction_chain[:predicted_away_goals])

# Look at the first 10 simulated outcomes
println("First 10 simulated home goals: ", Int.(home_goals_dist[1:10]))
println("First 10 simulated away goals: ", Int.(away_goals_dist[1:10]))


using StatsBase

# 3. Now, your plotting code will work correctly 📊
bar(
    countmap(home_goals_dist),
    title="Posterior Predictive Distribution for Home Goals",
    xlabel="Number of Goals",
    ylabel="Frequency",
    legend=false
)


using StatsBase

# Start with the raw goal samples
home_goals_dist = vec(prediction_chain[:predicted_home_goals])
away_goals_dist = vec(prediction_chain[:predicted_away_goals])
total_samples = length(home_goals_dist)

# 1. Get the frequency counts for each outcome
home_goal_counts = countmap(home_goals_dist)
away_goal_counts = countmap(away_goals_dist)

# 2. Normalize the counts to get the PMF (a dictionary of Goal => Probability)
home_pmf = Dict(goal => count / total_samples for (goal, count) in home_goal_counts)
away_pmf = Dict(goal => count / total_samples for (goal, count) in away_goal_counts)

println("--- Home Goal PMF ---")
display(home_pmf)
println("\n--- Away Goal PMF ---")
display(away_pmf)


# Initialize probabilities for each market
prob_home_win = 0.0
prob_draw = 0.0
prob_away_win = 0.0
prob_over_25 = 0.0

# Iterate through every possible scoreline defined by our PMFs
for (h, prob_h) in home_pmf
    for (a, prob_a) in away_pmf
        # Calculate the joint probability for this specific scoreline (h, a)
        joint_prob = prob_h * prob_a

        # Add this probability to the appropriate market's total
        if h > a
            prob_home_win += joint_prob
        elseif h == a
            prob_draw += joint_prob
        else # h < a
            prob_away_win += joint_prob
        end

        if h + a > 2.5
            prob_over_25 += joint_prob
        end
    end
end


println("--- Final Market Probabilities (from PMFs) ---")
println("Home Win:      $(round(prob_home_win * 100, digits=1))%")
println("Draw:          $(round(prob_draw * 100, digits=1))%")
println("Away Win:      $(round(prob_away_win * 100, digits=1))%")
println("Over 2.5 Goals: $(round(prob_over_25 * 100, digits=1))%")


println("Home Win:      $(round(prob_home_win * 100, digits=1))%")
println("Draw:          $(round(prob_draw * 100, digits=1))%")
println("Away Win:      $(round(prob_away_win * 100, digits=1))%")
println("Over 2.5 Goals: $(round(prob_over_25 * 100, digits=1))%")


# --- Initialize all market probabilities ---
model_probs = Dict{String, Float64}()

# Base outcomes
model_probs["home_win"] = 0.0
model_probs["draw"] = 0.0
model_probs["away_win"] = 0.0

# Over/Under markets
model_probs["over_05"] = 0.0
model_probs["over_15"] = 0.0
model_probs["over_25"] = 0.0
model_probs["over_35"] = 0.0
model_probs["over_45"] = 0.0
model_probs["over_55"] = 0.0
model_probs["over_65"] = 0.0
model_probs["over_75"] = 0.0
model_probs["over_85"] = 0.0


# Other markets
model_probs["btts_yes"] = 0.0

# --- Iterate through the joint distribution to calculate probabilities ---
for (h, prob_h) in home_pmf, (a, prob_a) in away_pmf
    joint_prob = prob_h * prob_a
    total_goals = h + a

    # 1X2
    if h > a; model_probs["home_win"] += joint_prob; end
    if h == a; model_probs["draw"] += joint_prob; end
    if h < a; model_probs["away_win"] += joint_prob; end

    # BTTS
    if h > 0 && a > 0; model_probs["btts_yes"] += joint_prob; end

    # Over/Under
    if total_goals > 0.5; model_probs["over_05"] += joint_prob; end
    if total_goals > 1.5; model_probs["over_15"] += joint_prob; end
    if total_goals > 2.5; model_probs["over_25"] += joint_prob; end
    if total_goals > 3.5; model_probs["over_35"] += joint_prob; end
    if total_goals > 4.5; model_probs["over_45"] += joint_prob; end
    if total_goals > 5.5; model_probs["over_55"] += joint_prob; end
    if total_goals > 6.5; model_probs["over_65"] += joint_prob; end
    if total_goals > 7.5; model_probs["over_75"] += joint_prob; end
    if total_goals > 8.5; model_probs["over_85"] += joint_prob; end
end

# --- Calculate derived probabilities ---
# Double Chance
model_probs["dc_1X"] = model_probs["home_win"] + model_probs["draw"]
model_probs["dc_X2"] = model_probs["away_win"] + model_probs["draw"]
model_probs["dc_12"] = model_probs["home_win"] + model_probs["away_win"]

# Draw No Bet (and Asian Handicap 0)
model_probs["dnb_1"] = model_probs["home_win"] / (1 - model_probs["draw"])
model_probs["dnb_2"] = model_probs["away_win"] / (1 - model_probs["draw"])

# BTTS No
model_probs["btts_no"] = 1 - model_probs["btts_yes"]


o = filter(row -> row.match_id==m.match_id, data_store.odds)


# Add new columns to your DataFrame, initialized as `missing`
o.model_prob = Vector{Union{Missing, Float64}}(missing, nrow(o))
o.value = Vector{Union{Missing, Float64}}(missing, nrow(o))

# Loop through each row to find the matching probability and calculate value
for r in eachrow(o)
    prob = missing
    mg, cn, cg = r.market_group, r.choice_name, r.choice_group

    if mg == "1X2"
        if cn == "1"; prob = model_probs["home_win"]; end
        if cn == "X"; prob = model_probs["draw"]; end
        if cn == "2"; prob = model_probs["away_win"]; end
    elseif mg == "Double chance"
        if cn == "1X"; prob = model_probs["dc_1X"]; end
        if cn == "12"; prob = model_probs["dc_12"]; end
        if cn == "X2"; prob = model_probs["dc_X2"]; end
    elseif mg == "Draw no bet" || (mg == "Asian Handicap" && contains(cn, "(0)"))
        if cn == "1" || contains(cn, "Hamilton"); prob = model_probs["dnb_1"]; end
        if cn == "2" || contains(cn, "Raith"); prob = model_probs["dnb_2"]; end
    elseif mg == "Both teams to score"
        if cn == "Yes"; prob = model_probs["btts_yes"]; end
        if cn == "No"; prob = model_probs["btts_no"]; end
    elseif mg == "Match goals"
        line = string(cg) # Convert line like 2.5 to "25" for key
        line_key = "over_" * replace(line, "." => "")[1:2]
        if cn == "Over"
            prob = model_probs[line_key]
        elseif cn == "Under"
            prob = 1 - model_probs[line_key]
        end
    end

    # If we found a matching probability, calculate value
    if !ismissing(prob)
        r.model_prob = prob
        r.value = (prob * r.decimal_odds) - 1
    end
end

# Filter for bets we could price and sort by value
value_bets = filter(row -> !ismissing(row.value), o)
sort!(value_bets, :value, rev=true)

value_bets.model_odds = round.( 1 ./ value_bets.model_prob, digits=2)

println(value_bets[!, [:market_group, :market_name, :choice_name, :choice_group, :winning, :decimal_odds, :model_odds, :model_prob, :value]])

# --- Option B: Fast Approximate Inference with ADVI ---
# training_method_advi = BayesianFootball.Training.ADVIMethod(n_iterations=30000)
# advi_result = BayesianFootball.Training.train(model_to_train, model_data, training_method_advi)
# println("\n--- ADVI Complete ---")
# # We can extract samples from the resulting approximation
# advi_samples = rand(advi_result, 1000)
# println("Approximated mean for home_adv: ", mean(advi_samples, dims=1)[1,1]) # Example


# --- Option C: Point Estimate with MAP ---
# training_method_map = BayesianFootball.Training.MAPMethod()
# map_result = BayesianFootball.Training.train(model_to_train, model_data, training_method_map)
# println("\n--- MAP Estimation Complete ---")
# println(map_result)
