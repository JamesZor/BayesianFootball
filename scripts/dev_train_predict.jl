
# Load your package
using BayesianFootball
using Turing
using DataFrames 
using StatsPlots, StatsBase


data_store = BayesianFootball.Data.load_default_datastore()

# reduce data size for testing.
filter!(row-> row.season=="24/25", data_store.matches)
filter!(row -> row.tournament_id in [54, 55], data_store.matches)

# get the features 
feature_set = BayesianFootball.Features.create_features(data_store)


## Models 

# model 1 simple 
static_poisson = BayesianFootball.Models.PreGame.StaticPoisson()
turing_model_1 = BayesianFootball.Models.PreGame.build_turing_model(static_poisson, feature_set)

# model 2 simplex - 

static_simplex_poisson = BayesianFootball.Models.PreGame.StaticSimplexPoisson()
turing_model_2 = BayesianFootball.Models.PreGame.build_turing_model(static_simplex_poisson, feature_set)

## training 

training_method_nuts = BayesianFootball.Sampling.NUTSMethod(n_samples=2_000, n_chains=4, n_warmup=500)


chain_simple = BayesianFootball.Sampling.train(turing_model_1, training_method_nuts)
chain_simplex = BayesianFootball.Sampling.train(turing_model_2, training_method_nuts)


## Plots 
# forest 

# Assuming 'chain_simplex' is your Chains object

# 1. Get all parameter names from the chain object
all_params = names(chain_simplex)

# 2. Filter this list to keep only the names starting with "log_α" or "log_β"
params_to_plot = filter(
    name -> startswith(string(name), "log_α") || startswith(string(name), "log_β"),
    all_params
)


params_to_plot_alpha = filter(
    name -> startswith(string(name), "log_α"),
    all_params
)

params_to_plot_beta = filter(
    name -> startswith(string(name), "log_β") ,
    all_params
)



# 3. Create the forest plot using this generated list of parameters
p = forestplot(chain_simplex, params_to_plot_alpha)

###
# --- Step 1: Create the reverse mapping from ID to team name ---
# The original map is String => Int. We need Int => String for lookups.
reverse_team_map = Dict(id => name for (name, id) in feature_set.team_map)

# --- Step 2: Generate the custom labels for the plot ---
# We will create a new vector of strings to use as labels.
alpha_labels = map(params_to_plot_alpha) do param_symbol
    s = string(param_symbol)
    # Use a regular expression to find the number in brackets, e.g., in "log_α[14]"
    m = match(r"\[(\d+)\]", s)

    if m !== nothing
        # If we found a number, parse it to an Int
        id = parse(Int, m.captures[1])
        # Look up the team name in our reverse map
        return get(reverse_team_map, id, s) # Fallback to original string if ID not found
    else
        # If no number (e.g., for :log_α_scale), just use the original name
        return s
    end
end

# --- Step 3: Create the forest plot with custom labels ---
# The yticks argument takes a tuple: (tick_positions, tick_labels)
# We must reverse the labels because the plot draws from the top down.
p_alpha = forestplot(
    chain_simplex,
    params_to_plot_alpha,
    yticks = (1:length(alpha_labels), reverse(alpha_labels)),
    title = "Team Attack Strengths (log α)"
)

# You can do the exact same for the beta parameters
beta_labels = map(params_to_plot_beta) do param_symbol
    s = string(param_symbol)
    m = match(r"\[(\d+)\]", s)
    if m !== nothing
        id = parse(Int, m.captures[1])
        return get(reverse_team_map, id, s)
    else
        return s
    end
end

p_beta = forestplot(
    chain_simplex,
    params_to_plot_beta,
    yticks = (1:length(beta_labels), reverse(beta_labels)),
    title = "Team Defence Strengths (log β)"
)
# 1. Get the summary statistics for the chain
summary_stats = summarystats(chain_simplex)

# 2. Sort the alpha parameters by their median value (descending)
# Note: We filter out :log_α_scale for sorting purposes
alpha_params_only = filter(p -> occursin("[", string(p)), params_to_plot_alpha)
sorted_alpha_params = sort(
    alpha_params_only,
    by = p -> summary_stats[p, :mean], # or :median
    rev = true # Highest strength at the top
)

# 3. Generate labels in the new sorted order
sorted_alpha_labels = map(sorted_alpha_params) do param_symbol
    s = string(param_symbol)
    m = match(r"\[(\d+)\]", s)
    id = parse(Int, m.captures[1])
    return reverse_team_map[id]
end

# 4. Plot the sorted results
p_sorted = forestplot(
    chain_simplex,
    sorted_alpha_params,
    # yticks = (1:length(sorted_alpha_labels), reverse(sorted_alpha_labels)),
    title = "Team Attack Strengths (Sorted)"
)

summary_stats = summarystats(chain_simple)

# 2. Sort the alpha parameters by their median value (descending)
# Note: We filter out :log_α_scale for sorting purposes
alpha_params_only = filter(p -> occursin("[", string(p)), params_to_plot_alpha)
sorted_alpha_params = sort(
    alpha_params_only,
    by = p -> summary_stats[p, :mean], # or :median
    rev = true # Highest strength at the top
)

# 3. Generate labels in the new sorted order
sorted_alpha_labels = map(sorted_alpha_params) do param_symbol
    s = string(param_symbol)
    m = match(r"\[(\d+)\]", s)
    id = parse(Int, m.captures[1])
    return reverse_team_map[id]
end

# 4. Plot the sorted results
p_sorted1 = forestplot(
    chain_simple,
    sorted_alpha_params,
    # yticks = (1:length(sorted_alpha_labels), reverse(sorted_alpha_labels)),
    title = "Team Attack Strengths (Sorted)"
)


#= 
--- predictions
=#

#### model 1 
# v1 
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

# v2 
@model function static_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors ---
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- Identifiability Constraint ---
    log_α := log_α_raw .- mean(log_α_raw)
    log_β := log_β_raw .- mean(log_β_raw)

    # --- Calculate Goal Rates (this works for any number of games) ---
    log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
    log_μs = log_α[away_ids] .+ log_β[home_ids]

    if !ismissing(home_goals)
        # TRAINING CASE: This part is already correct for multiple games
        for i in eachindex(home_goals)
            home_goals[i] ~ LogPoisson(log_λs[i])
            away_goals[i] ~ LogPoisson(log_μs[i])
        end
    else
        home_goals = similar(home_ids, Int)
        away_goals = similar(away_ids, Int)
        
        # Loop over each game we want to predict
        for i in eachindex(home_ids)
            home_goals[i] ~ LogPoisson(log_λs[i])
            away_goals[i] ~ LogPoisson(log_μs[i])
        end
    end
   
    return nothing
end

# v3 
# not working 
@model function static_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors ---
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- Identifiability Constraint ---
    log_α := log_α_raw .- mean(log_α_raw)
    log_β := log_β_raw .- mean(log_β_raw)

    # --- Calculate Goal Rates (this works for any number of games) ---
    log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
    log_μs = log_α[away_ids] .+ log_β[home_ids]

    if !ismissing(home_goals)
        # TRAINING CASE: This part is already correct for multiple games
        for i in eachindex(home_goals)
            home_goals[i] ~ LogPoisson(log_λs[i])
            away_goals[i] ~ LogPoisson(log_μs[i])
        end
    else
# PREDICTION CASE: Vectorized version ⚽️
        # This is much faster than a loop!
        home_goals .~ LogPoisson.(log_λs)
        away_goals .~ LogPoisson.(log_μs)
    end
   
    return nothing
end




### model 2 
@model function static_simplex_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors ---
      log_α_scale ~ Normal(0, 10)
      log_β_scale ~ Normal(0, 10)
      home_adv ~ Normal(log(1.3), 0.2)

      # --- Non-Centered Parameterization for Identifiability ---
      # Sample n-1 raw parameters from a standard normal distribution
      # These are independent of the scale! This is the key.
      α_raw_offsets ~ MvNormal(n_teams - 1, 1.0)
      β_raw_offsets ~ MvNormal(n_teams - 1, 1.0)

      # Deterministically create the full n-team vectors
      # that sum to zero.
      α_offsets = vcat(α_raw_offsets, -sum(α_raw_offsets))
      β_offsets = vcat(β_raw_offsets, -sum(β_raw_offsets))

      # Apply the scale and mean AFTER sampling.
      # This transformation happens "outside" the sampler's main work.
      log_α := log_α_scale .* α_offsets
      log_β := log_β_scale .* β_offsets


      # --- Calculate Goal Rates ---
      log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
      log_μs = log_α[away_ids] .+ log_β[home_ids]

      # --- Likelihood ---
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



a = rand(1:380)
m = data_store.matches[a, :]

ht = feature_set.team_map[m.home_team]
at = feature_set.team_map[m.away_team]

# model 1 
prediction_model = static_poisson_model(feature_set.n_teams, [ht], [at], missing, missing)
prediction_chain = predict(prediction_model, chain_simple)
prediction_chain

# 1. Select the games you want to predict (e.g., the first 5)
games_to_predict = data_store.matches

# 2. Get the corresponding team IDs from the feature_set map
# We use a list comprehension to get an array of IDs
home_team_ids = [feature_set.team_map[team] for team in games_to_predict.home_team];
away_team_ids = [feature_set.team_map[team] for team in games_to_predict.away_team];

# 3. Instantiate the new model with the arrays of team IDs
prediction_model_multi = static_poisson_model(
    feature_set.n_teams, 
    home_team_ids,         # Pass the array of home IDs
    away_team_ids,         # Pass the array of away IDs
    missing, 
    missing
)

# 4. Run the prediction
# This uses your existing trained chain (chain_simple)
prediction_chain_multi = predict(prediction_model_multi, chain_simple;  parallel=true)



# model 2 

prediction_model_2 = static_simplex_poisson_model(feature_set.n_teams, [ht], [at], missing, missing)
prediction_chain_2 = predict(prediction_model_2, chain_simplex)
prediction_chain_2

















#####



using MCMCChains
using Statistics # For mean and quantile

"""
Calculates the posterior predictive distribution for various betting markets.

Arguments:
- `prediction_chain`: A `Chains` object from `Turing.predict`. Must contain
  `:home_goals` and `:away_goals`.

Returns:
- A `Dict{String, Vector{Float64}}` where each vector contains the outcome
  (0.0 for loss, 1.0 for win) for each sample in the prediction chain.
"""
function calculate_betting_distributions(prediction_chain::Chains)
    # --- Extract the raw goal predictions for each sample ---
    home_goals = vec(prediction_chain[:home_goals])
    away_goals = vec(prediction_chain[:away_goals])
    total_goals = home_goals .+ away_goals

    # --- Initialize the results dictionary ---
    market_dists = Dict{String, Vector{Float64}}()

    # --- Calculate distributions using vectorized operations ---

    # 1X2 Market (Home/Draw/Away)
    # The dot performs an element-wise comparison. Float64.() converts true/false to 1.0/0.0
    market_dists["home_win"] = Float64.(home_goals .> away_goals)
    market_dists["draw"] = Float64.(home_goals .== away_goals)
    market_dists["away_win"] = Float64.(home_goals .< away_goals)

    # Over/Under Markets
    market_dists["over_05"] = Float64.(total_goals .> 0.5)
    market_dists["over_15"] = Float64.(total_goals .> 1.5)
    market_dists["over_25"] = Float64.(total_goals .> 2.5)
    market_dists["over_35"] = Float64.(total_goals .> 3.5)
    market_dists["over_45"] = Float64.(total_goals .> 4.5)
    
    # Both Teams To Score (BTTS)
    # .& is the element-wise AND operator
    market_dists["btts_yes"] = Float64.((home_goals .> 0) .& (away_goals .> 0))
    market_dists["btts_no"] = 1.0 .- market_dists["btts_yes"]

    # Double Chance (DC)
    market_dists["dc_1X"] = market_dists["home_win"] .+ market_dists["draw"]
    market_dists["dc_X2"] = market_dists["draw"] .+ market_dists["away_win"]
    market_dists["dc_12"] = market_dists["home_win"] .+ market_dists["away_win"]

    return market_dists
end





market_dists = calculate_betting_distributions(prediction_chain)
over_25_dist = market_dists["over_25"];
# The mean of this 0/1 vector is your posterior probability
prob_over_25 = mean(over_25_dist)

# The implied odds are the inverse of the probability
odds_over_25 = 1 / prob_over_25

println("--- Over 2.5 Goals Analysis ---")
println("Posterior Mean Probability: $(round(prob_over_25; digits=3))")
println("Implied Fair Odds: $(round(odds_over_25; digits=3))")


###

market_dists_2 = calculate_betting_distributions(prediction_chain_2)
over_25_dist_2 = market_dists_2["over_25"];
# The mean of this 0/1 vector is your posterior probability
prob_over_25_2 = mean(over_25_dist_2)

# The implied odds are the inverse of the probability
odds_over_25_2 = 1 / prob_over_25_2

println("--- Over 2.5 Goals Analysis ---")
println("Posterior Mean Probability: $(round(prob_over_25_2; digits=3))")
println("Implied Fair Odds: $(round(odds_over_25_2; digits=3))")



### v2 
using MCMCChains
using Statistics
using DataFrames

"""
Calculates the posterior predictive distribution for various betting markets.
This version correctly computes the distribution for conditional markets like DNB.

Arguments:
- `prediction_chain`: A `Chains` object from `Turing.predict`.

Returns:
- A `Dict{String, Vector{Float64}}` where each vector contains the outcome
  for each sample in the prediction chain.
"""
function calculate_betting_distributions(prediction_chain::Chains)
    # --- Extract the raw goal predictions for each sample ---
    home_goals = vec(prediction_chain[:home_goals])
    away_goals = vec(prediction_chain[:away_goals])

    # --- Calculate base outcomes as distributions of 0s and 1s ---
    home_win_dist = Float64.(home_goals .> away_goals)
    draw_dist = Float64.(home_goals .== away_goals)
    away_win_dist = Float64.(home_goals .< away_goals)
    
    # --- Initialize the results dictionary ---
    market_dists = Dict{String, Vector{Float64}}()

    # --- Store base outcomes ---
    market_dists["home_win"] = home_win_dist
    market_dists["draw"] = draw_dist
    market_dists["away_win"] = away_win_dist

    # --- Calculate and store derived markets ---
    total_goals_dist = home_goals .+ away_goals
    market_dists["over_05"] = Float64.(total_goals_dist .> 0.5)
    market_dists["over_15"] = Float64.(total_goals_dist .> 1.5)
    market_dists["over_25"] = Float64.(total_goals_dist .> 2.5)
    market_dists["over_35"] = Float64.(total_goals_dist .> 3.5)
    
    market_dists["btts_yes"] = Float64.((home_goals .> 0) .& (away_goals .> 0))
    market_dists["btts_no"] = 1.0 .- market_dists["btts_yes"]
    
    market_dists["dc_1X"] = home_win_dist .+ draw_dist
    market_dists["dc_X2"] = draw_dist .+ away_win_dist
    
    # --- Correctly calculate DNB as a conditional probability distribution ---
    # P(Home Win | Not a Draw) for each sample
    # We replace NaNs (from 0/0) with 0, as it implies no non-draw outcomes happened
    prob_not_draw_dist = 1.0 .- draw_dist
    dnb_1_dist = home_win_dist ./ prob_not_draw_dist
    dnb_2_dist = away_win_dist ./ prob_not_draw_dist
    
    market_dists["dnb_1"] = replace(dnb_1_dist, NaN => 0.0)
    market_dists["dnb_2"] = replace(dnb_2_dist, NaN => 0.0)

    return market_dists
end




"""
Compares betting market predictions from multiple models.

Arguments:
- `model_predictions`: A Dictionary mapping model names (String) to their
  prediction `Chains` objects.
- `ci_level`: The credible interval level (e.g., 0.90 for a 90% CI).

Returns:
- A `DataFrame` summarizing the probabilities and odds for each market
  from each model.
"""
function compare_model_predictions(
    model_predictions;
    ci_level::Float64 = 0.90
)
    # Define the quantiles for the credible interval
    lower_q = (1.0 - ci_level) / 2.0
    upper_q = 1.0 - lower_q

    # Prepare arrays to build the long-format DataFrame
    markets, models, mean_probs, prob_cis, mean_odds, odds_cis = [], [], [], [], [], []

    # Iterate over each model provided in the dictionary
    for (model_name, pred_chain) in model_predictions
        println("Processing model: $model_name...")
        market_dists = calculate_betting_distributions(pred_chain)

        # Iterate over each betting market for the current model
        for market_name in sort(collect(keys(market_dists)))
            dist = market_dists[market_name]

            # --- Calculate Statistics ---
            # Probability stats
            mean_prob = mean(dist)
            prob_ci = (quantile(dist, lower_q), quantile(dist, upper_q))
            
            # Odds stats (inverse of probability)
            # Handle cases where probability is 0 or 1 to avoid infinite odds
            mean_odd = mean_prob > 0 ? 1 / mean_prob : Inf
            odds_ci = (
                prob_ci[2] > 0 ? 1 / prob_ci[2] : Inf,
                prob_ci[1] > 0 ? 1 / prob_ci[1] : Inf
            )

            # --- Append results to our arrays ---
            push!(markets, market_name)
            push!(models, model_name)
            push!(mean_probs, round(mean_prob; digits=3))
            push!(prob_cis, "($(round(prob_ci[1]; digits=3)), $(round(prob_ci[2]; digits=3)))")
            push!(mean_odds, round(mean_odd; digits=3))
            push!(odds_cis, "($(round(odds_ci[1]; digits=3)), $(round(odds_ci[2]; digits=3)))")
        end
    end

    # --- Assemble the final DataFrame ---
    return DataFrame(
        Market = markets,
        Model = models,
        MeanProb = mean_probs,
        ProbCI = prob_cis,
        MeanOdds = mean_odds,
        OddsCI = odds_cis
    )
end


# 1. Create a dictionary of your models' predictions
model_predictions = Dict(
    "StaticPoisson" => prediction_chain,
    "StaticSimplex" => prediction_chain_2
)

# 2. Generate the comparison DataFrame
comparison_df = compare_model_predictions(model_predictions; ci_level = 0.30)

# 3. Display the results
display(comparison_df)




"""
Compares betting market predictions from multiple models and returns a
wide-format DataFrame, built directly by iterating through markets.

Arguments:
- `model_predictions`: A Dictionary mapping model names (String) to their
  prediction `Chains` objects.
- `ci_level`: The credible interval level (e.g., 0.95 for a 95% CI).

Returns:
- A wide-format `DataFrame` summarizing the predictions.
"""
function compare_models_by_market(
    model_predictions;
    ci_level::Float64 = 0.95
)
    # --- Step 1: Pre-calculate all distributions to avoid redundant work ---
    all_dists = Dict(
        name => calculate_betting_distributions(chain)
        for (name, chain) in model_predictions
    )

    # --- Step 2: Get a sorted list of all unique markets and models ---
    # We can get the market list from the first model's results
    market_names = sort(collect(keys(first(all_dists).second)))
    model_names = sort(collect(keys(model_predictions)))

    # --- Step 3: Iterate through markets, building one row at a time ---
    all_rows = []
    quantiles = ((1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0)

    for market in market_names
        # Create a dictionary to hold all data for the current market's row
        row_data = Dict{Symbol, Any}(:Market => market)

        # Inner loop: iterate through models to populate the row
        for model in model_names
            dist = all_dists[model][market]

            # Calculate stats using the Beta distribution for a proper credible interval
            n_samples = length(dist)
            n_successes = sum(dist)
            posterior_prob_dist = Beta(n_successes + 1, n_samples - n_successes + 1)
            
            mean_prob = mean(posterior_prob_dist)
            prob_ci = quantile(posterior_prob_dist, [quantiles[1], quantiles[2]])
            mean_odd = mean_prob > 0 ? 1 / mean_prob : Inf
            odds_ci = (prob_ci[2] > 0 ? 1 / prob_ci[2] : Inf, prob_ci[1] > 0 ? 1 / prob_ci[1] : Inf)

            # Add the stats for this model to the row dictionary with unique keys
            row_data[Symbol("MeanProb_", model)] = round(mean_prob, digits=3)
            row_data[Symbol("ProbCI_", model)] = "($(round(prob_ci[1], digits=3)), $(round(prob_ci[2], digits=3)))"
            row_data[Symbol("MeanOdds_", model)] = round(mean_odd, digits=3)
        end
        push!(all_rows, row_data)
    end

    # --- Step 4: Convert the array of dictionaries to a DataFrame ---
    final_df = DataFrame(all_rows)

    # --- Step 5: Enforce a logical column order ---
    ordered_cols = [:Market]
    for model in model_names
        push!(ordered_cols, Symbol("MeanProb_", model))
        push!(ordered_cols, Symbol("ProbCI_", model))
        push!(ordered_cols, Symbol("MeanOdds_", model))
    end
    
    return final_df[!, ordered_cols]
end



comparison_df = compare_models_by_market(model_predictions; ci_level = 0.95)


"""
Finds the corresponding decimal odds for a standardized market name from
the bookmaker's odds DataFrame.

This is a helper function to handle the complex mapping.
"""
function find_market_odds(market_key::String, odds_df::DataFrame)
    # 1X2, BTTS, DNB, DC markets
    simple_mappings = Dict(
        "home_win" => ("Full time", "1"),
        "draw"     => ("Full time", "X"),
        "away_win" => ("Full time", "2"),
        "btts_yes" => ("Both teams to score", "Yes"),
        "btts_no"  => ("Both teams to score", "No"),
        "dnb_1"    => ("Draw no bet", "1"),
        "dnb_2"    => ("Draw no bet", "2"),
        "dc_1X"    => ("Double chance", "1X"),
        "dc_X2"    => ("Double chance", "X2"),
        "dc_12"    => ("Double chance", "12")
    )

    if haskey(simple_mappings, market_key)
        market, choice = simple_mappings[market_key]
        res = filter(row -> row.market_name == market && row.choice_name == choice, odds_df)
        return isempty(res) ? missing : first(res).decimal_odds
    end

    # Handle Over/Under markets
    if startswith(market_key, "over_") || startswith(market_key, "under_")
        parts = split(market_key, '_')
        choice = parts[1] == "over" ? "Over" : "Under"
        line = parse(Float64, parts[2]) / 10.0 # Convert "25" to 2.5

        res = filter(row -> row.market_name == "Match goals" &&
                           row.choice_name == choice &&
                           row.choice_group == line, odds_df)
        return isempty(res) ? missing : first(res).decimal_odds
    end

    return missing # Return missing if no mapping is found
end


"""
Compares model predictions to market odds and calculates Expected Value (EV).
"""
function calculate_ev_comparison(model_df::DataFrame, market_odds_df::DataFrame)
    results = []
    model_names = unique([split(name, "_")[2] for name in names(model_df) if occursin("MeanProb", name)])

    for row in eachrow(model_df)
        market_key = row.Market
        market_odds = find_market_odds(market_key, market_odds_df)

        # Skip if we couldn't find a corresponding market odd
        ismissing(market_odds) && continue

        # Start building the output row
        output_row = Dict{Symbol, Any}(
            :Market => market_key,
            :MarketOdds => market_odds
        )

        # Calculate EV for each model
        for model in model_names
            model_prob = row[Symbol("MeanProb_", model)]
            model_odds = row[Symbol("MeanOdds_", model)]

            # EV = (Probability * Payout) - Stake
            # Where Payout = MarketOdds and Stake = 1
            ev = (model_prob * market_odds) - 1.0

            output_row[Symbol("ModelProb_", model)] = model_prob
            output_row[Symbol("ModelOdds_", model)] = model_odds
            output_row[Symbol("EV_", model)] = round(ev, digits=3)
        end
        push!(results, output_row)
    end

    return DataFrame(results)
end



o = filter(row -> row.match_id==m.match_id, data_store.odds)
m

comparison_df = compare_models_by_market(model_predictions; ci_level = 0.95)
ev_df = calculate_ev_comparison(comparison_df, o)

sort(ev_df, :EV_StaticSimplex, rev=true)

