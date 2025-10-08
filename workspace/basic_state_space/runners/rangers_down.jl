using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions, Plots
using CSV 

# utils 

include("/home/james/bet_project/models_julia/workspace/basic_state_space/utils/scot_matchday_utils.jl")

# models
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson_ha.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA
using .AR1PoissonHA




all_model_paths = Dict(
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_poisson_ha_20251004-111854",
  "ssm_neg_bin" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_neg_bin_ha_20251004-122001",
)


loaded_models_all = load_models_from_paths(all_model_paths)



m1 = loaded_models_all["ssm_neg_bin"]

mapping = m1.result.mapping;
chain = m1.result.chains_sequence[1];

posterior_samples = BayesianFootball.extract_posterior_samples(
    m1.config.model_def,
    chain.ft,
    mapping
);

last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

home_team="falkirk-fc"
away_team="rangers"

match_to_predict = DataFrame(
    home_team="falkirk-fc",
    away_team="rangers",
    tournament_id=54,
    global_round = next_round,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)

features = BayesianFootball.create_master_features(match_to_predict, mapping);
predictions = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);

print_1x2(predictions)

print_under(predictions)

plot_attack_defence(home_team, away_team, m1, posterior_samples)

const DATA_PATH = "/home/james/bet_project/football/scotland_football"
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

function add_global_round_column!(matches_df::DataFrame)
    sort!(matches_df, :match_date)
    num_matches = nrow(matches_df)
    global_rounds = Vector{Int}(undef, num_matches)
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(matches_df))
        home_team, away_team = row.home_team, row.away_team
        if home_team in teams_in_current_round || away_team in teams_in_current_round
            global_round_counter += 1
            empty!(teams_in_current_round)
        end
        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team, away_team)
    end
    
    matches_df.global_round = global_rounds
    println("✅ Successfully added `:global_round` column. Found $(global_round_counter) unique time steps.")
    return matches_df
end

add_global_round_column!(data_store.matches)



function plot_attack_defence(team1_name, team2_name, loaded_model, posterior_samples)

team1_id = loaded_model.result.mapping.team[team1_name]
team2_id = loaded_model.result.mapping.team[team2_name]

# --- 2. Get the full time-series of the parameters ---
log_α_centered = posterior_samples.log_α_centered
log_β_centered = posterior_samples.log_β_centered
n_rounds = posterior_samples.n_rounds

# --- 3. Calculate the posterior mean AND STANDARD DEVIATION over time ---
# Mean calculations
team1_attack_mean = vec(mean(log_α_centered[:, team1_id, :], dims=1))
team1_defense_mean = vec(mean(log_β_centered[:, team1_id, :], dims=1))
team2_attack_mean = vec(mean(log_α_centered[:, team2_id, :], dims=1))
team2_defense_mean = vec(mean(log_β_centered[:, team2_id, :], dims=1))

# Standard deviation calculations
team1_attack_std = vec(std(log_α_centered[:, team1_id, :], dims=1))
team1_defense_std = vec(std(log_β_centered[:, team1_id, :], dims=1))
team2_attack_std = vec(std(log_α_centered[:, team2_id, :], dims=1))
team2_defense_std = vec(std(log_β_centered[:, team2_id, :], dims=1))


# --- 4. Create the 1x2 plot with ribbons ---
p = plot(
    layout=(1, 2),
    # size=(1400, 500),
    size=(900, 500),
    legend=:bottomleft,
    # link=:y,
    xlabel="Global Round"
)

# Subplot 1: Attacking Strength
plot!(p[1], 1:n_rounds, team1_attack_mean,
    ribbon = 1 .* team1_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,                # Make ribbon transparent
    label = team1_name,
    title = "Attacking Strength (log α)",
    ylabel = "Parameter Value",
    lw = 2
)
plot!(p[1], 1:n_rounds, team2_attack_mean,
    ribbon = 1 .* team2_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
    lw = 2
)

# Subplot 2: Defensive Strength
plot!(p[2], 1:n_rounds, team1_defense_mean,
    ribbon = 1 .* team1_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team1_name,
    title = "Defensive Strength (log β)",
    lw = 2
)
plot!(p[2], 1:n_rounds, team2_defense_mean,
    ribbon = 1 .* team2_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
    lw = 2
)


end 


names(data_store.matches)
""" 
julia> names(data_store.matches)
20-element Vector{String}:
 "tournament_id"
 "season_id"
 "season"
 "match_id"
 "tournament_slug"
 "home_team"
 "away_team"
 "home_score"
 "away_score"
 "home_score_ht"
 "away_score_ht"
 "match_date"
 "round"
 "winner_code"
 "has_xg"
 "has_stats"
 "match_hour"
 "match_dayofweek"
 "match_month"
 "global_round"
"""
