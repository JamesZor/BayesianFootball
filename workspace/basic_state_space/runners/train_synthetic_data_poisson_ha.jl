using BayesianFootball
using Turing
using Plots
using Statistics
using DataFrames

# Performance libraries
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# --- 1. SETUP AND INCLUDES ---

# Include our refactored modules
include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson.jl")
using .AR1Poisson

include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson_ha.jl")
using .AR1PoissonHA

include("/home/james/bet_project/models_julia/workspace/basic_state_space/utils/utils.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/utils/plots.jl")
using .SSMUtils



synth_data = generate_synthetic_multi_season_data_ha(
    n_teams=10,
    n_seasons=2,
    rounds_per_season=38,
    season_to_season_volatility=0.02, 
    home_adv_volatility=0.01,
    seed=41
)


plot(
    1:synth_data.n_rounds,
    synth_data.true_log_α[1:10, :]', 
    # label=["1","2","3","4","5","6","7","8","9","10"],
    xlabel="Round",
    ylabel="True Log Attacking Strength (log α)",
    title="Simulated Team Strengths Over Multiple Seasons",
    legend=:outertopright,
    linewidth=2
)

plot(
    1:synth_data.n_rounds,
    synth_data.true_log_β[1:10, :]', 
    # label=["1","2","3","4","5","6","7","8","9","10"],
    xlabel="Round",
    ylabel="True Log Defense Strength (log Defense)",
    title="Simulated Team Strengths Over Multiple Seasons",
    legend=:outertopright,
    linewidth=2
)


plot(
    1:synth_data.n_rounds,
    synth_data.true_log_home_adv, 
    # label=["1","2","3","4","5","6","7","8","9","10"],
    xlabel="Round",
    ylabel="True Log Home advantage Strength (log σ)",
    title="Simulated Team Strengths Over Multiple Seasons",
    legend=:outertopright,
    linewidth=2
)


matches_df = DataFrame(
    global_round = synth_data.global_round,
    home_team_ids = synth_data.home_team_ids,
    away_team_ids = synth_data.away_team_ids
);


ar1_model_def = AR1PoissonModel()
ar1ha_model_def = AR1PoissonHAModel()


features = (
    global_round = synth_data.global_round,
    home_team_ids = synth_data.home_team_ids,
    away_team_ids = synth_data.away_team_ids,
    league_ids = synth_data.league_ids,
    n_teams = synth_data.n_teams,
    n_leagues = synth_data.n_leagues
)



ar1_turing_model = AR1Poisson.build_turing_model(
    ar1_model_def,
    features,
    synth_data.home_goals,
    synth_data.away_goals
)

ar1ha_turing_model = AR1PoissonHA.build_turing_model(
    ar1ha_model_def,
    features,
    synth_data.home_goals,
    synth_data.away_goals
)

ar1_chain = sample(ar1_turing_model, NUTS(0.65), 100, progress=true)
ar1ha_chain = sample(ar1ha_turing_model, NUTS(0.65), 100, progress=true)

# mapping = BayesianFootball.MappedData(
#     Dict(string(i) => i for i in 1:10),
#     Dict(string(i) => i for i in 1:1)
# )
#
# ar1_post = BayesianFootball.extract_posterior_samples(
#     ar1_model_def,
#     ar1_chain,
#     mapping
# )
#
# ar1ha_post = BayesianFootball.extract_posterior_samples(
#     ar1ha_model_def,
#     ar1ha_chain,
#     mapping
# )
#

a = get_processed_parameters(ar1ha_chain, synth_data);

team1_plot = plot_team_dashboard(1, synth_data, a.log_α_centered,  a.log_β_centered)


# --- Extract the static home advantage from the first model's chain ---
# The ar1_chain contains a single distribution for the home advantage.
log_home_adv_static_samples = vec(Array(ar1_chain[:log_home_adv]))

# Calculate the mean and standard deviation for the static estimate
static_ha_mean = mean(log_home_adv_static_samples)
static_ha_std = std(log_home_adv_static_samples)
static_ha_upper = static_ha_mean + 1.96 * static_ha_std
static_ha_lower = static_ha_mean - 1.96 * static_ha_std

# --- Create the Comparison Plot ---

p = plot(
    xlabel="Global Round",
    ylabel="Log Home Advantage",
    legend=:outertopright,
    title="Model Comparison: Dynamic vs. Static Home Advantage"
)

# 1. Plot the TRUE, evolving home advantage from the synthetic data
plot!(p, 1:synth_data.n_rounds, synth_data.true_log_home_adv,
    label="True Value (Dynamic)",
    lw=3,
    color=:black,
    ls=:dash
)

# 2. Plot the DYNAMIC model's estimate (from your existing code)
dynamic_ha_mean = vec(mean(a.log_home_adv_raw, dims=1))
dynamic_ha_std = vec(std(a.log_home_adv_raw, dims=1))

plot!(p, 1:synth_data.n_rounds, dynamic_ha_mean,
    ribbon=1.96 * dynamic_ha_std,
    label="Dynamic Model Estimate",
    color=:crimson,
    fillalpha=0.2,
    lw=2
)

# 3. Plot the STATIC model's estimate as horizontal lines
hline!(p, [static_ha_mean],
    label="Static Model Mean",
    color=:dodgerblue,
    lw=2.5
)

hline!(p, [static_ha_lower, static_ha_upper],
    label="Static Model 95% CI",
    color=:dodgerblue,
    ls=:dash,
    lw=1.5
)

# Display the final plot
display(p)


# --- Process both chains ---

# Process the AR1-HA model chain
ar1ha_params = get_processed_parameters(ar1ha_chain, synth_data);

# Process the simpler AR1 model chain
ar1_params = get_centered_parameters_from_static_ha_model(ar1_chain, synth_data);


# --- Generate the comparison plot for a specific team (e.g., Team 2) ---
team_to_plot = 10
comparison_plot = plot_model_comparison(team_to_plot, synth_data, ar1ha_params, ar1_params)

# Display the plot
display(comparison_plot)




####
# testing to use the models predict functins 
####
include("/home/james/bet_project/models_julia/workspace/basic_state_space/utils/model_eval_poisson.jl")
using .ModelEvaluation

dummy_chains = BayesianFootball.TrainedChains(ar1_chain, ar1_chain, "samples", 1)
dummy_chains_ha = BayesianFootball.TrainedChains(ar1ha_chain, ar1ha_chain, "samples", 1)

test_set_matches = ModelEvaluation.generate_test_set(synth_data)
test_set_matches = generate_test_set1(synth_data)

match = test_set_matches[1]

mapping = BayesianFootball.MappedData(
    Dict("Team " * string(i) => i for i in 1:synth_data.n_teams),
    Dict("League " * string(i) => i for i in 1:synth_data.n_leagues),
)

predict_df = DataFrame(
          home_team="Team " * string(match.home_team), # Use string names
          away_team="Team " * string(match.away_team),
          tournament_id="League 1",
          global_round = maximum(synth_data.global_round) + 1,
          home_score_ht = 0, 
          away_score_ht = 0, 
          home_score = 0, 
          away_score = 0, 
      )

features = BayesianFootball.create_master_features(predict_df, mapping)

predictions = BayesianFootball.predict(ar1_model_def, dummy_chains, features, mapping)
# predict not working 

# Extract posterior samples once, using dispatch to call the correct method
posterior_samples_ft = BayesianFootball.extract_posterior_samples(ar1_model_def, dummy_chains.ft, mapping);
posterior_samples_ht = BayesianFootball.extract_posterior_samples(ar1_model_def, dummy_chains.ht, mapping);

# Generate FT and HT predictions
ft_predict = BayesianFootball._predict_match_ft(ar1_model_def, dummy_chains.ft, features, posterior_samples_ft);
ht_predict = BayesianFootball._predict_match_ht(ar1_model_def, dummy_chains.ht, features, posterior_samples_ht);

m_prediction = BayesianFootball.Predictions.MatchLinePredictions(ht_predict, ft_predict);

mean( m_prediction.ft.home)
mean( m_prediction.ft.away)

mean( predictions.ft.home)
mean( predictions.ft.away)
match

mean( predictions.ft.under_35)



posterior_samples_ft_ha = BayesianFootball.extract_posterior_samples(ar1ha_model_def, dummy_chains_ha.ft, mapping);
posterior_samples_ht_ha = BayesianFootball.extract_posterior_samples(ar1ha_model_def, dummy_chains_ha.ht, mapping);

# Generate FT and HT predictions
ft_predict_ha = BayesianFootball._predict_match_ft(ar1ha_model_def, dummy_chains_ha.ft, features, posterior_samples_ft_ha);
ht_predict_ha = BayesianFootball._predict_match_ht(ar1ha_model_def, dummy_chains_ha.ht, features, posterior_samples_ht_ha);

m_prediction_ha = BayesianFootball.Predictions.MatchLinePredictions(ht_predict_ha, ft_predict_ha);


mean( m_prediction_ha.ft.home)
mean( m_prediction_ha.ft.away)
match
