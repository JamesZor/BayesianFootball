using BayesianFootball
using Turing
using Plots
using Statistics
using DataFrames

# Performance libraries
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)



include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA

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

ar1_binom_model_def = AR1NegativeBinomialHAModel()

features = (
    global_round = synth_data.global_round,
    home_team_ids = synth_data.home_team_ids,
    away_team_ids = synth_data.away_team_ids,
    league_ids = synth_data.league_ids,
    n_teams = synth_data.n_teams,
    n_leagues = synth_data.n_leagues
)



ar1_negbin_turing_model = AR1NegativeBinomialHA.build_turing_model(
    ar1_binom_model_def,
    features,
    synth_data.home_goals,
    synth_data.away_goals
)



ar1_binom_chain = sample(ar1_negbin_turing_model, NUTS(0.65), 100, progress=true)

ar1ha_params = get_processed_parameters(ar1ha_chain, synth_data);
ar1_binom_parameters =  get_processed_parameters(ar1_binom_chain, synth_data);


team_to_plot =6
comparison_plot = plot_model_comparison(team_to_plot, synth_data, ar1ha_params, ar1_binom_parameters)


dummy_chains_negb =  BayesianFootball.TrainedChains(ar1_binom_chain, ar1_binom_chain, "samples", 1)

predictions_negbi = BayesianFootball.predict(ar1_binom_model_def, dummy_chains_negb, features, mapping);



mean( m_prediction_ha.ft.home)
mean( m_prediction_ha.ft.away)

mean( predictions_negbi.ft.home)
mean( predictions_negbi.ft.away)

