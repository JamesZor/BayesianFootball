
using BayesianFootball
using Turing
using Plots
using Statistics
using DataFrames

# Performance libraries
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)



include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/bssm_poisson.jl")
using .BSSMPoisson

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


bssm_poisson_model_def  = BSSMPoissonModel()



features = (
    global_round = synth_data.global_round,
    home_team_ids = synth_data.home_team_ids,
    away_team_ids = synth_data.away_team_ids,
    league_ids = synth_data.league_ids,
    n_teams = synth_data.n_teams,
    n_leagues = synth_data.n_leagues
)



bssm_poisson_turing_model = BSSMPoisson.build_turing_model(
    bssm_poisson_model_def,
    features,
    synth_data.home_goals,
    synth_data.away_goals
)



bssm_p_chains = sample(bssm_poisson_turing_model, NUTS(0.65), 500, progress=true)



dummy_chains_bssm =  BayesianFootball.TrainedChains(bssm_p_chains, bssm_p_chains, "samples", 1)



predictions_bssm = BayesianFootball.predict(bssm_poisson_model_def, dummy_chains_negb, features, mapping);





