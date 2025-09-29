# workspace/basic_state_space/runners/train_synthetic_poisson.jl
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
include("/home/james/bet_project/models_julia/workspace/basic_state_space/utils/utils.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/utils/plots.jl")
using .SSMUtils

# Have not added yet
# include("../analysis/plotting.jl")
# using .SSMPlots

# --- 2. GENERATE SYNTHETIC DATA ---
synth_data = generate_synthetic_data(n_teams=10, n_rounds=38)

synth_data = generate_synthetic_multi_season_data(
    n_teams=10,
    n_seasons=3,
    rounds_per_season=38,
    season_to_season_volatility=0.02, # A higher value creates more dramatic shifts
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

# Convert the generated data into a DataFrame, which our model pipeline expects
#
#
#
# This mimics the structure of the real data
#  i dont think we need 
# matches_df = DataFrame(
#     global_round = vcat([fill(r, 5) for r in 1:synth_data.n_rounds]...), # 5 matches per round
#     home_team_ids = synth_data.home_team_ids,
#     away_team_ids = synth_data.away_team_ids
# );

matches_df = DataFrame(
    global_round = synth_data.global_round,
    home_team_ids = synth_data.home_team_ids,
    away_team_ids = synth_data.away_team_ids
);

# --- 3. TRAIN THE MODEL ---

println("Training AR1 Poisson model on synthetic data...")

# Instantiate the model definition
model_def = AR1PoissonModel()

# The `build_turing_model` function expects features as a NamedTuple
features = (
    global_round = matches_df.global_round,
    home_team_ids = matches_df.home_team_ids,
    away_team_ids = matches_df.away_team_ids,
    n_teams = synth_data.n_teams
)

# Build the Turing model instance
turing_model = AR1Poisson.build_turing_model(
    model_def,
    features,
    synth_data.home_goals,
    synth_data.away_goals
)

# Sample from the model (using a small number of samples for a quick test)
chain = sample(turing_model, NUTS(0.65), 500, progress=true)

println("Training complete.")

# --- 4. ANALYZE AND VISUALIZE RESULTS ---

println("Extracting posterior samples and plotting results...")


log_α_centered, log_β_centered = get_centered_parameters(chain, synth_data);


# Now use our plotting function to visualize the results for Team 1
# This team's attack strength was designed to improve over the season
team1_plot = plot_team_dashboard(1, synth_data, log_α_centered,  log_β_centered)

plot_multiple_dashboards([1,2,3], synth_data, log_α_centered, log_β_centered) 

# Save the plot



#### compare with the static poisson model 

home_team_ids_flat, away_team_ids_flat, home_goals_flat , away_goals_flat =  get_maher_inputs( synth_data) 
static_turing_model = basic_maher_model(
    home_team_ids_flat, away_team_ids_flat, home_goals_flat, away_goals_flat, synth_data.n_teams
)

chain_static = sample(static_turing_model, NUTS(0.65), 500)

log_α_static, log_β_static = get_static_parameters(chain_static)

plot_team_dashboard(1, synth_data, log_α_centered, log_α_static,  log_β_centered, log_β_static)

plot_multiple_dashboards([1,2], synth_data, log_α_centered, log_α_static, log_β_centered, log_β_static) 



