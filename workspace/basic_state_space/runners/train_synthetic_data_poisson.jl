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

# Convert the generated data into a DataFrame, which our model pipeline expects
# This mimics the structure of the real data
matches_df = DataFrame(
    global_round = vcat([fill(r, 5) for r in 1:synth_data.n_rounds]...), # 5 matches per round
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
chain = sample(turing_model, NUTS(0.65), 100, progress=true)

println("Training complete.")

# --- 4. ANALYZE AND VISUALIZE RESULTS ---

println("Extracting posterior samples and plotting results...")

"""
    get_raw_parameters(chain_dynamic, data)

Reconstructs the full, uncentered (raw) posterior trajectories for α and β,
and the volatility parameters σ, from a dynamic model chain.
"""
function get_raw_parameters(chain_dynamic, data)
    n_samples, n_teams, n_rounds = size(chain_dynamic,1), data.n_teams, data.n_rounds
    
    # --- 1. Reconstruct Sigmas ---
    μ_log_σ_attack = vec(Array(chain_dynamic[:μ_log_σ_attack]))
    τ_log_σ_attack = vec(Array(chain_dynamic[:τ_log_σ_attack]))
    z_log_σ_attack = Array(group(chain_dynamic, :z_log_σ_attack))
    log_σ_attack_samples = μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack
    σ_attack_samples = exp.(log_σ_attack_samples)

    μ_log_σ_defense = vec(Array(chain_dynamic[:μ_log_σ_defense]))
    τ_log_σ_defense = vec(Array(chain_dynamic[:τ_log_σ_defense]))
    z_log_σ_defense = Array(group(chain_dynamic, :z_log_σ_defense))
    log_σ_defense_samples = μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense
    σ_defense_samples = exp.(log_σ_defense_samples)

    # --- 2. Reconstruct Raw Trajectories ---
    ρ_attack = vec(Array(chain_dynamic["ρ_attack"]))
    ρ_defense = vec(Array(chain_dynamic["ρ_defense"]))
    initial_α_z = Array(group(chain_dynamic, :initial_α_z))
    initial_β_z = Array(group(chain_dynamic, :initial_β_z))
    z_α = Array(group(chain_dynamic, :z_α))
    z_β = Array(group(chain_dynamic, :z_β))

    log_α_raw = Array{Float64}(undef, n_samples, n_teams, n_rounds)
    log_β_raw = Array{Float64}(undef, n_samples, n_teams, n_rounds)

    for s in 1:n_samples
        z_α_mat_s = reshape(z_α[s, :], n_teams, n_rounds)
        z_β_mat_s = reshape(z_β[s, :], n_teams, n_rounds)

        log_α_raw_t0_s = initial_α_z[s, :] * sqrt(0.5)
        log_β_raw_t0_s = initial_β_z[s, :] * sqrt(0.5)
        
        for t in 1:n_rounds
            if t == 1
                log_α_raw[s, :, 1] = log_α_raw_t0_s .+ z_α_mat_s[:, 1] .* σ_attack_samples[s, :]
                log_β_raw[s, :, 1] = log_β_raw_t0_s .+ z_β_mat_s[:, 1] .* σ_defense_samples[s, :]
            else
                log_α_raw[s, :, t] = ρ_attack[s] * log_α_raw[s, :, t-1] .+ z_α_mat_s[:, t] .* σ_attack_samples[s, :]
                log_β_raw[s, :, t] = ρ_defense[s] * log_β_raw[s, :, t-1] .+ z_β_mat_s[:, t] .* σ_defense_samples[s, :]
            end
        end
    end

    return (
        log_α_raw = log_α_raw, 
        log_β_raw = log_β_raw, 
        σ_attack = σ_attack_samples, 
        σ_defense = σ_defense_samples
    )
end


function get_centered_parameters(chain_dynamic, data)
    # Get all the raw, uncentered parameters
    raw_params = get_raw_parameters(chain_dynamic, data)
    log_α_raw = raw_params.log_α_raw
    log_β_raw = raw_params.log_β_raw
    
    n_samples, _, n_rounds = size(log_α_raw)
    
    # Center the results for plotting
    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)
    for s in 1:n_samples, t in 1:n_rounds
        log_α_centered[s, :, t] = log_α_raw[s, :, t] .- mean(log_α_raw[s, :, t])
        log_β_centered[s, :, t] = log_β_raw[s, :, t] .- mean(log_β_raw[s, :, t])
    end
    
    return log_α_centered, log_β_centered
end



log_α_centered, log_β_centered = get_centered_parameters(chain, synth_data)

"""
    get_team_goal_history(team_number, data)

Extracts the number of goals scored and conceded by a specific team in each round.

# Arguments
- `team_number::Int`: The ID of the team to track.
- `data`: The data object containing match details (`home_team_ids`, `away_team_ids`, etc.).

# Returns
- `(goals_scored, goals_conceded)`: A tuple of two vectors, where each vector contains 
  the respective goal count for the team in each round.
"""
function get_team_goal_history(team_number::Int, data)
    n_rounds = data.n_rounds
    goals_scored = Vector{Int}(undef, n_rounds)
    goals_conceded = Vector{Int}(undef, n_rounds)

    for t in 1:n_rounds
        # Find if the team played home or away this round
        home_pos = findfirst(isequal(team_number), data.home_team_ids[t])
        away_pos = findfirst(isequal(team_number), data.away_team_ids[t])

        if !isnothing(home_pos)
            # Team played at home
            goals_scored[t] = data.home_goals[t][home_pos]
            goals_conceded[t] = data.away_goals[t][home_pos]
        elseif !isnothing(away_pos)
            # Team played away
            goals_scored[t] = data.away_goals[t][away_pos]
            goals_conceded[t] = data.home_goals[t][away_pos]
        else
            # Team did not play this round (unlikely in this simulation)
            goals_scored[t] = 0 
            goals_conceded[t] = 0
        end
    end
    
    return goals_scored, goals_conceded
end



"""
    plot_team_dashboard(team_number, data, α_dynamic, α_static, β_dynamic, β_static)

Generates a 2x2 dashboard for a single team, comparing true parameters, model estimates,
and actual match goals.

# Arguments
- `team_number`: The team to plot.
- `data`: The main data object.
- `α_dynamic`, `α_static`: Posterior samples for the attack parameter from dynamic and static models.
- `β_dynamic`, `β_static`: Posterior samples for the defense parameter from dynamic and static models.
"""
function plot_team_dashboard(team_number, data, α_dynamic, β_dynamic)
    
    # 1. Get the goal history for the specified team
    goals_scored, goals_conceded = get_team_goal_history(team_number, data)
    
    # 2. Initialize the 2x2 plot layout
    # link=:x synchronizes the x-axis across all subplots
    p = plot(layout=(2, 2), size=(1400, 800), legend=:topleft,
             titlefontsize=11, tickfontsize=8, link=:x)

    # --- COLUMN 1: ATTACKING PERFORMANCE ---
    
    # Plot [1, 1]: Attack Parameter (log α)
    plot!(p[1, 1], 1:data.n_rounds, data.true_log_α[team_number, :],
        label="True log α", lw=3, color=:black,
        title="Team $team_number Attacking Parameter (log α)",
        ylabel="Parameter Value")

    plot!(p[1, 1], 1:data.n_rounds, mean(α_dynamic[:, team_number, :], dims=1)',
        ribbon=std(α_dynamic[:, team_number, :], dims=1)',
        label="Dynamic Estimate", color=:dodgerblue, fillalpha=0.2)

    # Plot [2, 1]: Goals Scored (Bar Chart)
    bar!(p[2, 1], 1:data.n_rounds, goals_scored,
         label="Goals Scored", color=:dodgerblue, alpha=0.7,
         ylabel="Goals",
         # xlabel="Round", 
         legend=:topleft,
         ylims=(0, max(maximum(goals_scored), maximum(goals_conceded)) + 1)
    )

    # --- COLUMN 2: DEFENSIVE PERFORMANCE ---

    # Plot [1, 2]: Defense Parameter (log β)
    plot!(p[1, 2], 1:data.n_rounds, data.true_log_β[team_number, :],
        label="True log β", lw=3, color=:black,
        title="Team $team_number Defensive Parameter (log β)")

    plot!(p[1, 2], 1:data.n_rounds, mean(β_dynamic[:, team_number, :], dims=1)',
        ribbon=std(β_dynamic[:, team_number, :], dims=1)',
        label="Dynamic Estimate", color=:crimson, fillalpha=0.2)
        
    # Plot [2, 2]: Goals Conceded (Bar Chart)
    bar!(p[2, 2], 1:data.n_rounds, goals_conceded,
         label="Goals Conceded", color=:crimson, alpha=0.7,
         # xlabel="Round",
         legend=:topleft,
         ylims=(0, max(maximum(goals_scored), maximum(goals_conceded)) + 1)
    )
    
    return p
end




# Now use our plotting function to visualize the results for Team 1
# This team's attack strength was designed to improve over the season
team1_plot = plot_team_dashboard(2, synth_data, log_α_centered,  log_β_centered)
display(team1_plot) # This will show the plot in the REPL or plotting pane

# Save the plot
savefig(team1_plot, "team1_synthetic_recovery.png")

println("\n✅ Script finished. Plot saved to team1_synthetic_recovery.png")
