using Plots 
using StatsPlots
using Turing


####
# extract poisson basic 
#####

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

### 

using MCMCChains
using Statistics

"""
    get_raw_parameters_ha(chain_dynamic, data)

Reconstructs the raw, uncentered posterior trajectories for attack, defense,
and the dynamic home advantage from a model chain.
"""
function get_raw_parameters_ha(chain_dynamic, data)
    n_samples = size(chain_dynamic, 1) * size(chain_dynamic, 3)
    n_teams = data.n_teams
    n_rounds = data.n_rounds
    n_leagues = data.n_leagues

    # --- 1. Reconstruct Sigmas (Attack, Defense, Home Advantage) ---
    μ_log_σ_attack = vec(Array(chain_dynamic[:μ_log_σ_attack]))
    τ_log_σ_attack = vec(Array(chain_dynamic[:τ_log_σ_attack]))
    z_log_σ_attack = Array(group(chain_dynamic, :z_log_σ_attack))
    σ_attack_samples = exp.(μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack)

    μ_log_σ_defense = vec(Array(chain_dynamic[:μ_log_σ_defense]))
    τ_log_σ_defense = vec(Array(chain_dynamic[:τ_log_σ_defense]))
    z_log_σ_defense = Array(group(chain_dynamic, :z_log_σ_defense))
    σ_defense_samples = exp.(μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense)
    
    # NEW: Reconstruct Home Advantage Sigma
    μ_log_σ_home = vec(Array(chain_dynamic[:μ_log_σ_home]))
    σ_home_samples = exp.(μ_log_σ_home)

    # --- 2. Reconstruct Raw Trajectories ---
    ρ_attack = vec(Array(chain_dynamic["ρ_attack"]))
    ρ_defense = vec(Array(chain_dynamic["ρ_defense"]))
    ρ_home = vec(Array(chain_dynamic["ρ_home"])) # NEW

    initial_α_z = Array(group(chain_dynamic, :initial_α_z))
    initial_β_z = Array(group(chain_dynamic, :initial_β_z))
    initial_home_z = Array(group(chain_dynamic, :initial_home_z)) # NEW
    
    z_α = Array(group(chain_dynamic, :z_α))
    z_β = Array(group(chain_dynamic, :z_β))
    z_home = Array(group(chain_dynamic, :z_home)) # NEW

    log_α_raw = Array{Float64}(undef, n_samples, n_teams, n_rounds)
    log_β_raw = Array{Float64}(undef, n_samples, n_teams, n_rounds)
    log_home_adv_raw = Array{Float64}(undef, n_samples, n_leagues, n_rounds) # NEW

    for s in 1:n_samples
        z_α_mat_s = reshape(z_α[s, :], n_teams, n_rounds)
        z_β_mat_s = reshape(z_β[s, :], n_teams, n_rounds)
        z_home_mat_s = reshape(z_home[s, :], n_leagues, n_rounds) # NEW

        log_α_raw_t0_s = initial_α_z[s, :] * sqrt(0.5)
        log_β_raw_t0_s = initial_β_z[s, :] * sqrt(0.5)
        log_home_adv_raw_t0_s = log(1.3) .+ initial_home_z[s, :] * sqrt(0.1) # NEW
        
        for t in 1:n_rounds
            if t == 1
                log_α_raw[s, :, 1] = log_α_raw_t0_s .+ z_α_mat_s[:, 1] .* σ_attack_samples[s, :]
                log_β_raw[s, :, 1] = log_β_raw_t0_s .+ z_β_mat_s[:, 1] .* σ_defense_samples[s, :]
                log_home_adv_raw[s, :, 1] = log_home_adv_raw_t0_s .+ z_home_mat_s[:, 1] .* σ_home_samples[s] # NEW
            else
                log_α_raw[s, :, t] = ρ_attack[s] * log_α_raw[s, :, t-1] .+ z_α_mat_s[:, t] .* σ_attack_samples[s, :]
                log_β_raw[s, :, t] = ρ_defense[s] * log_β_raw[s, :, t-1] .+ z_β_mat_s[:, t] .* σ_defense_samples[s, :]
                log_home_adv_raw[s, :, t] = ρ_home[s] * log_home_adv_raw[s, :, t-1] .+ z_home_mat_s[:, t] .* σ_home_samples[s] # NEW
            end
        end
    end

    return (
        log_α_raw = log_α_raw, 
        log_β_raw = log_β_raw, 
        log_home_adv_raw = log_home_adv_raw, # NEW
        σ_attack = σ_attack_samples, 
        σ_defense = σ_defense_samples
    )
end


"""
    get_processed_parameters(chain_dynamic, data)

Takes a model chain, reconstructs the raw parameter trajectories, and returns
the centered team strengths along with the raw home advantage trajectory.
"""
function get_processed_parameters(chain_dynamic, data)
    # Get all the raw, uncentered parameters
    raw_params = get_raw_parameters_ha(chain_dynamic, data)
    log_α_raw = raw_params.log_α_raw
    log_β_raw = raw_params.log_β_raw
    log_home_adv_raw = raw_params.log_home_adv_raw # Pass through home advantage
    
    n_samples, _, n_rounds = size(log_α_raw)
    
    # Center the team-specific results for plotting and identifiability
    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)
    for s in 1:n_samples, t in 1:n_rounds
        log_α_centered[s, :, t] = log_α_raw[s, :, t] .- mean(log_α_raw[s, :, t])
        log_β_centered[s, :, t] = log_β_raw[s, :, t] .- mean(log_β_raw[s, :, t])
    end
    
    # Return all processed parameters
    return (
        log_α_centered = log_α_centered,
        log_β_centered = log_β_centered,
        log_home_adv_raw = log_home_adv_raw
    )
end




###


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

####
# PLots 
####

# --- Comparison Plotting Function ---

function plot_compare_a_b(team_number, data, α_dynamic, α_static, β_dynamic, β_static)
    p = plot(layout=(2, 1), legend=:outertopright, size=(800, 600),
             titlefontsize=10, tickfontsize=8)

    # --- Alpha (Attack) Plot ---
    plot!(p[1], 1:data.n_rounds, data.true_log_α[team_number, :],
        label="True log α", lw=3, color=:black,
        title="Team $team_number Attack Rating (log α)",
        ylabel="Attack Parameter")

    plot!(p[1], 1:data.n_rounds, mean(α_dynamic[:, team_number, :], dims=1)',
        ribbon=std(α_dynamic[:, team_number, :], dims=1)',
        label="Dynamic Estimate", color=:blue, fillalpha=0.2)

    mean_static_α = mean(α_static[:, team_number])
    std_static_α = std(α_static[:, team_number])
    plot!(p[1], 1:data.n_rounds, fill(mean_static_α, data.n_rounds),
        ribbon=std_static_α,
        label="Static Estimate", color=:red, linestyle=:dash, lw=2, fillalpha=0.2)

    # --- Beta (Defense) Plot ---
    plot!(p[2], 1:data.n_rounds, data.true_log_β[team_number, :],
        label="True log β", lw=3, color=:black,
        title="Team $team_number Defense Rating (log β)",
        xlabel="Round", ylabel="Defense Parameter")

    plot!(p[2], 1:data.n_rounds, mean(β_dynamic[:, team_number, :], dims=1)',
        ribbon=std(β_dynamic[:, team_number, :], dims=1)',
        label="Dynamic Estimate", color=:blue, fillalpha=0.2)

    mean_static_β = mean(β_static[:, team_number])
    std_static_β = std(β_static[:, team_number])
    plot!(p[2], 1:data.n_rounds, fill(mean_static_β, data.n_rounds),
        ribbon=std_static_β,
        label="Static Estimate", color=:red, linestyle=:dash, lw=2, fillalpha=0.2)
    
    return p
end

function get_centered_parameters_from_static_ha_model(chain_dynamic, data)
    # This uses the same raw parameter extraction logic
    raw_params = get_raw_parameters(chain_dynamic, data)
    log_α_raw = raw_params.log_α_raw
    log_β_raw = raw_params.log_β_raw
    
    n_samples, _, n_rounds = size(log_α_raw)
    
    # Center the results
    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)
    for s in 1:n_samples, t in 1:n_rounds
        log_α_centered[s, :, t] = log_α_raw[s, :, t] .- mean(log_α_raw[s, :, t])
        log_β_centered[s, :, t] = log_β_raw[s, :, t] .- mean(log_β_raw[s, :, t])
    end
    
    return (log_α_centered = log_α_centered, log_β_centered = log_β_centered)
end

function plot_model_comparison(team_number, data, ar1ha_params, ar1_params)
    p = plot(layout=(2, 1), legend=:outertopright, size=(1000, 700),
             titlefontsize=11, tickfontsize=8, left_margin=5Plots.mm)

    # --- Alpha (Attack) Plot ---
    plot!(p[1], 1:data.n_rounds, data.true_log_α[team_number, :],
          label="True Value", lw=3, color=:black, ls=:dash,
          title="Team $team_number Attack Rating (log α)",
          ylabel="Attack Parameter")

    # Plot estimate from the AR1-HA model
    plot!(p[1], 1:data.n_rounds, vec(mean(ar1ha_params.log_α_centered[:, team_number, :], dims=1)),
          ribbon=vec(std(ar1ha_params.log_α_centered[:, team_number, :], dims=1)),
          label="Dynamic HA Model", color=:crimson, fillalpha=0.2, lw=2)

    # Plot estimate from the simple AR1 model
    plot!(p[1], 1:data.n_rounds, vec(mean(ar1_params.log_α_centered[:, team_number, :], dims=1)),
          ribbon=vec(std(ar1_params.log_α_centered[:, team_number, :], dims=1)),
          label="Static HA Model", color=:dodgerblue, fillalpha=0.2, lw=2)

    # --- Beta (Defense) Plot ---
    plot!(p[2], 1:data.n_rounds, data.true_log_β[team_number, :],
          label="True Value", lw=3, color=:black, ls=:dash,
          title="Team $team_number Defense Rating (log β)",
          xlabel="Global Round", ylabel="Defense Parameter")
          
    # Plot estimate from the AR1-HA model
    plot!(p[2], 1:data.n_rounds, vec(mean(ar1ha_params.log_β_centered[:, team_number, :], dims=1)),
          ribbon=vec(std(ar1ha_params.log_β_centered[:, team_number, :], dims=1)),
          label="Dynamic HA Model", color=:crimson, fillalpha=0.2, lw=2)

    # Plot estimate from the simple AR1 model
    plot!(p[2], 1:data.n_rounds, vec(mean(ar1_params.log_β_centered[:, team_number, :], dims=1)),
          ribbon=vec(std(ar1_params.log_β_centered[:, team_number, :], dims=1)),
          label="Static HA Model", color=:dodgerblue, fillalpha=0.2, lw=2)
    
    return p
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

"""
    plot_multiple_dashboards(teams, data, α_dynamic, α_static, β_dynamic, β_static)

Creates a vertical layout of team dashboards for a given list of team IDs.

# Arguments
- `teams`: A vector of team IDs to plot (e.g., [1, 2, 5]).
- `data`, `α_dynamic`, etc.: The same arguments required by the single dashboard function.
"""
function plot_multiple_dashboards(teams, data, α_dynamic, β_dynamic)
    
    # Create an empty list to hold each team's dashboard plot
    plot_list = []

    # Loop through each requested team ID
    for team_id in teams
        # Generate the 2x2 dashboard for the current team
        p_team = plot_team_dashboard(
            team_id, 
            data, 
            α_dynamic, 
            β_dynamic, 
        )
        # Add the generated plot to our list
        push!(plot_list, p_team)
    end

    # Arrange all the plots in the list into a single column
    # The layout is (number_of_rows, number_of_columns)
    # The final size will need to be tall to accommodate all the plots.
    n_teams = length(teams)
    final_plot = plot(plot_list..., 
                      layout = (n_teams, 1), 
                      size = (1400, 750 * n_teams)
    )

    return final_plot
end

function plot_team_dashboard(team_number, data, α_dynamic, α_static, β_dynamic, β_static)
    
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

    mean_static_α = mean(α_static[:, team_number])
    std_static_α = std(α_static[:, team_number])
    plot!(p[1, 1], 1:data.n_rounds, fill(mean_static_α, data.n_rounds),
        ribbon=std_static_α,
        label="Static Estimate", color=:deepskyblue4, linestyle=:dash, lw=2, fillalpha=0.2)

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

    mean_static_β = mean(β_static[:, team_number])
    std_static_β = std(β_static[:, team_number])
    plot!(p[1, 2], 1:data.n_rounds, fill(mean_static_β, data.n_rounds),
        ribbon=std_static_β,
        label="Static Estimate", color=:darkred, linestyle=:dash, lw=2, fillalpha=0.2)
        
    # Plot [2, 2]: Goals Conceded (Bar Chart)
    bar!(p[2, 2], 1:data.n_rounds, goals_conceded,
         label="Goals Conceded", color=:crimson, alpha=0.7,
         # xlabel="Round",
         legend=:topleft,
         ylims=(0, max(maximum(goals_scored), maximum(goals_conceded)) + 1)
    )
    
    return p
end


"""
    plot_multiple_dashboards(teams, data, α_dynamic, α_static, β_dynamic, β_static)

Creates a vertical layout of team dashboards for a given list of team IDs.

# Arguments
- `teams`: A vector of team IDs to plot (e.g., [1, 2, 5]).
- `data`, `α_dynamic`, etc.: The same arguments required by the single dashboard function.
"""
function plot_multiple_dashboards(teams, data, α_dynamic, α_static, β_dynamic, β_static)
    
    # Create an empty list to hold each team's dashboard plot
    plot_list = []

    # Loop through each requested team ID
    for team_id in teams
        # Generate the 2x2 dashboard for the current team
        p_team = plot_team_dashboard(
            team_id, 
            data, 
            α_dynamic, 
            α_static, 
            β_dynamic, 
            β_static
        )
        # Add the generated plot to our list
        push!(plot_list, p_team)
    end

    # Arrange all the plots in the list into a single column
    # The layout is (number_of_rows, number_of_columns)
    # The final size will need to be tall to accommodate all the plots.
    n_teams = length(teams)
    final_plot = plot(plot_list..., 
                      layout = (n_teams, 1), 
                      size = (1400, 750 * n_teams)
    )

    return final_plot
end

