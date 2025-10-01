# ===================================================================
# ## 1. Setup and Libraries
# ===================================================================
using Turing, Distributions, LinearAlgebra, Random
using Plots, StatsPlots, MCMCChains
using ReverseDiff, Memoization
using PrettyTables, Printf, Statistics

# Configure Turing for better performance with this complex model
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# ===================================================================
# ## 2. Data Generation Function
# ===================================================================
function generate_synthetic_data(;
    n_teams::Int=10, 
    n_rounds::Int=38, 
    seed::Int=123
)
    Random.seed!(seed)
    
    true_log_α = zeros(n_teams, n_rounds)
    true_log_β = zeros(n_teams, n_rounds)

    for t in 1:n_rounds
        true_log_α[1, t] = -1.0 + 2.0 * (t / n_rounds)
        true_log_β[2, t] = -1.0 + 2.0 * (t / n_rounds)
    end
    
    for t in 1:n_rounds
        true_log_α[:, t] .-= mean(true_log_α[:, t])
        true_log_β[:, t] .-= mean(true_log_β[:, t])
    end
    
    true_α = exp.(true_log_α)
    true_β = exp.(true_log_β)
    true_home_adv = 1.3

    home_team_ids_all, away_team_ids_all = [], []
    home_goals_all, away_goals_all = [], []
    teams = 1:n_teams
    
    for t in 1:n_rounds
        opponents = shuffle(teams)
        home_teams_round = opponents[1:div(n_teams, 2)]
        away_teams_round = opponents[div(n_teams, 2)+1:end]
        
        home_goals_round, away_goals_round = [], []

        for k in 1:length(home_teams_round)
            i = home_teams_round[k]
            j = away_teams_round[k]
            λ = true_α[i, t] * true_β[j, t] * true_home_adv
            μ = true_α[j, t] * true_β[i, t]
            push!(home_goals_round, rand(Poisson(λ)))
            push!(away_goals_round, rand(Poisson(μ)))
        end
        
        push!(home_team_ids_all, home_teams_round)
        push!(away_team_ids_all, away_teams_round)
        push!(home_goals_all, home_goals_round)
        push!(away_goals_all, away_goals_round)
    end
    
    return (
        home_team_ids=home_team_ids_all,
        away_team_ids=away_team_ids_all,
        home_goals=home_goals_all,
        away_goals=away_goals_all,
        n_teams=n_teams,
        n_rounds=n_rounds,
        true_log_α=true_log_α,
        true_log_β=true_log_β
    )
end



# ===================================================================
# ## Improved Multi-Season Data Generation Function
# ===================================================================
function generate_multi_season_data(;
    n_teams::Int=10,
    n_seasons::Int=3,
    rounds_per_season::Int=38,
    season_to_season_volatility::Float64=0.4,
    seed::Int=123
)
    Random.seed!(seed)

    # Calculate total number of rounds across all seasons
    total_rounds = n_seasons * rounds_per_season

    # Initialize strength matrices for all rounds
    true_log_α = zeros(n_teams, total_rounds)
    true_log_β = zeros(n_teams, total_rounds)

    # True home advantage parameter (constant)
    true_home_adv = 1.3
    teams = 1:n_teams

    # --- Main Simulation Loop ---
    current_round_index = 0

    # Outer loop: Iterate through each season
    for s in 1:n_seasons
        # At the start of each season, generate a new trend slope for each team.
        # This is the core logic for changing dynamics between seasons.
        # A positive slope means the team improves during this season.
        # A negative slope means the team declines during this season.
        attack_slopes = rand(Normal(0, season_to_season_volatility), n_teams)
        defense_slopes = rand(Normal(0, season_to_season_volatility), n_teams)

        # Inner loop: Iterate through each round within the current season
        for r in 1:rounds_per_season
            current_round_index += 1
            t = current_round_index

            if t == 1
                # For the very first round, the strength is just the slope
                true_log_α[:, t] = attack_slopes
                true_log_β[:, t] = defense_slopes
            else
                # For subsequent rounds, build upon the previous round's strength
                # using the slope assigned for the *current season*.
                true_log_α[:, t] = true_log_α[:, t-1] + attack_slopes
                true_log_β[:, t] = true_log_β[:, t-1] + defense_slopes
            end

            # Apply the sum-to-zero constraint at every time step for identifiability
            true_log_α[:, t] .-= mean(true_log_α[:, t])
            true_log_β[:, t] .-= mean(true_log_β[:, t])
        end
    end

    # Convert log-space strengths to natural scale for Poisson rate
    true_α = exp.(true_log_α)
    true_β = exp.(true_log_β)

    # --- Generate Match Data (similar to original function) ---
    home_team_ids_all, away_team_ids_all = [], []
    home_goals_all, away_goals_all = [], []

    for t in 1:total_rounds
        # Create fixtures for the round
        opponents = shuffle(teams)
        home_teams_round = opponents[1:div(n_teams, 2)]
        away_teams_round = opponents[div(n_teams, 2)+1:end]

        home_goals_round, away_goals_round = [], []

        for k in 1:length(home_teams_round)
            i = home_teams_round[k]
            j = away_teams_round[k]

            # Calculate Poisson rates using the strengths for the specific round 't'
            λ = true_α[i, t] * true_β[j, t] * true_home_adv
            μ = true_α[j, t] * true_β[i, t]

            push!(home_goals_round, rand(Poisson(λ)))
            push!(away_goals_round, rand(Poisson(μ)))
        end

        push!(home_team_ids_all, home_teams_round)
        push!(away_team_ids_all, away_teams_round)
        push!(home_goals_all, home_goals_round)
        push!(away_goals_all, away_goals_round)
    end

    # Return data in a format compatible with downstream processes
    return (
        home_team_ids=home_team_ids_all,
        away_team_ids=away_team_ids_all,
        home_goals=home_goals_all,
        away_goals=away_goals_all,
        n_teams=n_teams,
        n_rounds=total_rounds, # Note: this key now refers to total rounds
        true_log_α=true_log_α,
        true_log_β=true_log_β
    )
end

"""
    generate_test_set(data, true_home_adv)

Generates a future test set with ground truth outcomes.
"""
function generate_test_set(data, true_home_adv=1.3)
    n_teams = data.n_teams
    n_rounds = data.n_rounds
    
    # --- 1. Get true strengths for the future round (t = n_rounds + 1) ---
    # This is a simplified continuation of the data generation process.
    # We take the last known trend and project it one step forward.
    attack_slopes = data.true_log_α[:, n_rounds] - data.true_log_α[:, n_rounds-1]
    defense_slopes = data.true_log_β[:, n_rounds] - data.true_log_β[:, n_rounds-1]

    true_log_α_future = data.true_log_α[:, n_rounds] + attack_slopes
    true_log_β_future = data.true_log_β[:, n_rounds] + defense_slopes

    true_log_α_future .-= mean(true_log_α_future)
    true_log_β_future .-= mean(true_log_β_future)

    true_α_future = exp.(true_log_α_future)
    true_β_future = exp.(true_log_β_future)

    # --- 2. Create all possible fixtures ---
    test_set = []
    for i in 1:n_teams, j in 1:n_teams
        if i == j continue end
        
        # --- 3. Calculate true Poisson rates and simulate one outcome ---
        λ_true = true_α_future[i] * true_β_future[j] * true_home_adv
        μ_true = true_α_future[j] * true_β_future[i]
        
        true_home_goals = rand(Poisson(λ_true))
        true_away_goals = rand(Poisson(μ_true))

        push!(test_set, (
            home_team=i, 
            away_team=j, 
            true_home_goals=true_home_goals, 
            true_away_goals=true_away_goals
        ))
    end
    
    return test_set
end



# ===================================================================
# ## 3. Model and Plotting Function Definitions
# ===================================================================

# --- Dynamic AR(1) Model ---
@model function dynamic_maher_model(
    home_team_ids, away_team_ids, home_goals, away_goals, n_teams, n_rounds
)
    ρ_attack ~ Beta(10, 1.5)
    ρ_defense ~ Beta(10, 1.5)

    μ_log_σ_attack ~ Normal(-2.5, 0.5) 
    τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5) 
    τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)

    z_log_σ_attack ~ MvNormal(zeros(n_teams), I)
    z_log_σ_defense ~ MvNormal(zeros(n_teams), I)
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack * τ_log_σ_attack
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense * τ_log_σ_defense
    σ_attack = exp.(log_σ_attack)
    σ_defense = exp.(log_σ_defense)
    
    log_home_adv ~ Normal(log(1.3), 0.2)

    initial_α_z ~ MvNormal(zeros(n_teams), I)
    initial_β_z ~ MvNormal(zeros(n_teams), I)
    log_α_raw_t0 = initial_α_z * sqrt(0.5)
    log_β_raw_t0 = initial_β_z * sqrt(0.5)

    z_α ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β_mat = reshape(z_β, n_teams, n_rounds)

    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)

    for t in 1:n_rounds
        if t == 1
            log_α_raw[:, 1] = log_α_raw_t0 .+ z_α_mat[:, 1] .* σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 .+ z_β_mat[:, 1] .* σ_defense
        else
            log_α_raw[:, t] = ρ_attack * log_α_raw[:, t-1] .+ z_α_mat[:, t] .* σ_attack
            log_β_raw[:, t] = ρ_defense * log_β_raw[:, t-1] .+ z_β_mat[:, t] .* σ_defense
        end

        log_α_t = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_t = log_β_raw[:, t] .- mean(log_β_raw[:, t])
        
        home_ids = home_team_ids[t]
        away_ids = away_team_ids[t]
        log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv
        log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
        home_goals[t] .~ LogPoisson.(log_λs)
        away_goals[t] .~ LogPoisson.(log_μs)
    end
end

# --- Static Model ---
@model function basic_maher_model(
    home_team_ids, away_team_ids, home_goals, away_goals, n_teams
)
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_home_adv ~ Normal(log(1.3), 0.2)
    
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    home_ids = home_team_ids
    away_ids = away_team_ids
    log_λs = log_α[home_ids] .+ log_β[away_ids] .+ log_home_adv
    log_μs = log_α[away_ids] .+ log_β[home_ids]

    home_goals .~ LogPoisson.(log_λs)
    away_goals .~ LogPoisson.(log_μs)
end

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

# ===================================================================
# . Helper and Plotting Functions
# ===================================================================

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


# ===================================================================
# ## 4. Extract the centered parameters from dynamic AR1 model
# ===================================================================

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



# ===================================================================
# ## 6. Predictive Evaluation Framework
# ===================================================================

"""
    generate_dynamic_predictions(chain_dynamic, data, test_set)

Generates one-step-ahead posterior predictive simulations for the dynamic model.
"""
function generate_dynamic_predictions(chain_dynamic, data, test_set)
    # 1. Get all necessary raw parameters from the chain
    raw_params = get_raw_parameters(chain_dynamic, data)
    log_α_raw_samples = raw_params.log_α_raw
    log_β_raw_samples = raw_params.log_β_raw
    σ_attack_samples = raw_params.σ_attack
    σ_defense_samples = raw_params.σ_defense
    
    ρ_attack_s = vec(Array(chain_dynamic["ρ_attack"]))
    ρ_defense_s = vec(Array(chain_dynamic["ρ_defense"]))
    
    n_samples = size(chain_dynamic, 1)
    predictions = Dict()

    for match in test_set
        i, j = match.home_team, match.away_team
        simulated_outcomes = []
        
        for s in 1:n_samples
            # 2. Propagate the RAW AR(1) process one step forward
            last_α_raw = log_α_raw_samples[s, :, end] # Use final raw state
            last_β_raw = log_β_raw_samples[s, :, end]
            
            future_α_raw = ρ_attack_s[s] .* last_α_raw .+ randn(data.n_teams) .* σ_attack_samples[s, :]
            future_β_raw = ρ_defense_s[s] .* last_β_raw .+ randn(data.n_teams) .* σ_defense_samples[s, :]

            # 3. Center for identifiability before calculating rates
            future_log_α = future_α_raw .- mean(future_α_raw)
            future_log_β = future_β_raw .- mean(future_β_raw)
            
            log_home_adv = mean(vec(Array(chain_dynamic["log_home_adv"])))

            # 4. Calculate rates and simulate one outcome
            log_λ = future_log_α[i] + future_log_β[j] + log_home_adv
            log_μ = future_log_α[j] + future_log_β[i]
            
            push!(simulated_outcomes, (hg=rand(Poisson(exp(log_λ))), ag=rand(Poisson(exp(log_μ)))))
        end
        predictions[match] = simulated_outcomes
    end
    return predictions
end
# end of mods 


"""
    generate_static_predictions(chain_static, test_set)

Generates posterior predictive simulations for the static model.
"""
function generate_static_predictions(chain_static, test_set)
    log_α_s = Array(group(chain_static, :log_α_raw)) .- mean(Array(group(chain_static, :log_α_raw)), dims=2)
    log_β_s = Array(group(chain_static, :log_β_raw)) .- mean(Array(group(chain_static, :log_β_raw)), dims=2)
    log_home_adv_s = vec(Array(chain_static["log_home_adv"]))

    n_samples = size(chain_static, 1)
    predictions = Dict()

    for match in test_set
        i, j = match.home_team, match.away_team
        simulated_outcomes = []

        for s in 1:n_samples
            log_λ = log_α_s[s, i] + log_β_s[s, j] + log_home_adv_s[s]
            log_μ = log_α_s[s, j] + log_β_s[s, i]
            push!(simulated_outcomes, (hg=rand(Poisson(exp(log_λ))), ag=rand(Poisson(exp(log_μ)))))
        end
        predictions[match] = simulated_outcomes
    end
    return predictions
end


"""
    simulations_to_probabilities(simulations)

Converts raw goal simulations into market probabilities for a single match.
"""
function simulations_to_probabilities(simulations)
    n_sims = length(simulations)
    
    # 1x2 Probs
    p_hw = count(s -> s.hg > s.ag, simulations) / n_sims
    p_d = count(s -> s.hg == s.ag, simulations) / n_sims
    p_aw = count(s -> s.hg < s.ag, simulations) / n_sims
    
    # Correct Score Probs
    score_counts = Dict()
    for s in simulations
        score = (s.hg, s.ag)
        score_counts[score] = get(score_counts, score, 0) + 1
    end
    p_cs = Dict(score => count / n_sims for (score, count) in score_counts)
    
    return (p_1x2 = [p_hw, p_d, p_aw], p_cs = p_cs)
end

"""
    evaluate_predictions(predictions, test_set; n_bins=10)

Calculates Log Score, Brier Score, and ECE for a set of predictions.
"""
function evaluate_predictions(predictions, test_set; n_bins=10)
    log_scores, brier_scores = [], []
    
    # For ECE
    bin_counts = zeros(n_bins)
    bin_correct = zeros(n_bins)
    bin_confidence = zeros(n_bins)
    
    for match in test_set
        true_hg, true_ag = match.true_home_goals, match.true_away_goals
        probs = simulations_to_probabilities(predictions[match])
        
        # --- Log Score ---
        true_score = (true_hg, true_ag)
        p_true_score = get(probs.p_cs, true_score, 1e-9) # Add smoother for zero prob
        push!(log_scores, log(p_true_score))
        
        # --- Brier Score ---
        outcome_vec = true_hg > true_ag ? [1,0,0] : (true_hg == true_ag ? [0,1,0] : [0,0,1])
        push!(brier_scores, sum((probs.p_1x2 - outcome_vec).^2))
        
        # --- ECE Data Aggregation (for 1x2 market) ---
        for (i, p) in enumerate(probs.p_1x2)
            if p == 0 continue end
            bin_idx = min(n_bins, ceil(Int, p * n_bins))
            bin_counts[bin_idx] += 1
            bin_confidence[bin_idx] += p
            if outcome_vec[i] == 1
                bin_correct[bin_idx] += 1
            end
        end
    end
    
    # --- Calculate final ECE ---
    ece = 0.0
    total_preds = sum(bin_counts)
    for i in 1:n_bins
        if bin_counts[i] > 0
            avg_conf = bin_confidence[i] / bin_counts[i]
            accuracy = bin_correct[i] / bin_counts[i]
            ece += bin_counts[i] * abs(avg_conf - accuracy)
        end
    end
    ece /= total_preds
    
    # Return average scores
    return (
        log_score = mean(log_scores),
        brier_score = mean(brier_scores),
        ece = ece
    )
end





"""
    display_results(dynamic_scores, static_scores)

Prints a formatted table comparing the evaluation scores of the two models.
"""
function display_results(dynamic_scores, static_scores)
    header = ["Metric", "Dynamic Model", "Static Model", "Winner"]
    data = Matrix(undef, 3, 4)

    metrics = ["Log Score", "Brier Score", "ECE"]
    ds = [dynamic_scores.log_score, dynamic_scores.brier_score, dynamic_scores.ece]
    ss = [static_scores.log_score, static_scores.brier_score, static_scores.ece]
    
    for i in 1:3
        data[i, 1] = metrics[i]
        data[i, 2] = @sprintf("%.4f", ds[i])
        data[i, 3] = @sprintf("%.4f", ss[i])
        
        # Higher is better for Log Score, lower is better for others
        is_better = i == 1 ? (ds[i] > ss[i]) : (ds[i] < ss[i])
        data[i, 4] = is_better ? "Dynamic" : "Static"
    end
    
    println("\n--- Predictive Performance Evaluation ---")
    pretty_table(data, header=header)
end



# ===================================================================
# ## notepad - main script follows after 
# ===================================================================
#  generate_multi_season_data

data = generate_multi_season_data(
    n_teams=10,
    n_seasons=3,
    rounds_per_season=30,
    season_to_season_volatility=0.02, # A higher value creates more dramatic shifts
    seed=42
)

println("Dimensions of true_log_α: ", size(data.true_log_α))
println("Total number of rounds simulated: ", data.n_rounds)


# --- Visualize the Team Strength Dynamics ---
plot(
    1:data.n_rounds,
    data.true_log_α[1:10, :]', 
    # label=["1","2","3","4","5","6","7","8","9","10"],
    xlabel="Round",
    ylabel="True Log Attacking Strength (log α)",
    title="Simulated Team Strengths Over Multiple Seasons",
    legend=:outertopright,
    linewidth=2
)

plot(
    1:data.n_rounds,
    data.true_log_β[1:10, :]', 
    # label=["1","2","3","4","5","6","7","8","9","10"],
    xlabel="Round",
    ylabel="True Log Defense Strength (log Defense)",
    title="Simulated Team Strengths Over Multiple Seasons",
    legend=:outertopright,
    linewidth=2
)


# ===================================================================
# ## 4. Main Script
# ===================================================================

# --- 4.1. Generate Data ---
data = generate_synthetic_data(n_teams=10, n_rounds=50)
# v2 data gen 
data = generate_multi_season_data(
    n_teams=10,
    n_seasons=3,
    rounds_per_season=50,
    season_to_season_volatility=0.02, # A higher value creates more dramatic shifts
    seed=42
)

# --- 4.2. Fit and Process DYNAMIC Model ---
dynamic_model_instance = dynamic_maher_model(
    data.home_team_ids, data.away_team_ids, data.home_goals, data.away_goals,
    data.n_teams, data.n_rounds
)
chain_dynamic = sample(dynamic_model_instance, NUTS(0.65), 500)

log_α_centered_reconstructed, log_β_centered_reconstructed = get_centered_parameters(chain_dynamic, data)


# --- 4.3. Fit and Process STATIC Model ---
home_team_ids_flat = reduce(vcat, data.home_team_ids)
away_team_ids_flat = reduce(vcat, data.away_team_ids)
home_goals_flat = reduce(vcat, data.home_goals)
away_goals_flat = reduce(vcat, data.away_goals)

static_model_instance = basic_maher_model(
    home_team_ids_flat, away_team_ids_flat, home_goals_flat, away_goals_flat, data.n_teams
)
chain_static = sample(static_model_instance, NUTS(0.65), 500)

log_α_raw_static = Array(group(chain_static, :log_α_raw))
log_α_static = log_α_raw_static .- mean(log_α_raw_static, dims=2)

log_β_raw_static = Array(group(chain_static, :log_β_raw))
log_β_static = log_β_raw_static .- mean(log_β_raw_static, dims=2)

# --- 4.4. Visualize and Compare ---
team_to_plot =3

comparison_plot = plot_compare_a_b(
    team_to_plot, 
    data, 
    log_α_centered_reconstructed, # Use centered results
    log_α_static, 
    log_β_centered_reconstructed, # Use centered results
    log_β_static
)

display(comparison_plot)



# ===================================================================
# ## 4.4. Visualize and Compare
# ===================================================================

team_to_plot = 1

# Use the dashboard plot function
dashboard_plot = plot_team_dashboard(
    team_to_plot, 
    data, 
    log_α_centered_reconstructed, 
    log_α_static, 
    log_β_centered_reconstructed, 
    log_β_static
)

# ===================================================================
# ## 4.5. Visualize and Compare Multiple Teams
# ===================================================================

# Define the list of teams you want to see
teams_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

# Generate the combined plot
multi_team_plot = plot_multiple_dashboards(
    teams_to_plot, 
    data, 
    log_α_centered_reconstructed, 
    log_α_static, 
    log_β_centered_reconstructed, 
    log_β_static
)

# savefig(multi_team_plot, "/home/james/bet_project/multi_team_dashboards_all.png")



#### mods 
# helper func 

# ===================================================================
# ## 5. Run Full Predictive Evaluation
# ===================================================================


# 1. Generate the hold-out test set
test_set = generate_test_set(data)

# 2. Generate predictions from both models
dynamic_preds = generate_dynamic_predictions(chain_dynamic, data, test_set)
static_preds = generate_static_predictions(chain_static, test_set)

# 3. Evaluate the predictions
dynamic_scores = evaluate_predictions(dynamic_preds, test_set)
static_scores = evaluate_predictions(static_preds, test_set)

# 4. Display the results
display_results(dynamic_scores, static_scores)
