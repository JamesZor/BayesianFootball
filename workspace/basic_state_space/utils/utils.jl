# workspace/basic_state_space/utils/utils.jl

module SSMUtils

using DataFrames
using Dates
using Random
using PrettyTables, Printf, Statistics
using Plots, StatsPlots, MCMCChains
using Distributions

export add_global_round_column!, generate_synthetic_data

"""
    add_global_round_column!(matches_df::DataFrame)

Adds a `:global_round` column in-place to the DataFrame.
"""
function add_global_round_column!(matches_df::DataFrame)
    sort!(matches_df, :match_date)
    num_matches = nrow(matches_df)
    global_rounds = Vector{Int}(undef, num_matches)
    
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(matches_df))
        home_team = row.home_team
        away_team = row.away_team

        if home_team in teams_in_current_round || away_team in teams_in_current_round
            global_round_counter += 1
            empty!(teams_in_current_round)
        end

        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team)
        push!(teams_in_current_round, away_team)
    end

    matches_df.global_round = global_rounds
    println("✅ Successfully added `:global_round` column. Found $(global_round_counter) unique time steps.")
    return matches_df
end

"""
    generate_synthetic_data(; n_teams=10, n_rounds=38, seed=123)

Generates a synthetic football season with known parameters.
"""
function generate_synthetic_data(;
    n_teams::Int=10, 
    n_rounds::Int=38, 
    seed::Int=123
)
    Random.seed!(seed)
    
    # --- Define True Parameters ---
    true_log_α = zeros(n_teams, n_rounds)
    true_log_β = zeros(n_teams, n_rounds)

    # Make two teams' strengths change over time
    for t in 1:n_rounds
        true_log_α[1, t] = -1.0 + 2.0 * (t / n_rounds) # Team 1 attack improves
        true_log_β[2, t] = -1.0 + 2.0 * (t / n_rounds) # Team 2 defense improves
    end
    
    # Apply sum-to-zero constraint
    for t in 1:n_rounds
        true_log_α[:, t] .-= mean(true_log_α[:, t])
        true_log_β[:, t] .-= mean(true_log_β[:, t])
    end
    
    true_home_adv = 1.3
    
    # --- Generate Match Schedule and Goals ---
    home_teams = []; away_teams = []; home_goals = []; away_goals = []; rounds = []
    
    for t in 1:n_rounds
        teams = shuffle(1:n_teams)
        for i in 1:2:n_teams
            ht = teams[i]; at = teams[i+1]
            
            λ_home = exp(true_log_α[ht, t] + true_log_β[at, t] + log(true_home_adv))
            λ_away = exp(true_log_α[at, t] + true_log_β[ht, t])
            
            push!(home_teams, ht); push!(away_teams, at)
            push!(home_goals, rand(Poisson(λ_home))); push!(away_goals, rand(Poisson(λ_away)))
            push!(rounds, t)
        end
    end

    return (
        home_team_ids=home_teams, away_team_ids=away_teams,
        home_goals=home_goals, away_goals=away_goals,
        n_teams=n_teams, n_rounds=n_rounds,
        true_log_α=true_log_α, true_log_β=true_log_β
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


end # end module
