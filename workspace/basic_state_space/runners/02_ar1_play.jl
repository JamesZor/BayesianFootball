# ===================================================================
# ## 1. Setup and Libraries
# ===================================================================
using Turing, Distributions, LinearAlgebra, Random
using Plots, StatsPlots, MCMCChains
using ReverseDiff, Memoization

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
# ## notepad - main script follows after 
# ===================================================================
#  generate_multi_season_data

data = generate_multi_season_data(
    n_teams=10,
    n_seasons=3,
    rounds_per_season=50,
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

# Reconstruct sigmas
μ_log_σ_attack_samples = vec(Array(chain_dynamic[:μ_log_σ_attack]))
τ_log_σ_attack_samples = vec(Array(chain_dynamic[:τ_log_σ_attack]))
z_log_σ_attack_samples = Array(group(chain_dynamic, :z_log_σ_attack))
log_σ_attack_samples = μ_log_σ_attack_samples .+ z_log_σ_attack_samples .* τ_log_σ_attack_samples
σ_attack_samples = exp.(log_σ_attack_samples)

μ_log_σ_defense_samples = vec(Array(chain_dynamic[:μ_log_σ_defense]))
τ_log_σ_defense_samples = vec(Array(chain_dynamic[:τ_log_σ_defense]))
z_log_σ_defense_samples = Array(group(chain_dynamic, :z_log_σ_defense))
log_σ_defense_samples = μ_log_σ_defense_samples .+ z_log_σ_defense_samples .* τ_log_σ_defense_samples
σ_defense_samples = exp.(log_σ_defense_samples)

# Reconstruct trajectories
ρ_attack_samples = vec(Array(chain_dynamic["ρ_attack"]))
ρ_defense_samples = vec(Array(chain_dynamic["ρ_defense"]))
initial_α_z_samples = Array(group(chain_dynamic, :initial_α_z))
initial_β_z_samples = Array(group(chain_dynamic, :initial_β_z))
z_α_samples = Array(group(chain_dynamic, :z_α))
z_β_samples = Array(group(chain_dynamic, :z_β))

n_samples, n_teams, n_rounds = size(chain_dynamic,1), data.n_teams, data.n_rounds
log_α_raw_reconstructed = Array{Float64}(undef, n_samples, n_teams, n_rounds)
log_β_raw_reconstructed = Array{Float64}(undef, n_samples, n_teams, n_rounds)

for s in 1:n_samples
    ρ_attack_s, σ_attack_s = ρ_attack_samples[s], σ_attack_samples[s, :]
    initial_α_z_s, z_α_mat_s = initial_α_z_samples[s, :], reshape(z_α_samples[s, :], n_teams, n_rounds)
    ρ_defense_s, σ_defense_s = ρ_defense_samples[s], σ_defense_samples[s, :]
    initial_β_z_s, z_β_mat_s = initial_β_z_samples[s, :], reshape(z_β_samples[s, :], n_teams, n_rounds)

    log_α_raw_t0_s = initial_α_z_s * sqrt(0.5)
    log_β_raw_t0_s = initial_β_z_s * sqrt(0.5)
    
    temp_log_α_raw = Matrix{Float64}(undef, n_teams, n_rounds)
    temp_log_β_raw = Matrix{Float64}(undef, n_teams, n_rounds)

    for t in 1:n_rounds
        if t == 1
            temp_log_α_raw[:, 1] = log_α_raw_t0_s .+ z_α_mat_s[:, 1] .* σ_attack_s
            temp_log_β_raw[:, 1] = log_β_raw_t0_s .+ z_β_mat_s[:, 1] .* σ_defense_s
        else
            temp_log_α_raw[:, t] = ρ_attack_s * temp_log_α_raw[:, t-1] .+ z_α_mat_s[:, t] .* σ_attack_s
            temp_log_β_raw[:, t] = ρ_defense_s * temp_log_β_raw[:, t-1] .+ z_β_mat_s[:, t] .* σ_defense_s
        end
    end
    log_α_raw_reconstructed[s, :, :] = temp_log_α_raw
    log_β_raw_reconstructed[s, :, :] = temp_log_β_raw
end

# Center the dynamic results for plotting
log_α_centered_reconstructed = similar(log_α_raw_reconstructed)
log_β_centered_reconstructed = similar(log_β_raw_reconstructed)
for s in 1:n_samples, t in 1:n_rounds
    log_α_centered_reconstructed[s, :, t] = log_α_raw_reconstructed[s, :, t] .- mean(log_α_raw_reconstructed[s, :, t])
    log_β_centered_reconstructed[s, :, t] = log_β_raw_reconstructed[s, :, t] .- mean(log_β_raw_reconstructed[s, :, t])
end

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
team_to_plot =4

comparison_plot = plot_compare_a_b(
    team_to_plot, 
    data, 
    log_α_centered_reconstructed, # Use centered results
    log_α_static, 
    log_β_centered_reconstructed, # Use centered results
    log_β_static
)

display(comparison_plot)



