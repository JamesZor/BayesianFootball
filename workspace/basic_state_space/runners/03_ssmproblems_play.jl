# ===================================================================
# ## 1. Setup and Libraries
# ===================================================================
using Turing
using SSMProblems
using GeneralisedFilters
using Distributions, LinearAlgebra, Random
using Plots, StatsPlots, MCMCChains
using PrettyTables, Printf, Statistics

# Configure Turing for better performance
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# ===================================================================
# ## 2. Data Generation (Copied from your script)
# ===================================================================

function generate_multi_season_data(;
    n_teams::Int=10,
    n_seasons::Int=3,
    rounds_per_season::Int=38,
    season_to_season_volatility::Float64=0.4,
    seed::Int=123
)
    Random.seed!(seed)
    total_rounds = n_seasons * rounds_per_season
    true_log_α = zeros(n_teams, total_rounds)
    true_log_β = zeros(n_teams, total_rounds)
    true_home_adv = 1.3
    teams = 1:n_teams

    current_round_index = 0
    for s in 1:n_seasons
        attack_slopes = rand(Normal(0, season_to_season_volatility), n_teams)
        defense_slopes = rand(Normal(0, season_to_season_volatility), n_teams)
        for r in 1:rounds_per_season
            current_round_index += 1
            t = current_round_index
            if t == 1
                true_log_α[:, t] = attack_slopes
                true_log_β[:, t] = defense_slopes
            else
                true_log_α[:, t] = true_log_α[:, t-1] + attack_slopes
                true_log_β[:, t] = true_log_β[:, t-1] + defense_slopes
            end
            true_log_α[:, t] .-= mean(true_log_α[:, t])
            true_log_β[:, t] .-= mean(true_log_β[:, t])
        end
    end

    true_α = exp.(true_log_α)
    true_β = exp.(true_log_β)
    home_team_ids_all, away_team_ids_all = [], []
    home_goals_all, away_goals_all = [], []

    for t in 1:total_rounds
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
        n_rounds=total_rounds,
        true_log_α=true_log_α,
        true_log_β=true_log_β
    )
end 

# ===================================================================
# ## 3. Model Definition using SSMProblems.jl
# ===================================================================

# The state is a vector containing all log_α and log_β values
# xₜ = [log_α₁,...,log_αₙ, log_β₁,...,log_βₙ]
const FootballState = Vector{<:Real}

# --- 3.1. Latent Dynamics: The AR(1) process ---
# This struct holds the static parameters for the AR(1) transition.
struct AR1Dynamics{V<:AbstractVector} <: LatentDynamics
    ρ_α::Real
    ρ_β::Real
    σ_α::V
    σ_β::V
end

# This function defines how the state evolves from t-1 to t.
function SSMProblems.distribution(dyn::AR1Dynamics, t::Int, prev_state::FootballState; kwargs...)
    n_teams = length(dyn.σ_α)
    
    # Unpack previous state
    log_α_prev = @view prev_state[1:n_teams]
    log_β_prev = @view prev_state[n_teams+1:end]
    
    # Calculate new mean based on AR(1) process
    μ_α_new = dyn.ρ_α .* log_α_prev
    μ_β_new = dyn.ρ_β .* log_β_prev
    μ_new = [μ_α_new; μ_β_new]
    
    # Covariance is diagonal, with team-specific volatilities
    Σ_new = Diagonal(vcat(dyn.σ_α, dyn.σ_β).^2)
    
    return MvNormal(μ_new, Σ_new)
end

# --- 3.2. Observation Process: The Poisson Likelihood ---
# This struct represents the observation model. It can be empty since
# we will pass the actual match data in via keyword arguments.
struct MaherObservationProcess <: ObservationProcess end

# This function calculates the log-likelihood of the observed goals for a
# given round, given a specific state (a particle).

function SSMProblems.logdensity(
    proc::MaherObservationProcess, t::Int, state::FootballState, obs; 
    data, log_home_adv, kwargs... # Added kwargs... to accept unused keyword arguments
)
    n_teams = div(length(state), 2)
    
    log_α_t = state[1:n_teams] .- mean(@view state[1:n_teams])
    log_β_t = state[n_teams+1:end] .- mean(@view state[n_teams+1:end])
    
    home_ids = data.home_team_ids[t]
    away_ids = data.away_team_ids[t]
    home_goals = data.home_goals[t]
    away_goals = data.away_goals[t]

    total_loglik = 0.0
    for i in 1:length(home_ids)
        h_team = home_ids[i]
        a_team = away_ids[i]
        
        log_λ = log_α_t[h_team] + log_β_t[a_team] + log_home_adv
        log_μ = log_α_t[a_team] + log_β_t[h_team]
        
        total_loglik += logpdf(LogPoisson(log_λ), home_goals[i])
        total_loglik += logpdf(LogPoisson(log_μ), away_goals[i])
    end
    
    return total_loglik
end

# --- 3.3. Initial State Prior ---
# Defines the distribution of the state at t=0.

struct FootballPrior <: StatePrior
    dist::MvNormal
end

function SSMProblems.distribution(prior::FootballPrior; kwargs...)
    return prior.dist
end

# ===================================================================
# ## 4. Inference Definition: Particle MCMC with Turing
# ===================================================================

@model function pmmh_football_model(data, n_particles)
    # --- Priors for Static Parameters ---
    ρ_attack ~ Beta(10, 1.5)
    ρ_defense ~ Beta(10, 1.5)

    μ_log_σ_attack ~ Normal(-2.5, 0.5) 
    τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5) 
    τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)

    z_log_σ_attack ~ MvNormal(zeros(data.n_teams), I)
    z_log_σ_defense ~ MvNormal(zeros(data.n_teams), I)
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense
    σ_attack = exp.(log_σ_attack)
    σ_defense = exp.(log_σ_defense)
    
    log_home_adv ~ Normal(log(1.3), 0.2)

    # --- Construct the State Space Model ---
    dyn = AR1Dynamics(ρ_attack, ρ_defense, σ_attack, σ_defense)
    obs = MaherObservationProcess()
    
    prior_dist = MvNormal(zeros(2 * data.n_teams), 0.5 * I)
    prior = FootballPrior(prior_dist)
    
    ssm = StateSpaceModel(prior, dyn, obs)
    
    dummy_obs = fill(nothing, data.n_rounds)
    
    # CORRECTED: Unpack the tuple returned by `filter`. We only need the second element.
    # The `_` is a standard Julia convention to indicate we are discarding the first element.
    _, loglik_estimate = GeneralisedFilters.filter(
        ssm,
        BootstrapFilter(n_particles),
        dummy_obs; 
        data=data, 
        log_home_adv=log_home_adv
    )

    # --- Add Log-Likelihood to Turing ---
    Turing.@addlogprob! loglik_estimate
end

# ===================================================================
# ## 5. Main Execution Script
# ===================================================================
# --- 5.1. Generate Synthetic Data ---
println("--- Generating Synthetic Data ---")
data = generate_multi_season_data(
    n_teams=10,
    n_seasons=2,
    rounds_per_season=20, # Reduced for faster run-time
    season_to_season_volatility=0.03,
    seed=42
)

# --- 5.2. Define and Sample the Model ---
n_particles = 500 # Increase for better accuracy, decrease for speed
n_samples = 1000   # MCMC samples

model_instance = pmmh_football_model(data, n_particles)

# Use the PMMH sampler from Turing, which is designed for this.
# It requires a particle filter to be run for the parameters inside `Turing.@model`.
chain = sample(model_instance, MH(), n_samples)

println("\n--- Sampling Complete ---")
display(chain)

# --- 5.3. Visualize Results ---
# We can plot the posteriors of the static parameters.
# We can't easily check parameter recovery for the hierarchical `σ`s,
# but we can check the persistence `ρ` parameters.
p = plot(chain, [:ρ_attack, :ρ_defense, :log_home_adv], 
         title="Posteriors for Static Parameters", size=(800, 600))
display(p)


####
using StatsBase
using GeneralisedFilters 


# ===================================================================
# ## 1. Define the Final, Correct Callback Structure
# ===================================================================

using StatsBase
using GeneralisedFilters

# (The TrajectoryCallback struct definition remains the same as the last version)
mutable struct TrajectoryCallback <: GeneralisedFilters.AbstractCallback
    history_ref::Ref{Vector{Any}}
end

function (cb::TrajectoryCallback)(t, particles)
    push!(cb.history_ref[], deepcopy(particles))
end


function reconstruct_dynamic_trajectory(chain, data, n_particles)
    println("\n--- Reconstructing dynamic trajectory ---")
    
    # 1. Extract posterior mean static parameters
    params = (
        ρ_attack = mean(chain["ρ_attack"]),
        ρ_defense = mean(chain["ρ_defense"]),
        μ_log_σ_attack = mean(chain["μ_log_σ_attack"]),
        τ_log_σ_attack = mean(chain["τ_log_σ_attack"]),
        μ_log_σ_defense = mean(chain["μ_log_σ_defense"]),
        τ_log_σ_defense = mean(chain["τ_log_σ_defense"]),
        z_log_σ_attack = vec(mean(Array(group(chain, "z_log_σ_attack")), dims=1)),
        z_log_σ_defense = vec(mean(Array(group(chain, "z_log_σ_defense")), dims=1)),
        log_home_adv = mean(chain["log_home_adv"]),
    )

    log_σ_attack = params.μ_log_σ_attack .+ params.z_log_σ_attack .* params.τ_log_σ_attack
    log_σ_defense = params.μ_log_σ_defense .+ params.z_log_σ_defense .* params.τ_log_σ_defense
    σ_attack = exp.(log_σ_attack)
    σ_defense = exp.(log_σ_defense)

    # 2. Build the State-Space Model
    dyn = AR1Dynamics(params.ρ_attack, params.ρ_defense, σ_attack, σ_defense)
    obs = MaherObservationProcess()
    prior = FootballPrior(MvNormal(zeros(2 * data.n_teams), 0.5 * I))
    ssm = StateSpaceModel(prior, dyn, obs)

    # 3. Create the history vector and the callback object holding a Ref to it
    particle_history_vec = Any[]
    callback_obj = TrajectoryCallback(Ref(particle_history_vec))

    # 4. Run the filter
    dummy_obs = fill(nothing, data.n_rounds)
    GeneralisedFilters.filter(
        ssm,
        BootstrapFilter(n_particles),
        dummy_obs; 
        data=data, 
        log_home_adv=params.log_home_adv,
        callback=callback_obj
    )
    
    # 5. Process the collected history
    # CORRECTED: Access the vector via `callback_obj.history_ref[]`
    n_states = 2 * data.n_teams
    n_rounds = data.n_rounds
    trajectory_history = zeros(n_states, n_rounds)
    for t in 1:n_rounds
        particles = particle_history_vec[t]
        states_matrix = hcat(particles.particles...)
        weights = particles.weights
        trajectory_history[:, t] = mapslices(x -> mean(x, Weights(weights)), states_matrix, dims=2)
    end

    # 6. Unpack and return results
    n_teams = data.n_teams
    log_α_raw = trajectory_history[1:n_teams, :]
    log_β_raw = trajectory_history[n_teams+1:end, :]

    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)
    for t in 1:n_rounds
        log_α_centered[:, t] = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_centered[:, t] = log_β_raw[:, t] .- mean(log_β_raw[:, t])
    end
    
    log_α_dynamic_mean = reshape(log_α_centered', 1, n_teams, n_rounds)
    log_β_dynamic_mean = reshape(log_β_centered', 1, n_teams, n_rounds)
    
    println("--- Trajectory reconstruction complete ---")
    return log_α_dynamic_mean, log_β_dynamic_mean
end
# ===================================================================
# ## Main Plotting Execution
# ===================================================================

# This assumes you have already run the sampler and have `chain` and `data` objects.
# It also assumes the `get_team_goal_history` and `plot_team_dashboard` functions
# from your original script are defined.

# 1. Reconstruct the dynamic estimates for log_α and log_β
# We need to reshape the output to match the format the plotting function expects
# The plotting function expects an array with dimensions (samples, teams, rounds).
# Since we only have the mean, we'll create a dummy "samples" dimension of size 1.
log_α_dynamic, log_β_dynamic = reconstruct_dynamic_trajectory(chain, data, n_particles) # Use more particles for a smoother plot

# 2. Create "pseudo-static" estimates for comparison
# We'll average the dynamic estimates over all rounds to get a single static value.
log_α_static = mean(log_α_dynamic, dims=3)
log_β_static = mean(log_β_dynamic, dims=3)

# 3. Choose a team and generate the plot
team_to_plot = 1
dashboard_plot = plot_team_dashboard(
    team_to_plot, 
    data, 
    log_α_dynamic, 
    log_α_static, 
    log_β_dynamic, 
    log_β_static
)

display(dashboard_plot)
