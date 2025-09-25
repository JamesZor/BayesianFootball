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


function data_to_gpu(data)
    if !CUDA.functional()
        return data
    end
    return (home_team_ids=[cu(t) for t in data.home_team_ids], away_team_ids=[cu(t) for t in data.away_team_ids], home_goals=[cu(t) for t in data.home_goals], away_goals=[cu(t) for t in data.away_goals], n_teams=data.n_teams, n_rounds=data.n_rounds, true_log_α=data.true_log_α, true_log_β=data.true_log_β)
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

function SSMProblems.logdensity(
    proc::MaherObservationProcess, t::Int, state::FootballState, obs; 
    data, log_home_adv, kwargs...
)
    n_teams = div(length(state), 2)
    
    # Move the particle's state vector to the GPU
    state_gpu = cu(state)
    
    # --- THE FIX IS HERE ---
    # 1. Create views for the alpha and beta parts of the state vector.
    log_α_view = @view state_gpu[1:n_teams]
    log_β_view = @view state_gpu[n_teams+1:end]
    
    # 2. Now, use these views to calculate the centered parameters.
    log_α_t = log_α_view .- mean(log_α_view)
    log_β_t = log_β_view .- mean(log_β_view)
    # -----------------------
    
    # Get the data for the current time step 't' (already on GPU)
    home_ids = data.home_team_ids[t]
    away_ids = data.away_team_ids[t]
    home_goals = data.home_goals[t]
    away_goals = data.away_goals[t]

    # Vectorized calculation (all arrays are now CuArrays)
    log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv
    log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
    
    # The logpdf and sum are performed efficiently on the GPU
    total_loglik = sum(logpdf.(LogPoisson.(log_λs), home_goals)) +
                   sum(logpdf.(LogPoisson.(log_μs), away_goals))
    
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
n_particles = 2000 # Increase for better accuracy, decrease for speed
n_samples = 500   # MCMC samples
# v2
using MCMCChains

n_samples = 1000
n_chains = 4 # Run 4 chains

# The 'MCMCThreads()' part enables parallel sampling
model_instance = pmmh_football_model(data, n_particles)

# Use the PMMH sampler from Turing, which is designed for this.
# It requires a particle filter to be run for the parameters inside `Turing.@model`.
chain = sample(model_instance, MH(), n_samples)

chain = sample(model_instance, MH(), MCMCThreads(), n_samples, n_chains)
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

# Copied from 02_ar1_play.jl for convenience
function get_team_goal_history(team_number::Int, data)
    n_rounds = data.n_rounds
    goals_scored = Vector{Int}(undef, n_rounds)
    goals_conceded = Vector{Int}(undef, n_rounds)

    for t in 1:n_rounds
        home_pos = findfirst(isequal(team_number), data.home_team_ids[t])
        away_pos = findfirst(isequal(team_number), data.away_team_ids[t])

        if !isnothing(home_pos)
            goals_scored[t] = data.home_goals[t][home_pos]
            goals_conceded[t] = data.away_goals[t][home_pos]
        elseif !isnothing(away_pos)
            goals_scored[t] = data.away_goals[t][away_pos]
            goals_conceded[t] = data.home_goals[t][away_pos]
        else
            goals_scored[t] = 0 
            goals_conceded[t] = 0
        end
    end
    return goals_scored, goals_conceded
end

# Copied from 02_ar1_play.jl
function plot_team_dashboard(team_number, data, α_dynamic, β_dynamic)
    goals_scored, goals_conceded = get_team_goal_history(team_number, data)
    p = plot(layout=(2, 2), size=(1400, 800), legend=:topleft, titlefontsize=11, tickfontsize=8, link=:x)

    # Attack Plot
    plot!(p[1, 1], 1:data.n_rounds, data.true_log_α[team_number, :], label="True log α", lw=3, color=:black, title="Team $team_number Attacking Parameter (log α)", ylabel="Parameter Value")
    plot!(p[1, 1], 1:data.n_rounds, α_dynamic[1, team_number, :], label="SSM Estimate", color=:dodgerblue)
    
    # Goals Scored Plot
    bar!(p[2, 1], 1:data.n_rounds, goals_scored, label="Goals Scored", color=:dodgerblue, alpha=0.7, ylabel="Goals", ylims=(0, max(maximum(goals_scored), maximum(goals_conceded)) + 1))
    
    # Defense Plot
    plot!(p[1, 2], 1:data.n_rounds, data.true_log_β[team_number, :], label="True log β", lw=3, color=:black, title="Team $team_number Defensive Parameter (log β)")
    plot!(p[1, 2], 1:data.n_rounds, β_dynamic[1, team_number, :], label="SSM Estimate", color=:crimson)
    
    # Goals Conceded Plot
    bar!(p[2, 2], 1:data.n_rounds, goals_conceded, label="Goals Conceded", color=:crimson, alpha=0.7, ylims=(0, max(maximum(goals_scored), maximum(goals_conceded)) + 1))
    
    return p
end


####
# ===================================================================
# ## Corrected Callback and Trajectory Reconstruction
# ===================================================================

using StatsBase
using GeneralisedFilters
using Random # Make sure Random is imported for default_rng()

# 1. Correctly define the TrajectoryCallback
#    - It should have a constructor for initialization.
#    - The function call signature must match one of the triggers from GeneralisedFilters.jl,
#      like `PostUpdateCallback`, to be activated at the right step.

mutable struct TrajectoryCallback <: GeneralisedFilters.AbstractCallback
    # Stores the history of particles at each step
    history::Vector{Any}
end

# Add a constructor to easily initialize the callback
TrajectoryCallback() = TrajectoryCallback([])

# Define the method that will be called by the filter loop.
# This signature ensures it runs *after* the 'update' step for each observation `t`.
function (cb::TrajectoryCallback)(
    model,
    filter_algo,
    t::Integer,
    particles, # This is the state, a `WeightedParticles` object
    observation,
    ::GeneralisedFilters.PostUpdateCallback; # The specific trigger
    kwargs...
)
    # Save a copy of the particle state at time t
    push!(cb.history, deepcopy(particles))
end


# 2. Update the reconstruction function to use the new callback correctly

function reconstruct_dynamic_trajectory(chain, data, n_particles)
    println("\n--- Reconstructing dynamic trajectory ---")

    # 1. Extract posterior mean of static parameters from the MCMC chain
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

    # 2. Build the State-Space Model with these mean parameters
    dyn = AR1Dynamics(params.ρ_attack, params.ρ_defense, σ_attack, σ_defense)
    obs = MaherObservationProcess()
    prior = FootballPrior(MvNormal(zeros(2 * data.n_teams), 0.5 * I))
    ssm = StateSpaceModel(prior, dyn, obs)

    # 3. Initialize the corrected callback
    callback_obj = TrajectoryCallback()

    # 4. Run the particle filter with the callback
    dummy_obs = fill(nothing, data.n_rounds)
    # The filter function returns the final state and the log-likelihood estimate
    final_state, ll = GeneralisedFilters.filter(
        Random.default_rng(), # It's good practice to provide a random number generator
        ssm,
        BootstrapFilter(n_particles),
        dummy_obs;
        data=data,
        log_home_adv=params.log_home_adv,
        callback=callback_obj # Pass the callback object
    )

    # 5. Process the collected history from the callback
    n_states = 2 * data.n_teams
    n_rounds = data.n_rounds
    trajectory_history = zeros(n_states, n_rounds)
    for t in 1:n_rounds
        # The history is now stored directly in the callback object
        particles = callback_obj.history[t]
        
        # Extract states and weights for computing the mean
        states_matrix = hcat(particles.particles...)
        particle_weights = StatsBase.weights(particles) # Use the weights() function
        
        # Calculate the weighted mean for the state at time t
        for i in 1:n_states
            trajectory_history[i, t] = mean(states_matrix[i, :], particle_weights)
        end
    end

    # 6. Unpack and return results for plotting
    n_teams = data.n_teams
    log_α_raw = trajectory_history[1:n_teams, :]
    log_β_raw = trajectory_history[n_teams+1:end, :]

    # Center the parameters for identifiability
    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)
    for t in 1:n_rounds
        log_α_centered[:, t] = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_centered[:, t] = log_β_raw[:, t] .- mean(log_β_raw[:, t])
    end

    # Reshape for compatibility with your plotting functions
    log_α_dynamic_mean = reshape(log_α_centered', 1, n_teams, n_rounds)
    log_β_dynamic_mean = reshape(log_β_centered', 1, n_teams, n_rounds)

    println("--- Trajectory reconstruction complete ---")
    return log_α_dynamic_mean, log_β_dynamic_mean
end

n_particles_viz = 500

log_α_dynamic, log_β_dynamic = reconstruct_dynamic_trajectory(chain, data, n_particles_viz)



team_to_plot = 1
dashboard_plot = plot_team_dashboard(team_to_plot, data, log_α_dynamic, log_β_dynamic)






####


# ===================================================================
# ## 1. Setup and Libraries
# ===================================================================
using Turing
using SSMProblems
using GeneralisedFilters
using Distributions, LinearAlgebra, Random, Statistics
using Plots, StatsPlots, MCMCChains
using PrettyTables, Printf
using CUDA
using LogExpFunctions # For logsumexp
using StatsBase # For weights

# Configure Turing for better performance
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Check for GPU
if CUDA.functional()
    CUDA.allowscalar(false) # Important for performance!
    println("✅ GPU is ready! The model will run on the GPU.")
else
    println("⚠️ GPU not found. The model will run on the CPU.")
end

# ===================================================================
# ## 2. Data Generation & GPU Transfer
# ===================================================================
#
function generate_multi_season_data(; n_teams::Int=10, n_seasons::Int=3, rounds_per_season::Int=38, season_to_season_volatility::Float64=0.4, seed::Int=123)
    Random.seed!(seed)
    total_rounds = n_seasons * rounds_per_season
    true_log_α = zeros(n_teams, total_rounds)
    true_log_β = zeros(n_teams, total_rounds)
    true_home_adv = 1.3
    teams = 1:n_teams
    home_team_ids_all = Vector{Vector{Int}}()
    away_team_ids_all = Vector{Vector{Int}}()
    home_goals_all = Vector{Vector{Int}}()
    away_goals_all = Vector{Vector{Int}}()
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
                true_log_α[:, t] = true_log_α[:, t-1] .+ attack_slopes
                true_log_β[:, t] = true_log_β[:, t-1] .+ defense_slopes
            end
            true_log_α[:, t] .-= mean(true_log_α[:, t])
            true_log_β[:, t] .-= mean(true_log_β[:, t])
        end
    end
    true_α = exp.(true_log_α)
    true_β = exp.(true_log_β)
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
    return (home_team_ids=home_team_ids_all, away_team_ids=away_team_ids_all, home_goals=home_goals_all, away_goals=away_goals_all, n_teams=n_teams, n_rounds=total_rounds, true_log_α=true_log_α, true_log_β=true_log_β)
end

function data_to_gpu(data)
    if !CUDA.functional()
        return data
    end
    return (home_team_ids=[cu(t) for t in data.home_team_ids], away_team_ids=[cu(t) for t in data.away_team_ids], home_goals=[cu(t) for t in data.home_goals], away_goals=[cu(t) for t in data.away_goals], n_teams=data.n_teams, n_rounds=data.n_rounds, true_log_α=data.true_log_α, true_log_β=data.true_log_β)
end

# ===================================================================
# ## 3. State-Space Model Definition
# ===================================================================

# --- Structs for the SSM components ---
struct AR1Dynamics{V<:AbstractVector} <: LatentDynamics
    ρ_α::Real; ρ_β::Real; σ_α::V; σ_β::V
end

struct MaherObservationProcess <: ObservationProcess end

struct FootballPrior <: StatePrior
    dist::MvNormal
end

SSMProblems.distribution(prior::FootballPrior; kwargs...) = prior.dist

# --- BATCH Implementations for GPU Performance ---
function SSMProblems.batch_simulate(rng::AbstractRNG, dyn::AR1Dynamics, t::Int, prev_states::CuArray; kwargs...)
    n_teams = length(dyn.σ_α)
    n_particles = size(prev_states, 2)
    log_α_prev = @view prev_states[1:n_teams, :]
    log_β_prev = @view prev_states[n_teams+1:end, :]
    μ_α_new = dyn.ρ_α .* log_α_prev
    μ_β_new = dyn.ρ_β .* log_β_prev
    μ_new = vcat(μ_α_new, μ_β_new)
    Σ_new_diag = vcat(dyn.σ_α, dyn.σ_β) # NOTE: Assumes σ is already on the correct device
    noise = CUDA.randn(eltype(μ_new), 2 * n_teams, n_particles) .* Σ_new_diag
    return μ_new .+ noise
end

function SSMProblems.batch_logdensity(proc::MaherObservationProcess, t::Int, states::CuArray, obs; data, log_home_adv, kwargs...)
    n_teams = div(size(states, 1), 2)
    log_α_raw = @view states[1:n_teams, :]
    log_β_raw = @view states[n_teams+1:end, :]
    
    # --- THE FIX IS HERE: Use GPU-native sum and division for mean ---
    log_α_t = log_α_raw .- (sum(log_α_raw; dims=1) ./ n_teams)
    log_β_t = log_β_raw .- (sum(log_β_raw; dims=1) ./ n_teams)
    # -----------------------------------------------------------------
    
    home_ids = data.home_team_ids[t]
    away_ids = data.away_team_ids[t]
    home_goals = data.home_goals[t]
    away_goals = data.away_goals[t]
    
    log_λs = log_α_t[home_ids, :] .+ log_β_t[away_ids, :] .+ log_home_adv
    log_μs = log_α_t[away_ids, :] .+ log_β_t[home_ids, :]
    
    loglik_hg = sum(logpdf.(LogPoisson.(log_λs), home_goals); dims=1)
    loglik_ag = sum(logpdf.(LogPoisson.(log_μs), away_goals); dims=1)
    
    return loglik_hg .+ loglik_ag
end

# ===================================================================
# ## 4. Custom GPU Particle Filter
# ===================================================================

# --- A GPU-native container for particles ---
mutable struct BatchParticles{P<:CuArray, W<:CuArray}
    particles::P
    log_weights::W
    ancestors::CuVector{Int64}
end

# --- Hooks to teach GeneralisedFilters how to resample our container ---
StatsBase.weights(state::BatchParticles) = softmax(state.log_weights)

function GeneralisedFilters.construct_new_state(states::BatchParticles, idxs::CuArray{<:Integer})
    new_particles = states.particles[:, idxs]
    new_log_weights = CUDA.zeros(Float32, length(idxs))
    return BatchParticles(new_particles, new_log_weights, idxs)
end

# --- The custom filter algorithm struct ---
struct BatchBootstrapFilter{RS<:GeneralisedFilters.AbstractResampler} <: GeneralisedFilters.AbstractParticleFilter
    N::Int
    resampler::RS
end

BatchBootstrapFilter(N::Int; resampler=GeneralisedFilters.Systematic()) = BatchBootstrapFilter(N, resampler)

# --- Methods to define the filter's behavior ---
function GeneralisedFilters.initialise(rng::AbstractRNG, model::StateSpaceModel, algo::BatchBootstrapFilter; kwargs...)
    prior_dist = SSMProblems.distribution(model.prior; kwargs...)
    particles = CUDA.functional() ? cu(rand(rng, prior_dist, algo.N)) : rand(rng, prior_dist, algo.N)
    log_weights = CUDA.functional() ? CUDA.zeros(Float32, algo.N) : zeros(Float32, algo.N)
    ancestors = CUDA.functional() ? CuArray(Int64(1):Int64(algo.N)) : collect(Int64(1):Int64(algo.N))
    return BatchParticles(particles, log_weights, ancestors)
end

function GeneralisedFilters.predict(rng::AbstractRNG, model::StateSpaceModel, algo::BatchBootstrapFilter, t::Integer, state::BatchParticles, obs; kwargs...)
    state.particles = SSMProblems.batch_simulate(rng, model.dyn, t, state.particles; kwargs...)
    return state
end

function GeneralisedFilters.update(model::StateSpaceModel, algo::BatchBootstrapFilter, t::Integer, state::BatchParticles, obs; kwargs...)
    log_increments_gpu = SSMProblems.batch_logdensity(model.obs, t, state.particles, obs; kwargs...)
    state.log_weights .+= vec(log_increments_gpu)
    log_marginalisation = logsumexp(state.log_weights)
    ll_increment = log_marginalisation - log(algo.N)
    return state, ll_increment
end

# ===================================================================
# ## 5. Turing Model Definition
# ===================================================================

@model function pmmh_football_model(data, n_particles)
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

    # Conditionally move σ vectors to GPU inside the model
    σ_attack_dev = CUDA.functional() ? cu(σ_attack) : σ_attack
    σ_defense_dev = CUDA.functional() ? cu(σ_defense) : σ_defense
    dyn = AR1Dynamics(ρ_attack, ρ_defense, σ_attack_dev, σ_defense_dev)
    
    obs = MaherObservationProcess()
    prior = FootballPrior(MvNormal(zeros(2 * data.n_teams), 0.5 * I))
    ssm = StateSpaceModel(prior, dyn, obs)
    dummy_obs = fill(nothing, data.n_rounds)
    
    _, loglik_estimate = GeneralisedFilters.filter(
        ssm,
        BatchBootstrapFilter(n_particles),
        dummy_obs;
        data=data,
        log_home_adv=log_home_adv
    )
    Turing.@addlogprob! loglik_estimate
end

# ===================================================================
# ## 6. Main Execution Script
# ===================================================================
println("--- Generating Synthetic Data ---")
cpu_data = generate_multi_season_data(n_teams=10, n_seasons=2, rounds_per_season=20, seed=42)
data = data_to_gpu(cpu_data)

n_particles = 500
n_samples = 100
n_chains = 2

model_instance = pmmh_football_model(data, n_particles)

println("--- Starting MCMC Sampling with $n_particles particles ---")
chain = sample(model_instance, PG(n_particles), MCMCThreads(), n_samples, n_chains)
# chain = sample(model_instance, HC(), MCMCThreads(), n_samples, n_chains)
println("\n--- Sampling Complete ---")
display(chain)




GC.gc()
CUDA.reclaim()

# ===================================================================
# ## 1. Setup and Libraries
# ===================================================================
using Turing
using SSMProblems
using GeneralisedFilters
using Distributions, LinearAlgebra, Random, Statistics
using Plots, StatsPlots, MCMCChains
using PrettyTables, Printf
using CUDA
using LogExpFunctions # For softmax and logsumexp

# Configure Turing for better performance
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Check for GPU
if CUDA.functional()
    CUDA.allowscalar(false)
    println("✅ GPU is ready! The model will run on the GPU.")
else
    println("⚠️ GPU not found. The model will run on the CPU.")
end

# ===================================================================
# ## 2. Data Generation and GPU Transfer
# ===================================================================

function generate_multi_season_data(; n_teams::Int=10, n_seasons::Int=3, rounds_per_season::Int=38, season_to_season_volatility::Float64=0.4, seed::Int=123)
    Random.seed!(seed)
    total_rounds = n_seasons * rounds_per_season
    true_log_α = zeros(n_teams, total_rounds)
    true_log_β = zeros(n_teams, total_rounds)
    true_home_adv = 1.3
    teams = 1:n_teams
    home_team_ids_all = Vector{Vector{Int}}()
    away_team_ids_all = Vector{Vector{Int}}()
    home_goals_all = Vector{Vector{Int}}()
    away_goals_all = Vector{Vector{Int}}()
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
                true_log_α[:, t] = true_log_α[:, t-1] .+ attack_slopes
                true_log_β[:, t] = true_log_β[:, t-1] .+ defense_slopes
            end
            true_log_α[:, t] .-= mean(true_log_α[:, t])
            true_log_β[:, t] .-= mean(true_log_β[:, t])
        end
    end
    true_α = exp.(true_log_α)
    true_β = exp.(true_log_β)
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
    return (home_team_ids=home_team_ids_all, away_team_ids=away_team_ids_all, home_goals=home_goals_all, away_goals=away_goals_all, n_teams=n_teams, n_rounds=total_rounds, true_log_α=true_log_α, true_log_β=true_log_β)
end

function data_to_gpu(data)
    if !CUDA.functional()
        return data
    end
    return (home_team_ids=[cu(t) for t in data.home_team_ids], away_team_ids=[cu(t) for t in data.away_team_ids], home_goals=[cu(t) for t in data.home_goals], away_goals=[cu(t) for t in data.away_goals], n_teams=data.n_teams, n_rounds=data.n_rounds, true_log_α=data.true_log_α, true_log_β=data.true_log_β)
end


# ===================================================================
# ## 3. Model Definition using SSMProblems.jl
# ===================================================================

const FootballState = Vector{<:Real}

struct AR1Dynamics{V<:AbstractVector} <: LatentDynamics
    ρ_α::Real; ρ_β::Real; σ_α::V; σ_β::V
end

struct MaherObservationProcess <: ObservationProcess end

struct FootballPrior <: StatePrior
    dist::MvNormal
end



function SSMProblems.distribution(dyn::AR1Dynamics, t::Int, prev_state::FootballState; kwargs...)
    n_teams = length(dyn.σ_α)
    log_α_prev = @view prev_state[1:n_teams]
    log_β_prev = @view prev_state[n_teams+1:end]
    μ_α_new = dyn.ρ_α .* log_α_prev
    μ_β_new = dyn.ρ_β .* log_β_prev
    μ_new = [μ_α_new; μ_β_new]
    Σ_new = Diagonal(vcat(dyn.σ_α, dyn.σ_β).^2)
    return MvNormal(μ_new, Σ_new)
end

# --- GPU-Accelerated logdensity function ---
function SSMProblems.logdensity(proc::MaherObservationProcess, t::Int, state::FootballState, obs; data, log_home_adv, kwargs...)
    n_teams = div(length(state), 2)
    state_gpu = cu(state)
    log_α_view = @view state_gpu[1:n_teams]
    log_β_view = @view state_gpu[n_teams+1:end]
    log_α_t = log_α_view .- mean(log_α_view)
    log_β_t = log_β_view .- mean(log_β_view)
    home_ids, away_ids = data.home_team_ids[t], data.away_team_ids[t]
    home_goals, away_goals = data.home_goals[t], data.away_goals[t]
    log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv
    log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
    total_loglik = sum(logpdf.(LogPoisson.(log_λs), home_goals)) + sum(logpdf.(LogPoisson.(log_μs), away_goals))
    return total_loglik
end

# ===================================================================
# ## 4. Turing Model
# ===================================================================

@model function pmmh_football_model(data, n_particles)
    # Priors
    ρ_attack ~ Beta(10, 1.5); ρ_defense ~ Beta(10, 1.5)
    μ_log_σ_attack ~ Normal(-2.5, 0.5); τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5); τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)
    z_log_σ_attack ~ MvNormal(zeros(Float32, data.n_teams), 1.0f0 * I) # Use Float32 for memory
    z_log_σ_defense ~ MvNormal(zeros(Float32, data.n_teams), 1.0f0 * I) # Use Float32 for memory
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense
    σ_attack = exp.(log_σ_attack); σ_defense = exp.(log_σ_defense)
    log_home_adv ~ Normal(log(1.3), 0.2)

    # Construct SSM
    dyn = AR1Dynamics(ρ_attack, ρ_defense, σ_attack, σ_defense)
    obs = MaherObservationProcess()
    prior = FootballPrior(MvNormal(zeros(Float32, 2 * data.n_teams), 0.5f0 * I)) # Use Float32
    ssm = StateSpaceModel(prior, dyn, obs)
    dummy_obs = fill(nothing, data.n_rounds)

    # Run Particle Filter
    _, loglik_estimate = GeneralisedFilters.filter(
        ssm,
        BootstrapFilter(n_particles), # Use the standard, library-provided filter
        dummy_obs;
        data=data,
        log_home_adv=log_home_adv
    )
    Turing.@addlogprob! loglik_estimate
end

# ===================================================================
# ## 5. Main Execution Script
# ===================================================================
println("--- Generating Synthetic Data ---")
cpu_data = generate_multi_season_data(n_teams=10, n_seasons=2, rounds_per_season=20, season_to_season_volatility=0.03, seed=42)
println("--- Moving data to GPU ---")
data = data_to_gpu(cpu_data)

# --- Find a particle count that fits your VRAM ---
# Start here and increase if it works, decrease if you run out of memory.
n_particles = 1000
n_samples = 100
n_chains = 2

model_instance = pmmh_football_model(data, n_particles)

println("--- Starting MCMC Sampling with Particle Gibbs ---")
# --- Use the PG() sampler! ---
chain = sample(model_instance, PG(n_particles), MCMCThreads(), n_samples, n_chains)

println("\n--- Sampling Complete ---")
display(chain)


### v2 
# ===================================================================
# ## 1. Setup and Libraries
# ===================================================================
using Turing
using SSMProblems
using GeneralisedFilters
using Distributions, LinearAlgebra, Random, Statistics
using Plots, StatsPlots, MCMCChains
using PrettyTables, Printf
using CUDA
using LogExpFunctions

# Configure Turing for better performance
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Check for GPU
if CUDA.functional()
    CUDA.allowscalar(false)
    println("✅ GPU is ready! The model will run on the GPU.")
else
    println("⚠️ GPU not found. The model will run on the CPU.")
end

# ===================================================================
# ## 2. Data Generation and GPU Transfer
# ===================================================================

function generate_multi_season_data(; n_teams::Int=10, n_seasons::Int=3, rounds_per_season::Int=38, season_to_season_volatility::Float64=0.4, seed::Int=123)
    Random.seed!(seed)
    total_rounds = n_seasons * rounds_per_season
    true_log_α = zeros(n_teams, total_rounds)
    true_log_β = zeros(n_teams, total_rounds)
    true_home_adv = 1.3
    teams = 1:n_teams
    home_team_ids_all = Vector{Vector{Int}}()
    away_team_ids_all = Vector{Vector{Int}}()
    home_goals_all = Vector{Vector{Int}}()
    away_goals_all = Vector{Vector{Int}}()
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
                true_log_α[:, t] = true_log_α[:, t-1] .+ attack_slopes
                true_log_β[:, t] = true_log_β[:, t-1] .+ defense_slopes
            end
            true_log_α[:, t] .-= mean(true_log_α[:, t])
            true_log_β[:, t] .-= mean(true_log_β[:, t])
        end
    end
    true_α = exp.(true_log_α)
    true_β = exp.(true_log_β)
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
    return (home_team_ids=home_team_ids_all, away_team_ids=away_team_ids_all, home_goals=home_goals_all, away_goals=away_goals_all, n_teams=n_teams, n_rounds=total_rounds, true_log_α=true_log_α, true_log_β=true_log_β)
end

function data_to_gpu(data)
    if !CUDA.functional()
        return data
    end
    return (home_team_ids=[cu(t) for t in data.home_team_ids], away_team_ids=[cu(t) for t in data.away_team_ids], home_goals=[cu(t) for t in data.home_goals], away_goals=[cu(t) for t in data.away_goals], n_teams=data.n_teams, n_rounds=data.n_rounds, true_log_α=data.true_log_α, true_log_β=data.true_log_β)
end


# ===================================================================
# ## 3. Model Definition using SSMProblems.jl
# ===================================================================

const FootballState = Vector{<:Real}

struct AR1Dynamics{V<:AbstractVector} <: LatentDynamics
    ρ_α::Real; ρ_β::Real; σ_α::V; σ_β::V
end

struct MaherObservationProcess <: ObservationProcess end

struct FootballPrior <: StatePrior
    dist::MvNormal
end

SSMProblems.distribution(prior::FootballPrior; kwargs...) = prior.dist

# --- FIX: This function was missing, causing PG to hang ---
function SSMProblems.distribution(dyn::AR1Dynamics, t::Int, prev_state::FootballState; kwargs...)
    n_teams = length(dyn.σ_α)
    log_α_prev = @view prev_state[1:n_teams]
    log_β_prev = @view prev_state[n_teams+1:end]
    μ_α_new = dyn.ρ_α .* log_α_prev
    μ_β_new = dyn.ρ_β .* log_β_prev
    μ_new = [μ_α_new; μ_β_new]
    Σ_new = Diagonal(vcat(dyn.σ_α, dyn.σ_β).^2)
    return MvNormal(μ_new, Σ_new)
end

function SSMProblems.logdensity(proc::MaherObservationProcess, t::Int, state::FootballState, obs; data, log_home_adv, kwargs...)
    n_teams = div(length(state), 2)
    state_gpu = cu(state)
    log_α_view = @view state_gpu[1:n_teams]
    log_β_view = @view state_gpu[n_teams+1:end]
    log_α_t = log_α_view .- mean(log_α_view)
    log_β_t = log_β_view .- mean(log_β_view)
    home_ids, away_ids = data.home_team_ids[t], data.away_team_ids[t]
    home_goals, away_goals = data.home_goals[t], data.away_goals[t]
    log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv
    log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
    total_loglik = sum(logpdf.(LogPoisson.(log_λs), home_goals)) + sum(logpdf.(LogPoisson.(log_μs), away_goals))
    return total_loglik
end

# ===================================================================
# ## 4. Turing Model
# ===================================================================

@model function pmmh_football_model(data, n_particles)
    # Priors
    ρ_attack ~ Beta(10, 1.5); ρ_defense ~ Beta(10, 1.5)
    μ_log_σ_attack ~ Normal(-2.5, 0.5); τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5); τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)
    z_log_σ_attack ~ MvNormal(zeros(Float32, data.n_teams), 1.0f0 * I)
    z_log_σ_defense ~ MvNormal(zeros(Float32, data.n_teams), 1.0f0 * I)
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense
    σ_attack = exp.(log_σ_attack); σ_defense = exp.(log_σ_defense)
    log_home_adv ~ Normal(log(1.3), 0.2)

    # Construct SSM
    dyn = AR1Dynamics(ρ_attack, ρ_defense, σ_attack, σ_defense)
    obs = MaherObservationProcess()
    prior = FootballPrior(MvNormal(zeros(Float32, 2 * data.n_teams), 0.5f0 * I))
    ssm = StateSpaceModel(prior, dyn, obs)
    dummy_obs = fill(nothing, data.n_rounds)

    # Run Particle Filter
    _, loglik_estimate = GeneralisedFilters.filter(
        ssm,
        BootstrapFilter(n_particles),
        dummy_obs;
        data=data,
        log_home_adv=log_home_adv
    )
    Turing.@addlogprob! loglik_estimate
end

# ===================================================================
# ## 5. Main Execution Script
# ===================================================================
println("--- Generating Synthetic Data ---")
cpu_data = generate_multi_season_data(n_teams=10, n_seasons=1, rounds_per_season=20, season_to_season_volatility=0.03, seed=42)
println("--- Moving data to GPU ---")
data = data_to_gpu(cpu_data)

# Find a particle count that fits your VRAM (e.g., 8000 for 4GB)
n_particles = 100
n_samples = 50
n_chains = 4

model_instance = pmmh_football_model(data, n_particles)

println("--- Starting MCMC Sampling with Particle Gibbs ---")
chain = sample(model_instance, PG(n_particles), MCMCThreads(), n_samples, n_chains)

println("\n--- Sampling Complete ---")
display(chain)
"""
n_particles = 50
Log evidence      = 0.0
Iterations        = 1:1:5
Number of chains  = 1
Samples per chain = 5
Wall duration     = 226.74 seconds
Compute duration  = 226.74 seconds

# run 2 
n_particles = 10
Chains MCMC chain (5×29×2 Array{Float64, 3}):
Iterations        = 1:1:5
Number of chains  = 2
Samples per chain = 5
Wall duration     = 9.47 seconds
Compute duration  = 18.69 seconds
Summary Statistics
           parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
               Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

             ρ_attack    0.8486    0.0889       NaN        NaN        NaN    1.1195           NaN
            ρ_defense    0.8592    0.1320       NaN        NaN        NaN    1.9233           NaN
       μ_log_σ_attack   -2.4923    0.6568       NaN        NaN        NaN    2.1541           NaN
       τ_log_σ_attack    0.1395    0.0740       NaN        NaN        NaN    1.9588           NaN
      μ_log_σ_defense   -2.4561    0.4465       NaN        NaN        NaN    2.2483           NaN
      τ_log_σ_defense    0.1249    0.0826       NaN        NaN        NaN    1.4699           NaN
    z_log_σ_attack[1]   -0.4459    0.6922       NaN        NaN        NaN    2.0617           NaN
    z_log_σ_attack[2]    0.6092    1.3153       NaN        NaN        NaN    1.9780           NaN
    z_log_σ_attack[3]   -0.3888    0.9152       NaN        NaN        NaN    3.0899           NaN
    z_log_σ_attack[4]   -0.1138    0.9358       NaN        NaN        NaN    1.6555           NaN
    z_log_σ_attack[5]   -0.1801    1.1212       NaN        NaN        NaN    1.4153           NaN
    z_log_σ_attack[6]    0.0090    0.5775       NaN        NaN        NaN    1.6543           NaN
    z_log_σ_attack[7]   -0.1338    1.1753       NaN        NaN        NaN    1.6130           NaN
    z_log_σ_attack[8]    0.2301    0.5136       NaN        NaN        NaN    1.5973           NaN
    z_log_σ_attack[9]   -0.4414    1.1621       NaN        NaN        NaN    0.8858           NaN
   z_log_σ_attack[10]   -0.1458    0.9880       NaN        NaN        NaN    1.7162           NaN
   z_log_σ_defense[1]    0.2895    0.7465       NaN        NaN        NaN    1.3649           NaN
   z_log_σ_defense[2]    0.2298    0.7205       NaN        NaN        NaN    2.2660           NaN
   z_log_σ_defense[3]    0.2157    1.2728       NaN        NaN        NaN    1.6543           NaN
   z_log_σ_defense[4]    0.8922    0.7611       NaN        NaN        NaN    1.4034           NaN
   z_log_σ_defense[5]    0.1703    1.1265       NaN        NaN        NaN    1.2343           NaN
   z_log_σ_defense[6]    0.6836    0.8921       NaN        NaN        NaN    1.1260           NaN
   z_log_σ_defense[7]    0.0234    0.9809       NaN        NaN        NaN    2.7077           NaN
   z_log_σ_defense[8]   -0.6236    0.8938       NaN        NaN        NaN    1.7762           NaN
   z_log_σ_defense[9]    0.1592    0.6544       NaN        NaN        NaN    0.9727           NaN
  z_log_σ_defense[10]    0.2652    1.0858       NaN        NaN        NaN    1.3420           NaN
         log_home_adv    0.3073    0.1524       NaN        NaN        NaN    2.1731           NaN




# run 3
Iterations        = 1:1:10
Number of chains  = 2
Samples per chain = 10
Wall duration     = 14.04 seconds
Compute duration  = 26.94 seconds

Summary Statistics
           parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
               Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

             ρ_attack    0.8742    0.0788    0.0155    26.0206    21.7391    0.9358        0.9660
            ρ_defense    0.8529    0.0817    0.0160    26.0206    21.7391    1.0555        0.9660
       μ_log_σ_attack   -2.4212    0.5786    0.1134    26.0206    21.7391    0.9113        0.9660
       τ_log_σ_attack    0.0994    0.0834    0.0164    26.0206    21.7391    0.9710        0.9660
      μ_log_σ_defense   -2.5355    0.4802    0.0941    26.0206    21.7391    0.9640        0.9660
      τ_log_σ_defense    0.1365    0.1013    0.0198    26.0206    21.7391    0.9202        0.9660
    z_log_σ_attack[1]   -0.1591    0.7734    0.1534    26.0206    21.7391    1.2427        0.9660
    z_log_σ_attack[2]    0.3906    0.8972    0.1759    26.0206    21.7391    0.9163        0.9660
    z_log_σ_attack[3]    0.3934    0.9717    0.2308    18.5569    21.7391    1.0621        0.6889
    z_log_σ_attack[4]   -0.1137    1.0379    0.2363    15.2739    21.7391    1.1488        0.5670
    z_log_σ_attack[5]   -0.2317    0.9840    0.1929    26.0206    21.7391    1.0413        0.9660
    z_log_σ_attack[6]    0.2170    1.2939    0.2537    26.0206    21.7391    1.0186        0.9660
    z_log_σ_attack[7]   -0.4001    0.7472    0.1876    15.1892    21.7391    1.1239        0.5639
    z_log_σ_attack[8]   -0.0364    0.9688    0.1899    26.0206    26.0206    1.0184        0.9660
    z_log_σ_attack[9]    0.0735    0.8566    0.1954    18.8972    21.7391    1.1502        0.7015
   z_log_σ_attack[10]    0.2633    0.9650    0.2260    18.6474    21.7391    1.0528        0.6923
   z_log_σ_defense[1]   -0.5863    0.8894    0.2545    11.4159    26.0206    1.3675        0.4238
   z_log_σ_defense[2]   -0.0807    1.1288    0.2792    16.8458    21.7391    1.0262        0.6254
   z_log_σ_defense[3]   -0.3604    0.8007    0.1570    26.0206    26.0206    0.9873        0.9660
   z_log_σ_defense[4]   -0.1386    0.8333    0.1831    23.5940    21.7391    1.0730        0.8759
   z_log_σ_defense[5]   -0.0564    1.1646    0.2283    26.0206    21.7391    0.9091        0.9660
   z_log_σ_defense[6]    0.0001    1.0769    0.2119    22.7258    21.7391    1.0676        0.8437
   z_log_σ_defense[7]   -0.1125    1.1248    0.2410    23.6525    26.0206    1.1250        0.8781
   z_log_σ_defense[8]    0.2381    1.1159    0.2188    26.0206    26.0206    0.9339        0.9660
   z_log_σ_defense[9]    0.0433    0.8587    0.1683    26.0206    21.7391    0.9346        0.9660
  z_log_σ_defense[10]    0.2543    1.1172    0.2550    20.2739    21.7391    0.9990        0.7526
         log_home_adv    0.3518    0.2077    0.0497    17.9783    21.7391    0.9925        0.6674




Iterations        = 1:1:50
Number of chains  = 2
Samples per chain = 50
Wall duration     = 75.91 seconds
Compute duration  = 148.32 seconds

Summary Statistics
           parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
               Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

             ρ_attack    0.8714    0.0859    0.0100   101.8025    57.6991    1.0111        0.6864
            ρ_defense    0.8856    0.0878    0.0088   109.6958   116.6190    0.9854        0.7396
       μ_log_σ_attack   -2.4311    0.4927    0.0520    90.1555    79.8920    1.0207        0.6079
       τ_log_σ_attack    0.1825    0.1348    0.0150    81.0545    77.5311    1.0017        0.5465
      μ_log_σ_defense   -2.4569    0.4500    0.0476    90.1431    80.5222    0.9959        0.6078
      τ_log_σ_defense    0.1530    0.1132    0.0121    86.8653    79.8920    1.0174        0.5857
    z_log_σ_attack[1]    0.0973    1.0026    0.1073    90.3665    76.8314    1.0041        0.6093
    z_log_σ_attack[2]    0.0895    1.0981    0.1245    85.0230    74.3035    1.0107        0.5733
    z_log_σ_attack[3]    0.0346    0.9838    0.1368    46.7167   116.6190    1.0453        0.3150
    z_log_σ_attack[4]   -0.0113    1.0175    0.1027    90.5429   114.3895    0.9885        0.6105
    z_log_σ_attack[5]    0.0661    1.0142    0.1038    93.6403   120.6815    1.0103        0.6314
    z_log_σ_attack[6]    0.1163    1.0373    0.1144    75.9143    70.6421    1.0266        0.5118
    z_log_σ_attack[7]   -0.0098    1.0403    0.1007    95.9616    83.9262    1.0133        0.6470
    z_log_σ_attack[8]    0.0260    0.9029    0.0867   107.5992   115.9769    1.0116        0.7255
    z_log_σ_attack[9]    0.0558    0.9462    0.0898   108.8444   112.8619    1.0095        0.7339
   z_log_σ_attack[10]   -0.1546    1.0923    0.1040   103.2277    71.9381    1.0410        0.6960
   z_log_σ_defense[1]   -0.1587    0.9624    0.0921   111.8887    81.1696    1.0103        0.7544
   z_log_σ_defense[2]    0.0323    1.0222    0.1112    85.9677    57.6991    1.0035        0.5796
   z_log_σ_defense[3]   -0.0130    1.1175    0.1008   120.2870    78.1991    1.0271        0.8110
   z_log_σ_defense[4]    0.0552    0.9743    0.0926   110.1932   117.5318    0.9989        0.7430
   z_log_σ_defense[5]    0.0371    0.9520    0.1229    59.6764   117.2611    1.0287        0.4024
   z_log_σ_defense[6]   -0.1147    1.0113    0.1165    71.3184    57.6991    1.0044        0.4809
   z_log_σ_defense[7]   -0.0473    1.0293    0.0976   136.9907   111.3907    1.0045        0.9236
   z_log_σ_defense[8]    0.0978    0.9981    0.1020    98.6793    57.9303    1.0071        0.6653
   z_log_σ_defense[9]    0.1205    0.8693    0.0865    96.0332    74.3426    0.9943        0.6475
  z_log_σ_defense[10]    0.2547    0.9915    0.1253    61.1284    76.2797    1.0285        0.4121
         log_home_adv    0.2556    0.1787    0.0209    76.0763   114.3895    0.9922        0.5129


n_particles = 10
Iterations        = 1:1:50
Number of chains  = 4
Samples per chain = 50
Wall duration     = 143.85 seconds
Compute duration  = 551.91 seconds
Summary Statistics
           parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
               Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

             ρ_attack    0.8725    0.1027    0.0087   127.8355   112.9982    1.0175        0.2316
            ρ_defense    0.8727    0.0907    0.0069   192.9721   187.4326    0.9991        0.3496
       μ_log_σ_attack   -2.4924    0.4897    0.0407   144.3363   183.8036    1.0191        0.2615
       τ_log_σ_attack    0.1597    0.1108    0.0079   194.2534   194.7938    1.0094        0.3520
      μ_log_σ_defense   -2.4737    0.5039    0.0391   166.7737   148.2983    0.9914        0.3022
      τ_log_σ_defense    0.1746    0.1364    0.0109   157.0763   178.3410    1.0296        0.2846
    z_log_σ_attack[1]    0.0604    1.0489    0.0824   156.8968   138.1430    1.0183        0.2843
    z_log_σ_attack[2]    0.0141    1.0213    0.0748   186.0321   238.1859    0.9907        0.3371
    z_log_σ_attack[3]   -0.1212    0.9740    0.0753   168.4033   122.7618    1.0095        0.3051
    z_log_σ_attack[4]    0.1647    1.0577    0.0968   115.5319   191.6640    1.0277        0.2093
    z_log_σ_attack[5]    0.0550    1.0179    0.0805   159.5690   136.0771    1.0156        0.2891
    z_log_σ_attack[6]   -0.0888    0.9607    0.0717   177.5809   194.7938    0.9948        0.3218
    z_log_σ_attack[7]    0.1251    0.9377    0.0713   173.1110   128.2012    1.0189        0.3137
    z_log_σ_attack[8]    0.0576    0.9585    0.1066    96.3383   134.8121    1.0411        0.1746
    z_log_σ_attack[9]    0.0546    1.0117    0.0739   195.8773   153.2241    1.0113        0.3549
   z_log_σ_attack[10]   -0.0110    1.0010    0.0936   147.7289   151.4174    1.0380        0.2677
   z_log_σ_defense[1]   -0.0179    0.8897    0.0742   143.2584   195.6592    1.0240        0.2596
   z_log_σ_defense[2]    0.0147    1.0097    0.0775   174.7953   193.0045    1.0211        0.3167
   z_log_σ_defense[3]    0.0099    0.9854    0.0740   177.6430   191.2300    1.0045        0.3219
   z_log_σ_defense[4]   -0.0629    0.9776    0.0668   210.6118   170.0970    0.9948        0.3816
   z_log_σ_defense[5]    0.0220    1.0775    0.0803   182.1490   187.3910    1.0045        0.3300
   z_log_σ_defense[6]   -0.0042    1.0849    0.0809   179.7004   190.7989    1.0072        0.3256
   z_log_σ_defense[7]   -0.0488    0.9171    0.0798   132.6773   187.8022    1.0060        0.2404
   z_log_σ_defense[8]    0.1273    1.0988    0.0831   176.1932   195.7076    1.0126        0.3192
   z_log_σ_defense[9]   -0.0340    1.0465    0.0782   186.8427    91.5449    0.9969        0.3385
  z_log_σ_defense[10]   -0.0638    0.9439    0.0707   181.3432   146.3585    1.0055        0.3286
         log_home_adv    0.2680    0.1927    0.0225    75.6170    76.9231    1.0491        0.1370



n_particles = 100
Log evidence      = 0.0
Iterations        = 1:1:1
Number of chains  = 1
Samples per chain = 1
Wall duration     = 162.06 seconds
Compute duration  = 162.06 seconds

Summary Statistics
           parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
               Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

             ρ_attack    0.7597       NaN       NaN        NaN        NaN       NaN           NaN
            ρ_defense    0.8498       NaN       NaN        NaN        NaN       NaN           NaN
       μ_log_σ_attack   -2.9818       NaN       NaN        NaN        NaN       NaN           NaN
       τ_log_σ_attack    0.1540       NaN       NaN        NaN        NaN       NaN           NaN
      μ_log_σ_defense   -3.1237       NaN       NaN        NaN        NaN       NaN           NaN
      τ_log_σ_defense    0.3192       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[1]   -0.8992       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[2]   -0.5881       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[3]    0.2166       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[4]   -0.5181       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[5]    0.6894       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[6]    0.5246       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[7]   -1.0631       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[8]    0.9490       NaN       NaN        NaN        NaN       NaN           NaN
    z_log_σ_attack[9]   -2.1709       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_attack[10]    0.5569       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[1]   -1.1506       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[2]   -0.3400       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[3]    0.4988       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[4]   -1.3166       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[5]    0.9739       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[6]   -0.1635       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[7]    0.2451       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[8]    0.6619       NaN       NaN        NaN        NaN       NaN           NaN
   z_log_σ_defense[9]    0.2068       NaN       NaN        NaN        NaN       NaN           NaN
  z_log_σ_defense[10]   -1.0888       NaN       NaN        NaN        NaN       NaN           NaN
         log_home_adv    0.0944       NaN       NaN        NaN        NaN       NaN           NaN
"""

# ===================================================================
# ## 6. Trajectory Reconstruction and Visualization
# ===================================================================
using StatsBase # For Weights
using GeneralisedFilters: AbstractCallback, PostUpdateCallback

# --- Step 1: Define a CPU-ONLY observation process ---
# We create a new type to distinguish it from the GPU version.
struct MaherObservationProcessCPU <: ObservationProcess end

# Define a logdensity method specifically for this CPU type.
# This version is a simple for-loop and does NO GPU operations.
function SSMProblems.logdensity(
    proc::MaherObservationProcessCPU, t::Int, state::FootballState, obs;
    data, log_home_adv, kwargs...
)
    n_teams = div(length(state), 2)
    log_α_t = @view(state[1:n_teams]) .- mean(@view(state[1:n_teams]))
    log_β_t = @view(state[n_teams+1:end]) .- mean(@view(state[n_teams+1:end]))
    
    home_ids = data.home_team_ids[t]
    away_ids = data.away_team_ids[t]
    home_goals = data.home_goals[t]
    away_goals = data.away_goals[t]

    total_loglik = 0.0
    for i in 1:length(home_ids)
        h_team, a_team = home_ids[i], away_ids[i]
        log_λ = log_α_t[h_team] + log_β_t[a_team] + log_home_adv
        log_μ = log_α_t[a_team] + log_β_t[h_team]
        total_loglik += logpdf(LogPoisson(log_λ), home_goals[i])
        total_loglik += logpdf(LogPoisson(log_μ), away_goals[i])
    end
    
    return total_loglik
end


# --- Step 2: Define the Callback to store particle history ---
mutable struct TrajectoryCallback <: AbstractCallback
    history::Vector{Vector{FootballState}} # Store CPU vectors
end

TrajectoryCallback() = TrajectoryCallback([])

# This function is called by the filter after each update step
function (cb::TrajectoryCallback)(
    model, filter_algo, t::Integer, particles, obs, ::PostUpdateCallback; kwargs...
)
    # The filter is running on the CPU, so particles are already CPU vectors
    push!(cb.history, deepcopy(particles.particles))
end


# --- Step 3: The Main Reconstruction Function ---
function reconstruct_dynamic_trajectory(chain, cpu_data, n_particles_viz)
    println("\n--- Reconstructing dynamic trajectory ---")

    # 1. Extract posterior mean of static parameters
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

    # 2. Build the SSM using the CPU-specific observation process
    dyn = AR1Dynamics(params.ρ_attack, params.ρ_defense, σ_attack, σ_defense)
    obs_cpu = MaherObservationProcessCPU() # USE THE CPU VERSION
    prior = FootballPrior(MvNormal(zeros(Float32, 2 * cpu_data.n_teams), 0.5f0 * I))
    ssm = StateSpaceModel(prior, dyn, obs_cpu)

    # 3. Run the filter on the CPU with the callback
    callback_obj = TrajectoryCallback()
    dummy_obs = fill(nothing, cpu_data.n_rounds)
    
    final_state, ll = GeneralisedFilters.filter(
        ssm,
        BootstrapFilter(n_particles_viz),
        dummy_obs;
        data=cpu_data,
        log_home_adv=params.log_home_adv,
        callback=callback_obj
    )

    # 4. Process the stored history
    n_states = 2 * cpu_data.n_teams
    n_rounds = cpu_data.n_rounds
    trajectory_history = zeros(n_states, n_rounds)
    
    for t in 1:n_rounds
        particles_at_t = hcat(callback_obj.history[t]...) # Convert Vector{Vector} to Matrix
        weights_at_t = Weights(fill(1.0 / n_particles_viz, n_particles_viz))
        for i in 1:n_states
            trajectory_history[i, t] = mean(particles_at_t[i, :], weights_at_t)
        end
    end

    # 5. Unpack and return for plotting
    n_teams = cpu_data.n_teams
    log_α_raw = trajectory_history[1:n_teams, :]
    log_β_raw = trajectory_history[n_teams+1:end, :]
    log_α_centered = log_α_raw .- mean(log_α_raw, dims=1)
    log_β_centered = log_β_raw .- mean(log_β_raw, dims=1)
    log_α_dynamic_mean = reshape(log_α_centered', 1, n_teams, n_rounds)
    log_β_dynamic_mean = reshape(log_β_centered', 1, n_teams, n_rounds)

    println("--- Trajectory reconstruction complete ---")
    return log_α_dynamic_mean, log_β_dynamic_mean
end
# --- Step 3: Add the plotting code from 02_ar1_play.jl ---
# You will need to copy over your `plot_team_dashboard` and other plotting
# helper functions from your `02_ar1_play.jl` script.
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


# --- Step 4: Call the reconstruction and plot ---
log_α_dynamic, log_β_dynamic = reconstruct_dynamic_trajectory(chain, cpu_data, 1000)
dashboard_plot = plot_team_dashboard(1, cpu_data, log_α_dynamic, log_β_dynamic)
