# src/dynamic_maher.jl
using Turing, Distributions, LinearAlgebra, Random, Plots, StatsPlots

@model function dynamic_maher_model(
    home_team_ids, away_team_ids,
    home_goals, away_goals,
    n_teams, n_rounds
)
    # --- HYPER-PRIORS (Still tight) ---
    σ_attack ~ Truncated(Normal(0, 0.1), 0, Inf)
    σ_defense ~ Truncated(Normal(0, 0.1), 0, Inf)
    
    # --- GLOBAL PARAMETERS ---
    log_home_adv ~ Normal(log(1.3), 0.2)
    home_adv = exp(log_home_adv)

    # --- INITIAL STATE (NON-CENTERED) ---
    # 1. Sample the standardized initial states
    initial_α_z ~ MvNormal(zeros(n_teams), I)
    initial_β_z ~ MvNormal(zeros(n_teams), I)
    
    # 2. Construct the actual initial states (log_α_raw_t0, log_β_raw_t0)
    # This is equivalent to log_α_raw_t0 ~ MvNormal(zeros, 0.5 * I), but more stable
    log_α_raw_t0 = initial_α_z * sqrt(0.5)
    log_β_raw_t0 = initial_β_z * sqrt(0.5)

    # --- STATE EVOLUTION NOISE (NON-CENTERED) ---
    # 1. Sample the standardized evolution "shocks" for all time steps
    z_α ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I)
    # Reshape them into matrices for easier use in the loop
    z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β_mat = reshape(z_β, n_teams, n_rounds)

    # --- STATE VARIABLES & LIKELIHOOD ---
    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)
    ϵ = 1e-6

    for t in 1:n_rounds
        # 2. Construct the evolving parameters from the noise
        if t == 1
            # Evolve from t0, using the first column of our noise matrix
            log_α_raw[:, 1] = log_α_raw_t0 + z_α_mat[:, 1] * σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 + z_β_mat[:, 1] * σ_defense
        else
            # Evolve from t-1, using the t-th column of our noise matrix
            log_α_raw[:, t] = log_α_raw[:, t-1] + z_α_mat[:, t] * σ_attack
            log_β_raw[:, t] = log_β_raw[:, t-1] + z_β_mat[:, t] * σ_defense
        end

        # Apply sum-to-zero constraint and calculate likelihood (as before)
        log_α_t = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_t = log_β_raw[:, t] .- mean(log_β_raw[:, t])
        
        α_t = exp.(log_α_t)
        β_t = exp.(log_β_t)

        for k in 1:length(home_goals[t])
            i = home_team_ids[t][k]
            j = away_team_ids[t][k]
            
            λ = α_t[i] * β_t[j] * home_adv + ϵ
            μ = α_t[j] * β_t[i] + ϵ
            
            home_goals[t][k] ~ Poisson(λ)
            away_goals[t][k] ~ Poisson(μ)
        end
    end
end


# v2 more stable maybe 
@model function dynamic_maher_model(
    home_team_ids, away_team_ids,
    home_goals, away_goals,
    n_teams, n_rounds
)
    # --- HYPER-PRIORS (Still tight) ---
    σ_attack ~ Truncated(Normal(0, 0.1), 0, Inf)
    σ_defense ~ Truncated(Normal(0, 0.1), 0, Inf)
    
    # --- GLOBAL PARAMETERS ---
    log_home_adv ~ Normal(log(1.3), 0.2)
    # No need to exponentiate home_adv here anymore

    # --- INITIAL STATE (NON-CENTERED) ---
    initial_α_z ~ MvNormal(zeros(n_teams), I)
    initial_β_z ~ MvNormal(zeros(n_teams), I)
    
    log_α_raw_t0 = initial_α_z * sqrt(0.5)
    log_β_raw_t0 = initial_β_z * sqrt(0.5)

    # --- STATE EVOLUTION NOISE (NON-CENTERED) ---
    z_α ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β_mat = reshape(z_β, n_teams, n_rounds)

    # --- STATE VARIABLES & LIKELIHOOD ---
    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)
    # The small epsilon 'ϵ' is no longer needed.

    for t in 1:n_rounds
        if t == 1
            log_α_raw[:, 1] = log_α_raw_t0 + z_α_mat[:, 1] * σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 + z_β_mat[:, 1] * σ_defense
        else
            log_α_raw[:, t] = log_α_raw[:, t-1] + z_α_mat[:, t] * σ_attack
            log_β_raw[:, t] = log_β_raw[:, t-1] + z_β_mat[:, t] * σ_defense
        end

        log_α_t = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_t = log_β_raw[:, t] .- mean(log_β_raw[:, t])
        
        # We no longer calculate α_t and β_t via exp.
        # We now calculate the log-rates directly.

        for k in 1:length(home_goals[t])
            i = home_team_ids[t][k]
            j = away_team_ids[t][k]
            
            # --- MODIFIED LIKELIHOOD ---
            # Calculate the log-rates instead of the rates
            log_λ = log_α_t[i] + log_β_t[j] + log_home_adv
            log_μ = log_α_t[j] + log_β_t[i]
            
            # Use the numerically stable LogPoisson distribution
            home_goals[t][k] ~ LogPoisson(log_λ)
            away_goals[t][k] ~ LogPoisson(log_μ)
        end
    end
end


### v3 
using Turing, Distributions, LinearAlgebra, Random
using ReverseDiff, Memoization

# 1. Set the AD backend and enable the cache
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

@model function dynamic_maher_model(
    home_team_ids, away_team_ids,
    home_goals, away_goals,
    n_teams, n_rounds
)
    # --- HYPER-PRIORS (Still tight) ---
    σ_attack ~ Truncated(Normal(0, 0.1), 0, Inf)
    σ_defense ~ Truncated(Normal(0, 0.1), 0, Inf)
    
    # --- GLOBAL PARAMETERS ---
    log_home_adv ~ Normal(log(1.3), 0.2)
    # No need to exponentiate home_adv here anymore

    # --- INITIAL STATE (NON-CENTERED) ---
    initial_α_z ~ MvNormal(zeros(n_teams), I)
    initial_β_z ~ MvNormal(zeros(n_teams), I)
    
    log_α_raw_t0 = initial_α_z * sqrt(0.5)
    log_β_raw_t0 = initial_β_z * sqrt(0.5)

    # --- STATE EVOLUTION NOISE (NON-CENTERED) ---
    z_α ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I)
    z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β_mat = reshape(z_β, n_teams, n_rounds)

    # --- STATE VARIABLES & LIKELIHOOD ---
    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)

    for t in 1:n_rounds
        if t == 1
            log_α_raw[:, 1] = log_α_raw_t0 + z_α_mat[:, 1] * σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 + z_β_mat[:, 1] * σ_defense
        else
            log_α_raw[:, t] = log_α_raw[:, t-1] + z_α_mat[:, t] * σ_attack
            log_β_raw[:, t] = log_β_raw[:, t-1] + z_β_mat[:, t] * σ_defense
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


###########

function generate_synthetic_data(;
    n_teams::Int=10, 
    n_rounds::Int=38, 
    seed::Int=123
)
    Random.seed!(seed)
    
    # --- Define True Evolving Parameters ---
    true_log_α = zeros(n_teams, n_rounds)
    true_log_β = zeros(n_teams, n_rounds)

    # Create trends for specific teams
    for t in 1:n_rounds
        # Team 1's attack rating steadily increases
        true_log_α[1, t] = -1.0 + 2.0 * (t / n_rounds)
        
        # Team 2's defense rating steadily worsens (a higher log_β means a worse defense)
        true_log_β[2, t] = -1.0 + 2.0 * (t / n_rounds)
    end
    
    # Apply sum-to-zero constraint to true parameters
    for t in 1:n_rounds
        true_log_α[:, t] .-= mean(true_log_α[:, t])
        true_log_β[:, t] .-= mean(true_log_β[:, t])
    end
    
    true_α = exp.(true_log_α)
    true_β = exp.(true_log_β)
    true_home_adv = 1.3

    # --- Generate Match Schedule and Scores ---
    home_team_ids_all, away_team_ids_all = [], []
    home_goals_all, away_goals_all = [], []

    teams = 1:n_teams
    
    for t in 1:n_rounds
        # Simple schedule: each team plays one random opponent per round
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
        home_team_ids = home_team_ids_all,
        away_team_ids = away_team_ids_all,
        home_goals = home_goals_all,
        away_goals = away_goals_all,
        n_teams = n_teams,
        n_rounds = n_rounds,
        true_log_α = true_log_α,
        true_log_β = true_log_β
    )
end


### runner 

data = generate_synthetic_data(n_teams=20, n_rounds=36)
# sense check here
# plot( 1:data.n_rounds, data.true_log_α[1, :], label="True log α (Team 1)", lw=3, color=:black)

model_instance = dynamic_maher_model(
    data.home_team_ids, data.away_team_ids,
    data.home_goals, data.away_goals,
    data.n_teams, data.n_rounds
)

chain = sample(model_instance, NUTS(0.65), 1000)


"""
# n_rounds = 38 , v2 (log poisson )
julia> chain = sample(model_instance, NUTS(0.65), 500)
┌ Info: Found initial step size
└   ϵ = 7.105427357601002e-16
Sampling 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:23:41
Chains MCMC chain (500×795×1 Array{Float64, 3}):

Iterations        = 251:1:750
Number of chains  = 1
Samples per chain = 500
Wall duration     = 1421.25 seconds
Compute duration  = 1421.25 seconds

v3 n_rounds = 38 v3 log poisson with cache reversediff 
julia> chain = sample(model_instance, NUTS(0.65), 500)
┌ Info: Found initial step size
└   ϵ = 0.0015625
Sampling 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:26
Chains MCMC chain (500×795×1 Array{Float64, 3}):

Iterations        = 251:1:750
Number of chains  = 1
Samples per chain = 500
Wall duration     = 27.15 seconds
Compute duration  = 27.15 seconds
"""

# save the chain 
using JLD2
jldsave("/home/james/bet_project/models_julia/workspace/basic_state_space/data/basic_SSM_DP_steps_500.jld2"; chain) 

# plt 


"""
Extracts all samples for a vector/matrix parameter from a Chains object
by finding all parameter names that start with a given base string.
"""
function extract_samples(chain::Chains, base::String)
    # Find all parameter names that start with the base string, followed by a '['
    # This is a robust way to find all elements of the vector `base`.
    regex = Regex("^" * base * "\\[")
    param_symbols = filter(s -> !isnothing(match(regex, string(s))), names(chain))

    if isempty(param_symbols)
        error("No parameters found with base name '$base'")
    end
    
    # Extract the data and reshape to a 2D matrix: (n_samples x n_parameters)
    arr = Array(chain[param_symbols])
    return reshape(arr, :, size(arr, 2))
end


# 1. Extract the sampled components using our new helper function
σ_attack_samples = vec(Array(chain["σ_attack"]))
initial_α_z_samples = extract_samples(chain, "initial_α_z")
z_α_samples = extract_samples(chain, "z_α")

σ_defense_samples = vec(Array(chain["σ_defense"]))
initial_β_z_samples = extract_samples(chain, "initial_β_z")
z_β_samples = extract_samples(chain, "z_β")


# 2. Reconstruct the trajectories (this section is now the same as before)
n_samples = size(chain, 1)
n_teams = data.n_teams
n_rounds = data.n_rounds




log_α_raw_reconstructed = Array{Float64}(undef, n_samples, n_teams, n_rounds)
log_β_raw_reconstructed = Array{Float64}(undef, n_samples, n_teams, n_rounds)



for s in 1:n_samples
    σ_attack_s = σ_attack_samples[s]
    initial_α_z_s = initial_α_z_samples[s, :]
    z_α_mat_s = reshape(z_α_samples[s, :], n_teams, n_rounds)
    
    σ_defense_s = σ_defense_samples[s]
    initial_β_z_s = initial_β_z_samples[s, :]
    z_β_mat_s = reshape(z_β_samples[s, :], n_teams, n_rounds)

    log_α_raw_t0_s = initial_α_z_s * sqrt(0.5)
    log_β_raw_t0_s = initial_β_z_s * sqrt(0.5)
    
    temp_log_α_raw = Matrix{Float64}(undef, n_teams, n_rounds)
    temp_log_β_raw = Matrix{Float64}(undef, n_teams, n_rounds)

    for t in 1:n_rounds
        if t == 1
            temp_log_α_raw[:, 1] = log_α_raw_t0_s + z_α_mat_s[:, 1] * σ_attack_s
            temp_log_β_raw[:, 1] = log_β_raw_t0_s + z_β_mat_s[:, 1] * σ_defense_s
        else
            temp_log_α_raw[:, t] = temp_log_α_raw[:, t-1] + z_α_mat_s[:, t] * σ_attack_s
            temp_log_β_raw[:, t] = temp_log_β_raw[:, t-1] + z_β_mat_s[:, t] * σ_defense_s
        end
    end
    
    log_α_raw_reconstructed[s, :, :] = temp_log_α_raw
    log_β_raw_reconstructed[s, :, :] = temp_log_β_raw
end


# 3. Apply the sum-to-zero constraint
log_α_samples = log_α_raw_reconstructed .- mean(log_α_raw_reconstructed, dims=2)
log_β_samples = log_β_raw_reconstructed .- mean(log_β_raw_reconstructed, dims=2)

# 4. Plotting (unchanged) # 
p = plot(layout=(2, 1), legend=:outertopright, size=(800, 600))
plot!(p[1], 1:data.n_rounds, data.true_log_α[1, :], label="True log α (Team 1)", lw=3, color=:black)
plot!(p[1], 1:data.n_rounds, mean(log_α_samples[:, 1, :], dims=1)', ribbon=std(log_α_samples[:, 1, :], dims=1)', label="Estimated log α (Team 1)", title="Team 1 Attack Parameter (log α)", xlabel="Round", ylabel="Parameter Value")
plot!(p[2], 1:data.n_rounds, data.true_log_β[2, :], label="True log β (Team 2)", lw=3, color=:black)
plot!(p[2], 1:data.n_rounds, mean(log_β_samples[:, 2, :], dims=1)', ribbon=std(log_β_samples[:, 2, :], dims=1)', label="Estimated log β (Team 2)", title="Team 2 Defense Parameter (log β)", xlabel="Round", ylabel="Parameter Value")
display(p)



### static model 
# 2. The Static Model (as you provided)
@model function basic_maher_model_raw(
    home_team_ids, away_team_ids, home_goals, away_goals, n_teams
)
    σ_attack ~ Truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    
    σ_defense ~ Truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    
    log_γ ~ Normal(log(1.3), 0.2) # Home advantage
    
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    α = exp.(log_α)
    β = exp.(log_β)
    γ = exp(log_γ)
    
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]
        j = away_team_ids[k]
        
        λ = α[i] * β[j] * γ
        μ = α[j] * β[i]
        
        home_goals[k] ~ Poisson(λ)
        away_goals[k] ~ Poisson(μ)
    end
end


home_team_ids_flat = reduce(vcat, data.home_team_ids)
away_team_ids_flat = reduce(vcat, data.away_team_ids)
home_goals_flat = reduce(vcat, data.home_goals)
away_goals_flat = reduce(vcat, data.away_goals)

static_model_instance = basic_maher_model_raw(
    home_team_ids_flat, away_team_ids_flat, home_goals_flat, away_goals_flat,
    data.n_teams
)
chain_static = sample(static_model_instance, NUTS(0.65), 500)


# --- COMPARISON PLOTTING ---
chain_dynamic = chain;
log_α_dynamic = log_α_samples;
log_β_dynamic = log_β_samples;

log_α_raw_static = extract_samples( chain_static, "log_α_raw");
log_α_static = log_α_raw_static .- mean(log_α_raw_static, dims=2);

log_β_raw_static = extract_samples( chain_static, "log_β_raw");
log_β_static = log_β_raw_static .- mean(log_β_raw_static, dims=2);




# C) Create the comparison plot
team_number = 5 
p = plot(layout=(1, 1), legend=:outertopright, size=(800, 400))

# Plot 1: True evolving parameter
plot!(p, 1:data.n_rounds, data.true_log_α[team_number, :],
    label="True log α (Team $team_number)", lw=3, color=:black,
    title="Comparison of Dynamic vs. Static Model for Team $team_number Attack",
    xlabel="Round", ylabel="Attack Parameter (log α)")

# Plot 2: Dynamic model estimate (a trajectory)
plot!(p, 1:data.n_rounds, mean(log_α_dynamic[:, team_number, :], dims=1)',
    ribbon=std(log_α_dynamic[:, team_number, :], dims=1)',
    label="Dynamic Estimate", color=:blue)

# Plot 3: Static model estimate (a constant average)
mean_static = mean(log_α_static[:, team_number])
std_static = std(log_α_static[:, team_number])
plot!(p, 1:data.n_rounds, fill(mean_static, data.n_rounds),
    ribbon=std_static,
    label="Static Estimate", color=:red, linestyle=:dash, lw=2)


display(p)



###
team_number = 1

p = plot(layout=(2, 1), legend=:outertopright, size=(800, 600))

plot!(p[1], 1:data.n_rounds, data.true_log_α[team_number, :],
    label="True log α (Team $team_number)", lw=3, color=:black,
    title="Comparison of Dynamic vs. Static Model for Team $team_number",
    xlabel="Round", ylabel="Attack Parameter (log α)")

plot!(p[2], 1:data.n_rounds, data.true_log_β[team_number, :],
    label="True log α (Team $team_number)", lw=3, color=:black,
    title="Comparison of Dynamic vs. Static Model for Team $team_number",
    xlabel="Round", ylabel="Defense Parameter (log β)")

# Plot 2: Dynamic model estimate (a trajectory)
plot!(p[1], 1:data.n_rounds, mean(log_α_dynamic[:, team_number, :], dims=1)',
    ribbon=std(log_α_dynamic[:, team_number, :], dims=1)',
    label="Dynamic Estimate", color=:blue)

plot!(p[2], 1:data.n_rounds, mean(log_β_dynamic[:, team_number, :], dims=1)',
    ribbon=std(log_α_dynamic[:, team_number, :], dims=1)',
    label="Dynamic Estimate", color=:blue)

# Plot 3: Static model estimate (a constant average)
mean_static_α = mean(log_α_static[:, team_number])
std_static_α = std(log_α_static[:, team_number])
mean_static_β = mean(log_β_static[:, team_number])
std_static_β = std(log_β_static[:, team_number])
plot!(p[1], 1:data.n_rounds, fill(mean_static_α, data.n_rounds),
    ribbon=std_static_α,
    label="Static Estimate", color=:red, linestyle=:dash, lw=2)

plot!(p[2], 1:data.n_rounds, fill(mean_static_β, data.n_rounds),
    ribbon=std_static_β,
    label="Static Estimate", color=:red, linestyle=:dash, lw=2)

function plot_compare_a_b(team_number)
p = plot(layout=(2, 1), legend=:outertopright, size=(800, 600))

plot!(p[1], 1:data.n_rounds, data.true_log_α[team_number, :],
    label="True log α (Team $team_number)", lw=3, color=:black,
    title="Comparison of Dynamic vs. Static Model for Team $team_number",
    xlabel="Round", ylabel="Attack Parameter (log α)")

plot!(p[2], 1:data.n_rounds, data.true_log_β[team_number, :],
    label="True log α (Team $team_number)", lw=3, color=:black,
    title="Comparison of Dynamic vs. Static Model for Team $team_number",
    xlabel="Round", ylabel="Defense Parameter (log β)")

# Plot 2: Dynamic model estimate (a trajectory)
plot!(p[1], 1:data.n_rounds, mean(log_α_dynamic[:, team_number, :], dims=1)',
    ribbon=std(log_α_dynamic[:, team_number, :], dims=1)',
    label="Dynamic Estimate", color=:blue)

plot!(p[2], 1:data.n_rounds, mean(log_β_dynamic[:, team_number, :], dims=1)',
    ribbon=std(log_α_dynamic[:, team_number, :], dims=1)',
    label="Dynamic Estimate", color=:blue)

# Plot 3: Static model estimate (a constant average)
mean_static_α = mean(log_α_static[:, team_number])
std_static_α = std(log_α_static[:, team_number])
mean_static_β = mean(log_β_static[:, team_number])
std_static_β = std(log_β_static[:, team_number])
plot!(p[1], 1:data.n_rounds, fill(mean_static_α, data.n_rounds),
    ribbon=std_static_α,
    label="Static Estimate", color=:red, linestyle=:dash, lw=2)

plot!(p[2], 1:data.n_rounds, fill(mean_static_β, data.n_rounds),
    ribbon=std_static_β,
    label="Static Estimate", color=:red, linestyle=:dash, lw=2)

end

plot_compare_a_b(20)

