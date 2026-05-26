# current_development/meta_model/r01_meta_model.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, "../../"))
using Revise
using BayesianFootball
using DataFrames
using Dates
using Turing
# using StatsPlots

# Include the loader
include("l01_meta_model.jl")


save_dir = "./data/copula_ab_test/"

saved_files = Experiments.list_experiments(save_dir, data_dir="")
result = Experiments.load_experiment(saved_files, 1)

println("=== Meta Model (Layer 2) Prototype ===")

# 1. Load Data (Synthetic for prototyping)
println("1. Generating Synthetic Ledger with a 'Bad' Regime in the middle...")
n_bets = 500
start_date = Date(2025, 8, 1)
synthetic_dates = start_date .+ Day.(rand(0:200, n_bets))
sort!(synthetic_dates)

# Week assignment
W = [(d - start_date).value ÷ 7 + 1 for d in synthetic_dates]
n_weeks = maximum(W)

# Synthetic L1 probabilities
P_L1 = rand(Uniform(0.45, 0.65), n_bets)

# Synthetic Outcomes with a regime shift
# The model has a 10% edge globally, but during weeks 10-20 it loses its edge
true_theta = zeros(Float64, n_weeks)
for w in 1:n_weeks
    if 10 <= w <= 20
        true_theta[w] = -0.8 # Bad regime
    else
        true_theta[w] = 0.2  # Good regime
    end
end

Y = zeros(Int, n_bets)
for i in 1:n_bets
    logit_pi = logit(P_L1[i]) + true_theta[W[i]]
    pi = logistic(logit_pi)
    Y[i] = rand(Bernoulli(pi))
end

synthetic_ledger = DataFrame(
    match_id = 1:n_bets,
    date = synthetic_dates,
    is_winner = Y,
    P_L1 = P_L1,
    W = W
)

# 2. Build and Sample Model
println("2. Building Meta Model...")
meta_data = MetaModelData(
    synthetic_ledger.is_winner,
    synthetic_ledger.P_L1,
    synthetic_ledger.W,
    n_weeks
)

model = build_meta_model(meta_data)

println("3. Sampling (NUTS)... This may take a moment.")
chain = sample(model, NUTS(0.65), 500)

# Extract Theta Walk manually since θ is a deterministic variable
theta_matrix = extract_theta(chain, n_weeks)
theta_means = vec(mean(theta_matrix, dims=1))

println("\n--- Theta (Regime Shift) Recovery ---")
println("True Theta (Bad Regime): -0.8")
println("True Theta (Good Regime): 0.2")
println("Recovered Theta (Week 15 - Bad): ", round(theta_means[15], digits=3))
println("Recovered Theta (Week 5 - Good):  ", round(theta_means[5], digits=3))


#=
julia> println("\n--- Theta (Regime Shift) Recovery ---")

--- Theta (Regime Shift) Recovery ---

julia> println("True Theta (Bad Regime): -0.8")
True Theta (Bad Regime): -0.8

julia> println("True Theta (Good Regime): 0.2")
True Theta (Good Regime): 0.2

julia> println("Recovered Theta (Week 15 - Bad): ", round(theta_means[15], digits=3))
Recovered Theta (Week 15 - Bad): -0.647

julia> println("Recovered Theta (Week 5 - Good):  ", round(theta_means[5], digits=3))
Recovered Theta (Week 5 - Good):  0.568
=#


# 3. Seamless Shifted Kelly Demonstration
println("\n4. Demonstrating Shifted Posterior Kelly...")
# Imagine a new match with odds 2.0
odds = 2.0
# Layer 1 gave us a full posterior around 0.55 (so we have an edge initially)
l1_mcmc_posterior = rand(Normal(0.55, 0.05), 500)

# Get Week 15 (Bad Regime) Meta Model samples
alpha_chain = vec(Array(chain[:α]))
beta_chain = vec(Array(chain[:β]))
theta_bad_chain = theta_matrix[:, 15]

shifted_bad_posterior = shift_posterior(l1_mcmc_posterior, alpha_chain, beta_chain, theta_bad_chain)

println("Original L1 Mean: ", round(mean(l1_mcmc_posterior), digits=3))
println("Shifted Meta Mean (Bad Regime): ", round(mean(shifted_bad_posterior), digits=3))

#=
julia> println("Original L1 Mean: ", round(mean(l1_mcmc_posterior), digits=3))
Original L1 Mean: 0.547

julia> println("Shifted Meta Mean (Bad Regime): ", round(mean(shifted_bad_posterior), digits=3))
Shifted Meta Mean (Bad Regime): 0.359
=#


# Use the existing BayesianFootball Signals module!
try
    s = BayesianFootball.Signals.BayesianKelly(min_edge=0.0)
    original_stake = BayesianFootball.Signals.compute_stake(s, l1_mcmc_posterior, odds)
    shifted_stake = BayesianFootball.Signals.compute_stake(s, shifted_bad_posterior, odds)

    println("Original L1 Stake: ", round(original_stake * 100, digits=2), "%")
    println("Shifted Meta Stake (Bad Regime): ", round(shifted_stake * 100, digits=2), "%")
    
    if shifted_stake < original_stake
        println("SUCCESS: The Meta Model correctly shrunk the stake during a poor predictive regime!")
    else
        println("WARNING: Stake did not shrink. Check model parameters.")
    end
catch e
    println("Error running BayesianKelly (module might not be loaded properly in the prototype script): ", e)
end




#=
julia> println("Original L1 Mean: ", round(mean(l1_mcmc_posterior), digits=3))
Original L1 Mean: 0.547

julia> println("Shifted Meta Mean (Bad Regime): ", round(mean(shifted_bad_posterior), digits=3))
Shifted Meta Mean (Bad Regime): 0.359

julia> 

julia> s = BayesianFootball.Signals.BayesianKelly(min_edge=0.0)
Signal Strategy: BayesianKelly
├─ Parameters: min_edge=0.0
└─ Logic: BayesianKelly (Baker-McHale 2013) with 0.0% Min Edge Filter.

julia> original_stake = BayesianFootball.Signals.compute_stake(s, l1_mcmc_posterior, odds)
0.04917263787744085

julia> shifted_stake = BayesianFootball.Signals.compute_stake(s, shifted_bad_posterior, odds)
0.0

julia> println("Original L1 Stake: ", round(original_stake * 100, digits=2), "%")
Original L1 Stake: 4.92%

julia> println("Shifted Meta Stake (Bad Regime): ", round(shifted_stake * 100, digits=2), "%")
Shifted Meta Stake (Bad Regime): 0.0%

julia> if shifted_stake < original_stake
           println("SUCCESS: The Meta Model correctly shrunk the stake during a poor predictive regime!")
       else
           println("WARNING: Stake did not shrink. Check model parameters.")
       end
SUCCESS: The Meta Model correctly shrunk the stake during a poor predictive regime!
=#

