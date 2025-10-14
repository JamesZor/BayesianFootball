# scripts/debug_static_model.jl
# --- A standalone script to debug the static model logic ---

println("--- Setting up debug environment ---")

using Pkg
Pkg.activate(".")

# Manually load dependencies
using DataFrames, Turing, LinearAlgebra

# --- 1. Manually include the necessary modules ---
# We include the files directly to bypass the package system for faster iteration.
using BayesianFootball
# Use the modules we just loaded

println("✅ Environment ready.")

# --- 2. Load data and create features (same as before) ---
println("\n--- Loading data and features ---")
data_store = BayesianFootball.Data.load_default_datastore()
feature_set = BayesianFootball.Features.create_features(data_store)
println("✅ Data and features ready.")

# --- 3. Prepare data for the static model ---
home_ids = vcat(feature_set.round_home_ids...)
away_ids = vcat(feature_set.round_away_ids...)
home_goals = vcat(feature_set.round_home_goals...)
away_goals = vcat(feature_set.round_away_goals...)
n_teams = feature_set.n_teams

# --- 4. Define the Turing Model Directly ---
# This is the core logic, isolated from the framework.
@model function static_maher_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors ---
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_home_adv ~ Normal(log(1.3), 0.2)
    
    # --- Identifiability Constraint ---
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    # --- Calculate Goal Rates ---
    log_λs = log_α[home_ids] .+ log_β[away_ids] .+ log_home_adv
    log_μs = log_α[away_ids] .+ log_β[home_ids]
    
    # --- Likelihood ---
    # With recent Turing.jl versions, we use a loop for the likelihood
    # instead of the broadcasted `.~` syntax for clarity and correctness.
    for i in 1:length(home_goals)
        home_goals[i] ~ LogPoisson(log_λs[i])
        away_goals[i] ~ LogPoisson(log_μs[i])
    end
end

# --- 5. Instantiate and Sample ---
println("\n--- Building and sampling the model ---")
turing_model = static_maher_model(n_teams, home_ids, away_ids, home_goals, away_goals)

# Sample for a few iterations to test
chain = sample(turing_model, NUTS(0.65), 10, progress=true)

println("✅ Sampling complete!")
println("\n--- Chain Summary ---")
println(chain)


# --- 4. Define Model Components ---

# Goal rate calculation is a "pure" helper function (no `~` inside).
function calculate_goal_rates(log_α, log_β, log_home_adv, home_ids, away_ids)
    log_λs = log_α[home_ids] .+ log_β[away_ids] .+ log_home_adv
    log_μs = log_α[away_ids] .+ log_β[home_ids]
    return log_λs, log_μs
end

# Priors submodel is correct as is.
@model function set_priors(n_teams)
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_home_adv ~ Normal(log(1.3), 0.2)
    
    return (log_α_raw=log_α_raw, log_β_raw=log_β_raw, log_home_adv=log_home_adv)
end

# The main model now uses the @submodel macro.
@model function static_maher_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # 1. Call the priors submodel using the `to_submodel` function.
    #    The return value of `set_priors` will be assigned to `priors`.
    priors ~ to_submodel(set_priors(n_teams))
    
    # 2. Apply identifiability constraint using the captured values.
    #    This part remains the same and is now correct.
    log_α = priors.log_α_raw .- mean(priors.log_α_raw)
    log_β = priors.log_β_raw .- mean(priors.log_β_raw)
    
    # 3. Use the pure helper to calculate goal rates.
    log_λs, log_μs = calculate_goal_rates(log_α, log_β, priors.log_home_adv, home_ids, away_ids)
    
    # 4. Define the likelihood.
    for i in 1:length(home_goals)
        home_goals[i] ~ LogPoisson(log_λs[i])
        away_goals[i] ~ LogPoisson(log_μs[i])
    end
end

# --- 6. Instantiate and Sample ---
println("\n--- Building and sampling the modular model ---")
turing_model = static_maher_model(n_teams, home_ids, away_ids, home_goals, away_goals)

# Sample to test the new structure
chain = sample(turing_model, NUTS(0.65), 10, progress=true)

println("✅ Sampling complete!")
println("\n--- Chain Summary ---")
println(chain)


# --- Likelihood Submodels ---
# 1. The original Poisson likelihood, now as a submodel
@model function poisson_likelihood(log_λs, log_μs, home_goals, away_goals)
    for i in 1:length(home_goals)
        home_goals[i] ~ LogPoisson(log_λs[i])
        away_goals[i] ~ LogPoisson(log_μs[i])
    end
end

# 2. A new Negative Binomial likelihood submodel
@model function negative_binomial_likelihood(log_λs, log_μs, home_goals, away_goals)
    # Add a prior for the dispersion parameter ϕ.
    # Exponential is a common choice for a positive-constrained parameter.
    ϕ ~ Exponential(1.0)
    
    for i in 1:length(home_goals)
        # The NegativeBinomial is parameterized by mean (μ) and dispersion (ϕ).
        # We must exponentiate our log-rates to get the mean.
        home_goals[i] ~ NegativeBinomial(exp(log_λs[i] + 1), ϕ)
        away_goals[i] ~ NegativeBinomial(exp(log_μs[i] + 1 ), ϕ)
    end
end

@model function static_maher_model(
    n_teams, 
    home_ids, 
    away_ids, 
    home_goals, 
    away_goals;
    likelihood_family::Symbol=:poisson  # Add a keyword argument
)
    # 1. Call the priors submodel (no change here)
    priors ~ to_submodel(set_priors(n_teams))
    
    # 2. Apply identifiability constraint (no change here)
    log_α = priors.log_α_raw .- mean(priors.log_α_raw)
    log_β = priors.log_β_raw .- mean(priors.log_β_raw)
    
    # 3. Calculate goal rates (no change here)
    log_λs, log_μs = calculate_goal_rates(log_α, log_β, priors.log_home_adv, home_ids, away_ids)
    
    # 4. Conditionally call the appropriate likelihood submodel
    if likelihood_family == :poisson
        # We use a dummy variable `_` because we don't need the return value.
        a ~ to_submodel(poisson_likelihood(log_λs, log_μs, home_goals, away_goals))
    elseif likelihood_family == :neg_bin
        a ~ to_submodel(negative_binomial_likelihood(log_λs, log_μs, home_goals, away_goals))
    else
        error("Unknown likelihood family: $likelihood_family")
    end
end
println("\n--- Sampling with Poisson Likelihood ---")
turing_model_poisson = static_maher_model(
    n_teams, home_ids, away_ids, home_goals, away_goals, 
    likelihood_family=:poisson
)
chain_poisson = sample(turing_model_poisson, NUTS(0.65), 100)
println(chain_poisson)


println("\n--- Sampling with Negative Binomial Likelihood ---")
turing_model_nb = static_maher_model(
    n_teams, home_ids, away_ids, home_goals, away_goals, 
    likelihood_family=:neg_bin
)
chain_nb = sample(turing_model_nb, NUTS(0.65), 10)
println(chain_nb)
