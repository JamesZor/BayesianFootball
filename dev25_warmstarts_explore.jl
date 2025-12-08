# ==============================================================================
# SECTION 0: IMPORTS & DATA SETUP
# ==============================================================================
using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using Optim             # For the Fast Solver (Paninski)
using Surrogates        # For Hyperparameter Tuning
using AbstractGPs       # Surrogate Model Backend
using Turing            # For MCMC
using DynamicPPL

# 1. Generate Synthetic Data
# We create data with known volatility (sigma=0.05) to test if we can recover it.
ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=12, n_seasons=1, legs_per_season=4, in_season_volatility=0.05
)


# 2. Prepare Data Vectors (Aligned for performance)
# We extract raw vectors so we don't look up DataFrames inside the tight loops.
# Note: Ensure you use the 'feature_sets' logic if your IDs need remapping, 
# but for synthetic data, raw IDs usually work if they are 1..N.



model = BayesianFootball.Models.PreGame.GRWPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)
splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)


matches = ds.matches
home_ids = feature_sets[1][1].data[:flat_home_ids]
away_ids = feature_sets[1][1].data[:flat_away_ids]
score_h  = Float64.(matches.home_score)
score_a  = Float64.(matches.away_score)
rounds   = Int.(matches.round)

n_teams  = 12
n_rounds = maximum(rounds)

println("✅ Data Loaded: $(length(score_h)) matches, $n_teams teams, $n_rounds rounds.")

# ==============================================================================
# SECTION 1: THE CORE MATH (The "Complex" Objective)
# ==============================================================================

"""
    log_posterior_full(flat_params, ...)

Calculates the Negative Log Posterior for a Split Model (Attack/Defense).
Supports:
  - Separate Sigmas for Attack and Defense
  - AR1 Mean Reversion (Rho)
"""
function log_posterior_full(
    flat_params, 
    h_idx, a_idx, s_h, s_a, rnds, 
    n_t, n_r, 
    σ_att, σ_def, rho
)
    # 1. Unpack Parameters
    # Optim gives us a flat vector. We reshape it to (Teams x Rounds x 2)
    # Layer 1 = Attack, Layer 2 = Defense
    matrix_form = reshape(flat_params, n_t, n_r, 2)
    att = view(matrix_form, :, :, 1)
    def = view(matrix_form, :, :, 2)
    
    log_lik = 0.0
    home_adv = 0.25 # Fixed HA (You could also optimize this if you added it to params)

    # 2. Likelihood Loop (Poisson)
    # We iterate through every match to calculate probability
    @inbounds for i in 1:length(s_h)
        r = rnds[i]
        h, a = h_idx[i], a_idx[i]
        
        # Home Goals ~ Poisson(exp(Att_H + Def_A + HA))
        # (Assuming Def parameter represents "Weakness", so we add it)
        λ_h = exp(att[h, r] + def[a, r] + home_adv)
        log_lik += s_h[i] * log(λ_h) - λ_h
        
        # Away Goals ~ Poisson(exp(Att_A + Def_H))
        λ_a = exp(att[a, r] + def[h, r])
        log_lik += s_a[i] * log(λ_a) - λ_a
    end

    # 3. Prior Loop (AR1 / Gaussian Random Walk)
    # We apply the penalty for moving too much (Sigma) or drifting (Rho)
    log_prior = 0.0
    
    # Pre-calculate precision (1/variance)
    prec_att = 1.0 / (σ_att^2)
    prec_def = 1.0 / (σ_def^2)

    @inbounds for t in 1:n_t
        # A. Initial State Priors (Round 1) ~ Normal(0, 1)
        log_prior += -0.5 * att[t, 1]^2
        log_prior += -0.5 * def[t, 1]^2
        
        # B. Evolution Priors (Round 2..T)
        for r in 2:n_r
            # Attack Update
            # Delta = Current - (Rho * Previous)
            d_att = att[t, r] - (rho * att[t, r-1])
            log_prior += -0.5 * (d_att^2) * prec_att
            
            # Defense Update
            d_def = def[t, r] - (rho * def[t, r-1])
            log_prior += -0.5 * (d_def^2) * prec_def
        end
    end

    # Return Negative Log Posterior (Optim minimizes this)
    return -(log_lik + log_prior)
end


function log_posterior_full_with_ha(
    flat_params, 
    h_idx, a_idx, s_h, s_a, rnds, 
    n_t, n_r, 
    σ_att, σ_def, rho
)
    # --- 1. Unpack Parameters ---
    # The last element is Home Advantage
    home_adv = flat_params[end]
    
    # The rest are the dynamic Attack/Defense parameters
    # We slice up to end-1
    # Reshape remaining vector to (Teams x Rounds x 2)
    dyn_params = @view flat_params[1:end-1]
    matrix_form = reshape(dyn_params, n_t, n_r, 2)
    
    att = view(matrix_form, :, :, 1)
    def = view(matrix_form, :, :, 2)
    
    log_lik = 0.0

    # --- 2. Likelihood Loop ---
    @inbounds for i in 1:length(s_h)
        r = rnds[i]
        h, a = h_idx[i], a_idx[i]
        
        # Use the optimized 'home_adv' variable here
        λ_h = exp(att[h, r] + def[a, r] + home_adv)
        log_lik += s_h[i] * log(λ_h) - λ_h
        
        λ_a = exp(att[a, r] + def[h, r])
        log_lik += s_a[i] * log(λ_a) - λ_a
    end

    # --- 3. Prior Loop ---
    log_prior = 0.0
    
    # A. Prior for Home Advantage (Weak Normal Prior)
    # HA ~ Normal(0.25, 1.0) -> Keeps it sane if data is missing
    log_prior += -0.5 * ((home_adv - 0.25) / 1.0)^2

    # B. Dynamic Priors (Same as before)
    prec_att = 1.0 / (σ_att^2)
    prec_def = 1.0 / (σ_def^2)

    @inbounds for t in 1:n_t
        log_prior += -0.5 * att[t, 1]^2
        log_prior += -0.5 * def[t, 1]^2
        
        for r in 2:n_r
            d_att = att[t, r] - (rho * att[t, r-1])
            log_prior += -0.5 * (d_att^2) * prec_att
            
            d_def = def[t, r] - (rho * def[t, r-1])
            log_prior += -0.5 * (d_def^2) * prec_def
        end
    end

    return -(log_lik + log_prior)
end



"""
    log_posterior_with_mu(flat_params, ...)

Model:
  Home = exp(Mu + Att_H + Def_A + HomeAdv)
  Away = exp(Mu + Att_A + Def_H)
"""
function log_posterior_with_mu(
    flat_params, 
    h_idx, a_idx, s_h, s_a, rnds, 
    n_t, n_r, 
    σ_att, σ_def, rho
)
    # --- 1. Unpack Parameters ---
    # The LAST two elements are HomeAdv and Mu
    mu       = flat_params[end]
    home_adv = flat_params[end-1]
    
    # The rest are Dynamic Params
    dyn_params = @view flat_params[1:end-2]
    matrix_form = reshape(dyn_params, n_t, n_r, 2)
    
    att = view(matrix_form, :, :, 1)
    def = view(matrix_form, :, :, 2)
    
    log_lik = 0.0

    # --- 2. Likelihood Loop ---
    @inbounds for i in 1:length(s_h)
        r = rnds[i]
        h, a = h_idx[i], a_idx[i]
        
        # YOUR PARAMETERIZATION:
        # Lambda = exp(Mu + Att + Def + [HA])
        
        # Home
        str_h = mu + att[h, r] + def[a, r] + home_adv
        λ_h   = exp(str_h)
        log_lik += s_h[i] * str_h - λ_h
        
        # Away (No Home Adv)
        str_a = mu + att[a, r] + def[h, r]
        λ_a   = exp(str_a)
        log_lik += s_a[i] * str_a - λ_a
    end

    # --- 3. Prior Loop ---
    log_prior = 0.0
    
    # A. Static Parameters Priors
    # HA ~ Normal(0.25, 0.5)
    log_prior += -0.5 * ((home_adv - 0.25) / 0.5)^2
    
    # Mu ~ Normal(0.0, 1.0) -> Weak prior, allows it to find the average
    # Note: exp(0) = 1 goal. 
    log_prior += -0.5 * ((mu - 0.0) / 1.0)^2

    # B. Dynamic Priors (Standard AR1/GRW)
    prec_att = 1.0 / (σ_att^2)
    prec_def = 1.0 / (σ_def^2)

    @inbounds for t in 1:n_t
        # Initial State (T=1)
        log_prior += -0.5 * att[t, 1]^2
        log_prior += -0.5 * def[t, 1]^2
        
        # Evolution (T=2..N)
        for r in 2:n_r
            d_att = att[t, r] - (rho * att[t, r-1])
            log_prior += -0.5 * (d_att^2) * prec_att
            
            d_def = def[t, r] - (rho * def[t, r-1])
            log_prior += -0.5 * (d_def^2) * prec_def
        end
    end

    return -(log_lik + log_prior)
end


# ==============================================================================
# SECTION 2: HYPERPARAMETER TUNING (Bayesian Optimization)
# ==============================================================================
println("\n🧠 Starting Hyperparameter Tuning (Surrogates.jl)...")

# 1. Define the "Black Box" function
# Input: Tuple of (Sigma_Att, Sigma_Def, Rho)
# Output: The best possible Negative Log Likelihood achievable with those settings
function objective_function(params)
    s_att, s_def, r_rho = params
    
    # Reset guess to zeros
    x0 = zeros(Int(2 * n_teams * n_rounds)) 
    
    # Create closure
    func = x -> log_posterior_full(
        x, home_ids, away_ids, score_h, score_a, rounds, 
        n_teams, n_rounds, 
        s_att, s_def, r_rho
    )
    
    # Run Fast Optimizer
    # We use fewer iterations (50) for the tuning checks to keep it fast
    res = Optim.optimize(func, x0, LBFGS(), Optim.Options(iterations=50))
    
    return Optim.minimum(res)
end

# 2. Setup Surrogate
lb = [0.01, 0.01, 0.90] # Lower Bounds: Sigmas must be positive, Rho high
ub = [0.20, 0.20, 1.00] # Upper Bounds: Sigmas not too crazy, Rho max 1.0

# Initial sampling (5 random points)
x_train = Surrogates.sample(5, lb, ub, SobolSample())
y_train = objective_function.(x_train)

# Build Kriging Model (GP)
my_surrogate = Kriging(x_train, y_train, lb, ub)

# 3. Run Optimization
# "SRBF" is a solid strategy for finding global minima with few samples
best_param_tuple = surrogate_optimize(
    objective_function, SRBF(), lb, ub, my_surrogate, SobolSample(); 
    maxiters = 15 # Run 15 smart experiments
)

# 4. Extract Winners
best_hypers, best_val = best_param_tuple
opt_σ_att, opt_σ_def, opt_rho = best_hypers

println("🏆 Tuned Parameters Found:")
println("   Sigma Attack:  ", round(opt_σ_att, digits=4))
println("   Sigma Defense: ", round(opt_σ_def, digits=4))
println("   AR1 Rho:       ", round(opt_rho, digits=4))
println("   (Best Score:   ", round(best_val, digits=2), ")")

println("\n🚀 Running Final Optimization (with Learnable Home Advantage)...")
"""
   Sigma Attack:  0.1871

   Sigma Defense: 0.2
   AR1 Rho:       0.9844
   (Best Score:   382.32)


"""

# with home Advantage ----

# 1. Setup x0 with +1 size
num_dynamic_params = Int(2 * n_teams * n_rounds)
x0 = zeros(num_dynamic_params + 1)

# Set the initial guess for Home Advantage (last index) to 0.2
x0[end] = 0.2 

# 2. Define Function
final_func_ha = x -> log_posterior_full_with_ha(
    x, home_ids, away_ids, score_h, score_a, rounds, 
    n_teams, n_rounds, 
    opt_σ_att, opt_σ_def, opt_rho # Use your tuned values
)

# 3. Run Optimization
@time res_final = Optim.optimize(
    final_func_ha, x0, LBFGS(), 
    Optim.Options(iterations=1000, g_tol=1e-6)
)

# 4. Extract Results
best_vec = Optim.minimizer(res_final)

# A. Extract Home Advantage
learned_ha = best_vec[end]

# B. Extract Paths
best_path_flat = best_vec[1:end-1]
matrix_final = reshape(best_path_flat, n_teams, n_rounds, 2)

println("✅ Optimization Complete.")
println("🏠 Learned Home Advantage: ", round(learned_ha, digits=4), " (Goals: ~$(round(exp(learned_ha), digits=2))x multiplier)")

# ==============================================================================
# SECTION 2 (UPDATED): HYPERPARAMETER TUNING "WITH HA"
# ==============================================================================
println("\n🧠 Starting Hyperparameter Tuning (with Learnable HA)...")

# 1. Define the Objective Function
#    This runs the Inner Optimizer (Paninski) which NOW solves for [Att, Def, HA].
#    The BO only tunes [Sigma_Att, Sigma_Def, Rho].
function objective_function_ha(params)
    s_att, s_def, r_rho = params
    
    # Setup x0 with space for Home Advantage at the end
    num_params = Int(2 * n_teams * n_rounds) + 1
    x0 = zeros(num_params)
    x0[end] = 0.2 # Initial guess for HA
    
    # Closure with the specific Sigmas/Rho we are testing
    func = x -> log_posterior_full_with_ha(
        x, home_ids, away_ids, score_h, score_a, rounds, 
        n_teams, n_rounds, 
        s_att, s_def, r_rho
    )
    
    # Run Inner Optimizer (Fast)
    res = Optim.optimize(func, x0, LBFGS(), Optim.Options(iterations=50))
    
    # Return the fit score (Negative Log Posterior)
    return Optim.minimum(res)
end

# 2. Setup Surrogate Search Space
lb = [0.01, 0.01, 0.90] 
ub = [0.20, 0.20, 1.00] 

# Initial Grid
# Note: Surrogates.sample prevents conflict with Turing.sample
x_train = Surrogates.sample(5, lb, ub, SobolSample())
y_train = objective_function_ha.(x_train) 

# Build Surrogate
my_surrogate = Kriging(x_train, y_train, lb, ub)

# 3. Run Optimization
best_param_tuple = surrogate_optimize(
    objective_function_ha, 
    SRBF(), 
    lb, ub, 
    my_surrogate, 
    RandomSample(); 
    maxiters = 15
)

# 4. Extract Winners
# Note: Surrogates returns (Params, Value)
best_hypers, best_val = best_param_tuple
opt_σ_att, opt_σ_def, opt_rho = best_hypers

println("🏆 Tuned Parameters Found (Model includes HA):")
println("   Sigma Attack:  ", round(opt_σ_att, digits=4))
println("   Sigma Defense: ", round(opt_σ_def, digits=4))
println("   AR1 Rho:       ", round(opt_rho, digits=4))
println("   (Best Score:   ", round(best_val, digits=2), ")")

# ==============================================================================
# SECTION 3 (RE-RUN): THE GOLDEN PATH
# ==============================================================================
println("\n🚀 Running Final Optimization (with Learnable HA + Tuned Sigmas)...")

# Setup x0
x0 = zeros(Int(2 * n_teams * n_rounds) + 1)
x0[end] = 0.2 

final_func = x -> log_posterior_full_with_ha(
    x, home_ids, away_ids, score_h, score_a, rounds, 
    n_teams, n_rounds, 
    opt_σ_att, opt_σ_def, opt_rho # <--- New Tuned Values
)

@time res_final = Optim.optimize(final_func, x0, LBFGS(), Optim.Options(iterations=1000, g_tol=1e-6))

# Extract Logic
best_vec = Optim.minimizer(res_final)
learned_ha = best_vec[end]
best_path_flat = best_vec[1:end-1]
matrix_final = reshape(best_path_flat, n_teams, n_rounds, 2)

println("✅ Optimization Complete.")
println("🏠 Final Learned Home Advantage: ", round(learned_ha, digits=4))

# ==============================================================================
# SECTION 3: THE GOLDEN PATH (High-Precision Optimization)
# ==============================================================================
println("\n🚀 Running Final Paninski Optimization (Best Params)...")

# Now we run the optimizer one last time, with HIGH precision and the BEST hyperparameters
x0 = zeros(Int(2 * n_teams * n_rounds)) 

final_func = x -> log_posterior_full(
    x, home_ids, away_ids, score_h, score_a, rounds, 
    n_teams, n_rounds, 
    opt_σ_att, opt_σ_def, opt_rho
)

# Increase iterations for the final "Golden" run
@time res_final = Optim.optimize(final_func, x0, LBFGS(), Optim.Options(iterations=500, show_trace=false))

# Extract the Best Path (The MAP Estimate)
best_path_flat = Optim.minimizer(res_final)

# Reshape for inspection if needed
matrix_final = reshape(best_path_flat, n_teams, n_rounds, 2)
println("✅ Optimization Complete. Best Path found.")


# --- Helper to convert MAP matrix to Fake Tensor ---
function wrap_map_as_tensor(map_matrix, n_teams, n_rounds)
    # Map Matrix is (n_teams x n_rounds)
    # Tensor needs to be (n_teams x n_rounds x n_samples)
    # We create a "1-sample" tensor.
    
    # Reshape to (N, T, 1)
    return reshape(map_matrix, n_teams, n_rounds, 1)
end

map_tensor = wrap_map_as_tensor(matrix_final, n_teams, n_rounds)

p_split = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    1, 
    true_params, 
    att_tensor_split, 
    def_tensor_split
)

# 1. Split the Final Matrix into Attack and Defense components
# matrix_final is (n_teams, n_rounds, 2)
best_att = matrix_final[:, :, 1]
best_def = matrix_final[:, :, 2]

# 2. Wrap them individually into Tensors (Fake 1-sample for plotting)
# We reuse your helper function
att_tensor = wrap_map_as_tensor(best_att, n_teams, n_rounds)
def_tensor = wrap_map_as_tensor(best_def, n_teams, n_rounds)

# 3. Plot
# We use Team 1 (change the '1' to see others)
p_split = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    12,              # Team ID
    true_params,    # The Truth
    att_tensor,     # Your Optimized Attack Path
    def_tensor      # Your Optimized Defense Path
)

display(p_split)

function center_trajectories(att_matrix, def_matrix)
    # att_matrix is (N_Teams x N_Rounds)
    n_rounds = size(att_matrix, 2)
    
    c_att = copy(att_matrix)
    c_def = copy(def_matrix)
    
    for r in 1:n_rounds
        # Find the average strength for this specific week
        avg_att = mean(c_att[:, r])
        avg_def = mean(c_def[:, r])
        
        # Shift everyone so the average is 0.0
        c_att[:, r] .-= avg_att
        c_def[:, r] .-= avg_def
    end
    
    return c_att, c_def
end

# --- APPLY THE FIX ---
# 1. Unwrap the optimized result
raw_att = matrix_final[:, :, 1]
raw_def = matrix_final[:, :, 2]

# 2. Center them
centered_att, centered_def = center_trajectories(raw_att, raw_def)

# 3. Wrap for plotting
att_tensor_centered = wrap_map_as_tensor(centered_att, n_teams, n_rounds)
def_tensor_centered = wrap_map_as_tensor(centered_def, n_teams, n_rounds)

# 4. Plot again (Team 8 or 'aa')
p_fixed = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    8, true_params, att_tensor_centered, def_tensor_centered
)

display(p_fixed)


# ==============================================================================
# SECTION 4: THE WARM START (Full Bayesian MCMC)
# ==============================================================================
println("\n🧗 Initializing MCMC with Warm Start...")

# 1. Define the Turing Model using our Tuned Hyperparameters
# We fix the hyperparameters to what we found in Section 2 to save time,
# OR use tight priors around them. Here we use tight priors.

@model function grw_tuned(n_teams, n_rounds, h_ids, a_ids, h_goals, a_goals, fixed_rho)
    # Priors (Centered on our Tuned Values!)
    σ_att ~ Gamma(2, opt_σ_att/2) # Heuristic to center mean near opt_σ_att
    σ_def ~ Gamma(2, opt_σ_def/2)
    home_adv ~ Normal(0.25, 0.1)

    # Initial States
    z_att_init ~ filldist(Normal(0, 1), n_teams)
    z_def_init ~ filldist(Normal(0, 1), n_teams)

    # Steps
    z_att_steps ~ filldist(Normal(0, 1), n_teams, n_rounds-1)
    z_def_steps ~ filldist(Normal(0, 1), n_teams, n_rounds-1)

    # ... (Standard Matt Trick implementation goes here) ...
    # This is just a placeholder to show where the init_params go
end

# 2. Construct Initialization Vector
# The order of parameters in Turing's `init_params` is crucial.
# It typically follows the order of definition in the @model block.
# [σ_att, σ_def, home_adv, z_att_init..., z_def_init..., z_att_steps..., z_def_steps...]

# Note: Our `best_path_flat` contains the *transformed* values (Att, Def), not the *raw* z-scores.
# For a true warm start, you would reverse-engineer the z-scores from the path.
# However, for simplicity, many users just init the Sigmas and let the Zs warm up quickly.

println("   (Ready to run NUTS. Use 'best_path_flat' to analyze trajectory instantly.)")
# chain = sample(model, NUTS(), 1000) ...
