using BayesianFootball, Optim, LinearAlgebra


model = BayesianFootball.Models.PreGame.GRWPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)


# --- 1. PREP DATA (Aligning vectors) ---
# We use the raw match data to ensure alignment
matches = ds.matches
home_ids = feature_sets[1][1].data[:flat_home_ids]
away_ids = feature_sets[1][1].data[:flat_away_ids]
score_h  = matches.home_score
score_a  = matches.away_score
rounds   = Int.(matches.round) # Ensure Int

n_teams  = 12
n_rounds = Int(maximum(rounds)) # Ensure Int

# --- 2. DEFINE THE OBJECTIVE (Negative Log Posterior) ---
function log_posterior(flat_params, matches_h, matches_a, score_h, score_a, rounds, n_teams, n_rounds, σ_rw)
    # 1. Unpack Parameters
    # Reshape vector back to Matrix (Rows=Teams, Cols=Rounds)
    # params[t, r] = Strength of Team t in Round r
    params = reshape(flat_params, n_teams, n_rounds)
    
    log_lik = 0.0
    home_adv = 0.25 # Fixed Home Advantage for simplicity (or optimize it too)

    # 2. Likelihood (Poisson for every match)
    # Loop over all matches (using the length of the score vector)
    for i in 1:length(score_h)
        r = rounds[i]
        h = matches_h[i]
        a = matches_a[i]
        
        # Get dynamic strengths for this specific round
        str_h = params[h, r]
        str_a = params[a, r]

        # Home Goals: exp(H - A + HA)
        λ_h = exp(str_h - str_a + home_adv)
        log_lik += score_h[i] * log(λ_h) - λ_h
        
        # Away Goals: exp(A - H)
        λ_a = exp(str_a - str_h)
        log_lik += score_a[i] * log(λ_a) - λ_a
    end

    # 3. Prior (Gaussian Random Walk)
    # This creates the "Tridiagonal" structure Paninski exploits
    log_prior = 0.0
    
    # Prior for Round 1 (Initial State ~ Normal(0, 1))
    for t in 1:n_teams
        log_prior += -0.5 * (params[t, 1])^2
    end

    # Prior for Rounds 2..T (Random Walk ~ Normal(x_t-1, σ))
    # x_t ~ N(x_t-1, σ) => log_prob proportional to -(x_t - x_t-1)^2 / 2σ^2
    inv_var = 1.0 / (σ_rw^2)
    for t in 1:n_teams
        for r in 2:n_rounds
            diff = params[t, r] - params[t, r-1]
            log_prior += -0.5 * (diff^2) * inv_var
        end
    end

    # Return Negative Log Posterior (because Optim minimizes)
    return -(log_lik + log_prior)
end

# --- 3. RUN OPTIMIZATION ---
# Initial Guess: All teams are average (0.0)
x0 = zeros(Int(n_teams * n_rounds)) 

# Create a closure to lock in the data args
# We assume a volatility (sigma) of 0.05
obj_func = x -> log_posterior(x, home_ids, away_ids, score_h, score_a, rounds, n_teams, n_rounds, 0.05)

println("Starting Optimization (The Paninski Trick)...")
@time res = Optim.optimize(obj_func, x0, LBFGS(), Optim.Options(show_trace=true, iterations=200))

# --- 4. EXTRACT RESULTS ---
# Reshape the best vector back into (Teams x Rounds)
best_params = reshape(Optim.minimizer(res), n_teams, n_rounds)

println("Optimization Complete.")
println("Best strength for Team 1 in Round 1: ", best_params[1, 1])
println("Best strength for Team 1 in Round $(n_rounds): ", best_params[1, end])



# --- Helper to convert MAP matrix to Fake Tensor ---
function wrap_map_as_tensor(map_matrix, n_teams, n_rounds)
    # Map Matrix is (n_teams x n_rounds)
    # Tensor needs to be (n_teams x n_rounds x n_samples)
    # We create a "1-sample" tensor.
    
    # Reshape to (N, T, 1)
    return reshape(map_matrix, n_teams, n_rounds, 1)
end



# 1. Convert Optim result to Tensor format
# best_params is (12 x 44) -> (12 x 44 x 1)
map_tensor = wrap_map_as_tensor(best_params, n_teams, n_rounds)

# 2. Fake a "Defense" tensor (Zeros) just to satisfy the plot function signature
# (Since the simple Paninski example only solved for one parameter)
dummy_def = zeros(n_teams, n_rounds, 1)

# 3. Plot for Team 1
# We use your existing function!
p = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    3,              # Team ID
    true_params,    # The Truth
    map_tensor,     # Our Optimized Path (treated as "Attack")
    dummy_def       # Dummy
)

display(p)


"""
v2 improved 

"""

using Optim, LinearAlgebra

# --- 1. The Split Objective Function ---
function log_posterior_split(flat_params, matches_h, matches_a, score_h, score_a, rounds, n_teams, n_rounds, σ_rw)
    # The params vector is now twice as long: [All_Attacks... ; All_Defenses...]
    # We reshape it into a (Teams x Rounds x 2) tensor for easy indexing
    # matrix_form[:, :, 1] is Attack
    # matrix_form[:, :, 2] is Defense
    matrix_form = reshape(flat_params, n_teams, n_rounds, 2)
    att = view(matrix_form, :, :, 1)
    def = view(matrix_form, :, :, 2)
    
    log_lik = 0.0
    home_adv = 0.25 # You can also make this a parameter to optimize!

    # A. Likelihood (Poisson)
    for i in 1:length(score_h)
        r = rounds[i]
        h = matches_h[i]
        a = matches_a[i]
        
        # Model: Goal_Home ~ Poisson(exp(Att_Home + Def_Away + HA))
        # Note: We use +Def_Away because your synthetic data likely defines 'Def' as 'defensive weakness' (higher = concede more)
        str_h = att[h, r] + def[a, r] + home_adv
        λ_h = exp(str_h)
        log_lik += score_h[i] * str_h - λ_h # Log-Lik of Poisson
        
        # Model: Goal_Away ~ Poisson(exp(Att_Away + Def_Home))
        str_a = att[a, r] + def[h, r]
        λ_a = exp(str_a)
        log_lik += score_a[i] * str_a - λ_a
    end

    # B. Prior (Gaussian Random Walk)
    # We apply the penalty to both Attack and Defense chains
    inv_var = 1.0 / (σ_rw^2)
    log_prior = 0.0
    
    # Loop over Attack (k=1) and Defense (k=2)
    for k in 1:2
        layer = view(matrix_form, :, :, k)
        for t in 1:n_teams
            # T=1 Prior: Normal(0, 1)
            log_prior += -0.5 * layer[t, 1]^2 
            
            # T=2..N Random Walk: Normal(x_{t-1}, sigma)
            for r in 2:n_rounds
                delta = layer[t, r] - layer[t, r-1]
                log_prior += -0.5 * (delta^2) * inv_var
            end
        end
    end

    return -(log_lik + log_prior) # Minimize Negative Log Posterior
end

# --- 2. Run the Optimization ---

# A. Initialize (Size is now 2x larger)
# params: 2 * 12 teams * 44 rounds = 1056 parameters
x0 = zeros(Int(2 * n_teams * n_rounds)) 

# B. Wrap and Optimize
# We fix sigma at 0.05 to match your synthetic volatility
obj_func = x -> log_posterior_split(x, home_team_ids, away_team_ids, obs_goals, obs_goals, rounds, n_teams, n_rounds, 0.05)
# Note: In the likelihood loop above, I assumed 'obs_goals' contained both home and away scores interleaved or separate.
# Let's clarify the data passing:
# Pass `ds.matches.home_score` and `ds.matches.away_score` explicitly.

func_to_optimize = x -> log_posterior_split(
    x, 
    home_ids, 
    away_ids, 
    ds.matches.home_score, 
    ds.matches.away_score, 
    Int.(ds.matches.round), 
    n_teams, 
    n_rounds, 
    0.2
)

println("Optimizing Split Model (Att + Def)...")
@time res_split = Optim.optimize(func_to_optimize, x0, LBFGS(), Optim.Options(iterations=500, show_trace=true))

# --- 3. Extract Results ---
best_params_split = reshape(Optim.minimizer(res_split), n_teams, n_rounds, 2)
best_att = best_params_split[:, :, 1]
best_def = best_params_split[:, :, 2]

println("Split Optimization Complete.")

# 1. Wrap results
att_tensor_split = wrap_map_as_tensor(best_att, n_teams, n_rounds)
def_tensor_split = wrap_map_as_tensor(best_def, n_teams, n_rounds)

# 2. Plot for Team 1 (e.g., "aa")
p_split = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    1, 
    true_params, 
    att_tensor_split, 
    def_tensor_split
)

display(p_split)

# Tighter tolerances = More precise, usually takes more iterations
options = Optim.Options(
    iterations = 10_000,      # Hard limit (safety net)
    g_tol = 1e-8,            # Gradient Tolerance: Stop if gradient is smaller than this
    x_abstol = 1e-8,            # Input Tolerance: Stop if steps are smaller than this
    show_trace = true,       # Show progress
    show_every = 200          # Don't spam console, print every 50 steps
)

res = Optim.optimize(func_to_optimize, x0, LBFGS(), options)

# CHECKING THE RESULT PROGRAMMATICALLY
if Optim.converged(res)
    println("✅ Converged successfully in $(Optim.iterations(res)) steps.")
else
    println("⚠️ Did not converge! Final Gradient: $(Optim.g_residual(res))")
end

# --- 3. Extract Results ---
best_params_split = reshape(Optim.minimizer(res), n_teams, n_rounds, 2)
best_att = best_params_split[:, :, 1]
best_def = best_params_split[:, :, 2]

println("Split Optimization Complete.")

# 1. Wrap results
att_tensor_split = wrap_map_as_tensor(best_att, n_teams, n_rounds)
def_tensor_split = wrap_map_as_tensor(best_def, n_teams, n_rounds)

# 2. Plot for Team 1 (e.g., "aa")
p_split = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    2, 
    true_params, 
    att_tensor_split, 
    def_tensor_split
)



## Posterior Distributions
using ForwardDiff, LinearAlgebra

# 1. RUN OPTIMIZATION (As before, but with checks)
res_split = Optim.optimize(
    func_to_optimize, 
    x0, 
    LBFGS(), 
    Optim.Options(g_tol=1e-6, iterations=1000, show_trace=false)
)

if !Optim.converged(res_split)
    @warn "Optimization did not converge. Results may be inaccurate."
else
    println("Converged in $(Optim.iterations(res_split)) iterations.")
end

# --- 2. THE LAPLACE APPROXIMATION (Getting the Ribbons) ---

# A. Extract the Mode (Best Path)
μ_vec = Optim.minimizer(res_split)

# B. Calculate Curvature (Hessian)
# We use ForwardDiff to get the exact 2nd derivative of your objective function at the solution
println("Calculating Hessian for Uncertainty...")
H = ForwardDiff.hessian(func_to_optimize, μ_vec)

# C. Calculate Covariance (Inverse of Hessian)
# Note: Since we minimized Negative Log Posterior, H is the Precision Matrix.
# Sigma = inv(H)
Σ = inv(H) 

# D. Extract Standard Deviations (Diagonal elements)
# These are the "Standard Errors" for every single week for every team
σ_vec = sqrt.(diag(Σ))

# --- 3. REPACKAGING FOR PLOTTING ---

# Reshape Mean and Std vectors back to (Teams x Rounds x 2)
# 1 = Attack, 2 = Defense
μ_matrix = reshape(μ_vec, n_teams, n_rounds, 2)
σ_matrix = reshape(σ_vec, n_teams, n_rounds, 2)

# Helper to create "Fake Samples" for your plotter
# We generate 3 slices: [Mean - 1.96*Std, Mean, Mean + 1.96*Std]
# This tricks your existing plotter into drawing the correct 95% ribbon.
function stats_to_ribbon_tensor(means, stds)
    n_t, n_r = size(means)
    tensor = zeros(n_t, n_r, 3) # 3 "Samples"
    
    tensor[:, :, 1] = means .- 1.96 .* stds # Lower Bound (Sample 1)
    tensor[:, :, 2] = means                 # Mean (Sample 2, Median)
    tensor[:, :, 3] = means .+ 1.96 .* stds # Upper Bound (Sample 3)
    
    return tensor
end

# Convert Attack and Defense
att_with_uncertainty = stats_to_ribbon_tensor(μ_matrix[:,:,1], σ_matrix[:,:,1])
def_with_uncertainty = stats_to_ribbon_tensor(μ_matrix[:,:,2], σ_matrix[:,:,2])

# --- 4. PLOT ---
# Now your plot will have real uncertainty ribbons derived from the Hessian!
p = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    12,                  # Team 1
    true_params,        # Truth
    att_with_uncertainty, 
    def_with_uncertainty
)
display(p)

"""
# --------------------------------------------------------------------------------------
version one  - simple model 

# --------------------------------------------------------------------------------------
"""


###########
using BayesianFootball, SparseArrays, LinearAlgebra, Optim, ForwardDiff
# 1. Generate Data
ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=12, n_seasons=1, legs_per_season=4, in_season_volatility=0.05
)

# Convert to simple vectors for the optimizer
obs_goals = vcat(ds.matches.home_score, ds.matches.away_score)
home_team_ids = ds.matches.home_team_id
away_team_ids = ds.matches.away_team_id
rounds = ds.matches.round
n_teams = 12
n_rounds = maximum(rounds)

model = BayesianFootball.Models.PreGame.GRWPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)



splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)


# Convert to simple vectors for the optimizer
obs_goals = vcat(ds.matches.home_score, ds.matches.away_score)
home_team_ids = feature_sets[1][1].data[:flat_home_ids]
away_team_ids = feature_sets[1][1].data[:flat_away_ids]
rounds = ds.matches.round
n_teams = 12
n_rounds = maximum(rounds)


# ==============================================================================
# FAST MAP SOLVER (The "Paninski" Logic)
# ==============================================================================

# 1. Define the Negative Log Posterior (Target to Minimize)
function log_posterior(flat_params, n_teams, n_rounds, h_idx, a_idx, goals, σ_rw)
    # Reshape: (Teams x Rounds)
    # We store parameters as a long vector for Optim.jl
    # Structure: [Team1_T1...Team1_Tn, Team2_T1...]
    params = reshape(flat_params, n_teams, n_rounds)
    
    log_lik = 0.0
    
    # A. Likelihood (Poisson) - Vectorized
    # We construct the linear predictor for every match
    # λ = exp(Att_Home + Def_Away + HomeAdv) -> Simplified here to just Strength
    
    # *Note*: For full model, you'd split Att/Def params. 
    # This is a simplified "Team Strength" scalar example.
    for i in 1:length(goals)
        h, a, r = h_idx[i], a_idx[i], rounds[i]
        θ_h = params[h, r]
        θ_a = params[a, r] # Ideally Def parameter
        
        λ = exp(θ_h - θ_a) # Simple attacking model
        log_lik += goals[i] * log(λ) - λ # Poisson LogLik
    end

    # B. Prior (Gaussian Random Walk) - The "Tridiagonal" Part
    log_prior = 0.0
    for t in 1:n_teams
        # Initial state prior
        log_prior += -0.5 * (params[t, 1])^2 
        # Random Walk steps
        for r in 2:n_rounds
            # This (x_t - x_{t-1})^2 creates the band structure
            log_prior += -0.5 * ((params[t, r] - params[t, r-1]) / σ_rw)^2
        end
    end
    
    return -(log_lik + log_prior) # Return Negative Log Post
end

# 2. Run Optimization (Newton-Raphson)
# We use LBFGS or NewtonTrustRegion. For exact Paninski, we want Newton with Hessian.

# Initial Guess (Zeros)
x0 = zeros(Float64, n_teams * n_rounds)
x0 = zeros(Int(n_teams * n_rounds))

# Wrap function for Optim
obj_func = x -> log_posterior(x, n_teams, n_rounds, home_team_ids, away_team_ids, obs_goals, 0.05)

println("Optimizing path for $(n_teams) teams over $(n_rounds) rounds...")
@time res = Optim.optimize(obj_func, x0, LBFGS(), Optim.Options(iterations=100))

# 3. Extract the "Best Path"
map_path = reshape(Optim.minimizer(res), n_teams, n_rounds)

# This solves in SECONDS vs Hours.
#
#
#
## EXPLORATORY: StateSpaceDynamics.jl Interface
# Ref:
using StateSpaceDynamics

# 1. Format Data: Y must be (N_teams x N_rounds)
# Fill with 'missing' where teams didn't play (if supported) or 0 (and use masks)
Y = Matrix{Union{Float64, Missing}}(missing, n_teams, n_rounds)

# Fill goals
for row in eachrow(ds.matches)
    Y[row.home_team_id, row.round] = row.home_score
    # Note: This ignores the *opponent* dependency, reducing it to an AR1 intensity model
end

# 2. Define Model (PLDS)
# Latent State Dimension = 2 (Att/Def per team? Or latent factors?)
state_dim = 2 
obs_dim = n_teams

# Define PLDS Parameters (Initial guess)
# A (Dynamics), C (Observation), Q (Noise), x0, P0
model = PLDS(state_dim, obs_dim) 

# 3. Fit
# This attempts the Paninski optimization
# Warning: Without custom C_t matrices for matchups, this models individual team output,
# not the interaction (Att vs Def).
fit!(model, Y)
