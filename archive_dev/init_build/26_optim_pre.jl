using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using Optim             # For the Fast Solver (Paninski)
using Surrogates        # For Hyperparameter Tuning
using AbstractGPs       # Surrogate Model Backend
using Turing            # For MCMC
using DynamicPPL
using ThreadPinning



ds = BayesianFootball.load_scottish_data("24/25", split_week=0)



model = BayesianFootball.Models.PreGame.GRWPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :split_col 
)



data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

# ==============================================================================
# SECTION 0: PREP REAL DATA
# ==============================================================================

# 1. Extract Vectors from your Feature Set
#    (Using the output you pasted)
fs_data = feature_sets[1][1].data

home_ids = fs_data[:flat_home_ids]
away_ids = fs_data[:flat_away_ids]
score_h  = Float64.(fs_data[:flat_home_goals])
score_a  = Float64.(fs_data[:flat_away_goals])

# 2. Extract Metadata
#    We need the 'round' for every match. 
#    Your 'feature_sets' usually groups by round, but 'matches_df' has the raw list.
#    Let's align it carefully using the matches dataframe inside the feature set.
matches_df = fs_data[:matches_df]
rounds     = Int.(matches_df.round)

n_teams  = fs_data[:n_teams]  # 97
n_rounds = fs_data[:n_rounds] # 39
team_map = fs_data[:team_map] # Dict("celtic"=>1, ...)

# Reverse Map: ID -> Name (For plotting later)
id_to_name = Dict(v => k for (k, v) in team_map)

println("✅ Real Data Ready: $n_teams Teams, $n_rounds Rounds, $(length(score_h)) Matches.")


# ==============================================================================
# SECTION 1: PARALLEL TUNE HYPERPARAMETERS (Ask-Tell)
# ==============================================================================
#
function log_posterior_optimized(
    flat_params::AbstractVector{T}, 
    h_idx::Vector{Int}, a_idx::Vector{Int}, 
    s_h::Vector{Float64}, s_a::Vector{Float64}, 
    rnds::Vector{Int}, 
    n_t::Int, n_r::Int, 
    σ_att::Float64, σ_def::Float64, rho::Float64
) where T <: Real

    # --- 1. Unpack Static Parameters ---
    # Access directly from end of vector
    mu       = flat_params[end]
    home_adv = flat_params[end-1]
    
    # --- 2. Zero-Cost Reshape ---
    # Instead of creating views, we reshape the underlying data pointer.
    # Note: We only look at the dynamic part (length - 2)
    L_dyn = length(flat_params) - 2
    
    # We create a reshaped wrapper. This does NOT copy data.
    # Layout is (Teams, Rounds, 2) -> (Col, Row, Page)
    # This means iterating Teams (dim 1) is fastest (contiguous memory).
    matrix_form = reshape(view(flat_params, 1:L_dyn), n_t, n_r, 2)

    log_joint = 0.0

    # --- 3. Priors: Optimized Loop Order ---
    
    # A. Static Priors
    # HA ~ Normal(0.25, 0.5) -> transform division to multiplication
    log_joint -= 0.5 * abs2((home_adv - 0.25) * 2.0) 
    
    # Mu ~ Normal(0.0, 1.0)
    log_joint -= 0.5 * abs2(mu)

    # B. Dynamic Priors
    prec_att = 1.0 / abs2(σ_att)
    prec_def = 1.0 / abs2(σ_def)

    # B1. Initial State (Round 1)
    # Using @simd allows CPU to process multiple teams at once
    @inbounds @simd for t in 1:n_t
        log_joint -= 0.5 * (abs2(matrix_form[t, 1, 1]) + abs2(matrix_form[t, 1, 2]))
    end

    # B2. Evolution (Round 2..N)
    # CRITICAL FIX: We swap loops. 
    # Outer: Rounds, Inner: Teams.
    # This ensures we access memory [t, r] then [t+1, r] which are neighbors.
    @inbounds for r in 2:n_r
        @simd for t in 1:n_t
            # Attack (Type 1)
            curr_att = matrix_form[t, r, 1]
            prev_att = matrix_form[t, r-1, 1]
            log_joint -= 0.5 * abs2(curr_att - rho * prev_att) * prec_att
            
            # Defense (Type 2)
            curr_def = matrix_form[t, r, 2]
            prev_def = matrix_form[t, r-1, 2]
            log_joint -= 0.5 * abs2(curr_def - rho * prev_def) * prec_def
        end
    end

    # --- 4. Likelihood Loop ---
    # Poisson: k * log(λ) - λ 
    # log(λ) = strength. So: k * strength - exp(strength)
    
    @inbounds @simd for i in 1:length(s_h)
        h, a, r = h_idx[i], a_idx[i], rnds[i]
        
        # Direct lookup (Bounds check removed by @inbounds)
        att_h = matrix_form[h, r, 1]
        def_a = matrix_form[a, r, 2]
        
        att_a = matrix_form[a, r, 1]
        def_h = matrix_form[h, r, 2]

        # Home Likelihood
        str_h = mu + att_h + def_a + home_adv
        log_joint += s_h[i] * str_h - exp(str_h)
        
        # Away Likelihood
        str_a = mu + att_a + def_h
        log_joint += s_a[i] * str_a - exp(str_a)
    end

    return -log_joint # Return Negative Log Posterior for Minimizer
end




using Base.Threads
using LinearAlgebra

pinthreads(:cores)
BLAS.set_num_threads(1) 

println("\n🧠 Tuning Parameters for Scottish Football (Parallel Mode)...")
println("   Using $(Threads.nthreads()) Threads.")

# 0. Important: Prevent BLAS thread contention
#    Each individual optimization should be single-threaded so we can run
#    many optimizations in parallel.
# BLAS.set_num_threads(1)

# 1. Define Objective (Same as before)
function objective_function_real(params)
    s_att, s_def, r_rho = params
    num_params = Int(2 * n_teams * n_rounds) + 2
    x0 = zeros(num_params)
    x0[end] = 0.18   # Mu Guess
    x0[end-1] = 0.25 # HA Guess
    
    func = x -> log_posterior_with_mu(
        x, home_ids, away_ids, score_h, score_a, rounds, 
        n_teams, n_rounds, 
        s_att, s_def, r_rho
    )
    
    # We use a lower iteration count per check since we are doing many checks
    res = Optim.optimize(func, x0, LBFGS(), Optim.Options(iterations=50))
    return Optim.minimum(res)
end

# 2. Initialize Surrogate
lb = [0.01, 0.01, 0.90]
ub = [0.20, 0.20, 1.00]

# Initial samples (Random or Sobol)
initial_n = 10
x_train = Surrogates.sample(initial_n, lb, ub, SobolSample())

# Evaluate initial samples in parallel
println("   Evaluating initial $initial_n points...")
y_train = zeros(initial_n)
Threads.@threads for i in 1:initial_n
    y_train[i] = objective_function_real(x_train[i])
end

my_surrogate = Kriging(x_train, y_train, lb, ub)

# 3. The Ask-Tell Optimization Loop
n_batches = 5             # How many rounds of parallel checks
batch_size = Threads.nthreads() # Check 16 points at once (1 per thread)

println("   Starting Batch Optimization ($n_batches batches of $batch_size points)...")

for i in 1:n_batches
    # --- ASK STEP ---
    # We ask for 'batch_size' promising points.
    # We use "KrigingBeliever" strategy to halluncinate outcomes so the points spread out.
    new_x_points, _ = potential_optimal_points(
        EI(),                  # Expected Improvement (Standard for minimization)
        KrigingBeliever(),     # "Liar" strategy to allow batch sampling
        lb, ub, 
        my_surrogate, 
        RandomSample(),        # Sampler for the optimization of the acquisition function
        batch_size             # How many points to return
    )
    
    # --- EVALUATE STEP (Parallel) ---
    new_y_values = zeros(length(new_x_points))
    
    Threads.@threads for j in 1:length(new_x_points)
        # Evaluate the expensive function
        new_y_values[j] = objective_function_real(new_x_points[j])
    end
    
    # --- TELL STEP ---
    # Update the model with the real results
    update!(my_surrogate, new_x_points, new_y_values)
    
    # Logging
    current_min = minimum(my_surrogate.y)
    println("   Batch $i/$n_batches complete. Current Best: $(round(current_min, digits=4))")
end

# 4. Extract Best Result
best_val_idx = argmin(my_surrogate.y)
opt_σ_att, opt_σ_def, opt_rho = my_surrogate.x[best_val_idx]
best_val = my_surrogate.y[best_val_idx]

println("\n🏆 Parallel Tuned Results:")
println("   Sigma Att: ", round(opt_σ_att, digits=4))
println("   Sigma Def: ", round(opt_σ_def, digits=4))
println("   Rho:        ", round(opt_rho, digits=4))
println("   Best NLP:   ", round(best_val, digits=4))

# Reset BLAS threads to default if you do heavy matrix math later
BLAS.set_num_threads(Sys.CPU_THREADS)

"""
analysis 
"""

using Plots

println("\n🏃 Running Final Solver with Optimal Hyperparameters...")

# 1. Setup the final function with your WINNING parameters
#    Note: Use 'log_posterior_optimized' if you defined it, otherwise your original function.
final_func = x -> log_posterior_optimized(
    x, home_ids, away_ids, score_h, score_a, rounds, 
    n_teams, n_rounds, 
    opt_σ_att, opt_σ_def, opt_rho
)

# 2. Initial Guess (Same as before)
num_params = Int(2 * n_teams * n_rounds) + 2
x0_final = zeros(num_params)
x0_final[end] = 0.18   # Mu
x0_final[end-1] = 0.25 # HomeAdv

# 3. Run the "Fast Solver" (LBFGS)
#    We allow more iterations (1000) to ensure perfect convergence now that we know the hypers are good.
final_res = Optim.optimize(final_func, x0_final, LBFGS(), Optim.Options(iterations=100, show_trace=true))

# 4. Extract the Vector
flat_best = Optim.minimizer(final_res)
println("✅ Final Solver Complete. Minimum NLP: $(Optim.minimum(final_res))")


# 1. Separate Static vs Dynamic params
mu_final = flat_best[end]
ha_final = flat_best[end-1]

# 2. Reshape the dynamic part
#    Shape: (Teams, Rounds, 2) where 2 is [Attack, Defense]
n_dyn_params = length(flat_best) - 2
matrix_final = reshape(flat_best[1:n_dyn_params], n_teams, n_rounds, 2)

println("📊 Extracted Ratings:")
println("   League Average Goals (exp(Mu)): ", round(exp(mu_final), digits=3))
println("   Home Advantage Factor:          ", round(exp(ha_final), digits=3), "x")

function plot_team_strength(team_name_query, team_map, matrix_data, n_rounds, rounds_dates=nothing)
    # 1. Find the Team ID
    #    Case-insensitive search
    matches = filter(p -> occursin(lowercase(team_name_query), lowercase(p.first)), team_map)
    
    if isempty(matches)
        println("❌ Team not found.")
        return
    end
    
    t_name = first(keys(matches))
    t_id   = first(values(matches))
    
    # 2. Extract Data
    #    Slice [TeamID, All Rounds, Type]
    #    Type 1 = Attack, Type 2 = Defense
    att_path = matrix_data[t_id, :, 1]
    def_path = matrix_data[t_id, :, 2]
    
    xs = 1:n_rounds
    
    # 3. Create Plots
    #    Attack: Higher is better
    p1 = plot(xs, att_path, 
        title="$t_name Attack Strength", 
        label="Att (Log Scale)", 
        lw=3, color=:dodgerblue, legend=:topleft,
        ylabel="Log Strength (>0 is good)",
        fill=(0, 0.1, :dodgerblue) # Fill to 0 line
    )
    hline!(p1, [0.0], color=:black, linestyle=:dash, label="League Avg")
    
    #    Defense: LOWER is usually "conceding less", but in Poisson models:
    #    Higher Def parameter = Higher Lambda for Opponent = WORSE Defense.
    #    So, values < 0 are Good Defenses (Conceding less than average).
    p2 = plot(xs, def_path, 
        title="$t_name Defensive Vulnerability", 
        label="Def (Log Scale)", 
        lw=3, color=:firebrick, legend=:topleft,
        ylabel="Log Vulnerability (>0 is bad)",
        fill=(0, 0.1, :firebrick)
    )
    hline!(p2, [0.0], color=:black, linestyle=:dash, label="League Avg")

    # Combine
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# --- RUN PLOTS ---
# Try your big teams (Check your team_map keys to be sure of spelling)
display(plot_team_strength("Celtic", team_map, matrix_final, n_rounds))
display(plot_team_strength("Rangers", team_map, matrix_final, n_rounds))
display(plot_team_strength("Aberdeen", team_map, matrix_final, n_rounds))

display(plot_team_strength("Falkirk-fc", team_map, matrix_final, n_rounds))
display(plot_team_strength("Hamilton", team_map, matrix_final, n_rounds))
display(plot_team_strength("Livingston", team_map, matrix_final, n_rounds))

using Plots

# Create a grid of points
s_range = range(lb[1], ub[1], length=50) # Sigma Att
d_range = range(lb[2], ub[2], length=50) # Sigma Def

# Function to query the surrogate
# We fix Rho to the optimal value we found (opt_rho)
z_val(s, d) = my_surrogate([s, d, opt_rho])

# Calculate Heatmap
Z = [z_val(s, d) for s in s_range, d in d_range]

heatmap(s_range, d_range, Z', 
    title="Hyperparameter Loss Landscape (Rho fixed at $(round(opt_rho, digits=2)))",
    xlabel="Sigma Attack", ylabel="Sigma Defense",
    color=:viridis, right_margin=5Plots.mm
)

# Overlay the points the optimizer actually checked
scatter!([p[1] for p in my_surrogate.x], [p[2] for p in my_surrogate.x], 
    label="Sampled Points", color=:white, markerstrokecolor=:black
)

# Highlight the winner
scatter!([opt_σ_att], [opt_σ_def], 
    label="Best Found", color=:red, markersize=8, shape=:star5
)



"""
the mu 

"""
using DataFrames, Statistics

# 1. Aggregate Stats by Tournament
league_stats = combine(groupby(matches_df, :tournament_slug)) do sdf
    (
        n_games = nrow(sdf),
        avg_home_goals = mean(sdf.home_score),
        avg_away_goals = mean(sdf.away_score),
        home_win_pct = mean(sdf.home_score .> sdf.away_score)
    )
end

display(league_stats)


# ==============================================================================
# ROBUST SOLVER (Fixed Hypers)
# ==============================================================================

# 1. Set the "Truth" from your MCMC / Papers
fixed_σ_att = 0.055  # Example MCMC result
fixed_σ_def = 0.055
fixed_rho   = 0.98   # High memory (teams stay consistent)

println("🛡️ Running Robust Solver with Fixed Sigma = $fixed_σ_att ...")

# 2. Define the function with FIXED sigmas
#    (We only optimize the 7000+ team parameters, not the 3 hypers)
robust_func = x -> log_posterior_optimized(
    x, home_ids, away_ids, score_h, score_a, rounds, 
    n_teams, n_rounds, 
    fixed_σ_att, fixed_σ_def, fixed_rho
)

# 3. Optimize
#    Notice we don't use 'surrogate_optimize' here. We go straight to LBFGS.
num_params = Int(2 * n_teams * n_rounds) + 2
x0 = zeros(num_params)
x0[end] = 0.18   # Mu guess
x0[end-1] = 0.25 # HA guess

res = Optim.optimize(robust_func, x0, LBFGS(), Optim.Options(iterations=100, show_trace=true))
robust_solution = Optim.minimizer(res)

# 4. Extract and Plot (Same as before)
# ... (Use your extraction code here)

# 4. Extract the Vector
flat_best = Optim.minimizer(final_res)
println("✅ Final Solver Complete. Minimum NLP: $(Optim.minimum(final_res))")


# 1. Separate Static vs Dynamic params
mu_final = robust_solution[end]
ha_final = robust_solution[end-1]

# 2. Reshape the dynamic part
#    Shape: (Teams, Rounds, 2) where 2 is [Attack, Defense]
n_dyn_params = length(robust_solution) - 2
matrix_final = reshape(robust_solution[1:n_dyn_params], n_teams, n_rounds, 2)


display(plot_team_strength("Celtic", team_map, matrix_final, n_rounds))
display(plot_team_strength("Rangers", team_map, matrix_final, n_rounds))
display(plot_team_strength("Aberdeen", team_map, matrix_final, n_rounds))

display(plot_team_strength("Falkirk-fc", team_map, matrix_final, n_rounds))
display(plot_team_strength("Hamilton", team_map, matrix_final, n_rounds))
display(plot_team_strength("Livingston", team_map, matrix_final, n_rounds))




"""

"""
# ==============================================================================
# SECTION 1: TUNE HYPERPARAMETERS (Surrogates)
# ==============================================================================
println("\n🧠 Tuning Parameters for Scottish Football...")

# 1. Define Objective (Minimize Negative Log Posterior)
function objective_function_real(params)
    s_att, s_def, r_rho = params
    
    # x0 size: (2 * 97 teams * 39 rounds) + 2 static params (Mu, HA)
    num_params = Int(2 * n_teams * n_rounds) + 2
    x0 = zeros(num_params)
    x0[end] = 0.18   # Mu Guess
    x0[end-1] = 0.25 # HA Guess
    
    # Create Closure
    func = x -> log_posterior_with_mu(
        x, home_ids, away_ids, score_h, score_a, rounds, 
        n_teams, n_rounds, 
        s_att, s_def, r_rho
    )
    
    # Fast Run (50 iters)
    res = Optim.optimize(func, x0, LBFGS(), Optim.Options(iterations=50))
    return Optim.minimum(res)
end

# 2. Run BO
lb = [0.01, 0.01, 0.90]
ub = [0.20, 0.20, 1.00]

x_train = Surrogates.sample(5, lb, ub, SobolSample())
y_train = objective_function_real.(x_train)
my_surrogate = Kriging(x_train, y_train, lb, ub)

best_param_tuple = surrogate_optimize(
    objective_function_real, SRBF(), lb, ub, my_surrogate, RandomSample(); 
    maxiters = 15
)

best_hypers, best_val = best_param_tuple
opt_σ_att, opt_σ_def, opt_rho = best_hypers

println("🏆 Tuned Results:")
println("   Sigma Att: ", round(opt_σ_att, digits=4))
println("   Sigma Def: ", round(opt_σ_def, digits=4))
println("   Rho:       ", round(opt_rho, digits=4))



# ==============================================================================
# SECTION 3: PLOTTING
# ==============================================================================

function plot_real_team(name_query, team_map, matrix_final, n_rounds)
    # 1. Find ID
    # Simple partial match search
    candidates = filter(p -> occursin(lowercase(name_query), lowercase(p.first)), team_map)
    
    if isempty(candidates)
        println("❌ Team '$name_query' not found.")
        return
    end
    
    team_name = first(keys(candidates))
    team_id   = first(values(candidates))
    println("📊 Plotting for: $team_name (ID: $team_id)")

    # 2. Extract Paths
    # Note: These are centered around 'Mu' (League Average)
    att_path = matrix_final[team_id, :, 1]
    def_path = matrix_final[team_id, :, 2]
    
    x_axis = 1:n_rounds

    # 3. Plot
    p = plot(layout=(2,1), size=(800, 600), link=:x, titlefontsize=12)
    
    # Attack (Higher is Better)
    plot!(p[1], x_axis, att_path, 
        lw=3, color=:dodgerblue, label="Attack Rating",
        title="$team_name: Attack Strength (Log)", ylabel="Log Goals > Avg")
    hline!(p[1], [0.0], color=:gray, linestyle=:dash, label="League Avg")

    # Defense (Lower is Better - usually)
    # In our model: Higher Def Parameter = Concede MORE goals (Weakness)
    # So we flip the sign visually if you want "Higher = Better Defense", 
    # OR keep it raw: "Higher = Weaker". Let's keep it raw but label it "Def Weakness".
    plot!(p[2], x_axis, def_path, 
        lw=3, color=:firebrick, label="Defensive Weakness",
        title="$team_name: Defensive Weakness (Log)", ylabel="Log Goals Conceded > Avg")
    hline!(p[2], [0.0], color=:gray, linestyle=:dash, label="League Avg")
    
    display(p)
end

# --- EXAMPLE USAGE ---
# Try plotting the big teams
plot_real_team("celtic", team_map, matrix_final, n_rounds)
plot_real_team("rangers", team_map, matrix_final, n_rounds)
plot_real_team("aberdeen", team_map, matrix_final, n_rounds)


"""

notes olds stuff added 

"""


matches = ds.matches
home_ids = feature_sets[1][1].data[:flat_home_ids]
away_ids = feature_sets[1][1].data[:flat_away_ids]
score_h  = Float64.(matches.home_score)
score_a  = Float64.(matches.away_score)

rounds   = Int.(matches.round)

n_teams  = Int(feature_sets[1][1].data[:n_teams])
n_rounds = maximum(rounds)

println("✅ Data Loaded: $(length(score_h)) matches, $n_teams teams, $n_rounds rounds.")



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
# SECTION 2 (UPDATED): HYPERPARAMETER TUNING "WITH HA"
# ==============================================================================
println("\n🧠 Starting Hyperparameter Tuning (with Learnable HA)...")

# 1. Define the Objective Function
#    This runs the Inner Optimizer (Paninski) which NOW solves for [Att, Def, HA].
#    The BO only tunes [Sigma_Att, Sigma_Def, Rho].
function objective_function_mu(params)
    s_att, s_def, r_rho = params
    
    # Setup x0 with space for Home Advantage at the end
    num_params = Int(2 * n_teams * n_rounds) + 1
    x0 = zeros(num_params)
    x0[end] = 0.2 # Initial guess for HA
    
    # Closure with the specific Sigmas/Rho we are testing
    func = x -> log_posterior_full_with_mu(
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
y_train = objective_function_mu.(x_train) 

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



# 1. Setup x0 (Dynamic + 2 Static)
x0 = zeros(Int(2 * n_teams * n_rounds) + 2)

# Set initial guesses
# HomeAdv guess
x0[end-1] = 0.25 
# Mu guess (Log average goals). log(1.2) approx 0.18
x0[end]   = 0.18 

# 2. Define Closure
# Assuming you have your tuned hyperparameters (opt_σ_att, etc.)
final_func_mu = x -> log_posterior_with_mu(
    x, home_ids, away_ids, score_h, score_a, rounds, 
    n_teams, n_rounds, 
    opt_σ_att, opt_σ_def, opt_rho
)

# 3. Optimize
println("🚀 Optimizing Model with Mu parameter...")
@time res_mu = Optim.optimize(
    final_func_mu, x0, LBFGS(), 
    Optim.Options(iterations=1000, g_tol=1e-6)
)

# 4. Extract
best_vec = Optim.minimizer(res_mu)
learned_mu = best_vec[end]
learned_ha = best_vec[end-1]

println("✅ Done.")
println("   League Base Rate (Mu): $learned_mu  (Avg Goals: $(exp(learned_mu)))")
println("   Home Advantage:        $learned_ha")

