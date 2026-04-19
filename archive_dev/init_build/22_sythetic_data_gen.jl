# dev/22 - synthetic data

"""
Explore and develop some function to create synthetic data, 
So that we can control the teams strength parameters through the season to ensure 
that dynamic model process work, such as the Gaussian Random Walk, and the AR1.. 

Looking at the matches only, hence we should have a data_store stuct, which the odds and incidents 
are empty dataframes:

struct DataStore
    matches::DataFrame => synthetic data
    odds::DataFrame => empty dataframe 
    incidents::DataFrame => empty dataframe
end

Futhermore, the columns and types for the matches are - with left sorted items needed and 
right not for the synthetic data:

      :tournament_id => Int64,
      :season_id     => Int64,
      :season        => String7,
      :match_id      => Int64,
      :home_team     => String31, -> use a 2 character string "aa", "ab" ... "az"  "ba" 
      :away_team     => String31, -> same as home_team
      :home_score    => Float64,
      :away_score    => Float64,
      :round => Float64,
                  :match_date    => Date,
                  :tournament_slug => String15,
                  :home_score_ht => Float64,
                  :away_score_ht => Float64,
                  :winner_code => Float64,
                  :has_xg => Bool,
                  :has_stats => Bool,
                  :match_hour => Int64,
                  :match_dayofweek => Int64,
                  :match_month => Int64

"""
###### dev area 
using DataFrames
using Random
using Statistics
using Distributions
using Dates
using Printf
using Plots # Required for the dashboard

using BayesianFootball

# --- STRUCTS ---

struct DataStore
    matches::DataFrame
    odds::DataFrame    
    incidents::DataFrame 
end

struct TrueParameters
    # Matrices are (n_teams x n_total_rounds)
    α::Matrix{Float64} 
    β::Matrix{Float64}
    # Vector is (n_total_rounds)
    home_adv::Vector{Float64}
    # Helper to map "aa" -> 1
    team_names::Vector{String}
end

# --- HELPERS ---

function get_team_code(i::Int)
    idx = i - 1
    first_char = 'a' + div(idx, 26)
    second_char = 'a' + mod(idx, 26)
    return string(first_char, second_char)
end

# Standard Round Robin Generator
function generate_round_robin_schedule(n_teams::Int)
    n_teams_sched = isodd(n_teams) ? n_teams + 1 : n_teams
    half = div(n_teams_sched, 2)
    teams = collect(1:n_teams_sched)
    schedule = Tuple{Int, Int, Int}[] 

    rounds_per_leg = n_teams_sched - 1

    for r in 1:rounds_per_leg
        for i in 1:half
            t1 = teams[i]
            t2 = teams[n_teams_sched - i + 1]
            if t1 > n_teams || t2 > n_teams; continue; end
            
            if isodd(r)
                push!(schedule, (r, t1, t2))
            else
                push!(schedule, (r, t2, t1))
            end
        end
        last_team = pop!(teams)
        insert!(teams, 2, last_team)
    end
    return schedule
end

# --- MAIN GENERATOR ---

function generate_synthetic_data_with_params(;
    n_teams::Int=10,
    n_seasons::Int=2,
    legs_per_season::Int=2, 
    season_to_season_volatility::Float64=0.05,
    in_season_volatility::Float64=0.015,
    home_adv_volatility::Float64=0.01,
    base_home_adv::Float64=0.25, 
    start_year::Int=2020,
    seed::Int=42
)
    Random.seed!(seed)

    # 1. Setup DataFrames
    df_matches = DataFrame(
        :tournament_id => Int64[], :season_id => Int64[], :season => String[], :match_id => Int64[],
        :home_team => String[], :away_team => String[], 
        :home_score => Float64[], :away_score => Float64[], :round => Float64[],
        :match_date => Date[], :tournament_slug => String[],
        :home_score_ht => Float64[], :away_score_ht => Float64[],
        :winner_code => Float64[], :has_xg => Bool[], :has_stats => Bool[],
        :match_hour => Int64[], :match_dayofweek => Int64[], :match_month => Int64[]
    )
    
    # 2. Setup History Containers
    # We don't know exact total rounds easily without pre-calc, so we use vectors of vectors and reduce later
    history_α = Vector{Vector{Float64}}()
    history_β = Vector{Vector{Float64}}()
    history_ha = Float64[]

    # 3. Initialize Parameters
    true_α = rand(Normal(0, 0.2), n_teams)
    true_β = rand(Normal(0, 0.2), n_teams)
    true_α .-= mean(true_α)
    true_β .-= mean(true_β)

    current_match_id = 1
    current_date = Date(start_year, 8, 1)
    
    # Global round counter for plotting continuity
    global_round_counter = 0

    for s_idx in 1:n_seasons
        season_year_str = "$(start_year + s_idx - 1)/$(mod(start_year + s_idx, 100))"
        
        # Season Drift
        if s_idx > 1
            true_α .+= rand(Normal(0, season_to_season_volatility), n_teams)
            true_β .+= rand(Normal(0, season_to_season_volatility), n_teams)
            true_α .-= mean(true_α)
            true_β .-= mean(true_β)
        end

        current_home_adv = base_home_adv

        # Schedule Generation
        base_schedule = generate_round_robin_schedule(n_teams)
        rounds_in_single_leg = maximum(x[1] for x in base_schedule)
        
        full_schedule = Tuple{Int, Int, Int}[]
        for leg in 1:legs_per_season
            round_offset = (leg - 1) * rounds_in_single_leg
            for (r, t1, t2) in base_schedule
                actual_round = r + round_offset
                if isodd(leg)
                    push!(full_schedule, (actual_round, t1, t2))
                else
                    push!(full_schedule, (actual_round, t2, t1))
                end
            end
        end
        sort!(full_schedule, by = x -> x[1])

        # Round Loop
        unique_rounds = unique(map(x -> x[1], full_schedule))
        
        for r in unique_rounds
            global_round_counter += 1
            
            # Evolve Parameters (Random Walk)
            true_α .+= rand(Normal(0, in_season_volatility), n_teams)
            true_β .+= rand(Normal(0, in_season_volatility), n_teams)
            current_home_adv += rand(Normal(0, home_adv_volatility))
            
            # Re-center
            true_α .-= mean(true_α)
            true_β .-= mean(true_β)

            # --- RECORD HISTORY ---
            push!(history_α, copy(true_α))
            push!(history_β, copy(true_β))
            push!(history_ha, current_home_adv)

            # Play Matches
            round_matches = filter(x -> x[1] == r, full_schedule)
            current_date += Day(7)

            for (rnd, h_idx, a_idx) in round_matches
                λ_home = exp(true_α[h_idx] + true_β[a_idx] + current_home_adv)
                λ_away = exp(true_α[a_idx] + true_β[h_idx])

                h_score = rand(Poisson(λ_home))
                a_score = rand(Poisson(λ_away))
                winner = h_score > a_score ? 1.0 : (a_score > h_score ? 2.0 : 0.0)

                push!(df_matches, (
                    100, start_year + s_idx - 1, season_year_str, current_match_id,
                    get_team_code(h_idx), get_team_code(a_idx),
                    Float64(h_score), Float64(a_score), Float64(global_round_counter), # Note: using global round for continuity
                    current_date, "synthetic-lge",
                    0.0, 0.0, winner, false, false, 15, dayofweek(current_date), month(current_date)
                ))
                current_match_id += 1
            end
        end
    end

    # Convert history vectors of vectors to Matrices (Teams x Rounds)
    # reduce(hcat, ...) makes it (Teams x Rounds)
    matrix_α = reduce(hcat, history_α)
    matrix_β = reduce(hcat, history_β)
    
    team_list = [get_team_code(i) for i in 1:n_teams]

    ds = DataStore(df_matches, DataFrame(), DataFrame())
    tp = TrueParameters(matrix_α, matrix_β, history_ha, team_list)

    return ds, tp
end

# --- DASHBOARD FUNCTIONS ---

"""
    get_team_goal_history(team_name, data)
Helper to extract goals scored and conceded per global round.
"""
function get_team_goal_history(team_name::String, data::DataStore, total_rounds::Int)
    df = data.matches
    
    # Initialize zero arrays
    scored = zeros(Float64, total_rounds)
    conceded = zeros(Float64, total_rounds)
    
    # Fill based on match data
    # (Filter is slightly inefficient but fine for dashboards)
    team_matches = filter(row -> row.home_team == team_name || row.away_team == team_name, df)
    
    for r in eachrow(team_matches)
        rnd_idx = Int(r.round)
        if r.home_team == team_name
            scored[rnd_idx] = r.home_score
            conceded[rnd_idx] = r.away_score
        else
            scored[rnd_idx] = r.away_score
            conceded[rnd_idx] = r.home_score
        end
    end
    
    return scored, conceded
end

"""
    plot_team_dashboard(team_name_or_id, data_store, true_params)
Plots the Truth vs Outcomes for a specific team.
"""
function plot_team_dashboard(team_input, data::DataStore, params::TrueParameters)
    
    # Handle input (allow "aa" or 1)
    if typeof(team_input) == Int
        team_idx = team_input
        team_name = params.team_names[team_idx]
    else
        team_name = team_input
        team_idx = findfirst(==(team_input), params.team_names)
    end

    n_rounds = length(params.home_adv)
    
    # 1. Get Match Data
    goals_scored, goals_conceded = get_team_goal_history(team_name, data, n_rounds)

    # 2. Setup Plot
    p = plot(layout=(2, 2), size=(1000, 600), link=:x, margin=5Plots.mm)
    
    # --- Col 1: Attack ---
    # Top Left: Parameter Evolution
    plot!(p[1, 1], 1:n_rounds, params.α[team_idx, :], 
        label="True log(Att)", lw=2, color=:blue,
        title="Attack Strength ($team_name)", ylabel="Strength", legend=:topleft)
    
    # Bottom Left: Goals Scored
    bar!(p[2, 1], 1:n_rounds, goals_scored,
        label="Goals Scored", color=:blue, alpha=0.6, linecolor=nothing,
        ylabel="Goals", xlabel="Global Round", ylims=(0, 6))

    # --- Col 2: Defense ---
    # Top Right: Parameter Evolution
    plot!(p[1, 2], 1:n_rounds, params.β[team_idx, :], 
        label="True log(Def)", lw=2, color=:red,
        title="Defensive Weakness ($team_name)", ylabel="Strength")
        # Note: Higher Beta usually means conceding more in standard Att-Def models

    # Bottom Right: Goals Conceded
    bar!(p[2, 2], 1:n_rounds, goals_conceded,
        label="Goals Conceded", color=:red, alpha=0.6, linecolor=nothing,
        ylabel="Goals", xlabel="Global Round", ylims=(0, 6))

    return p
end

"""
    plot_multiple_dashboards(team_list, data_store, true_params)
Plots a tall stack of dashboards for multiple teams.
"""
function plot_multiple_dashboards(team_inputs, data::DataStore, params::TrueParameters)
    plots = []
    for t in team_inputs
        push!(plots, plot_team_dashboard(t, data, params))
    end
    
    n = length(team_inputs)
    plot(plots..., layout=(n, 1), size=(1000, 600*n))
end


### test area 
# Create a Scottish Championship style dataset (10 teams, 4 legs)
ds = generate_synthetic_match_data(
    n_teams = 10, 
    legs_per_season = 4 # Teams play each other 4 times
)

# Check the column names are clean
println(names(ds.matches)) 
# Output should be exact: ["tournament_id", "season_id", ...] 
# (No "_Int64" suffixes)

# Verify the number of games per team
using DataFrames
counts = combine(groupby(ds.matches, :home_team), nrow)
println(counts)
# Each team should have (10-1)*4 / 2 = 18 home games per season



# 1. Generate Data (Scottish style: 4 games per season)
ds, true_params = generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)

# 2. Plot Dashboard for Team 1 ("aa")
# You can pass 1 (Index) or "aa" (String)
p1 = plot_team_dashboard(1, ds, true_params)
display(p1)

# 3. Plot Dashboard for multiple teams ("aa" and "ab")
p2 = plot_multiple_dashboards(["aa", "ab", "ad"], ds, true_params)
display(p2)

#### added a synthetic-module to BayesianFootball 

ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)


p1 = BayesianFootball.SyntheticData.plot_team_dashboard(1, ds, true_params)



p2 = BayesianFootball.SyntheticData.plot_multiple_dashboards(["aa", "ab", "ad"], ds, true_params)

### 

using BayesianFootball
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 



model = BayesianFootball.Models.PreGame.StaticPoisson() 

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["2020/21"], :round, :sequential) #

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)


data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=3) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=2, n_warmup=500) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)


results = BayesianFootball.Training.train(model, training_config, feature_sets)



using Turing

r = results[1][1]
# Convert that directly to an array
alpha_matrix = Array(group(r, :log_α))
beta_matrix = Array(group(r, :log_β))

mean( alpha_matrix, dims=1)



########
using Plots
using Statistics

"""
    plot_parameter_comparison(alpha_matrix, beta_matrix, true_params; team_names=nothing)

Creates a side-by-side comparison of all teams. 
Checks if the Static Model's estimate overlaps with the Mean of the True Dynamic Parameter.
"""
function plot_parameter_comparison(
    alpha_post::Matrix, 
    beta_post::Matrix, 
    true_params; 
    title_tag=""
)
    n_teams = size(alpha_post, 2)
    teams = 1:n_teams
    team_labels = isnothing(true_params.team_names) ? string.(teams) : true_params.team_names

    # --- 1. Process Model Estimates ---
    # Calculate Mean and 95% Error Bars (approx +/- 2 std)
    α_est_mean = vec(mean(alpha_post, dims=1))
    α_est_std  = vec(std(alpha_post, dims=1))
    
    β_est_mean = vec(mean(beta_post, dims=1))
    β_est_std  = vec(std(beta_post, dims=1))

    # --- 2. Process True Parameters ---
    # Since the true parameters moved over time, we take the MEAN of the truth
    # to compare against the static estimate.
    α_true_mean = vec(mean(true_params.α, dims=2))
    β_true_mean = vec(mean(true_params.β, dims=2))

    # --- 3. Plotting ---
    # Layout: 2 Columns (Attack Left, Defense Right)
    p = plot(layout=(1, 2), size=(1000, 500), link=:y, margin=5Plots.mm)

    # -- Attack Plot --
    # Plot Model Estimates (Blue Dots with Error Bars)
    scatter!(p[1], teams, α_est_mean, yerr=1.96 .* α_est_std, 
        label="Model Estimate (95% CI)", color=:blue, ms=4, msw=0)
    
    # Plot True Means (Red Crosses)
    scatter!(p[1], teams, α_true_mean, 
        label="True Average", shape=:xcross, color=:red, ms=6, msw=2,
        xticks=(teams, team_labels), xrotation=45,
        title="Attack Strengths (Log α) $title_tag", ylabel="Parameter Value")

    # -- Defense Plot --
    scatter!(p[2], teams, β_est_mean, yerr=1.96 .* β_est_std, 
        label="Model Estimate (95% CI)", color=:blue, ms=4, msw=0)
    
    scatter!(p[2], teams, β_true_mean, 
        label="True Average", shape=:xcross, color=:red, ms=6, msw=2,
        xticks=(teams, team_labels), xrotation=45,
        title="Defense Weaknesses (Log β) $title_tag")

    return p
end


"""
    plot_static_fit_over_time(team_idx, alpha_post, beta_post, true_params)

Visualizes how the Static fit (straight line) attempts to cover the Dynamic Truth (wavy line).
"""
function plot_static_fit_over_time(team_idx::Int, alpha_post, beta_post, true_params)
    
    team_name = true_params.team_names[team_idx]
    n_rounds = size(true_params.α, 2)
    x_axis = 1:n_rounds

    # --- Model Stats ---
    # The model is static, so the mean/std is constant over time
    α_mu = mean(alpha_post[:, team_idx])
    α_sig = std(alpha_post[:, team_idx])
    
    β_mu = mean(beta_post[:, team_idx])
    β_sig = std(beta_post[:, team_idx])

    # --- Plotting ---
    p = plot(layout=(2, 1), size=(800, 600), link=:x)

    # Subplot 1: Attack
    # 1. Plot the "Ribbon" (The Static Estimate extended over time)
    plot!(p[1], x_axis, fill(α_mu, n_rounds), 
        ribbon=fill(1.96*α_sig, n_rounds), 
        fillalpha=0.2, color=:blue, lw=2, label="Static Model (95% CI)",
        title="Attack Evolution: $team_name", ylabel="Log α")
    
    # 2. Plot the "Truth" (The actual walking parameter)
    plot!(p[1], x_axis, true_params.α[team_idx, :], 
        color=:red, lw=2, label="True Dynamic Path")

    # Subplot 2: Defense
    plot!(p[2], x_axis, fill(β_mu, n_rounds), 
        ribbon=fill(1.96*β_sig, n_rounds), 
        fillalpha=0.2, color=:green, lw=2, label="Static Model (95% CI)",
        title="Defense Evolution: $team_name", ylabel="Log β")
    
    plot!(p[2], x_axis, true_params.β[team_idx, :], 
        color=:red, lw=2, label="True Dynamic Path")

    return p
end


# 1. Overview: Compare all teams at once
# This checks if your model successfully found the "Average" strength
p_overview = plot_parameter_comparison(alpha_matrix, beta_matrix, true_params)
display(p_overview)

# 2. Deep Dive: Look at specific teams
# This shows the "Straight Line" (Model) vs "Wavy Line" (Truth)
# Team 1 ("aa")
p_aa = plot_static_fit_over_time(1, alpha_matrix, beta_matrix, true_params)
display(p_aa)

# Team 2 ("ab")
p_ab = plot_static_fit_over_time(2, alpha_matrix, beta_matrix, true_params)
display(p_ab)


p_overview = BayesianFootball.SyntheticData.plot_parameter_comparison(alpha_matrix, beta_matrix, true_params)
display(p_overview)

# 2. Deep Dive: Look at specific teams
# This shows the "Straight Line" (Model) vs "Wavy Line" (Truth)
# Team 1 ("aa")
p_aa = BayesianFootball.SyntheticData.plot_static_fit_over_time(1, alpha_matrix, beta_matrix, true_params)
display(p_aa)

# Team 2 ("ab")
p_ab = BayesianFootball.SyntheticData.plot_static_fit_over_time(3, alpha_matrix, beta_matrix, true_params)
display(p_ab)



##########
# Gaussian random walk model 


using BayesianFootball
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 



model_grw = BayesianFootball.Models.PreGame.GRWPoisson() 

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["2020/21"], :round, :sequential) #

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)


data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model_grw, splitter_config)

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=3) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=2, n_warmup=500) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)


results_grw = BayesianFootball.Training.train(model_grw, training_config, feature_sets)



rg = results_grw[1][1]
rg
names(rg)
describe(rg[:tree_depth])

function reconstruct_vectorized(chain, n_teams, target_param_step=:z_att_steps, target_param_init=:z_att_init, target_sigma=:σ_att)
    
    # --- 1. PREPARE DIMENSIONS ---
    # Extract raw flattened data
    # Shape: (Samples, N_teams * N_steps) -> e.g., (1000, 350)
    Z_flat = Array(group(chain, target_param_step))
    
    n_total_samples = size(Z_flat, 1) # 1000
    n_steps = div(size(Z_flat, 2), n_teams) # 35
    
    # --- 2. CONSTRUCT TENSORS (RESHAPING) ---
    
    # A. The Steps Tensor (Δ)
    # Reshape logic: (Sample, Team, Time) -> Permute to (Team, Time, Sample)
    Z_steps = permutedims(reshape(Z_flat, n_total_samples, n_teams, n_steps), (2, 3, 1))
    
    # B. The Init Tensor (X0)
    # Shape: (Samples, N_teams) -> Permute to (Team, Sample) -> Reshape to (Team, 1, Sample)
    Z_init_raw = Array(group(chain, target_param_init))
    Z_init = reshape(permutedims(Z_init_raw, (2, 1)), n_teams, 1, n_total_samples)
    
    # C. The Sigma Vector (σ)
    # We must flatten the chains (500x2) into a single vector (1000)
    # Then reshape to (1, 1, 1000) to allow broadcasting across Teams and Time
    σ_vec = vec(Array(chain[target_sigma])) 
    σ_tensor = reshape(σ_vec, 1, 1, n_total_samples)

    # --- 3. ALGEBRAIC OPERATIONS ---

    # A. Scale the steps
    # (N, T, S) .* (1, 1, S) -> Broadcasting magic
    scaled_steps = Z_steps .* σ_tensor
    
    # B. Scale the init (Fixed 0.5 scaler from your model)
    scaled_init = Z_init .* 0.5
    
    # C. Concatenate: Join Init and Steps along Time axis (dim 2)
    # Result shape: (N, T+1, S)
    raw_walk = cat(scaled_init, scaled_steps, dims=2)
    
    # D. Integrate: Cumulative Sum along Time axis
    traj_raw = cumsum(raw_walk, dims=2)
    
    # E. Project: Center the teams (Mean along dim 1)
    # We subtract the mean of the teams from every team
    traj_centered = traj_raw .- mean(traj_raw, dims=1)
    
    return traj_centered
end

# Usage
att_tensor = reconstruct_vectorized(rg, 10, :z_att_steps, :z_att_init, :σ_att)
def_tensor = reconstruct_vectorized(rg, 10, :z_def_steps, :z_def_init, :σ_def)

att_tensor = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(rg, 10, :z_att_steps, :z_att_init, :σ_att)
def_tensor = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(rg, 10, :z_def_steps, :z_def_init, :σ_def)



println("Final Tensor Shape: ", size(att_tensor)) 
# Should be (10, 36, 1000) -> (Team, Time, Sample)
#
using Plots, Statistics

# Plot Attack strength over time for Team 1
team_id = 2
# Get median over samples (dim 3)
medians = vec(median(att_tensor[team_id, :, :], dims=2))
# Get 90% credible interval
lowers = vec(quantile.(eachrow(att_tensor[team_id, :, :]), 0.05))
uppers = vec(quantile.(eachrow(att_tensor[team_id, :, :]), 0.95))

plot(medians, ribbon=(medians .- lowers, uppers .- medians), 
     label="Team 1 Attack", xlabel="Round", ylabel="Strength")





# 
using Plots
using Statistics

# --- Helper: Extract Quantiles from Tensor (Team x Time x Sample) ---
function get_posterior_stats(tensor::Array{Float64, 3})
    # tensor dimensions: (n_teams, n_rounds, n_samples)
    
    # Calculate median (50%) and 90% CI (5% and 95%) along the sample dimension (dim 3)
    medians = dropdims(median(tensor, dims=3), dims=3)
    lower = dropdims(mapslices(x -> quantile(x, 0.05), tensor, dims=3), dims=3)
    upper = dropdims(mapslices(x -> quantile(x, 0.95), tensor, dims=3), dims=3)
    
    return medians, lower, upper
end

"""
    plot_dynamic_trajectory(team_id, true_params, dyn_att, dyn_def, stat_att=nothing, stat_def=nothing)

Plots the full time-series comparison:
1. TRUTH (Red Line)
2. DYNAMIC MODEL (Blue Ribbon)
3. STATIC MODEL (Green Dashed Ribbon - Optional)
"""
function plot_dynamic_trajectory(
    team_id::Int, 
    true_params, 
    dyn_att::Array{Float64, 3}, # (Team, Time, Sample)
    dyn_def::Array{Float64, 3};
    stat_att::Union{Matrix, Nothing}=nothing, # (Sample, Team)
    stat_def::Union{Matrix, Nothing}=nothing
)
    team_name = true_params.team_names[team_id]
    n_rounds = size(true_params.α, 2)
    x_axis = 1:n_rounds

    # --- 1. Process Dynamic Data ---
    # We slice [team_id, :, :] -> becomes (Time, Sample) -> Helper expects 3D, so let's just do it manually for the single team
    
    # Attack
    att_slice = dyn_att[team_id, :, :] 
    att_med = vec(median(att_slice, dims=2))
    att_lo  = vec(mapslices(x -> quantile(x, 0.05), att_slice, dims=2))
    att_hi  = vec(mapslices(x -> quantile(x, 0.95), att_slice, dims=2))

    # Defense
    def_slice = dyn_def[team_id, :, :]
    def_med = vec(median(def_slice, dims=2))
    def_lo  = vec(mapslices(x -> quantile(x, 0.05), def_slice, dims=2))
    def_hi  = vec(mapslices(x -> quantile(x, 0.95), def_slice, dims=2))

    # --- 2. Setup Plot ---
    p = plot(layout=(2,1), size=(900, 700), link=:x, margin=5Plots.mm)

    # --- Subplot 1: ATTACK ---
    # A. Dynamic
    plot!(p[1], x_axis, att_med, ribbon=(att_med .- att_lo, att_hi .- att_med),
        color=:dodgerblue, fillalpha=0.3, lw=2, label="GRW Model (Dynamic)")
    
    # B. Static (Optional)
    if !isnothing(stat_att)
        s_med = mean(stat_att[:, team_id]) # Scalar
        s_std = std(stat_att[:, team_id])
        plot!(p[1], x_axis, fill(s_med, n_rounds), 
              ribbon=fill(1.96*s_std, n_rounds),
              color=:green, fillalpha=0.15, linestyle=:dash, label="Static Avg")
    end

    # C. Truth (Last so it's on top)
    plot!(p[1], x_axis, true_params.α[team_id, :], 
        color=:red, lw=2, label="TRUE Path",
        title="Attack Evolution: $team_name", ylabel="Log Strength")

    # --- Subplot 2: DEFENSE ---
    # A. Dynamic
    plot!(p[2], x_axis, def_med, ribbon=(def_med .- def_lo, def_hi .- def_med),
        color=:dodgerblue, fillalpha=0.3, lw=2, label="")
    
    # B. Static
    if !isnothing(stat_def)
        s_med = mean(stat_def[:, team_id]) 
        s_std = std(stat_def[:, team_id])
        plot!(p[2], x_axis, fill(s_med, n_rounds), 
              ribbon=fill(1.96*s_std, n_rounds),
              color=:green, fillalpha=0.15, linestyle=:dash, label="")
    end

    # C. Truth
    plot!(p[2], x_axis, true_params.β[team_id, :], 
        color=:red, lw=2, label="",
        title="Defense Evolution: $team_name", ylabel="Log Strength")

    return p
end

"""
    plot_terminal_error(true_params, dyn_att, dyn_def, stat_att, stat_def)

Calculates the absolute error at the FINAL time step (T) for both models.
Shows which model is closer to reality at the end of the season.
"""
function plot_terminal_error(
    true_params, 
    dyn_att::Array{Float64, 3}, 
    dyn_def::Array{Float64, 3},
    stat_att::Matrix, # Static is required for comparison here
    stat_def::Matrix
)
    n_teams = length(true_params.team_names)
    teams = 1:n_teams
    labels = true_params.team_names

    # Get Index of Final Round
    T = size(true_params.α, 2)

    # --- Calculate Errors ---
    # 1. Dynamic Error: | Median_Est[T] - Truth[T] |
    dyn_att_end = vec(median(dyn_att[:, T, :], dims=2))
    dyn_err_att = abs.(dyn_att_end .- true_params.α[:, T])

    # 2. Static Error: | Mean_Est - Truth[T] |
    # (Checking how far the season average is from the final reality)
    stat_att_mean = vec(mean(stat_att, dims=1))
    stat_err_att  = abs.(stat_att_mean .- true_params.α[:, T])

    # --- Plotting ---
    # Grouped Bar Chart
    p = groupedbar(
        labels, 
        [dyn_err_att stat_err_att],
        label=["GRW (Dynamic)" "Static"],
        color=[:dodgerblue :green],
        title="Error at Final Time Step (Attack)",
        ylabel="Absolute Error |Est - True|",
        size=(900, 400),
        alpha=0.8
    )

    return p
end


# 1. Visualize the Trajectory for Team 2 ("ab")
# This is the money shot: You should see the Blue ribbon following the Red line, 
# while the Green line just cuts through the middle.
p1 = plot_dynamic_trajectory(
    5, 
    true_params, 
    att_tensor, 
    def_tensor; 
    stat_att=alpha_matrix, 
    stat_def=beta_matrix
)

display(p1)



p1 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    6, 
    true_params, 
    att_tensor, 
    def_tensor; 
    stat_att=alpha_matrix, 
    stat_def=beta_matrix
)


# 2. Compare Errors at the end of the season
# This proves that the Dynamic model is better at predicting the "Now" 
# than the Static model (which only knows the "Average of the Past").
using StatsPlots
p2 = plot_terminal_error(true_params, att_tensor, def_tensor, alpha_matrix, beta_matrix)


p2 = BayesianFootball.SyntheticData.plot_terminal_error(true_params, att_tensor, def_tensor, alpha_matrix, beta_matrix)


display(p2)


density(rg[:σ_att])
density(rg[:σ_att])


