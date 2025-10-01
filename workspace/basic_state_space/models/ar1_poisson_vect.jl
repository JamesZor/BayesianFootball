# models/ar1_poisson_ha_vectorized.jl
module AR1PoissonHAVectorized

using Turing
using LinearAlgebra
using Statistics
using SparseArrays
using BayesianFootball
using DataFrames
using MCMCChains # Needed for prediction logic

export AR1PoissonHAVectorizedModel

#=
--------------------------------------------------------------------------------
1.  MODEL DEFINITION
--------------------------------------------------------------------------------
=#

struct AR1PoissonHAVectorizedModel <: AbstractStateSpaceModel end

function BayesianFootball.get_required_features(::AR1PoissonHAVectorizedModel)
    # Requirements are the same as the old model
    return (
        :home_team_ids, :away_team_ids, :n_teams,
        :global_round, :league_ids, :n_leagues
    )
end

#=
--------------------------------------------------------------------------------
2.  TURING @MODEL IMPLEMENTATION (The Core Logic)
--------------------------------------------------------------------------------
=#

@model function ar1_poisson_ha_vectorized_turing(
    # --- Data ---
    home_goals, away_goals, # Flat vectors of all goals
    n_teams::Int, n_leagues::Int, n_rounds::Int,
    # Pre-computed selection matrices
    S_home_α, S_away_β, S_home_adv, # For home goal rate
    S_away_α, S_home_β,             # For away goal rate
    # Type parameter for stability
    ::Type{T}=Float64
) where {T}

    # --- Priors for AR(1) Process ---
    ρ_attack ~ Beta(10, 1.5); ρ_defense ~ Beta(10, 1.5); ρ_home ~ Beta(10, 1.5)

    μ_log_σ_attack ~ Normal(-2.5, 0.5); τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    z_log_σ_attack ~ MvNormal(zeros(T, n_teams), I)
    σ_attack = exp.(μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack)

    μ_log_σ_defense ~ Normal(-2.5, 0.5); τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)
    z_log_σ_defense ~ MvNormal(zeros(T, n_teams), I)
    σ_defense = exp.(μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense)
    
    μ_log_σ_home ~ Normal(-3.0, 0.5); σ_home = exp(μ_log_σ_home)

    # --- Initial Latent States ---
    initial_α_z ~ MvNormal(zeros(T, n_teams), I); log_α_raw_t0 = initial_α_z .* sqrt(T(0.5))
    initial_β_z ~ MvNormal(zeros(T, n_teams), I); log_β_raw_t0 = initial_β_z .* sqrt(T(0.5))
    initial_home_z ~ MvNormal(zeros(T, n_leagues), I); log_home_adv_raw_t0 = log(T(1.3)) .+ initial_home_z .* sqrt(T(0.1))

    # --- Innovations ---
    z_α ~ MvNormal(zeros(T, n_teams * n_rounds), I); z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β ~ MvNormal(zeros(T, n_teams * n_rounds), I); z_β_mat = reshape(z_β, n_teams, n_rounds)
    z_home ~ MvNormal(zeros(T, n_leagues * n_rounds), I); z_home_mat = reshape(z_home, n_leagues, n_rounds)

    # --- Vectorized State Evolution ---
    # Construct the lower-triangular AR(1) matrix for each rho
    L_α = LowerTriangular([t >= s ? ρ_attack^(t - s) : zero(T) for t in 1:n_rounds, s in 1:n_rounds])
    L_β = LowerTriangular([t >= s ? ρ_defense^(t - s) : zero(T) for t in 1:n_rounds, s in 1:n_rounds])
    L_h = LowerTriangular([t >= s ? ρ_home^(t - s) : zero(T) for t in 1:n_rounds, s in 1:n_rounds])
    
    # Evolve initial state + innovations for all teams/leagues and all rounds at once
    log_α_raw = (L_α * (z_α_mat .* σ_attack .+ log_α_raw_t0))' # Transpose to (n_rounds, n_teams)
    log_β_raw = (L_β * (z_β_mat .* σ_defense .+ log_β_raw_t0))' # Transpose to (n_rounds, n_teams)
    log_home_adv_raw = (L_h * (z_home_mat .* σ_home .+ log_home_adv_raw_t0))' # Transpose to (n_rounds, n_leagues)

    # --- Sum-to-zero Constraint (Vectorized) ---
    log_α_centered = log_α_raw .- mean(log_α_raw, dims=2)
    log_β_centered = log_β_raw .- mean(log_β_raw, dims=2)
    
    # Flatten the matrices to a single vector
    log_α_flat = vec(log_α_centered') # Transpose back to (n_teams, n_rounds) before flattening
    log_β_flat = vec(log_β_centered')
    log_home_adv_flat = vec(log_home_adv_raw')

    # --- Likelihood Calculation (Vectorized) ---
    log_λs = S_home_α * log_α_flat + S_away_β * log_β_flat + S_home_adv * log_home_adv_flat
    log_μs = S_away_α * log_α_flat + S_home_β * log_β_flat

    home_goals .~ LogPoisson.(log_λs)
    away_goals .~ LogPoisson.(log_μs)
end

#=
--------------------------------------------------------------------------------
3.  INTERFACE FUNCTION (The "glue")
--------------------------------------------------------------------------------
=#

"""
    build_turing_model(::AR1PoissonHAVectorizedModel, features, goals_home, goals_away)

Builds the turing model by first constructing the sparse selection matrices
and then passing them to the vectorized Turing @model.
"""
function BayesianFootball.build_turing_model(
    ::AR1PoissonHAVectorizedModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    # --- 1. Pre-compute Sparse Selection Matrices ---
    # This is the "grouping" step, done once.
    n_matches = length(goals_home)
    n_teams = features.n_teams
    n_leagues = features.n_leagues
    
    # We must find the *max* round in the *training data*
    n_rounds = maximum(features.global_round) 
    
    row_idx = 1:n_matches

    # Get the column index for each match in the flattened (n_teams * n_rounds) vector
    col_home_α = (features.global_round .- 1) .* n_teams .+ features.home_team_ids
    col_away_β = (features.global_round .- 1) .* n_teams .+ features.away_team_ids
    col_away_α = (features.global_round .- 1) .* n_teams .+ features.away_team_ids
    col_home_β = (features.global_round .- 1) .* n_teams .+ features.home_team_ids
    
    # Get column index for the (n_leagues * n_rounds) vector
    col_home_adv = (features.global_round .- 1) .* n_leagues .+ features.league_ids

    # Create the sparse matrices
    S_home_α   = sparse(row_idx, col_home_α, 1, n_matches, n_teams * n_rounds)
    S_away_β   = sparse(row_idx, col_away_β, 1, n_matches, n_teams * n_rounds)
    S_away_α   = sparse(row_idx, col_away_α, 1, n_matches, n_teams * n_rounds)
    S_home_β   = sparse(row_idx, col_home_β, 1, n_matches, n_teams * n_rounds)
    S_home_adv = sparse(row_idx, col_home_adv, 1, n_matches, n_leagues * n_rounds)

    # --- 2. Return the Instantiated Turing Model ---
    return ar1_poisson_ha_vectorized_turing(
        goals_home, goals_away,
        n_teams, n_leagues, n_rounds,
        S_home_α, S_away_β, S_home_adv,
        S_away_α, S_home_β
    )
end

#=
--------------------------------------------------------------------------------
4.  PREDICTION LOGIC (This MUST be updated)
--------------------------------------------------------------------------------
=#

# We need to re-write the posterior extraction and prediction
# functions to match the new vectorized model's parameters.

function BayesianFootball.extract_posterior_samples(
    ::AR1PoissonHAVectorizedModel,
    chain::Chains,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    n_teams = length(mapping.team)
    n_leagues = length(mapping.league)
    
    # Extract innovations to get n_rounds
    z_α_flat = BayesianFootball.extract_samples(chain, "z_α")
    n_rounds = size(z_α_flat, 2) ÷ n_teams
    
    # Extract AR(1) parameters
    ρ_attack=vec(Array(chain[:ρ_attack]));
    ρ_defense=vec(Array(chain[:ρ_defense]));
    ρ_home=vec(Array(chain[:ρ_home]))

    # Extract σ values
    σ_attack = exp.(vec(Array(chain[:μ_log_σ_attack])) .+ BayesianFootball.extract_samples(chain, "z_log_σ_attack") .* vec(Array(chain[:τ_log_σ_attack])))
    σ_defense = exp.(vec(Array(chain[:μ_log_σ_defense])) .+ BayesianFootball.extract_samples(chain, "z_log_σ_defense") .* vec(Array(chain[:τ_log_σ_defense])))
    σ_home = exp.(vec(Array(chain[:μ_log_σ_home])))

    # Extract initial states
    initial_α_z = BayesianFootball.extract_samples(chain, "initial_α_z")
    initial_β_z = BayesianFootball.extract_samples(chain, "initial_β_z")
    initial_home_z = BayesianFootball.extract_samples(chain, "initial_home_z")

    # Extract innovations
    z_α_mat = reshape(z_α_flat', n_teams, n_rounds, n_samples)
    z_β_mat = reshape(BayesianFootball.extract_samples(chain, "z_β")', n_teams, n_rounds, n_samples)
    z_home_mat = reshape(BayesianFootball.extract_samples(chain, "z_home")', n_leagues, n_rounds, n_samples)

    # Re-build the full latent states for all samples
    log_α_raw = Array{Float64,3}(undef, n_samples, n_teams, n_rounds)
    log_β_raw = Array{Float64,3}(undef, n_samples, n_teams, n_rounds)
    log_home_adv_raw = Array{Float64,3}(undef, n_samples, n_leagues, n_rounds)
    
    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)

    for s in 1:n_samples
        # Re-build L matrices for this sample's ρ
        L_α = LowerTriangular([t >= r ? ρ_attack[s]^(t - r) : 0.0 for t in 1:n_rounds, r in 1:n_rounds])
        L_β = LowerTriangular([t >= r ? ρ_defense[s]^(t - r) : 0.0 for t in 1:n_rounds, r in 1:n_rounds])
        L_h = LowerTriangular([t >= r ? ρ_home[s]^(t - r) : 0.0 for t in 1:n_rounds, r in 1:n_rounds])
        
        # Get initial states for this sample
        log_α_raw_t0 = initial_α_z[s,:] .* sqrt(0.5)
        log_β_raw_t0 = initial_β_z[s,:] .* sqrt(0.5)
        log_home_adv_raw_t0 = log(1.3) .+ initial_home_z[s,:] .* sqrt(0.1)
        
        # Get innovations for this sample
        z_α_s = z_α_mat[:, :, s]
        z_β_s = z_β_mat[:, :, s]
        z_h_s = z_home_mat[:, :, s]
        
        # Evolve all states at once
        α_raw_s = (L_α * (z_α_s .* σ_attack[s,:] .+ log_α_raw_t0))'
        β_raw_s = (L_β * (z_β_s .* σ_defense[s,:] .+ log_β_raw_t0))'
        h_raw_s = (L_h * (z_h_s .* σ_home[s] .+ log_home_adv_raw_t0))'
        
        # Save raw and centered states
        log_α_raw[s, :, :] = α_raw_s'
        log_β_raw[s, :, :] = β_raw_s'
        log_home_adv_raw[s, :, :] = h_raw_s'
        
        log_α_centered[s, :, :] = (α_raw_s .- mean(α_raw_s, dims=2))'
        log_β_centered[s, :, :] = (β_raw_s .- mean(β_raw_s, dims=2))'
    end

    return (
        n_rounds=n_rounds,
        ρ_attack=ρ_attack, ρ_defense=ρ_defense, ρ_home=ρ_home,
        σ_attack=σ_attack, σ_defense=σ_defense, σ_home=σ_home,
        log_α_raw=log_α_raw, log_β_raw=log_β_raw, log_home_adv_raw=log_home_adv_raw,
        log_α_centered=log_α_centered, log_β_centered=log_β_centered
    )
end

# This function becomes *much* simpler
function get_goal_rates(
    ::AR1PoissonHAVectorizedModel,
    samples::NamedTuple,
    i::Int, # Sample index
    features::NamedTuple
)
    home_idx=features.home_team_ids[1]
    away_idx=features.away_team_ids[1]
    league_idx=features.league_ids[1]
    current_round = features.global_round[1]
    
    local log_α_t, log_β_t, log_home_adv_i

    if current_round <= samples.n_rounds
        # Just look up the pre-computed value
        log_α_t = samples.log_α_centered[i, :, current_round]
        log_β_t = samples.log_β_centered[i, :, current_round]
        log_home_adv_i = samples.log_home_adv_raw[i, league_idx, current_round]
    else
        # Forecast one step ahead (same logic as your old model's 'else' block)
        n_teams = size(samples.log_α_raw, 2)
        n_leagues = size(samples.log_home_adv_raw, 2)
        
        last_α_raw = samples.log_α_raw[i, :, samples.n_rounds]
        last_β_raw = samples.log_β_raw[i, :, samples.n_rounds]
        last_home_adv_raw = samples.log_home_adv_raw[i, :, samples.n_rounds]
        
        future_α_raw = samples.ρ_attack[i] .* last_α_raw .+ randn(n_teams) .* samples.σ_attack[i,:]
        future_β_raw = samples.ρ_defense[i] .* last_β_raw .+ randn(n_teams) .* samples.σ_defense[i,:]
        future_home_adv_raw = samples.ρ_home[i] .* last_home_adv_raw .+ randn(n_leagues) .* samples.σ_home[i]

        log_α_t = future_α_raw .- mean(future_α_raw)
        log_β_t = future_β_raw .- mean(future_β_raw)
        log_home_adv_i = future_home_adv_raw[league_idx]
    end
    
    log_λ_home = log_α_t[home_idx] + log_β_t[away_idx] + log_home_adv_i
    log_λ_away = log_α_t[away_idx] + log_β_t[home_idx]
    
    return (λ_home = exp(log_λ_home), λ_away = exp(log_λ_away))
end

function BayesianFootball._predict_match_ft(
    model_def::AR1PoissonHAVectorizedModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home=zeros(n_samples); λ_away=zeros(n_samples); home_win_probs=zeros(n_samples); draw_probs=zeros(n_samples); away_win_probs=zeros(n_samples)
    under_05=zeros(n_samples); under_15=zeros(n_samples); under_25=zeros(n_samples); under_35=zeros(n_samples); btts=zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:3 for a in 0:3], ["other_home_win", "other_away_win", "other_draw"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i]=λ_h; λ_away[i]=λ_a
        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p); cs = BayesianFootball.calculate_correct_score_dict_ft(p)
        home_win_probs[i]=hda.hw; draw_probs[i]=hda.dr; away_win_probs[i]=hda.aw
        for (key, value) in cs; correct_score[key][i]=value; end
        under_05[i]=BayesianFootball.calculate_under_prob(p,0); under_15[i]=BayesianFootball.calculate_under_prob(p,1)
        under_25[i]=BayesianFootball.calculate_under_prob(p,2); under_35[i]=BayesianFootball.calculate_under_prob(p,3)
        btts[i] = BayesianFootball.calculate_btts(p)
    end

    return BayesianFootball.Predictions.MatchFTPredictions(λ_home, λ_away, home_win_probs, draw_probs, away_win_probs, correct_score, under_05, under_15, under_25, under_35, btts)
end

function BayesianFootball._predict_match_ht(
    model_def::AR1PoissonHAVectorizedModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home=zeros(n_samples); λ_away=zeros(n_samples); home_win_probs=zeros(n_samples); draw_probs=zeros(n_samples); away_win_probs=zeros(n_samples)
    under_05=zeros(n_samples); under_15=zeros(n_samples); under_25=zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:2 for a in 0:2], ["any_unquoted"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i]=λ_h; λ_away[i]=λ_a
        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p); cs = BayesianFootball.calculate_correct_score_dict_ht(p)
        home_win_probs[i]=hda.hw; draw_probs[i]=hda.dr; away_win_probs[i]=hda.aw
        for (key, value) in cs; correct_score[key][i]=value; end
        under_05[i]=BayesianFootball.calculate_under_prob(p,0); under_15[i]=BayesianFootball.calculate_under_prob(p,1)
        under_25[i]=BayesianFootball.calculate_under_prob(p,2)
    end

    return BayesianFootball.Predictions.MatchHTPredictions(λ_home, λ_away, home_win_probs, draw_probs, away_win_probs, correct_score, under_05, under_15, under_25)
end
end # end module
