# src/models/pregame/implementations/grw_double_neg_bin_delta.jl

using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics


export GRWNegativeBinomialDelta

Base.@kwdef struct GRWNegativeBinomialDelta <: AbstractDynamicNegBinModel
    # --- Global Baseline (Intercept) ---
    μ::Distribution = Normal(0.2, 0.5)

    # --- Home Advantage ---
    γ::Distribution = Normal(log(1.3), 0.2)
    
    # --- Dispersion (Negative Binomial) ---
    log_r_prior::Distribution = Normal(1.5, 1.0) 

    # --- NEW: Hierarchical Process Noise (Volatility) ---
    # 1. Global Mean Log-Volatility
    #    Targeting approx 0.05 on linear scale. 
    #    log(0.05) ≈ -3.0. We allow some uncertainty around this.
    log_σ_att_global::Distribution = Normal(-3.0, 0.5)
    log_σ_def_global::Distribution = Normal(-3.0, 0.5)

    # 2. Team Deviation (The "Delta")
    #    Controls how much teams are allowed to differ from the global average.
    #    Normal(0, 0.2) allows teams to be roughly ±20% different from the norm.
    δ_σ_att::Distribution = Normal(0, 0.2)
    δ_σ_def::Distribution = Normal(0, 0.2)
    
    # --- Initial State Hyperparameters (t=0) ---
    # We keep this as is, or you could make this hierarchical too if desired.
    σ_0::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

    z_init::Distribution = Normal(0,1)
    z_steps::Distribution = Normal(0,1)
end


function extract_parameters(
    model::GRWNegativeBinomialDelta, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer n_rounds
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    # --- 1. Extract Chains ---
    # Get the raw matrices/vectors from the chain
    # Note: These operations can be memory intensive; using `generated_quantities` is preferred
    # if possible, but this manual reconstruction allows full control.
    
    # Helper to get mean values from chain (Point Estimate Extraction)
    # If you want full posterior predictive, you must loop over samples.
    # Here we extract the MEAN parameters for prediction.
    γ_val = mean(chain[:γ])
    μ_val = mean(chain[:μ])
    r_val = mean(exp.(chain[:log_r]))
    
    # Reconstruct Process Noise Vectors
    log_σ_att_bar = mean(chain[:log_σ_att_bar])
    log_σ_def_bar = mean(chain[:log_σ_def_bar])
    
    # Extract Deltas (Vectors of size n_teams)
    # We retrieve the group and compute column means
    δ_att_vec = vec(mean(Array(group(chain, :δ_att)), dims=1))
    δ_def_vec = vec(mean(Array(group(chain, :δ_def)), dims=1))
    
    # Compute Effective Sigmas
    σ_att_vec = exp.(log_σ_att_bar .+ δ_att_vec)
    σ_def_vec = exp.(log_σ_def_bar .+ δ_def_vec)

    # --- 2. Reconstruct Trajectories (Point Estimate) ---
    z_att_init = vec(mean(Array(group(chain, :z_att_init)), dims=1))
    z_def_init = vec(mean(Array(group(chain, :z_def_init)), dims=1))
    
    # Reshape Steps: Chains returns flat 1x(N*T) usually, need to reshape
    z_att_steps_flat = vec(mean(Array(group(chain, :z_att_steps)), dims=1))
    z_def_steps_flat = vec(mean(Array(group(chain, :z_def_steps)), dims=1))
    
    # Reshape to (n_teams, n_rounds - 1)
    z_att_steps = reshape(z_att_steps_flat, n_teams, n_rounds - 1)
    z_def_steps = reshape(z_def_steps_flat, n_teams, n_rounds - 1)
    
    # Helper for σ_0
    σ_att_0 = mean(chain[:σ_att_0])
    σ_def_0 = mean(chain[:σ_def_0])

    # Calculate Paths
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0
    
    # Apply Team-Specific Process Noise
    scaled_steps_att = z_att_steps .* σ_att_vec
    scaled_steps_def = z_def_steps .* σ_def_vec
    
    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)
    
    # Center
    att_cube = att_raw .- mean(att_raw, dims=1)
    def_cube = def_raw .- mean(def_raw, dims=1)

    # --- 3. Build Prediction Dictionary ---
    ExtractionValue = NamedTuple{(:λ_h, :λ_a, :r), Tuple{Float64, Float64, Float64}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        t = row.match_week 
        t_idx = clamp(t, 1, n_rounds)

        att_h = att_cube[h_id, t_idx]
        def_a = def_cube[a_id, t_idx]
        att_a = att_cube[a_id, t_idx]
        def_h = def_cube[h_id, t_idx]

        λ_h = exp(μ_val + γ_val + att_h + def_a)
        λ_a = exp(μ_val + att_a + def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, r=r_val)
    end

    return extraction_dict
end


"""
    extract_volatility_analysis(model, feature_set, chain)

Extracts hierarchical volatility parameters.
Returns a DataFrame with one row per team, showing their specific stability profile.
"""
function extract_volatility_analysis(
    model::GRWNegativeBinomialDelta, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # --- 1. Setup ---
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    # Invert map to get names: ID -> Name
    id_to_team = Dict(v => k for (k, v) in team_map)

    # --- 2. Extract Posterior Samples ---
    # Global Baselines (Vectors of samples)
    log_σ_att_bar = vec(Array(chain[:log_σ_att_bar]))
    log_σ_def_bar = vec(Array(chain[:log_σ_def_bar]))

    # Team Deltas (Matrices: Samples x Teams)
    # 'group' grabs all δ_att[1]...δ_att[n] columns
    δ_att_mat = Array(group(chain, :δ_att))
    δ_def_mat = Array(group(chain, :δ_def))

    # --- 3. Compute Effective Sigmas (Sample-wise) ---
    # We must compute exp(global + delta) for every sample to preserve correlations.
    # Broadcasting: (Samples vector) .+ (Samples x Teams matrix) works column-wise.
    σ_att_mat = exp.(log_σ_att_bar .+ δ_att_mat)
    σ_def_mat = exp.(log_σ_def_bar .+ δ_def_mat)

    # --- 4. Summarize per Team ---
    teams = String[]
    
    # Delta Stats (Relative Stability)
    d_att_mean, d_att_lo, d_att_hi = Float64[], Float64[], Float64[]
    d_def_mean, d_def_lo, d_def_hi = Float64[], Float64[], Float64[]
    
    # Sigma Stats (Absolute Volatility)
    s_att_mean, s_att_lo, s_att_hi = Float64[], Float64[], Float64[]
    s_def_mean, s_def_lo, s_def_hi = Float64[], Float64[], Float64[]

    for i in 1:n_teams
        t_name = id_to_team[i]
        push!(teams, t_name)

        # --- Attack ---
        # Delta
        d_a_vals = view(δ_att_mat, :, i)
        push!(d_att_mean, mean(d_a_vals))
        push!(d_att_lo, quantile(d_a_vals, 0.05))
        push!(d_att_hi, quantile(d_a_vals, 0.95))
        
        # Effective Sigma
        s_a_vals = view(σ_att_mat, :, i)
        push!(s_att_mean, mean(s_a_vals))
        push!(s_att_lo, quantile(s_a_vals, 0.05))
        push!(s_att_hi, quantile(s_a_vals, 0.95))

        # --- Defense ---
        # Delta
        d_d_vals = view(δ_def_mat, :, i)
        push!(d_def_mean, mean(d_d_vals))
        push!(d_def_lo, quantile(d_d_vals, 0.05))
        push!(d_def_hi, quantile(d_d_vals, 0.95))
        
        # Effective Sigma
        s_d_vals = view(σ_def_mat, :, i)
        push!(s_def_mean, mean(s_d_vals))
        push!(s_def_lo, quantile(s_d_vals, 0.05))
        push!(s_def_hi, quantile(s_d_vals, 0.95))
    end

    # --- 5. Build DataFrame ---
    df = DataFrame(
        team = teams,
        # Attack
        att_delta = d_att_mean,
        att_sigma = s_att_mean,
        att_sigma_low = s_att_lo,
        att_sigma_high = s_att_hi,
        
        # Defense
        def_delta = d_def_mean,
        def_sigma = s_def_mean,
        def_sigma_low = s_def_lo,
        def_sigma_high = s_def_hi
    )
    
    # Sort by Attack Volatility (Descending) - easiest way to spot the "wild" teams
    sort!(df, :att_sigma, rev=true)

    return df
end
