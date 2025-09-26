# workspace/ar1_poisson/prediction.jl

module AR1Prediction

using ..AR1StateSpace
using BayesianFootball
using MCMCChains
using Statistics
using Distributions
using Turing # Required for LogPoisson

# Export the main entry point function
export predict_ar1_match_lines

#=
--------------------------------------------------------------------------------
1.  PARAMETER EXTRACTION
--------------------------------------------------------------------------------
This is the specialized implementation for the AR1 model. It reconstructs the
full time-series of raw (uncentered) parameters and also pre-calculates the
centered versions for efficient prediction lookups.
=#

function BayesianFootball.extract_posterior_samples(
    ::AR1PoissonModel,
    chain::Chains,
    mapping::MappedData
)
    # --- 1. Get dimensions ---
    n_samples = size(chain, 1) * size(chain, 3)
    n_teams = length(mapping.team)
    
    # Infer n_rounds from the dimensions of a time-varying parameter
    z_α_flat = BayesianFootball.extract_samples(chain, "z_α")
    n_rounds = size(z_α_flat, 2) ÷ n_teams
    
    # --- 2. Extract process parameters (ρ, σ) ---
    ρ_attack = vec(Array(chain[:ρ_attack]))
    ρ_defense = vec(Array(chain[:ρ_defense]))
    log_home_adv = vec(Array(chain[:log_home_adv]))

    # [cite_start]Reconstruct team-specific volatilities (σ) [cite: 45]
    μ_log_σ_attack = vec(Array(chain[:μ_log_σ_attack]))
    τ_log_σ_attack = vec(Array(chain[:τ_log_σ_attack]))
    z_log_σ_attack = BayesianFootball.extract_samples(chain, "z_log_σ_attack")
    σ_attack = exp.(μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack)

    μ_log_σ_defense = vec(Array(chain[:μ_log_σ_defense]))
    τ_log_σ_defense = vec(Array(chain[:τ_log_σ_defense]))
    z_log_σ_defense = BayesianFootball.extract_samples(chain, "z_log_σ_defense")
    σ_defense = exp.(μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense)

    # --- 3. Reconstruct the full raw state-space trajectories ---
    initial_α_z = BayesianFootball.extract_samples(chain, "initial_α_z")
    initial_β_z = BayesianFootball.extract_samples(chain, "initial_β_z")
    z_α_mat_reshaped = reshape(z_α_flat', n_teams, n_rounds, n_samples)
    z_β_flat = BayesianFootball.extract_samples(chain, "z_β")
    z_β_mat_reshaped = reshape(z_β_flat', n_teams, n_rounds, n_samples)

    log_α_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)
    log_β_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)

    for s in 1:n_samples
        log_α_raw_t0 = initial_α_z[s, :] .* sqrt(0.5)
        log_β_raw_t0 = initial_β_z[s, :] .* sqrt(0.5)
        
        # [cite_start]Evolve state over time for each sample [cite: 47]
        for t in 1:n_rounds
            if t == 1
                log_α_raw[s, :, 1] = log_α_raw_t0 .+ z_α_mat_reshaped[:, 1, s] .* σ_attack[s, :]
                log_β_raw[s, :, 1] = log_β_raw_t0 .+ z_β_mat_reshaped[:, 1, s] .* σ_defense[s, :]
            else
                log_α_raw[s, :, t] = ρ_attack[s] * log_α_raw[s, :, t-1] .+ z_α_mat_reshaped[:, t, s] .* σ_attack[s, :]
                log_β_raw[s, :, t] = ρ_defense[s] * log_β_raw[s, :, t-1] .+ z_β_mat_reshaped[:, t, s] .* σ_defense[s, :]
            end
        end
    end

    # --- 4. Pre-calculate centered parameters for efficiency ---
    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)
    for s in 1:n_samples, t in 1:n_rounds
        log_α_centered[s, :, t] = log_α_raw[s, :, t] .- mean(log_α_raw[s, :, t])
        log_β_centered[s, :, t] = log_β_raw[s, :, t] .- mean(log_β_raw[s, :, t])
    end

    return (
        n_rounds = n_rounds,
        log_home_adv = log_home_adv,
        ρ_attack = ρ_attack,
        ρ_defense = ρ_defense,
        σ_attack = σ_attack,
        σ_defense = σ_defense,
        log_α_raw = log_α_raw,
        log_β_raw = log_β_raw,
        log_α_centered = log_α_centered,
        log_β_centered = log_β_centered
    )
end


#=
--------------------------------------------------------------------------------
2.  TIME-AWARE GOAL RATE CALCULATION
--------------------------------------------------------------------------------
This is the core prediction logic. It checks if the match is in-sample or a
future match and gets the team strengths for the correct time step.
=#

function BayesianFootball.get_goal_rates(
    ::AR1PoissonModel,
    samples::NamedTuple,
    i::Int,
    features::NamedTuple
)
    home_idx = features.home_team_ids[1]
    away_idx = features.away_team_ids[1]
    current_round = features.global_round[1]
    
    local log_α_t, log_β_t

    if current_round <= samples.n_rounds
        # --- In-Sample Prediction ---
        # Simply look up the pre-calculated centered strengths for this round
        log_α_t = samples.log_α_centered[i, :, current_round]
        log_β_t = samples.log_β_centered[i, :, current_round]

    else
        # --- Future (T+1) Prediction ---
        # [cite_start]Get the raw state from the final round of data [cite: 79]
        last_α_raw = samples.log_α_raw[i, :, samples.n_rounds]
        last_β_raw = samples.log_β_raw[i, :, samples.n_rounds]
        
        # [cite_start]Propagate the state one step forward using the AR(1) process [cite: 80]
        n_teams = size(last_α_raw, 1)
        future_α_raw = samples.ρ_attack[i] .* last_α_raw .+ randn(n_teams) .* samples.σ_attack[i, :]
        future_β_raw = samples.ρ_defense[i] .* last_β_raw .+ randn(n_teams) .* samples.σ_defense[i, :]

        # [cite_start]Center the new state for identifiability before calculating rates [cite: 80]
        log_α_t = future_α_raw .- mean(future_α_raw)
        log_β_t = future_β_raw .- mean(future_β_raw)
    end
    
    # Calculate goal rates using the strengths for the correct time step
    log_home_adv_i = samples.log_home_adv[i]
    log_λ_home = log_α_t[home_idx] + log_β_t[away_idx] + log_home_adv_i
    log_λ_away = log_α_t[away_idx] + log_β_t[home_idx]
    
    return (λ_home = exp(log_λ_home), λ_away = exp(log_λ_away))
end


#=
--------------------------------------------------------------------------------
3.  PREDICTION LOOP & API INTEGRATION
--------------------------------------------------------------------------------
These functions follow the standard API, but they call our new time-aware
`get_goal_rates` function inside the loop.
=#

"""
Main entry point for AR1 model predictions.
"""
function predict_ar1_match_lines(
    model_def::AR1PoissonModel,
    round_chains::TrainedChains,
    features::NamedTuple,
    mapping::MappedData
)
    # Extract posterior samples once, including pre-calculated parameters
    posterior_samples = BayesianFootball.extract_posterior_samples(model_def, round_chains.ft, mapping)
    
    ht_predict = _predict_ar1_match_ht(model_def, round_chains.ht, features, posterior_samples)
    ft_predict = _predict_ar1_match_ft(model_def, round_chains.ft, features, posterior_samples)
    
    return BayesianFootball.Predictions.MatchLinePredictions(ht_predict, ft_predict)
end

function _predict_ar1_match_ft(
    model_def::AR1PoissonModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple # Pass in the pre-extracted samples
)
    n_samples = size(chain, 1) * size(chain, 3)
    # [cite_start]Initialize result structs using the pattern from api.jl [cite: 113, 114, 115, 116, 117]
    # ... (code for initializing containers like home_win_probs, correct_score, etc.)
    # This part is identical to the standard predict_match_ft in api.jl

    λ_home = zeros(n_samples); λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples); draw_probs = zeros(n_samples); away_win_probs = zeros(n_samples)
    under_05 = zeros(n_samples); under_15 = zeros(n_samples); under_25 = zeros(n_samples); under_35 = zeros(n_samples)
    btts = zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:3 for a in 0:3], ["other_home_win", "other_away_win", "other_draw"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        # Call the new time-aware function to get goal rates
        λ_h, λ_a = BayesianFootball.get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a

        # --- REUSE EXISTING FRAMEWORK LOGIC ---
        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ft(p)
        
        home_win_probs[i] = hda.hw; draw_probs[i] = hda.dr; away_win_probs[i] = hda.aw
        for (key, value) in cs; correct_score[key][i] = value; end
        
        under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
        under_35[i] = BayesianFootball.calculate_under_prob(p, 3)
        btts[i] = BayesianFootball.calculate_btts(p)
    end

    return BayesianFootball.Predictions.MatchFTPredictions(
        λ_home, λ_away, home_win_probs, draw_probs, away_win_probs,
        correct_score, under_05, under_15, under_25, under_35, btts
    )
end

function _predict_ar1_match_ht(
    model_def::AR1PoissonModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple # Pass in the pre-extracted samples
)
    n_samples = size(chain, 1) * size(chain, 3)
    # [cite_start]Initialize result structs using the pattern from api.jl [cite: 119, 120, 121, 122]
    # This part is identical to the standard predict_match_ht in api.jl
    λ_home = zeros(n_samples); λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples); draw_probs = zeros(n_samples); away_win_probs = zeros(n_samples)
    under_05 = zeros(n_samples); under_15 = zeros(n_samples); under_25 = zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:2 for a in 0:2], ["any_unquoted"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        λ_h, λ_a = BayesianFootball.get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a

        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ht(p)

        home_win_probs[i] = hda.hw; draw_probs[i] = hda.dr; away_win_probs[i] = hda.aw
        for (key, value) in cs; correct_score[key][i] = value; end

        under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
    end

    return BayesianFootball.Predictions.MatchHTPredictions(
        λ_home, λ_away, home_win_probs, draw_probs, away_win_probs,
        correct_score, under_05, under_15, under_25
    )
end

end # end module AR1Prediction




# workspace/neg_bin_ar1/prediction.jl

# Note: As with the AR(1) Poisson version, the 'module' wrapper has been omitted
# to ensure the code works correctly within your scripting environment and avoids type errors.
module AR1NegBiPrediction

using ..AR1NegativeBinomial
using BayesianFootball
using MCMCChains
using Statistics
using Distributions
using Turing # Required for NegativeBinomial PDF

export predict_ar1_neg_bin_match_lines 

#=
--------------------------------------------------------------------------------
0.  HELPER FUNCTION FOR NEGATIVE BINOMIAL
--------------------------------------------------------------------------------
This is a new helper, similar to compute_xScore, but for the NegBinomial.
=#

function _compute_xScore_neg_bin(μ_home::Number, μ_away::Number, ϕ::Number, max_goals::Int)
    p = zeros(max_goals + 1, max_goals + 1)
    for h in 0:max_goals, a in 0:max_goals
        # Use the Negative Binomial's probability density function (pdf)
        p[h+1, a+1] = pdf(NegativeBinomial(μ_home, ϕ), h) * pdf(NegativeBinomial(μ_away, ϕ), a)
    end
    return p
end

#=
--------------------------------------------------------------------------------
1.  PARAMETER EXTRACTION
--------------------------------------------------------------------------------
This is the specialized implementation for the AR1-NB model. It's identical
to the Poisson version, with the addition of extracting the `ϕ` parameter.
=#

function BayesianFootball.extract_posterior_samples(
    ::AR1NegativeBinomialModel,
    chain::Chains,
    mapping::MappedData
)
    # This function is identical to the Poisson version, with one addition.
    
    # --- 1. Get dimensions & Base Parameters (copied) ---
    n_samples = size(chain, 1) * size(chain, 3)
    n_teams = length(mapping.team)
    z_α_flat = BayesianFootball.extract_samples(chain, "z_α")
    n_rounds = size(z_α_flat, 2) ÷ n_teams
    
    ρ_attack = vec(Array(chain[:ρ_attack]))
    ρ_defense = vec(Array(chain[:ρ_defense]))
    log_home_adv = vec(Array(chain[:log_home_adv]))

    μ_log_σ_attack = vec(Array(chain[:μ_log_σ_attack]))
    τ_log_σ_attack = vec(Array(chain[:τ_log_σ_attack]))
    z_log_σ_attack = BayesianFootball.extract_samples(chain, "z_log_σ_attack")
    σ_attack = exp.(μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack)

    μ_log_σ_defense = vec(Array(chain[:μ_log_σ_defense]))
    τ_log_σ_defense = vec(Array(chain[:τ_log_σ_defense]))
    z_log_σ_defense = BayesianFootball.extract_samples(chain, "z_log_σ_defense")
    σ_defense = exp.(μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense)

    # --- NEW: Extract the dispersion parameter ---
    ϕ = vec(Array(chain[:ϕ]))

    # --- 3. Reconstruct State-Space Trajectories (copied) ---
    initial_α_z = BayesianFootball.extract_samples(chain, "initial_α_z")
    initial_β_z = BayesianFootball.extract_samples(chain, "initial_β_z")
    z_α_mat_reshaped = reshape(z_α_flat', n_teams, n_rounds, n_samples)
    z_β_flat = BayesianFootball.extract_samples(chain, "z_β")
    z_β_mat_reshaped = reshape(z_β_flat', n_teams, n_rounds, n_samples)

    log_α_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)
    log_β_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)

    for s in 1:n_samples
        log_α_raw_t0 = initial_α_z[s, :] .* sqrt(0.5)
        log_β_raw_t0 = initial_β_z[s, :] .* sqrt(0.5)
        for t in 1:n_rounds
            if t == 1
                log_α_raw[s, :, 1] = log_α_raw_t0 .+ z_α_mat_reshaped[:, 1, s] .* σ_attack[s, :]
                log_β_raw[s, :, 1] = log_β_raw_t0 .+ z_β_mat_reshaped[:, 1, s] .* σ_defense[s, :]
            else
                log_α_raw[s, :, t] = ρ_attack[s] * log_α_raw[s, :, t-1] .+ z_α_mat_reshaped[:, t, s] .* σ_attack[s, :]
                log_β_raw[s, :, t] = ρ_defense[s] * log_β_raw[s, :, t-1] .+ z_β_mat_reshaped[:, t, s] .* σ_defense[s, :]
            end
        end
    end
    
    # --- 4. Pre-calculate centered parameters (copied) ---
    log_α_centered = similar(log_α_raw)
    log_β_centered = similar(log_β_raw)
    for s in 1:n_samples, t in 1:n_rounds
        log_α_centered[s, :, t] = log_α_raw[s, :, t] .- mean(log_α_raw[s, :, t])
        log_β_centered[s, :, t] = log_β_raw[s, :, t] .- mean(log_β_raw[s, :, t])
    end

    return (
        ϕ = ϕ, # Add new parameter to the output
        n_rounds = n_rounds,
        log_home_adv = log_home_adv,
        ρ_attack = ρ_attack, ρ_defense = ρ_defense,
        σ_attack = σ_attack, σ_defense = σ_defense,
        log_α_raw = log_α_raw, log_β_raw = log_β_raw,
        log_α_centered = log_α_centered, log_β_centered = log_β_centered
    )
end

#=
--------------------------------------------------------------------------------
2.  TIME-AWARE GOAL RATE CALCULATION
--------------------------------------------------------------------------------
This function is identical to the Poisson version because the dispersion
parameter `ϕ` does not affect the mean goal rate.
=#

function BayesianFootball.get_goal_rates(
    ::AR1NegativeBinomialModel,
    samples::NamedTuple,
    i::Int,
    features::NamedTuple
)
    home_idx = features.home_team_ids[1]
    away_idx = features.away_team_ids[1]
    current_round = features.global_round[1]
    
    local log_α_t, log_β_t

    if current_round <= samples.n_rounds
        log_α_t = samples.log_α_centered[i, :, current_round]
        log_β_t = samples.log_β_centered[i, :, current_round]
    else
        last_α_raw = samples.log_α_raw[i, :, samples.n_rounds]
        last_β_raw = samples.log_β_raw[i, :, samples.n_rounds]
        
        n_teams = size(last_α_raw, 1)
        future_α_raw = samples.ρ_attack[i] .* last_α_raw .+ randn(n_teams) .* samples.σ_attack[i, :]
        future_β_raw = samples.ρ_defense[i] .* last_β_raw .+ randn(n_teams) .* samples.σ_defense[i, :]

        log_α_t = future_α_raw .- mean(future_α_raw)
        log_β_t = future_β_raw .- mean(future_β_raw)
    end
    
    log_home_adv_i = samples.log_home_adv[i]
    log_λ_home = log_α_t[home_idx] + log_β_t[away_idx] + log_home_adv_i
    log_λ_away = log_α_t[away_idx] + log_β_t[home_idx]
    
    return (λ_home = exp(log_λ_home), λ_away = exp(log_λ_away))
end

#=
--------------------------------------------------------------------------------
3.  PREDICTION LOOP & API INTEGRATION
--------------------------------------------------------------------------------
=#

function predict_ar1_neg_bin_match_lines(
    model_def::AR1NegativeBinomialModel,
    round_chains::TrainedChains,
    features::NamedTuple,
    mapping::MappedData
)
    posterior_samples = BayesianFootball.extract_posterior_samples(model_def, round_chains.ft, mapping)
    
    ht_predict = _predict_ar1_neg_bin_match_ht(model_def, round_chains.ht, features, mapping, posterior_samples)
    ft_predict = _predict_ar1_neg_bin_match_ft(model_def, round_chains.ft, features, mapping, posterior_samples)
    
    return BayesianFootball.Predictions.MatchLinePredictions(ht_predict, ft_predict)
end

function _predict_ar1_neg_bin_match_ft(
    model_def::AR1NegativeBinomialModel,
    chain::Chains,
    features::NamedTuple,
    mapping::MappedData,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home = zeros(n_samples); λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples); draw_probs = zeros(n_samples); away_win_probs = zeros(n_samples)
    under_05 = zeros(n_samples); under_15 = zeros(n_samples); under_25 = zeros(n_samples); under_35 = zeros(n_samples)
    btts = zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:3 for a in 0:3], ["other_home_win", "other_away_win", "other_draw"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        μ_h, μ_a = BayesianFootball.get_goal_rates(model_def, posterior_samples, i, features)
        ϕ_i = posterior_samples.ϕ[i]
        λ_home[i] = μ_h; λ_away[i] = μ_a

        # Use the new helper for Negative Binomial probabilities
        p = _compute_xScore_neg_bin(μ_h, μ_a, ϕ_i, 10)
        
        # The rest of the logic is identical
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ft(p)
        home_win_probs[i] = hda.hw; draw_probs[i] = hda.dr; away_win_probs[i] = hda.aw
        for (key, value) in cs; correct_score[key][i] = value; end
        under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
        under_35[i] = BayesianFootball.calculate_under_prob(p, 3)
        btts[i] = BayesianFootball.calculate_btts(p)
    end

    return BayesianFootball.Predictions.MatchFTPredictions(λ_home, λ_away, home_win_probs, draw_probs, away_win_probs, correct_score, under_05, under_15, under_25, under_35, btts)
end

function _predict_ar1_neg_bin_match_ht(
    model_def::AR1NegativeBinomialModel,
    chain::Chains,
    features::NamedTuple,
    mapping::MappedData,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home = zeros(n_samples); λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples); draw_probs = zeros(n_samples); away_win_probs = zeros(n_samples)
    under_05 = zeros(n_samples); under_15 = zeros(n_samples); under_25 = zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:2 for a in 0:2], ["any_unquoted"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        μ_h, μ_a = BayesianFootball.get_goal_rates(model_def, posterior_samples, i, features)
        ϕ_i = posterior_samples.ϕ[i]
        λ_home[i] = μ_h; λ_away[i] = μ_a

        # Use the new helper for Negative Binomial probabilities
        p = _compute_xScore_neg_bin(μ_h, μ_a, ϕ_i, 10)
        
        # The rest of the logic is identical
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ht(p)
        home_win_probs[i] = hda.hw; draw_probs[i] = hda.dr; away_win_probs[i] = hda.aw
        for (key, value) in cs; correct_score[key][i] = value; end
        under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
    end

    return BayesianFootball.Predictions.MatchHTPredictions(λ_home, λ_away, home_win_probs, draw_probs, away_win_probs, correct_score, under_05, under_15, under_25)
end

end 
