# src/prediction/api.jl

using ..BayesianFootball: AbstractModelDefinition, Predictions, MappedData, Chains
using Distributions

"""
Extract all samples for parameters matching a base string, e.g., "log_α_raw".
"""
function extract_samples(chain::Chains, base::String)
    nms = names(chain)
    matches = filter(n -> occursin(base, String(n)), nms)
    arr = Array(chain[matches])
    reshape(arr, :, size(arr, 2))  # flatten iterations × chains in first dimension
end

function compute_xScore(λ_home::Number, λ_away::Number, max_goals::Int64)
    p = zeros(max_goals+1, max_goals+1)
    for h in 0:max_goals, a in 0:max_goals
        p[h+1,a+1] = pdf(Poisson(λ_home), h) * pdf(Poisson(λ_away), a)
    end 
    return p
end

function calculate_1x2(p::Matrix{Float64}) 
    hw = 0.0
    dr = 0.0
    aw = 0.0
    for h in 1:(size(p,1)), a in 1:(size(p,2))
        if h > a
            hw += p[h, a]
        elseif h == a
            dr += p[h,a]
        else
            aw += p[h, a]
        end 
    end 
    return (hw = hw, dr = dr, aw = aw)
end

function calculate_correct_score_dict_ht(p::Matrix{Float64})
    # Initialize dict with all score combinations
    correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    
    # Standard scores (0-0 through 2-2)
    for h in 0:2, a in 0:2
        correct_scores[(h, a)] = p[h+1, a+1]
    end
    
    # Calculate any_unquoted probability
    any_unquoted = 0.0
    max_goals = size(p,1)-1
    for h in 0:max_goals, a in 0:max_goals
        if h > 2 || a > 2
            any_unquoted += p[h+1, a+1]
        end
    end
    correct_scores["any_unquoted"] = any_unquoted
    
    return correct_scores
end

function calculate_correct_score_dict_ft(p::Matrix{Float64})
    # Initialize dict with all score combinations
    correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    
    # Standard scores (0-0 through 3-3)
    for h in 0:3, a in 0:3
        correct_scores[(h, a)] = p[h+1, a+1]
    end
    
    # Calculate other scores
    other_home_win = 0.0
    other_away_win = 0.0
    other_draw = 0.0
    max_goals = size(p,1)-1
    
    for h in 0:max_goals, a in 0:max_goals
        if h > 3 || a > 3
            if h > a
                other_home_win += p[h+1, a+1]
            elseif a > h
                other_away_win += p[h+1, a+1]
            else  # h == a
                other_draw += p[h+1, a+1]
            end
        end
    end
    
    correct_scores["other_home_win"] = other_home_win
    correct_scores["other_away_win"] = other_away_win
    correct_scores["other_draw"] = other_draw
    
    return correct_scores
end

function calculate_under_prob(p::Matrix{Float64}, threshold::Int)
    prob_under = 0.0
    max_goals = size(p,1)-1
    for h in 0:max_goals, a in 0:max_goals
        total_goals = h + a
        if total_goals <= threshold
            prob_under += p[h+1, a+1]
        end
    end
    return prob_under
end

function calculate_btts(p::Matrix{Float64})
    btts_prob = 0.0
    max_goals = size(p,1)-1
    for h in 1:max_goals, a in 1:max_goals  # Start from 1 (both teams score)
        btts_prob += p[h+1, a+1]
    end
    return btts_prob
end

function predict_match_ft(
    model_def::AbstractModelDefinition,
    chain::Chains,
    features::NamedTuple,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    posterior_samples = extract_posterior_samples(model_def, chain)

    # Initialize result vectors
    λ_home = zeros(n_samples)
    λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples)
    draw_probs = zeros(n_samples)
    away_win_probs = zeros(n_samples)
    
    # Initialize correct score dict of vectors
    cs_keys = vcat(
        [(h, a) for h in 0:3 for a in 0:3],
        ["other_home_win", "other_away_win", "other_draw"]
    )
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(
        key => zeros(n_samples) for key in cs_keys
    )
    
    under_05 = zeros(n_samples)
    under_15 = zeros(n_samples)
    under_25 = zeros(n_samples)
    under_35 = zeros(n_samples)
    btts = zeros(n_samples)


    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a

        p = compute_xScore(λ_home[i], λ_away[i], 10)
        
        hda = calculate_1x2(p)
        cs = calculate_correct_score_dict_ft(p)
        
        home_win_probs[i] = hda.hw
        draw_probs[i] = hda.dr
        away_win_probs[i] = hda.aw
        
        # Assign correct scores
        for (key, value) in cs
            correct_score[key][i] = value
        end
        
        under_05[i] = calculate_under_prob(p, 0)
        under_15[i] = calculate_under_prob(p, 1)
        under_25[i] = calculate_under_prob(p, 2)
        under_35[i] = calculate_under_prob(p, 3)
        btts[i] = calculate_btts(p)

    end

    return Predictions.MatchFTPredictions(
        λ_home,
        λ_away,
        home_win_probs,
        draw_probs,
        away_win_probs,
        correct_score,
        under_05,
        under_15,
        under_25,
        under_35,
        btts
    )
end


function predict_match_ht(
    model_def::AbstractModelDefinition,
    chain::Chains,
    features::NamedTuple,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    posterior_samples = extract_posterior_samples(model_def, chain)
    
    # Initialize result vectors
    λ_home = zeros(n_samples)
    λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples)
    draw_probs = zeros(n_samples)
    away_win_probs = zeros(n_samples)
    
    # Initialize correct score dict of vectors
    cs_keys = vcat(
        [(h, a) for h in 0:2 for a in 0:2],
        ["any_unquoted"]
    )
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(
        key => zeros(n_samples) for key in cs_keys
    )
    
    under_05 = zeros(n_samples)
    under_15 = zeros(n_samples)
    under_25 = zeros(n_samples)

    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a

        p = compute_xScore(λ_home[i], λ_away[i], 10)
        
        hda = calculate_1x2(p)
        cs = calculate_correct_score_dict_ht(p)
        
        home_win_probs[i] = hda.hw
        draw_probs[i] = hda.dr
        away_win_probs[i] = hda.aw
        
        # Assign correct scores
        for (key, value) in cs
            correct_score[key][i] = value
        end
        
        under_05[i] = calculate_under_prob(p, 0)
        under_15[i] = calculate_under_prob(p, 1)
        under_25[i] = calculate_under_prob(p, 2)
    end
    return Predictions.MatchHTPredictions(
        λ_home,
        λ_away,
        home_win_probs,
        draw_probs,
        away_win_probs,
        correct_score,
        under_05,
        under_15,
        under_25
    )
end

"""
    predict_match_lines(model_def, round_chains, features, mapping)

Generates both HT and FT predictions for a single match.
"""
function predict_match_lines(
    model_def::AbstractModelDefinition,
    round_chains::TrainedChains,
    features::NamedTuple,
    mapping::MappedData
)
    ht_predict = predict_match_ht(model_def, round_chains.ht, features, mapping)
    ft_predict = predict_match_ft(model_def, round_chains.ft, features, mapping)
    return Predictions.MatchLinePredictions(ht_predict, ft_predict)
end
