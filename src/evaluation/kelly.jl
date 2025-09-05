# used with the types Kelly
##################################################
#  functions for kelly
##################################################
function under_to_over(under::Vector{Float64})
  return 1 .- under
end

"""
Calculate Bayesian Kelly fractions using a fully vectorized approach.
"""
function calculate_bayesian_kelly(
    model_probs::Vector{Float64},
    market_odds::Float64,
    commission_percent::Float64 = 0.02
)
    if market_odds <= 1.0
        return zeros(length(model_probs))
    end

    effective_odds = 1.0 + (market_odds - 1.0) * (1.0 - commission_percent)
    denominator = effective_odds - 1.0

    if denominator <= 0
        return zeros(length(model_probs))
    end

    # 1. Calculate all edges: (model_probs .* effective_odds .- 1.0)
    # 2. Divide by denominator: ... ./ denominator
    # 3. Set any negative results to 0.0: max.(0.0, ...)
    kelly_fractions = max.(0.0, (model_probs .* effective_odds .- 1.0) ./ denominator)

    return kelly_fractions
end

"""
Apply Bayesian Kelly calculations to all market groups for each period.
This module processes match odds and probabilities to calculate Kelly fractions.
"""

# Helper function to safely apply Kelly calculation
function safe_kelly_calculation(
    probs::Union{Vector{Float64}, Nothing},
    odds::Union{Float64, Nothing},
    config::Kelly.Config
)
    if isnothing(probs) || isnothing(odds)
        return nothing
    end
    return calculate_bayesian_kelly(probs, odds, config.market_commission_percent)
end

# Process correct score markets for HT
function process_ht_correct_score_kelly(
    correct_score_probs::Dict,
    correct_score_odds::Dict,
    config::Kelly.Config
)
    result = Kelly.CorrectScore()
    
    # Define HT correct score keys
    ht_keys = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        "any_unquoted"
    ]
    
    for key in ht_keys
        if haskey(correct_score_probs, key) && haskey(correct_score_odds, key)
            result[key] = safe_kelly_calculation(
                correct_score_probs[key],
                correct_score_odds[key],
                config
            )
        else
            result[key] = nothing
        end
    end
    
    return result
end

# Process correct score markets for FT
function process_ft_correct_score_kelly(
    correct_score_probs::Dict,
    correct_score_odds::Dict,
    config::Kelly.Config
)
    result = Kelly.CorrectScore()
    
    # Define FT correct score keys
    ft_keys = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2),
        (0,3), (1,3), (2,3), (3,3),
        "other_home_win", "other_away_win", "other_draw"
    ]
    
    for key in ft_keys
        if haskey(correct_score_probs, key) && haskey(correct_score_odds, key)
            result[key] = safe_kelly_calculation(
                correct_score_probs[key],
                correct_score_odds[key],
                config
            )
        else
            result[key] = nothing
        end
    end
    
    return result
end

"""
    apply_kelly_to_ht_markets(
        probs::MatchHTProbs,
        odds::MatchHTOdds,
        config::Kelly.Config
    ) -> Kelly.MatchHTKelly

Apply Bayesian Kelly calculations to all HT market groups.
"""
function apply_kelly_to_ht_markets(
    probs,  # Your probability structure for HT
    odds,   # Your odds structure for HT
    config::Kelly.Config
)
    # Main markets (1X2)
    home_kelly = safe_kelly_calculation(probs.home, odds.home, config)
    draw_kelly = safe_kelly_calculation(probs.draw, odds.draw, config)
    away_kelly = safe_kelly_calculation(probs.away, odds.away, config)
    
    # Correct scores
    correct_score_kelly = process_ht_correct_score_kelly(
        probs.correct_score,
        odds.correct_score,
        config
    )
    
    # Under/Over markets
    under_05_kelly = safe_kelly_calculation(probs.under_05, odds.under_05, config)
    over_05_kelly = safe_kelly_calculation(under_to_over(probs.under_05), odds.over_05, config)
    under_15_kelly = safe_kelly_calculation(probs.under_15, odds.under_15, config)
    over_15_kelly = safe_kelly_calculation(under_to_over(probs.under_15), odds.over_15, config)
    under_25_kelly = safe_kelly_calculation(probs.under_25, odds.under_25, config)
    over_25_kelly = safe_kelly_calculation(under_to_over(probs.under_25), odds.over_25, config)
    
    return Kelly.MatchHTKelly(
        home_kelly,
        draw_kelly,
        away_kelly,
        correct_score_kelly,
        under_05_kelly,
        over_05_kelly,
        under_15_kelly,
        over_15_kelly,
        under_25_kelly,
        over_25_kelly
    )
end

"""
    apply_kelly_to_ft_markets(
        probs::MatchFTProbs,
        odds::MatchFTOdds,
        config::Kelly.Config
    ) -> Kelly.MatchFTKelly

Apply Bayesian Kelly calculations to all FT market groups.
"""
function apply_kelly_to_ft_markets(
    probs,  # Your probability structure for FT
    odds,   # Your odds structure for FT
    config::Kelly.Config
)
    # Main markets (1X2)
    home_kelly = safe_kelly_calculation(probs.home, odds.home, config)
    draw_kelly = safe_kelly_calculation(probs.draw, odds.draw, config)
    away_kelly = safe_kelly_calculation(probs.away, odds.away, config)
    
    # Correct scores
    correct_score_kelly = process_ft_correct_score_kelly(
        probs.correct_score,
        odds.correct_score,
        config
    )
    
    # Under/Over markets
    under_05_kelly = safe_kelly_calculation(probs.under_05, odds.under_05, config)
    over_05_kelly = safe_kelly_calculation(under_to_over(probs.under_05), odds.over_05, config)
    under_15_kelly = safe_kelly_calculation(probs.under_15, odds.under_15, config)
    over_15_kelly = safe_kelly_calculation(under_to_over(probs.under_15), odds.over_15, config)
    under_25_kelly = safe_kelly_calculation(probs.under_25, odds.under_25, config)
    over_25_kelly = safe_kelly_calculation(under_to_over(probs.under_25), odds.over_25, config)
    under_35_kelly = safe_kelly_calculation(probs.under_35, odds.under_35, config)
    over_35_kelly = safe_kelly_calculation(under_to_over(probs.under_35), odds.over_35, config)

    # BTTS markets
    btts_yes_kelly = safe_kelly_calculation(probs.btts, odds.btts_yes, config)
    btts_no_kelly = safe_kelly_calculation(1 .- probs.btts, odds.btts_no, config)
    
    return Kelly.MatchFTKelly(
        home_kelly,
        draw_kelly,
        away_kelly,
        correct_score_kelly,
        under_05_kelly,
        over_05_kelly,
        under_15_kelly,
        over_15_kelly,
        under_25_kelly,
        over_25_kelly,
        under_35_kelly,
        over_35_kelly,
        btts_yes_kelly,
        btts_no_kelly
    )
end

"""
    apply_kelly_to_match(
        probs::MatchLineProbs,
        odds::MatchLineOdds,
        config::Kelly.Config
    ) -> Kelly.MatchLineKelly

Apply Bayesian Kelly calculations to all markets for both HT and FT periods.
"""
function apply_kelly_to_match(
    probs,  # Your probability structure with .ht and .ft
    odds,   # Your odds structure with .ht and .ft
    config::Kelly.Config
)
    ht_kelly = apply_kelly_to_ht_markets(probs.ht, odds.ht, config)
    ft_kelly = apply_kelly_to_ft_markets(probs.ft, odds.ft, config)
    
    return Kelly.MatchLineKelly(ht_kelly, ft_kelly)
end

"""
    process_matches_kelly(
        matches_probs::Dict{Int64, MatchLineProbs},
        matches_odds::Dict{Int64, MatchLineOdds},
        config::Kelly.Config
    ) -> Dict{Int64, Kelly.MatchLineKelly}

Process multiple matches and apply Kelly calculations to all markets.
Supports parallel processing using threading.
"""
function process_matches_kelly(
    matches_probs::Dict,
    matches_odds::Dict,
    config::Kelly.Config
)
    match_ids = collect(keys(matches_probs))
    n_matches = length(match_ids)
    n_threads = Threads.nthreads()
    
    # Each thread gets its own dictionary
    thread_results = Vector{Dict{Int64, Kelly.MatchLineKelly}}(undef, n_threads)
    
    Threads.@threads for tid in 1:n_threads
        # Each thread creates its own dictionary
        local_dict = Dict{Int64, Kelly.MatchLineKelly}()
        
        # Thread tid processes every n_threads-th match
        for idx in tid:n_threads:n_matches
            match_id = match_ids[idx]
            try
                if haskey(matches_odds, match_id)
                    match_kelly = apply_kelly_to_match(
                        matches_probs[match_id],
                        matches_odds[match_id],
                        config
                    )
                    local_dict[match_id] = match_kelly
                else
                    @warn "No odds found for match $match_id"
                end
            catch e
                @warn "Failed to process Kelly for match $match_id" exception=e
            end
        end
        
        thread_results[tid] = local_dict
    end
    
    # Merge all thread dictionaries
    return merge(thread_results...)
end


