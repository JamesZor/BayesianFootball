##################################################
#  Types for kelly
##################################################
#
module Kelly
const KellyChain = Union{Nothing, Vector{Float64}}
  # Type aliases matching Results module
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}

  struct Config 
    market_commission_percent::Number
    prob_value::Number 
  end
  
  struct MatchHTKelly
    home::KellyChain
    draw::KellyChain
    away::KellyChain
    correct_score::CorrectScore
    under_05::KellyChain
    over_05::KellyChain
    under_15::KellyChain
    over_15::KellyChain
    under_25::KellyChain
    over_25::KellyChain
  end
  
  struct MatchFTKelly
    home::KellyChain
    draw::KellyChain
    away::KellyChain
    correct_score::CorrectScore
    under_05::KellyChain
    over_05::KellyChain
    under_15::KellyChain
    over_15::KellyChain
    under_25::KellyChain
    over_25::KellyChain
    under_35::KellyChain
    over_35::KellyChain
    btts_yes::KellyChain
    btts_no::KellyChain
  end
  
  struct MatchLineKelly
    ht::MatchHTKelly
    ft::MatchFTKelly
  end
end

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


"""
ROI Calculation System with Kelly Quantile-based Betting
Computes ROI for all betting lines using Kelly fractions selected via quantiles
"""

using Statistics

##################################################
# ROI Types (mirrors Kelly structure but with MaybeFloat)
##################################################
module ROI
    const MaybeFloat = Union{Float64, Nothing}
    const CorrectScoreROI = Dict{Union{Tuple{Int,Int}, String}, MaybeFloat}
    
    struct MatchHTROI
        home::MaybeFloat
        draw::MaybeFloat
        away::MaybeFloat
        correct_score::CorrectScoreROI
        under_05::MaybeFloat
        over_05::MaybeFloat
        under_15::MaybeFloat
        over_15::MaybeFloat
        under_25::MaybeFloat
        over_25::MaybeFloat
    end
    
    struct MatchFTROI
        home::MaybeFloat
        draw::MaybeFloat
        away::MaybeFloat
        correct_score::CorrectScoreROI
        under_05::MaybeFloat
        over_05::MaybeFloat
        under_15::MaybeFloat
        over_15::MaybeFloat
        under_25::MaybeFloat
        over_25::MaybeFloat
        under_35::MaybeFloat
        over_35::MaybeFloat
        btts_yes::MaybeFloat
        btts_no::MaybeFloat
    end
    
    struct MatchLineROI
        ht::MatchHTROI
        ft::MatchFTROI
    end
    
    # Configuration for ROI calculations
    struct Config
        kelly_quantile::Float64  # e.g., 0.5 for median, 0.25 for conservative
        min_kelly_threshold::Float64  # minimum kelly to place bet (e.g., 0.01)
        commission_percent::Float64  # betting commission
    end
end

##################################################
# Kelly Quantile Selection
##################################################

"""
    select_kelly_fraction(
        kelly_chain::Union{Vector{Float64}, Nothing},
        quantile_c::Float64
    ) -> Union{Float64, Nothing}

Select the Kelly fraction at the specified quantile from the chain.
Returns nothing if chain is nothing, empty, or contains NaN/Inf values.
"""
function select_kelly_fraction(
    kelly_chain::Union{Vector{Float64}, Nothing},
    quantile_c::Float64
)
    if isnothing(kelly_chain) || isempty(kelly_chain)
        return nothing
    end
    
    # Filter out NaN, Inf, and -Inf values
    clean_chain = filter(x -> isfinite(x), kelly_chain)
    
    # Return nothing if no valid values remain
    if isempty(clean_chain)
        return nothing
    end
    
    # Also filter out negative values (Kelly fractions should be non-negative)
    clean_chain = filter(x -> x >= 0.0, clean_chain)
    
    if isempty(clean_chain)
        return nothing
    end
    
    return quantile(clean_chain, quantile_c)
end

##################################################
# ROI Calculation Functions
##################################################

"""
    calculate_single_bet_roi(
        bet_won::Union{Bool, Nothing},
        odds::Union{Float64, Nothing},
        kelly_fraction::Union{Float64, Nothing},
        commission_percent::Float64
    ) -> Union{Float64, Nothing}

Calculate ROI for a single bet given the outcome, odds, and Kelly fraction.
Returns nothing if any required input is nothing.
"""
function calculate_single_bet_roi(
    bet_won::Union{Bool, Nothing},
    odds::Union{Float64, Nothing},
    kelly_fraction::Union{Float64, Nothing},
    commission_percent::Float64
)
    # Check if we have all necessary data
    if isnothing(bet_won) || isnothing(odds) || isnothing(kelly_fraction)
        return nothing
    end
    
    # No bet placed if Kelly fraction is 0
    if kelly_fraction <= 0.0
        return nothing
    end
    
    # Calculate effective odds after commission
    effective_odds = 1.0 + (odds - 1.0) * (1.0 - commission_percent)
    
    # Calculate ROI
    if bet_won
        # Profit = (effective_odds - 1) * kelly_fraction
        # ROI = Profit / Stake = (effective_odds - 1) * kelly_fraction / kelly_fraction
        roi = effective_odds - 1.0
    else
        # Lost the entire stake
        roi = -1.0
    end
    
    return roi
end

"""
    calculate_market_roi(
        result::Union{Bool, Nothing},
        odds::Union{Float64, Nothing},
        kelly_chain::Union{Vector{Float64}, Nothing},
        config::ROI.Config
    ) -> Union{Float64, Nothing}

Calculate ROI for a market using quantile-selected Kelly fraction.
"""
function calculate_market_roi(
    result::Union{Bool, Nothing},
    odds::Union{Float64, Nothing},
    kelly_chain::Union{Vector{Float64}, Nothing},
    config::ROI.Config
)
    # Select Kelly fraction at specified quantile
    kelly_fraction = select_kelly_fraction(kelly_chain, config.kelly_quantile)
    
    # Check if Kelly meets minimum threshold
    if !isnothing(kelly_fraction) && kelly_fraction < config.min_kelly_threshold
        return nothing  # No bet placed
    end
    
    # Calculate ROI
    return calculate_single_bet_roi(
        result,
        odds,
        kelly_fraction,
        config.commission_percent
    )
end

##################################################
# Process Correct Score ROIs
##################################################

function process_ht_correct_score_roi(
    results::Dict,
    odds::Dict,
    kelly::Dict,
    config::ROI.Config
)
    roi_dict = ROI.CorrectScoreROI()
    
    ht_keys = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        "any_unquoted"
    ]
    
    for key in ht_keys
        if haskey(results, key) && haskey(odds, key) && haskey(kelly, key)
            roi_dict[key] = calculate_market_roi(
                results[key],
                odds[key],
                kelly[key],
                config
            )
        else
            roi_dict[key] = nothing
        end
    end
    
    return roi_dict
end

function process_ft_correct_score_roi(
    results::Dict,
    odds::Dict,
    kelly::Dict,
    config::ROI.Config
)
    roi_dict = ROI.CorrectScoreROI()
    
    ft_keys = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2),
        (0,3), (1,3), (2,3), (3,3),
        "other_home_win", "other_away_win", "other_draw"
    ]
    
    for key in ft_keys
        if haskey(results, key) && haskey(odds, key) && haskey(kelly, key)
            roi_dict[key] = calculate_market_roi(
                results[key],
                odds[key],
                kelly[key],
                config
            )
        else
            roi_dict[key] = nothing
        end
    end
    
    return roi_dict
end

##################################################
# Calculate ROI for All Markets
##################################################

"""
    calculate_ht_roi(
        results::MatchHTResults,
        odds::MatchHTOdds,
        kelly::Kelly.MatchHTKelly,
        config::ROI.Config
    ) -> ROI.MatchHTROI

Calculate ROI for all HT markets.
"""
function calculate_ht_roi(
    results,  # MatchHTResults
    odds,     # MatchHTOdds structure
    kelly,    # Kelly.MatchHTKelly
    config::ROI.Config
)
    # Main markets
    home_roi = calculate_market_roi(results.home, odds.home, kelly.home, config)
    draw_roi = calculate_market_roi(results.draw, odds.draw, kelly.draw, config)
    away_roi = calculate_market_roi(results.away, odds.away, kelly.away, config)
    
    # Correct scores
    correct_score_roi = process_ht_correct_score_roi(
        results.correct_score,
        odds.correct_score,
        kelly.correct_score,
        config
    )
    
    # Under/Over markets
    # For under markets, result is under_xx from results
    under_05_roi = calculate_market_roi(results.under_05, odds.under_05, kelly.under_05, config)
    under_15_roi = calculate_market_roi(results.under_15, odds.under_15, kelly.under_15, config)
    under_25_roi = calculate_market_roi(results.under_25, odds.under_25, kelly.under_25, config)
    
    # For over markets, result is NOT under_xx
    over_05_roi = calculate_market_roi(!isnothing(results.under_05) ? !results.under_05 : nothing, 
                                       odds.over_05, kelly.over_05, config)
    over_15_roi = calculate_market_roi(!isnothing(results.under_15) ? !results.under_15 : nothing,
                                       odds.over_15, kelly.over_15, config)
    over_25_roi = calculate_market_roi(!isnothing(results.under_25) ? !results.under_25 : nothing,
                                       odds.over_25, kelly.over_25, config)
    
    return ROI.MatchHTROI(
        home_roi,
        draw_roi,
        away_roi,
        correct_score_roi,
        under_05_roi,
        over_05_roi,
        under_15_roi,
        over_15_roi,
        under_25_roi,
        over_25_roi
    )
end

"""
    calculate_ft_roi(
        results::MatchFTResults,
        odds::MatchFTOdds,
        kelly::Kelly.MatchFTKelly,
        config::ROI.Config
    ) -> ROI.MatchFTROI

Calculate ROI for all FT markets.
"""
function calculate_ft_roi(
    results,  # MatchFTResults
    odds,     # MatchFTOdds structure
    kelly,    # Kelly.MatchFTKelly
    config::ROI.Config
)
    # Main markets
    home_roi = calculate_market_roi(results.home, odds.home, kelly.home, config)
    draw_roi = calculate_market_roi(results.draw, odds.draw, kelly.draw, config)
    away_roi = calculate_market_roi(results.away, odds.away, kelly.away, config)
    
    # Correct scores
    correct_score_roi = process_ft_correct_score_roi(
        results.correct_score,
        odds.correct_score,
        kelly.correct_score,
        config
    )
    
    # Under/Over markets
    under_05_roi = calculate_market_roi(results.under_05, odds.under_05, kelly.under_05, config)
    under_15_roi = calculate_market_roi(results.under_15, odds.under_15, kelly.under_15, config)
    under_25_roi = calculate_market_roi(results.under_25, odds.under_25, kelly.under_25, config)
    under_35_roi = calculate_market_roi(results.under_35, odds.under_35, kelly.under_35, config)
    
    over_05_roi = calculate_market_roi(!isnothing(results.under_05) ? !results.under_05 : nothing,
                                       odds.over_05, kelly.over_05, config)
    over_15_roi = calculate_market_roi(!isnothing(results.under_15) ? !results.under_15 : nothing,
                                       odds.over_15, kelly.over_15, config)
    over_25_roi = calculate_market_roi(!isnothing(results.under_25) ? !results.under_25 : nothing,
                                       odds.over_25, kelly.over_25, config)
    over_35_roi = calculate_market_roi(!isnothing(results.under_35) ? !results.under_35 : nothing,
                                       odds.over_35, kelly.over_35, config)
    
    # BTTS markets
    btts_yes_roi = calculate_market_roi(results.btts, odds.btts_yes, kelly.btts_yes, config)
    btts_no_roi = calculate_market_roi(!isnothing(results.btts) ? !results.btts : nothing,
                                       odds.btts_no, kelly.btts_no, config)
    
    return ROI.MatchFTROI(
        home_roi,
        draw_roi,
        away_roi,
        correct_score_roi,
        under_05_roi,
        over_05_roi,
        under_15_roi,
        over_15_roi,
        under_25_roi,
        over_25_roi,
        under_35_roi,
        over_35_roi,
        btts_yes_roi,
        btts_no_roi
    )
end

"""
    calculate_match_roi(
        results::MatchLinesResults,
        odds::MatchLineOdds,
        kelly::Kelly.MatchLineKelly,
        config::ROI.Config
    ) -> ROI.MatchLineROI

Calculate ROI for all markets in a match (both HT and FT).
"""
function calculate_match_roi(
    results,  # MatchLinesResults
    odds,     # MatchLineOdds
    kelly,    # Kelly.MatchLineKelly
    config::ROI.Config
)
    ht_roi = calculate_ht_roi(results.ht, odds.ht, kelly.ht, config)
    ft_roi = calculate_ft_roi(results.ft, odds.ft, kelly.ft, config)
    
    return ROI.MatchLineROI(ht_roi, ft_roi)
end

"""
    process_matches_roi(
        matches_results::Dict{Int64, MatchLinesResults},
        matches_odds::Dict{Int64, MatchLineOdds},
        matches_kelly::Dict{Int64, Kelly.MatchLineKelly},
        config::ROI.Config
    ) -> Dict{Int64, ROI.MatchLineROI}

Process ROI calculations for multiple matches using threading.
"""
function process_matches_roi(
    matches_results::Dict,
    matches_odds::Dict,
    matches_kelly::Dict,
    config::ROI.Config
)
    match_ids = collect(keys(matches_results))
    n_matches = length(match_ids)
    n_threads = Threads.nthreads()
    
    # Each thread gets its own dictionary
    thread_results = Vector{Dict{Int64, ROI.MatchLineROI}}(undef, n_threads)
    
    Threads.@threads for tid in 1:n_threads
        local_dict = Dict{Int64, ROI.MatchLineROI}()
        
        for idx in tid:n_threads:n_matches
            match_id = match_ids[idx]
            try
                if haskey(matches_odds, match_id) && haskey(matches_kelly, match_id)
                    match_roi = calculate_match_roi(
                        matches_results[match_id],
                        matches_odds[match_id],
                        matches_kelly[match_id],
                        config
                    )
                    local_dict[match_id] = match_roi
                else
                    @warn "Missing odds or Kelly for match $match_id"
                end
            catch e
                @warn "Failed to calculate ROI for match $match_id" exception=e
            end
        end
        
        thread_results[tid] = local_dict
    end
    
    return merge(thread_results...)
end

##################################################
# Analysis Functions
##################################################

"""
    analyze_roi_performance(
        roi_dict::Dict{Int64, ROI.MatchLineROI}
    ) -> NamedTuple

Analyze overall ROI performance across all matches and markets.
"""
function analyze_roi_performance(roi_dict::Dict{Int64, ROI.MatchLineROI})
    # Collect all ROIs by market type
    ht_home_rois = Float64[]
    ht_draw_rois = Float64[]
    ht_away_rois = Float64[]
    ft_home_rois = Float64[]
    ft_draw_rois = Float64[]
    ft_away_rois = Float64[]
    ft_btts_yes_rois = Float64[]
    ft_btts_no_rois = Float64[]
    
    for (match_id, match_roi) in roi_dict
        # HT markets
        !isnothing(match_roi.ht.home) && push!(ht_home_rois, match_roi.ht.home)
        !isnothing(match_roi.ht.draw) && push!(ht_draw_rois, match_roi.ht.draw)
        !isnothing(match_roi.ht.away) && push!(ht_away_rois, match_roi.ht.away)
        
        # FT markets
        !isnothing(match_roi.ft.home) && push!(ft_home_rois, match_roi.ft.home)
        !isnothing(match_roi.ft.draw) && push!(ft_draw_rois, match_roi.ft.draw)
        !isnothing(match_roi.ft.away) && push!(ft_away_rois, match_roi.ft.away)
        !isnothing(match_roi.ft.btts_yes) && push!(ft_btts_yes_rois, match_roi.ft.btts_yes)
        !isnothing(match_roi.ft.btts_no) && push!(ft_btts_no_rois, match_roi.ft.btts_no)
    end
    
    # Calculate statistics
    calc_stats = function(rois)
        if isempty(rois)
            return (mean=nothing, median=nothing, std=nothing, count=0, total_roi=nothing)
        end
        return (
            mean=mean(rois),
            median=median(rois),
            std=std(rois),
            count=length(rois),
            total_roi=sum(rois)
        )
    end
    
    return (
        ht_home = calc_stats(ht_home_rois),
        ht_draw = calc_stats(ht_draw_rois),
        ht_away = calc_stats(ht_away_rois),
        ft_home = calc_stats(ft_home_rois),
        ft_draw = calc_stats(ft_draw_rois),
        ft_away = calc_stats(ft_away_rois),
        ft_btts_yes = calc_stats(ft_btts_yes_rois),
        ft_btts_no = calc_stats(ft_btts_no_rois)
    )
end

##################################################
# Example Usage
##################################################

function example_usage()
    # Configuration with quantile selection
    config = ROI.Config(
        0.5,    # Use median Kelly (50th percentile) - moderate risk
        0.01,   # Minimum Kelly threshold of 1%
        0.02    # 2% commission
    )
    
    # For conservative betting, use lower quantile
    conservative_config = ROI.Config(0.25, 0.01, 0.02)
    
    # For aggressive betting, use higher quantile
    aggressive_config = ROI.Config(0.75, 0.01, 0.02)
    
    # Process single match
    # match_roi = calculate_match_roi(match_results, match_odds, match_kelly, config)
    
    # Process multiple matches
    # all_roi = process_matches_roi(matches_results, matches_odds, matches_kelly, config)
    
    # Analyze performance
    # performance = analyze_roi_performance(all_roi)
    
    println("ROI calculation system configured with quantile: ", config.kelly_quantile)
end




######



"""
Cumulative Wealth Tracking System
Tracks wealth evolution across different betting markets with visualization
"""

using DataFrames
using Dates
using Plots
using Statistics

##################################################
# Wealth Tracking Types
##################################################

module WealthTracking
    # Market line groups for tracking
    const MARKET_LINES_HT = [
        :home, :draw, :away,
        :under_05, :over_05,
        :under_15, :over_15,
        :under_25, :over_25
    ]
    
    const MARKET_LINES_FT = [
        :home, :draw, :away,
        :under_05, :over_05,
        :under_15, :over_15,
        :under_25, :over_25,
        :under_35, :over_35,
        :btts_yes, :btts_no
    ]
    
    const MARKET_GROUPS = Dict(
        :main_1x2 => [:home, :draw, :away],
        :under_over_05 => [:under_05, :over_05],
        :under_over_15 => [:under_15, :over_15],
        :under_over_25 => [:under_25, :over_25],
        :under_over_35 => [:under_35, :over_35],  # FT only
        :btts => [:btts_yes, :btts_no],  # FT only
        :correct_score => :correct_score  # Special handling
    )
end

##################################################
# Extract ROI values into flat structure
##################################################

"""
    extract_roi_values(match_roi::ROI.MatchLineROI) -> NamedTuple

Extract all ROI values from nested structure into a flat NamedTuple.
"""
function extract_roi_values(match_roi)
    # Extract HT values
    ht_values = Dict{Symbol, Union{Float64, Nothing}}()
    for field in WealthTracking.MARKET_LINES_HT
        value = getfield(match_roi.ht, field)
        ht_values[Symbol("ht_", field)] = value
    end
    
    # Extract FT values
    ft_values = Dict{Symbol, Union{Float64, Nothing}}()
    for field in WealthTracking.MARKET_LINES_FT
        value = getfield(match_roi.ft, field)
        ft_values[Symbol("ft_", field)] = value
    end
    
    # Extract correct scores (aggregate ROI for all CS bets)
    ht_cs_roi = aggregate_correct_score_roi(match_roi.ht.correct_score)
    ft_cs_roi = aggregate_correct_score_roi(match_roi.ft.correct_score)
    
    ht_values[:ht_correct_score] = ht_cs_roi
    ft_values[:ft_correct_score] = ft_cs_roi
    
    return merge(ht_values, ft_values)
end

"""
    aggregate_correct_score_roi(cs_dict::Dict) -> Union{Float64, Nothing}

Aggregate ROI from all correct score bets placed in a match.
Assumes equal stakes on each bet that was placed.
"""
function aggregate_correct_score_roi(cs_dict::Dict)
    rois = Float64[]
    for (key, roi) in cs_dict
        if !isnothing(roi)
            push!(rois, roi)
        end
    end
    
    if isempty(rois)
        return nothing
    else
        # Average ROI across all CS bets placed
        return mean(rois)
    end
end

##################################################
# Calculate Cumulative Wealth
##################################################

"""
    calculate_cumulative_wealth(
        all_roi::Dict{Int64, ROI.MatchLineROI},
        target_matches::DataFrame,
        initial_wealth::Float64 = 100.0
    ) -> DataFrame

Calculate cumulative wealth for each market line across all matches.
Returns a DataFrame with match info and wealth evolution.
"""
function calculate_cumulative_wealth(
    all_roi::Dict,
    target_matches::DataFrame,
    initial_wealth::Float64 = 100.0
)
    # Sort matches by date and validate dates
    sorted_matches = sort(target_matches, :match_date)
    
    # Check for invalid dates
    min_date = minimum(sorted_matches.match_date)
    max_date = maximum(sorted_matches.match_date)
    if min_date < Date(2000, 1, 1) || max_date > Date(2030, 1, 1)
        println("Warning: Found suspicious dates - Min: $min_date, Max: $max_date")
        # Filter out invalid dates
        sorted_matches = filter(row -> Date(2000, 1, 1) <= row.match_date <= Date(2030, 1, 1), sorted_matches)
        println("After filtering: $(nrow(sorted_matches)) matches remain")
    end
    
    # Initialize result DataFrame
    result_df = DataFrame()
    result_df.match_id = sorted_matches.match_id
    result_df.match_date = convert(Vector{Date}, sorted_matches.match_date)  # Ensure proper Date type
    result_df.home_team = sorted_matches.home_team
    result_df.away_team = sorted_matches.away_team
    
    # Get all market columns
    all_markets = [
        [Symbol("ht_", m) for m in WealthTracking.MARKET_LINES_HT];
        [Symbol("ft_", m) for m in WealthTracking.MARKET_LINES_FT];
        [:ht_correct_score, :ft_correct_score]
    ]
    
    # Initialize ROI and wealth columns
    for market in all_markets
        # ROI column
        roi_col = Symbol(market, "_roi")
        result_df[!, roi_col] = Union{Float64, Missing}[missing for _ in 1:nrow(result_df)]
        
        # Wealth column
        wealth_col = Symbol(market, "_wealth")
        result_df[!, wealth_col] = Float64[initial_wealth for _ in 1:nrow(result_df)]
    end
    
    # Process each match
    for (idx, row) in enumerate(eachrow(sorted_matches))
        match_id = row.match_id
        
        if !haskey(all_roi, match_id)
            # No ROI data for this match, wealth stays the same
            if idx > 1
                for market in all_markets
                    wealth_col = Symbol(market, "_wealth")
                    result_df[idx, wealth_col] = result_df[idx-1, wealth_col]
                end
            end
            continue
        end
        
        # Extract ROI values
        roi_values = extract_roi_values(all_roi[match_id])
        
        # Update each market
        for market in all_markets
            roi_col = Symbol(market, "_roi")
            wealth_col = Symbol(market, "_wealth")
            
            # Get ROI for this market
            market_roi = get(roi_values, market, nothing)
            
            if !isnothing(market_roi)
                result_df[idx, roi_col] = market_roi
                
                # Calculate new wealth
                # Assuming unit stake (1% of wealth or fixed fraction)
                stake_fraction = 0.01  # This could be the actual Kelly fraction
                if idx > 1
                    prev_wealth = result_df[idx-1, wealth_col]
                    stake = prev_wealth * stake_fraction
                    profit = stake * market_roi
                    result_df[idx, wealth_col] = prev_wealth + profit
                else
                    stake = initial_wealth * stake_fraction
                    profit = stake * market_roi
                    result_df[idx, wealth_col] = initial_wealth + profit
                end
            else
                # No bet placed, wealth stays the same
                if idx > 1
                    result_df[idx, wealth_col] = result_df[idx-1, wealth_col]
                end
            end
        end
    end
    
    # Forward fill wealth for matches with no bets
    for market in all_markets
        wealth_col = Symbol(market, "_wealth")
        for idx in 2:nrow(result_df)
            if ismissing(result_df[idx, Symbol(market, "_roi")])
                result_df[idx, wealth_col] = result_df[idx-1, wealth_col]
            end
        end
    end
    
    return result_df
end

##################################################
# Advanced Wealth Calculation with Kelly Scaling
##################################################

"""
    calculate_cumulative_wealth_kelly(
        all_roi::Dict{Int64, ROI.MatchLineROI},
        matches_kelly::Dict{Int64, Kelly.MatchLineKelly},
        target_matches::DataFrame,
        config::ROI.Config,
        initial_wealth::Float64 = 100.0,
        kelly_scale::Float64 = 0.25  # Scale Kelly fractions for safety
    ) -> DataFrame

Calculate cumulative wealth using actual Kelly fractions (scaled for safety).
"""
function calculate_cumulative_wealth_kelly(
    all_roi::Dict,
    matches_kelly::Dict,
    target_matches::DataFrame,
    config::ROI.Config,
    initial_wealth::Float64 = 100.0,
    kelly_scale::Float64 = 0.25  # Quarter Kelly for safety
)
    # Sort matches by date
    sorted_matches = sort(target_matches, :match_date)
    
    # Initialize result DataFrame
    result_df = DataFrame()
    result_df.match_id = sorted_matches.match_id
    result_df.match_date = sorted_matches.match_date
    result_df.home_team = sorted_matches.home_team
    result_df.away_team = sorted_matches.away_team
    
    # Get all market columns
    all_markets = [
        [Symbol("ht_", m) for m in WealthTracking.MARKET_LINES_HT];
        [Symbol("ft_", m) for m in WealthTracking.MARKET_LINES_FT];
        [:ht_correct_score, :ft_correct_score]
    ]
    
    # Initialize columns
    for market in all_markets
        result_df[!, Symbol(market, "_roi")] = Union{Float64, Missing}[missing for _ in 1:nrow(result_df)]
        result_df[!, Symbol(market, "_stake")] = Union{Float64, Missing}[missing for _ in 1:nrow(result_df)]
        result_df[!, Symbol(market, "_wealth")] = Float64[initial_wealth for _ in 1:nrow(result_df)]
    end
    
    # Track wealth for each market
    current_wealth = Dict{Symbol, Float64}()
    for market in all_markets
        current_wealth[market] = initial_wealth
    end
    
    # Process each match
    for (idx, row) in enumerate(eachrow(sorted_matches))
        match_id = row.match_id
        
        if !haskey(all_roi, match_id) || !haskey(matches_kelly, match_id)
            # No data, carry forward wealth
            for market in all_markets
                result_df[idx, Symbol(market, "_wealth")] = current_wealth[market]
            end
            continue
        end
        
        # Extract ROI and Kelly values
        roi_values = extract_roi_values(all_roi[match_id])
        kelly_match = matches_kelly[match_id]
        
        # Process each market
        for market in all_markets
            roi = get(roi_values, market, nothing)
            
            if !isnothing(roi)
                # Get Kelly fraction for this market
                kelly_fraction = get_kelly_for_market(kelly_match, market, config.kelly_quantile)
                
                if !isnothing(kelly_fraction) && kelly_fraction >= config.min_kelly_threshold
                    # Scale Kelly for safety
                    scaled_kelly = kelly_fraction * kelly_scale
                    
                    # Calculate stake and profit
                    stake = current_wealth[market] * scaled_kelly
                    profit = stake * roi
                    
                    # Update wealth
                    current_wealth[market] += profit
                    
                    # Record in DataFrame
                    result_df[idx, Symbol(market, "_roi")] = roi
                    result_df[idx, Symbol(market, "_stake")] = stake
                end
            end
            
            # Record current wealth
            result_df[idx, Symbol(market, "_wealth")] = current_wealth[market]
        end
    end
    
    return result_df
end

"""
    get_kelly_for_market(kelly_match, market::Symbol, quantile_c::Float64)

Extract Kelly fraction for a specific market.
"""
function get_kelly_for_market(kelly_match, market::Symbol, quantile_c::Float64)
    # Parse market symbol (e.g., :ft_home -> ft, home)
    market_str = string(market)
    parts = split(market_str, "_", limit=2)
    
    if length(parts) != 2
        return nothing
    end
    
    period = parts[1]
    market_type = parts[2]
    
    # Special handling for correct scores
    if market_type == "correct_score"
        return nothing  # Would need aggregate Kelly calculation
    end
    
    # Get the appropriate period
    period_kelly = period == "ht" ? kelly_match.ht : kelly_match.ft
    
    # Get the Kelly chain for this market
    kelly_chain = try
        getfield(period_kelly, Symbol(market_type))
    catch
        return nothing
    end
    
    # Select fraction at quantile
    return select_kelly_fraction(kelly_chain, quantile_c)
end

##################################################
# Visualization Functions
##################################################

"""
    plot_wealth_simple(
        wealth_df::DataFrame,
        columns::Vector{Symbol} = [:ft_home_wealth, :ft_draw_wealth, :ft_away_wealth];
        title::String = "Wealth Evolution",
        initial_wealth::Float64 = 100.0,
        date_range::Union{Nothing, Tuple{Date, Date}} = nothing
    ) -> Plots.Plot

Simple plot function where you directly specify which wealth columns to plot.
Optional date_range parameter to set consistent x-axis limits across multiple plots.
"""
function plot_wealth_simple(
    wealth_df::AbstractDataFrame,
    columns::Vector{Symbol} = [:ft_home_wealth, :ft_draw_wealth, :ft_away_wealth];
    title::String = "Wealth Evolution",
    initial_wealth::Float64 = 100.0,
    date_range::Union{Nothing, Tuple{Date, Date}} = nothing
)
    # Validate dates first
    dates = wealth_df.match_date
    valid_date_indices = [i for i in 1:length(dates) if !ismissing(dates[i]) && Date(2000,1,1) <= dates[i] <= Date(2030,1,1)]
    
    if isempty(valid_date_indices)
        println("Error: No valid dates found in the data")
        println("Date range: ", minimum(skipmissing(dates)), " to ", maximum(skipmissing(dates)))
        return plot(title="No valid data to plot")
    end
    
    # Filter to valid dates only
    wealth_df_clean = wealth_df[valid_date_indices, :]
    
    # Determine x-axis limits
    if !isnothing(date_range)
        xlims_tuple = Dates.value.([date_range[1], date_range[2]])
    else
        xlims_tuple = nothing
    end
    
    p = plot(
        title = title,
        xlabel = "Match Date",
        ylabel = "Wealth",
        legend = :outertopright,
        size = (1200, 600),
        grid = true,
        gridstyle = :dash,
        gridlinewidth = 0.5,
        xrotation = 45,  # Rotate x-axis labels for better readability
        bottom_margin = 5Plots.mm,  # Add margin for rotated labels
        xlims = xlims_tuple  # Set x-axis limits if provided
    )
    
    # Add horizontal line at initial wealth
    hline!([initial_wealth], label="Initial", color=:gray, linestyle=:dash, linewidth=1)
    
    # Plot each specified column
    for col in columns
        col_str = string(col)
        if col_str in names(wealth_df_clean)
            data = wealth_df_clean[!, col]
            dates_clean = wealth_df_clean.match_date
            
            # Filter out missing values
            valid_indices = .!ismissing.(data)
            if any(valid_indices)
                valid_dates = dates_clean[valid_indices]
                valid_data = Vector{Float64}(data[valid_indices])
                
                # Calculate ROI for label
                final_wealth = valid_data[end]
                roi_pct = round((final_wealth/initial_wealth - 1) * 100, digits=1)
                
                label = replace(col_str, "_wealth" => "") * " ($roi_pct%)"
                
                plot!(
                    p,
                    valid_dates,
                    valid_data,
                    label = label,
                    linewidth = 1,
                    marker = :circle,
                    markersize = 0.5
                )
            else
                println("Warning: No valid data for column $col_str")
            end
        else
            println("Warning: Column $col_str not found")
        end
    end
    
    return p
end

"""
    plot_wealth_comparison(
        wealth_dfs::Vector{DataFrame},
        labels::Vector{String},
        column::Symbol = :ft_home_wealth;
        title::String = "Wealth Comparison",
        initial_wealth::Float64 = 100.0
    ) -> Plots.Plot

Plot the same market across multiple wealth DataFrames for comparison.
Automatically uses consistent date range across all data.
"""
function plot_wealth_comparison(
    wealth_dfs::Vector{DataFrame},
    labels::Vector{String},
    column::Symbol = :ft_home_wealth;
    title::String = "Wealth Comparison",
    initial_wealth::Float64 = 100.0
)
    # Find overall date range
    all_dates = Date[]
    for df in wealth_dfs
        append!(all_dates, skipmissing(df.match_date))
    end
    
    if isempty(all_dates)
        return plot(title="No valid data to plot")
    end
    
    min_date = minimum(all_dates)
    max_date = maximum(all_dates)
    xlims_values = Dates.value.([min_date, max_date])
    
    p = plot(
        title = title * " - " * string(column),
        xlabel = "Match Date",
        ylabel = "Wealth",
        legend = :outertopright,
        size = (1200, 600),
        grid = true,
        gridstyle = :dash,
        gridlinewidth = 0.5,
        xrotation = 45,
        bottom_margin = 5Plots.mm,
        xlims = xlims_values  # Set consistent x-axis limits
    )
    
    # Add horizontal line at initial wealth
    hline!([initial_wealth], label="Initial", color=:gray, linestyle=:dash, linewidth=1)
    
    # Plot each DataFrame
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :cyan]
    
    for (i, (df, label)) in enumerate(zip(wealth_dfs, labels))
        col_str = string(column)
        if col_str in names(df)
            data = df[!, column]
            dates = df.match_date
            
            # Filter out missing values
            valid_indices = .!ismissing.(data) .& .!ismissing.(dates)
            if any(valid_indices)
                valid_dates = dates[valid_indices]
                valid_data = Vector{Float64}(data[valid_indices])
                
                # Calculate ROI for label
                final_wealth = valid_data[end]
                roi_pct = round((final_wealth/initial_wealth - 1) * 100, digits=1)
                
                plot_label = "$label ($roi_pct%)"
                color = colors[mod1(i, length(colors))]
                
                plot!(
                    p,
                    valid_dates,
                    valid_data,
                    label = plot_label,
                    linewidth = 2,
                    marker = :circle,
                    markersize = 2,
                    color = color
                )
            end
        end
    end
    
    return p
end

"""
    plot_wealth_evolution(
        wealth_df::DataFrame,
        market_group::Symbol = :main_1x2;
        title::String = "Wealth Evolution",
        initial_wealth::Float64 = 100.0
    ) -> Plots.Plot

Plot wealth evolution for a specific market group.
"""
function plot_wealth_evolution(
    wealth_df::DataFrame,
    market_group::Symbol = :main_1x2;
    title::String = "Wealth Evolution",
    initial_wealth::Float64 = 100.0
)
    # Get markets for this group
    markets = if market_group == :all_1x2
        [:ht_home, :ht_draw, :ht_away, :ft_home, :ft_draw, :ft_away]
    elseif haskey(WealthTracking.MARKET_GROUPS, market_group)
        group_markets = WealthTracking.MARKET_GROUPS[market_group]
        if group_markets == :correct_score
            [:ht_correct_score, :ft_correct_score]
        else
            # Add both HT and FT versions
            result = Symbol[]
            for m in group_markets
                # Check if market exists in both periods
                if m in WealthTracking.MARKET_LINES_HT
                    push!(result, Symbol("ht_", m))
                end
                if m in WealthTracking.MARKET_LINES_FT
                    push!(result, Symbol("ft_", m))
                end
            end
            result
        end
    else
        error("Unknown market group: $market_group")
    end
    
    println("Plotting markets: ", markets)  # Debug info
    
    # Create plot
    p = plot(
        title = title,
        xlabel = "Match Date",
        ylabel = "Wealth",
        legend = :outertopright,
        size = (1200, 600),
        grid = true,
        gridstyle = :dash,
        gridlinewidth = 0.5
    )
    
    # Add horizontal line at initial wealth
    hline!([initial_wealth], label="Initial", color=:gray, linestyle=:dash, linewidth=1)
    
    # Plot each market
    for market in markets
        wealth_col_str = string(market) * "_wealth"
        
        if wealth_col_str in names(wealth_df)
            println("Adding line for: ", wealth_col_str)  # Debug info
            
            # Get non-missing values for plotting
            data = wealth_df[!, wealth_col_str]
            dates = wealth_df.match_date
            
            # Filter out any missing values
            valid_indices = .!ismissing.(data)
            if any(valid_indices)
                plot!(
                    p,
                    dates[valid_indices],
                    Vector{Float64}(data[valid_indices]),
                    label = string(market),
                    linewidth = 2,
                    marker = :circle,
                    markersize = 2
                )
            else
                println("Warning: No valid data for ", wealth_col_str)
            end
        else
            println("Warning: Column ", wealth_col_str, " not found in DataFrame")
        end
    end
    
    return p
end

"""
    plot_all_market_groups(
        wealth_df::DataFrame;
        initial_wealth::Float64 = 100.0,
        config_label::String = ""
    ) -> NamedTuple

Create plots for all market groups.
"""
function plot_all_market_groups(
    wealth_df::DataFrame;
    initial_wealth::Float64 = 100.0,
    config_label::String = ""
)
    plots_dict = Dict{Symbol, Plots.Plot}()
    
    # Main 1X2 markets - use simple plot function for clarity
    plots_dict[:main_1x2] = plot_wealth_simple(
        wealth_df,
        [:ft_home_wealth, :ft_draw_wealth, :ft_away_wealth],
        title = "FT 1X2 Markets" * (isempty(config_label) ? "" : " - $config_label"),
        initial_wealth = initial_wealth
    )
    
    plots_dict[:main_1x2_ht] = plot_wealth_simple(
        wealth_df,
        [:ht_home_wealth, :ht_draw_wealth, :ht_away_wealth],
        title = "HT 1X2 Markets" * (isempty(config_label) ? "" : " - $config_label"),
        initial_wealth = initial_wealth
    )
    
    # Under/Over 0.5
    plots_dict[:under_over_05] = plot_wealth_simple(
        wealth_df,
        [:ft_under_05_wealth, :ft_over_05_wealth],
        title = "FT Under/Over 0.5 Goals" * (isempty(config_label) ? "" : " - $config_label"),
        initial_wealth = initial_wealth
    )
    
    # Under/Over 1.5
    plots_dict[:under_over_15] = plot_wealth_simple(
        wealth_df,
        [:ft_under_15_wealth, :ft_over_15_wealth],
        title = "FT Under/Over 1.5 Goals" * (isempty(config_label) ? "" : " - $config_label"),
        initial_wealth = initial_wealth
    )
    
    # Under/Over 2.5
    plots_dict[:under_over_25] = plot_wealth_simple(
        wealth_df,
        [:ft_under_25_wealth, :ft_over_25_wealth],
        title = "FT Under/Over 2.5 Goals" * (isempty(config_label) ? "" : " - $config_label"),
        initial_wealth = initial_wealth
    )
    
    # Under/Over 3.5 (FT only)
    if :ft_under_35_wealth in Symbol.(names(wealth_df))
        plots_dict[:under_over_35] = plot_wealth_simple(
            wealth_df,
            [:ft_under_35_wealth, :ft_over_35_wealth],
            title = "FT Under/Over 3.5 Goals" * (isempty(config_label) ? "" : " - $config_label"),
            initial_wealth = initial_wealth
        )
    end
    
    # BTTS markets (FT only)
    if :ft_btts_yes_wealth in Symbol.(names(wealth_df))
        plots_dict[:btts] = plot_wealth_simple(
            wealth_df,
            [:ft_btts_yes_wealth, :ft_btts_no_wealth],
            title = "BTTS Markets" * (isempty(config_label) ? "" : " - $config_label"),
            initial_wealth = initial_wealth
        )
    end
    
    # Correct scores
    if :ft_correct_score_wealth in Symbol.(names(wealth_df))
        plots_dict[:correct_score] = plot_wealth_simple(
            wealth_df,
            [:ht_correct_score_wealth, :ft_correct_score_wealth],
            title = "Correct Score Markets" * (isempty(config_label) ? "" : " - $config_label"),
            initial_wealth = initial_wealth
        )
    end
    
    # Combined plot showing best performers
    plots_dict[:summary] = create_summary_plot(wealth_df, initial_wealth, config_label)
    
    return (; plots_dict...)
end

"""
    create_summary_plot(wealth_df::DataFrame, initial_wealth::Float64, config_label::String)

Create a summary plot showing best and worst performing markets.
"""
function create_summary_plot(wealth_df::DataFrame, initial_wealth::Float64, config_label::String)
    # Calculate final wealth for each market
    final_wealths = Dict{Symbol, Float64}()
    
    for col in names(wealth_df)
        if endswith(string(col), "_wealth")
            market = Symbol(replace(string(col), "_wealth" => ""))
            final_wealths[market] = wealth_df[end, col]
        end
    end
    
    # Sort by performance
    sorted_markets = sort(collect(final_wealths), by=x->x[2], rev=true)
    
    # Take top 5 and bottom 5
    n_markets = min(5, length(sorted_markets) ÷ 2)
    best_markets = sorted_markets[1:n_markets]
    worst_markets = sorted_markets[end-n_markets+1:end]
    
    # Create plot
    p = plot(
        title = "Best & Worst Performing Markets" * (isempty(config_label) ? "" : " - $config_label"),
        xlabel = "Match Date",
        ylabel = "Wealth",
        legend = :outertopright,
        size = (1400, 700),
        grid = true
    )
    
    hline!([initial_wealth], label="Initial", color=:gray, linestyle=:dash, linewidth=1)
    
    # Plot best performers
    for (market, final_wealth) in best_markets
        wealth_col = Symbol(market, "_wealth")
        roi_pct = round((final_wealth/initial_wealth - 1) * 100, digits=1)
        plot!(
            wealth_df.match_date,
            wealth_df[!, wealth_col],
            label = "$market (+$roi_pct%)",
            linewidth = 2.5,
            marker = :circle,
            markersize = 2
        )
    end
    
    # Plot worst performers
    for (market, final_wealth) in worst_markets
        wealth_col = Symbol(market, "_wealth")
        roi_pct = round((final_wealth/initial_wealth - 1) * 100, digits=1)
        plot!(
            wealth_df.match_date,
            wealth_df[!, wealth_col],
            label = "$market ($roi_pct%)",
            linewidth = 1.5,
            linestyle = :dash,
            marker = :square,
            markersize = 2
        )
    end
    
    return p
end

##################################################
# Compare Different Configurations
##################################################

"""
    compare_configurations(
        matches_results::Dict,
        matches_odds::Dict,
        matches_kelly::Dict,
        target_matches::DataFrame,
        configs::Vector{ROI.Config};
        config_labels::Vector{String} = String[],
        initial_wealth::Float64 = 100.0
    ) -> NamedTuple

Compare performance across different ROI configurations.
"""
function compare_configurations(
    matches_results::Dict,
    matches_odds::Dict,
    matches_kelly::Dict,
    target_matches::DataFrame,
    configs::Vector,
    config_labels::Vector{String} = String[];
    initial_wealth::Float64 = 100.0,
    kelly_scale::Float64 = 0.25
)
    if isempty(config_labels)
        config_labels = ["Config $i" for i in 1:length(configs)]
    end
    
    all_results = Dict{String, Any}()
    
    for (i, config) in enumerate(configs)
        label = config_labels[i]
        
        # Calculate ROI for this config
        all_roi = process_matches_roi(
            matches_results,
            matches_odds,
            matches_kelly,
            config
        )
        
        # Calculate cumulative wealth
        wealth_df = calculate_cumulative_wealth_kelly(
            all_roi,
            matches_kelly,
            target_matches,
            config,
            initial_wealth,
            kelly_scale
        )
        
        # Create plots
        plots = plot_all_market_groups(
            wealth_df,
            initial_wealth = initial_wealth,
            config_label = label
        )
        
        all_results[label] = (
            roi = all_roi,
            wealth = wealth_df,
            plots = plots,
            config = config
        )
    end
    
    # Create comparison plot
    comparison_plot = create_comparison_plot(all_results, initial_wealth)
    
    return (
        results = all_results,
        comparison = comparison_plot
    )
end

"""
    create_comparison_plot(all_results::Dict, initial_wealth::Float64)

Create a plot comparing different configurations.
"""
function create_comparison_plot(all_results::Dict, initial_wealth::Float64)
    p = plot(
        title = "Configuration Comparison - 1X2 Markets",
        xlabel = "Match Date",
        ylabel = "Wealth",
        legend = :outertopright,
        size = (1400, 700),
        grid = true
    )
    
    hline!([initial_wealth], label="Initial", color=:gray, linestyle=:dash, linewidth=1)
    
    colors = [:blue, :red, :green, :orange, :purple, :brown]
    
    for (i, (label, result)) in enumerate(all_results)
        wealth_df = result.wealth
        color = colors[mod1(i, length(colors))]
        
        # Plot FT home for each config as example
        if :ft_home_wealth in names(wealth_df)
            final_wealth = wealth_df[end, :ft_home_wealth]
            roi_pct = round((final_wealth/initial_wealth - 1) * 100, digits=1)
            
            plot!(
                wealth_df.match_date,
                wealth_df.ft_home_wealth,
                label = "$label (FT Home): $roi_pct%",
                color = color,
                linewidth = 2,
                marker = :circle,
                markersize = 2
            )
        end
    end
    
    return p
end

##################################################
# Diagnostic Functions
##################################################

"""
    diagnose_date_issues(df::DataFrame)

Check for date formatting issues in the DataFrame.
"""
function diagnose_date_issues(df::DataFrame)
    if !(:match_date in Symbol.(names(df)))
        println("Error: No match_date column found")
        return
    end
    
    dates = df.match_date
    println("Date column type: ", typeof(dates))
    println("Total rows: ", length(dates))
    
    # Check for missing dates
    n_missing = count(ismissing, dates)
    println("Missing dates: ", n_missing)
    
    # Get non-missing dates
    valid_dates = skipmissing(dates)
    if !isempty(valid_dates)
        min_date = minimum(valid_dates)
        max_date = maximum(valid_dates)
        println("Date range: $min_date to $max_date")
        
        # Check for suspicious dates
        suspicious = [d for d in valid_dates if d < Date(2000,1,1) || d > Date(2030,1,1)]
        if !isempty(suspicious)
            println("⚠ Found $(length(suspicious)) suspicious dates:")
            for (i, d) in enumerate(suspicious[1:min(5, length(suspicious))])
                println("  - $d")
            end
        else
            println("✓ All dates appear valid")
        end
        
        # Show date distribution
        years = [year(d) for d in valid_dates]
        year_counts = Dict{Int, Int}()
        for y in years
            year_counts[y] = get(year_counts, y, 0) + 1
        end
        println("\nMatches by year:")
        for (y, c) in sort(collect(year_counts))
            println("  $y: $c matches")
        end
    end
end

"""
    fix_wealth_df_dates!(wealth_df::DataFrame)

Fix any date issues in the wealth DataFrame in-place.
"""
function fix_wealth_df_dates!(wealth_df::DataFrame)
    # Check current state
    println("Before fixing:")
    diagnose_date_issues(wealth_df)
    
    # Filter out invalid dates
    valid_rows = [i for i in 1:nrow(wealth_df) 
                  if !ismissing(wealth_df.match_date[i]) && 
                     Date(2000,1,1) <= wealth_df.match_date[i] <= Date(2030,1,1)]
    
    if length(valid_rows) < nrow(wealth_df)
        println("\nRemoving $(nrow(wealth_df) - length(valid_rows)) rows with invalid dates")
        filter!(row -> !ismissing(row.match_date) && 
                      Date(2000,1,1) <= row.match_date <= Date(2030,1,1), 
                wealth_df)
    end
    
    # Sort by date
    sort!(wealth_df, :match_date)
    
    println("\nAfter fixing:")
    println("Rows remaining: ", nrow(wealth_df))
    if nrow(wealth_df) > 0
        println("Date range: ", minimum(wealth_df.match_date), " to ", maximum(wealth_df.match_date))
    end
    
    return wealth_df
end

##################################################
# Example Usage
##################################################

function example_usage()
    # Example with different configurations
    configs = [
        ROI.Config(0.25, 0.01, 0.02),  # Conservative (25th percentile)
        ROI.Config(0.50, 0.01, 0.02),  # Moderate (median)
        ROI.Config(0.75, 0.01, 0.02),  # Aggressive (75th percentile)
    ]
    
    config_labels = ["Conservative (Q25)", "Moderate (Q50)", "Aggressive (Q75)"]
    
    # Compare configurations
    # comparison = compare_configurations(
    #     matches_results,
    #     matches_odds,
    #     matches_kelly,
    #     target_matches,
    #     configs,
    #     config_labels,
    #     initial_wealth = 100.0,
    #     kelly_scale = 0.25  # Quarter Kelly for safety
    # )
    
    # Access individual results
    # conservative_wealth = comparison.results["Conservative (Q25)"].wealth
    # moderate_plots = comparison.results["Moderate (Q50)"].plots
    
    # Display comparison
    # display(comparison.comparison)
    
    println("Wealth tracking system initialized")
end
