using StatsBase
##################################################
#  Types for kelly cude 
##################################################
module Cude 
  """
  A tensor-based structure for evaluating model performance across multiple
  matches, markets, and c-value thresholds simultaneously.
  """
  struct EvaluationCube
      # 3D Array: (match_idx, market_idx, c_idx) -> stake
      stakes::Array{Float64, 3}

      # 2D Array: (match_idx, market_idx) -> outcome (e.g., odds - 1 or -1)
      outcomes::Matrix{Float64}

      # Dimension Mappings (The "Axes Labels")
      match_ids::Vector{Int}
      markets::Vector{Symbol}
      c_values::Vector{Float64}

      # Helper for quick market name to index lookups
      market_map::Dict{Symbol, Int}

      # Constructor to build the market_map automatically
      function EvaluationCube(stakes, outcomes, match_ids, markets, c_values)
          market_map = Dict(market => i for (i, market) in enumerate(markets))
          new(stakes, outcomes, match_ids, markets, c_values, market_map)
      end
  end

end

##################################################
#  functions  for kelly cude 
##################################################
#
# Helper to parse the standardized market symbol
function parse_market_symbol(market_symbol::Symbol)
    parts = split(string(market_symbol), "_")
    period = Symbol(parts[1])
    market_type = Symbol(parts[2])
    
    if market_type == :cs
        key = (parse(Int, parts[3]), parse(Int, parts[4]))
        return (period, :correct_score, key)
    elseif market_type in [:other, :any]
        key = join(parts[2:end], "_")
        return (period, :correct_score, key)
    else
        key = Symbol(join(parts[2:end], "_"))
        return (period, key, nothing)
    end
end

# Helper to get the base value (for odds and kelly)
function _get_base_value(data_struct, market_symbol::Symbol)
    period, market_type, key = parse_market_symbol(market_symbol)
    period_data = getfield(data_struct, period)
    if market_type == :correct_score
        return get(getfield(period_data, market_type), key, nothing)
    else
        return getfield(period_data, market_type)
    end
end

# Specialized helper for getting the match result, now with BTTS fix
function _get_result_value(result_data::BayesianFootball.MatchLinesResults, market_symbol::Symbol)
    s_market = string(market_symbol)

    if contains(s_market, "_over_")
        under_symbol = Symbol(replace(s_market, "_over_" => "_under_"))
        under_result = _get_base_value(result_data, under_symbol)
        return isnothing(under_result) ? nothing : !under_result
    
    # NEW: Special case for ft_btts_yes, mapping to the `.btts` field
    elseif market_symbol == :ft_btts_yes
        return result_data.ft.btts
        
    elseif market_symbol == :ft_btts_no
        btts_yes_result = result_data.ft.btts
        return isnothing(btts_yes_result) ? nothing : !btts_yes_result
        
    else
        return _get_base_value(result_data, market_symbol)
    end
end


"""
    build_evaluation_cube(matches_kelly, matches_odds, matches_results, c_values)

Constructs an EvaluationCube. (Corrected version for BTTS field name).
"""
function build_evaluation_cube(
    matches_kelly::Dict,
    matches_odds::Dict,
    matches_results::Dict,
    c_values::AbstractVector{Float64}
)
    markets = [
        :ht_home, :ht_draw, :ht_away,
        :ht_under_05, :ht_over_05, :ht_under_15, :ht_over_15, :ht_under_25, :ht_over_25,
        :ft_home, :ft_draw, :ft_away,
        :ft_under_05, :ft_over_05, :ft_under_15, :ft_over_15,
        :ft_under_25, :ft_over_25, :ft_under_35, :ft_over_35,
        :ft_btts_yes, :ft_btts_no
    ]
    for h in 0:2, a in 0:2 push!(markets, Symbol("ht_cs_$(h)_$(a)")) end
    push!(markets, :ht_any_unquoted)
    for h in 0:3, a in 0:3 push!(markets, Symbol("ft_cs_$(h)_$(a)")) end
    push!(markets, :ft_other_home_win, :ft_other_away_win, :ft_other_draw)

    common_match_ids = intersect(keys(matches_kelly), keys(matches_odds), keys(matches_results))
    match_ids = collect(common_match_ids)
    
    num_matches = length(match_ids)
    num_markets = length(markets)
    num_cs = length(c_values)

    stakes = zeros(Float64, num_matches, num_markets, num_cs)
    outcomes = zeros(Float64, num_matches, num_markets)

    for (match_idx, match_id) in enumerate(match_ids)
        kelly_data = matches_kelly[match_id]
        odds_data = matches_odds[match_id]
        result_data = matches_results[match_id]

        for (market_idx, market_symbol) in enumerate(markets)
            kelly_chain = _get_base_value(kelly_data, market_symbol)
            odds = _get_base_value(odds_data, market_symbol)
            result = _get_result_value(result_data, market_symbol)

            if isnothing(kelly_chain) || isnothing(odds) || isnothing(result) || any(isnan, kelly_chain)
                continue
            end
            
            outcomes[match_idx, market_idx] = result ? (odds - 1.0) : -1.0
            stake_fractions = quantile(kelly_chain, c_values)
            stakes[match_idx, market_idx, :] = stake_fractions
        end
    end
    
    return Cude.EvaluationCube(stakes, outcomes, match_ids, markets, c_values)
end



##################################################
#  Types for the performance  
##################################################
module Performance
  const MaybeProfit = Union{Float64, Nothing}
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, MaybeProfit}

  # A mutable struct is used here so we can build it field-by-field in the loop
  mutable struct MatchHTPerformance
    home::MaybeProfit
    draw::MaybeProfit
    away::MaybeProfit
    correct_score::CorrectScore
    under_05::MaybeProfit
    over_05::MaybeProfit
    under_15::MaybeProfit
    over_15::MaybeProfit
    under_25::MaybeProfit
    over_25::MaybeProfit
    MatchHTPerformance() = new(nothing, nothing, nothing, CorrectScore(), nothing, nothing, nothing, nothing, nothing, nothing)
  end

  mutable struct MatchFTPerformance
    home::MaybeProfit
    draw::MaybeProfit
    away::MaybeProfit
    correct_score::CorrectScore
    under_05::MaybeProfit
    over_05::MaybeProfit
    under_15::MaybeProfit
    over_15::MaybeProfit
    under_25::MaybeProfit
    over_25::MaybeProfit
    under_35::MaybeProfit
    over_35::MaybeProfit
    btts_yes::MaybeProfit
    btts_no::MaybeProfit
    MatchFTPerformance() = new(nothing, nothing, nothing, CorrectScore(), nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
  end

  struct MatchLinePerformance
    ht::MatchHTPerformance
    ft::MatchFTPerformance
  end
end

### methods

"""
    calculate_performance_by_match(evaluation_cube, c_value)

Calculates the profit for each market in each match for a single, given c_value.

Returns a dictionary mapping match_id to a `Performance.MatchLinePerformance` struct.
"""
function calculate_performance_by_match(
    evaluation_cube::Cude.EvaluationCube,
    c_value::Float64
)
    # Find the index for the chosen c_value
    c_idx = findfirst(isapprox(c_value), evaluation_cube.c_values)
    if isnothing(c_idx)
        error("c_value $c_value not found in the evaluation cube.")
    end

    # --- Vectorized Calculation ---
    # 1. Slice the cube to get the stakes for just our chosen c_value
    stakes_at_c = evaluation_cube.stakes[:, :, c_idx]

    # 2. Calculate the profit for all matches and markets at once
    profits_matrix = stakes_at_c .* evaluation_cube.outcomes

    # --- Reshape the Results ---
    results_dict = Dict{Int, Performance.MatchLinePerformance}()
    
    for (match_idx, match_id) in enumerate(evaluation_cube.match_ids)
        # Create empty structs to hold the results for this match
        ht_performance = Performance.MatchHTPerformance()
        ft_performance = Performance.MatchFTPerformance()

        # Loop through all markets and assign the calculated profit to the correct field
        for (market_idx, market_symbol) in enumerate(evaluation_cube.markets)
            profit = profits_matrix[match_idx, market_idx]

            # Skip if the profit is zero (likely no bet or no data) to keep the struct clean
            if profit == 0.0
                continue
            end

            period, market_type, key = parse_market_symbol(market_symbol)
            perf_struct = (period == :ht) ? ht_performance : ft_performance

            if market_type == :correct_score
                getfield(perf_struct, :correct_score)[key] = profit
            else
                setfield!(perf_struct, market_type, profit)
            end
        end
        
        results_dict[match_id] = Performance.MatchLinePerformance(ht_performance, ft_performance)
    end

    return results_dict
end

# You will need this helper function from the previous step
function parse_market_symbol(market_symbol::Symbol)
    parts = split(string(market_symbol), "_")
    period = Symbol(parts[1])
    market_type = Symbol(parts[2])
    
    if market_type == :cs
        key = (parse(Int, parts[3]), parse(Int, parts[4]))
        return (period, :correct_score, key)
    elseif market_type in [:other, :any]
        key = join(parts[2:end], "_")
        return (period, :correct_score, key)
    else
        key = Symbol(join(parts[2:end], "_"))
        return (period, key, nothing)
    end
end


###

module PerformanceCube
  const MaybeProfitVector = Union{Vector{Float64}, Nothing}
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, MaybeProfitVector}

  mutable struct MatchHTPerformanceCube
    home::MaybeProfitVector
    draw::MaybeProfitVector
    away::MaybeProfitVector
    correct_score::CorrectScore
    under_05::MaybeProfitVector
    over_05::MaybeProfitVector
    under_15::MaybeProfitVector
    over_15::MaybeProfitVector
    under_25::MaybeProfitVector
    over_25::MaybeProfitVector
    MatchHTPerformanceCube() = new(nothing, nothing, nothing, CorrectScore(), nothing, nothing, nothing, nothing, nothing, nothing)
  end

  mutable struct MatchFTPerformanceCube
    home::MaybeProfitVector
    draw::MaybeProfitVector
    away::MaybeProfitVector
    correct_score::CorrectScore
    under_05::MaybeProfitVector
    over_05::MaybeProfitVector
    under_15::MaybeProfitVector
    over_15::MaybeProfitVector
    under_25::MaybeProfitVector
    over_25::MaybeProfitVector
    under_35::MaybeProfitVector
    over_35::MaybeProfitVector
    btts_yes::MaybeProfitVector
    btts_no::MaybeProfitVector
    MatchFTPerformanceCube() = new(nothing, nothing, nothing, CorrectScore(), nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
  end

  struct MatchLinePerformanceCube
    ht::MatchHTPerformanceCube
    ft::MatchFTPerformanceCube
  end
end


"""
    calculate_performance_cube_by_match(evaluation_cube)

Calculates the profit for each market across all c-values for each match.

Returns a dictionary mapping match_id to a `PerformanceCube.MatchLinePerformanceCube` struct.
"""
function calculate_performance_cube_by_match(
    evaluation_cube::Cude.EvaluationCube
)
    # --- Vectorized Calculation ---
    # 1. Reshape outcomes to be broadcastable with the 3D stakes cube
    (num_matches, num_markets, num_cs) = size(evaluation_cube.stakes)
    outcomes_reshaped = reshape(evaluation_cube.outcomes, (num_matches, num_markets, 1))

    # 2. Calculate the 3D profit cube for all matches, markets, and c-values at once
    profits_cube = evaluation_cube.stakes .* outcomes_reshaped

    # --- Reshape the Results ---
    results_dict = Dict{Int, PerformanceCube.MatchLinePerformanceCube}()
    
    for (match_idx, match_id) in enumerate(evaluation_cube.match_ids)
        ht_perf_cube = PerformanceCube.MatchHTPerformanceCube()
        ft_perf_cube = PerformanceCube.MatchFTPerformanceCube()

        for (market_idx, market_symbol) in enumerate(evaluation_cube.markets)
            # 3. Extract the entire profit vector for this bet
            profit_vector = profits_cube[match_idx, market_idx, :]

            # Only store the vector if a bet was actually placed for at least one c-value
            if any(p -> p != 0.0, profit_vector)
                period, market_type, key = parse_market_symbol(market_symbol)
                perf_struct = (period == :ht) ? ht_perf_cube : ft_perf_cube

                if market_type == :correct_score
                    getfield(perf_struct, :correct_score)[key] = profit_vector
                else
                    setfield!(perf_struct, market_type, vec(profit_vector))
                end
            end
        end
        
        results_dict[match_id] = PerformanceCube.MatchLinePerformanceCube(ht_perf_cube, ft_perf_cube)
    end

    return results_dict
end

# You will still need the parse_market_symbol helper from before




###
#
##################################################
#  Types for the performance  summary
##################################################
#
module PerformanceSummary
  # A vector of ROI values, one for each c-value
  const ROICurve = Vector{Float64}
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, ROICurve}

  mutable struct MatchHTPerformanceSummary
    home::ROICurve
    draw::ROICurve
    away::ROICurve
    correct_score::CorrectScore
    under_05::ROICurve
    over_05::ROICurve
    under_15::ROICurve
    over_15::ROICurve
    under_25::ROICurve
    over_25::ROICurve
    MatchHTPerformanceSummary() = new(ROICurve(), ROICurve(), ROICurve(), CorrectScore(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve())
  end

  mutable struct MatchFTPerformanceSummary
    home::ROICurve
    draw::ROICurve
    away::ROICurve
    correct_score::CorrectScore
    under_05::ROICurve
    over_05::ROICurve
    under_15::ROICurve
    over_15::ROICurve
    under_25::ROICurve
    over_25::ROICurve
    under_35::ROICurve
    over_35::ROICurve
    btts_yes::ROICurve
    btts_no::ROICurve
    MatchFTPerformanceSummary() = new(ROICurve(), ROICurve(), ROICurve(), CorrectScore(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve(), ROICurve())
  end

  struct MatchLinePerformanceSummary
    ht::MatchHTPerformanceSummary
    ft::MatchFTPerformanceSummary
  end
end

"""
    summarize_performance_roi(evaluation_cube)

Aggregates performance across all matches to get the total ROI curve for each market.

Returns a single `PerformanceSummary.MatchLinePerformanceSummary` struct.
"""
function summarize_performance_roi(evaluation_cube::Cude.EvaluationCube)
    # --- Vectorized Calculation ---
    (num_matches, num_markets, num_cs) = size(evaluation_cube.stakes)
    outcomes_reshaped = reshape(evaluation_cube.outcomes, (num_matches, num_markets, 1))

    # 1. Calculate the 3D profit cube
    profits_cube = evaluation_cube.stakes .* outcomes_reshaped

    # 2. Sum profits and stakes along the 'matches' dimension (dim=1)
    # This aggregates the results from all matches
    total_profits = dropdims(sum(profits_cube, dims=1), dims=1)
    total_stakes = dropdims(sum(evaluation_cube.stakes, dims=1), dims=1)
    
    # 3. Calculate the final ROI matrix (markets x c-values)
    # Handle cases where total_stakes is zero to avoid division errors
    total_roi_matrix = ifelse.(total_stakes .== 0, 0.0, total_profits ./ total_stakes)

    # --- Reshape into the Summary Struct ---
    ht_summary = PerformanceSummary.MatchHTPerformanceSummary()
    ft_summary = PerformanceSummary.MatchFTPerformanceSummary()

    for (market_idx, market_symbol) in enumerate(evaluation_cube.markets)
        # 4. Extract the full ROI curve for this market
        roi_curve = total_roi_matrix[market_idx, :]

        period, market_type, key = parse_market_symbol(market_symbol)
        summary_struct = (period == :ht) ? ht_summary : ft_summary

        if market_type == :correct_score
            getfield(summary_struct, :correct_score)[key] = roi_curve
        else
            setfield!(summary_struct, market_type, roi_curve)
        end
    end

    return PerformanceSummary.MatchLinePerformanceSummary(ht_summary, ft_summary)
end

# You will still need the parse_market_symbol helper from before



"""
    summarize_performance_ir(evaluation_cube)

Aggregates performance across all matches to get the Information Ratio curve for each market.
"""
function summarize_performance_ir(evaluation_cube::Cude.EvaluationCube)
    # --- Vectorized Calculation ---
    (num_matches, num_markets, num_cs) = size(evaluation_cube.stakes)
    outcomes_reshaped = reshape(evaluation_cube.outcomes, (num_matches, num_markets, 1))

    # 1. Calculate the 3D profit cube. Zeros represent "no-bet" decisions.
    profits_cube = evaluation_cube.stakes .* outcomes_reshaped

    # 2. Calculate the mean and standard deviation along the 'matches' dimension
    # This is the Active Return and Tracking Error, respectively.
    active_returns = dropdims(mean(profits_cube, dims=1), dims=1)
    tracking_errors = dropdims(std(profits_cube, dims=1), dims=1)
    
    # 3. Calculate the Information Ratio matrix (markets x c-values)
    ir_matrix = ifelse.(tracking_errors .== 0, 0.0, active_returns ./ tracking_errors)

    # --- Reshape into the Summary Struct ---
    ht_summary = PerformanceSummary.MatchHTPerformanceSummary()
    ft_summary = PerformanceSummary.MatchFTPerformanceSummary()

    for (market_idx, market_symbol) in enumerate(evaluation_cube.markets)
        ir_curve = ir_matrix[market_idx, :]
        period, market_type, key = parse_market_symbol(market_symbol)
        summary_struct = (period == :ht) ? ht_summary : ft_summary

        if market_type == :correct_score
            getfield(summary_struct, :correct_score)[key] = ir_curve
        else
            setfield!(summary_struct, market_type, ir_curve)
        end
    end

    return PerformanceSummary.MatchLinePerformanceSummary(ht_summary, ft_summary)
end


"""
    summarize_performance_sharpe(evaluation_cube)

Aggregates performance across all matches to get the Sharpe Ratio curve for each market.
"""
function summarize_performance_sharpe(evaluation_cube::Cude.EvaluationCube)
    (num_matches, num_markets, num_cs) = size(evaluation_cube.stakes)
    sharpe_matrix = zeros(Float64, num_markets, num_cs)

    # This calculation is harder to fully vectorize, so we loop through markets and c-values
    for market_idx in 1:num_markets
        for c_idx in 1:num_cs
            # Get the stakes and outcomes for this specific market/c-value slice
            stakes_col = evaluation_cube.stakes[:, market_idx, c_idx]
            outcomes_col = evaluation_cube.outcomes[:, market_idx]
            
            # Find the indices where a bet was actually placed
            bet_indices = findall(stakes_col .> 0)
            
            # Sharpe ratio requires at least 2 bets to calculate standard deviation
            if length(bet_indices) < 2
                sharpe_matrix[market_idx, c_idx] = 0.0
                continue
            end
            
            # The "return" of each bet is its outcome (profit/stake = outcome)
            bet_returns = outcomes_col[bet_indices]
            
            # Calculate Sharpe Ratio (assuming risk-free rate is 0)
            mean_return = mean(bet_returns)
            std_return = std(bet_returns)
            
            sharpe_matrix[market_idx, c_idx] = ifelse(std_return == 0, 0.0, mean_return / std_return)
        end
    end

    # --- Reshape into the Summary Struct ---
    ht_summary = PerformanceSummary.MatchHTPerformanceSummary()
    ft_summary = PerformanceSummary.MatchFTPerformanceSummary()

    for (market_idx, market_symbol) in enumerate(evaluation_cube.markets)
        sharpe_curve = sharpe_matrix[market_idx, :]
        period, market_type, key = parse_market_symbol(market_symbol)
        summary_struct = (period == :ht) ? ht_summary : ft_summary

        if market_type == :correct_score
            getfield(summary_struct, :correct_score)[key] = sharpe_curve
        else
            setfield!(summary_struct, market_type, sharpe_curve)
        end
    end

    return PerformanceSummary.MatchLinePerformanceSummary(ht_summary, ft_summary)
end




##############################
# Cum wealth 
##############################

module Wealth 
"""
A tensor-based structure for tracking cumulative wealth over time for each
market and c-value threshold.
"""
struct WealthCube
    # 3D Array: (time_idx, market_idx, c_idx) -> bankroll size
    # The time dimension is N_matches + 1 to include the initial bankroll.
    wealth::Array{Float64, 3}

    # Dimension Mappings (The "Axes Labels")
    # match_ids are sorted chronologically to represent the time axis.
    match_ids_sorted::Vector{Int}
    markets::Vector{Symbol}
    c_values::Vector{Float64}
    market_map::Dict{Symbol, Int}

    function WealthCube(wealth, match_ids_sorted, markets, c_values)
        market_map = Dict(market => i for (i, market) in enumerate(markets))
        new(wealth, match_ids_sorted, markets, c_values, market_map)
    end
end
end

using DataFrames

"""
    calculate_wealth_over_time(evaluation_cube, target_matches; initial_bankroll=100.0)

Simulates bankroll growth over time using proportional Kelly staking.
(Corrected version to scale stakes by current wealth).
"""
function calculate_wealth_over_time(
    evaluation_cube::Cude.EvaluationCube,
    target_matches::DataFrame;
    initial_bankroll::Float64=100.0
)
    # --- 1. Chronologically Sort the Matches (No change here) ---
    id_to_idx = Dict(id => i for (i, id) in enumerate(evaluation_cube.match_ids))
    sim_matches = filter(row -> haskey(id_to_idx, row.match_id), target_matches)
    sort!(sim_matches, :match_date)
    
    sorted_match_ids = sim_matches.match_id
    sorted_indices = [id_to_idx[id] for id in sorted_match_ids]

    # --- 2. Get Sorted Stakes and Outcomes (No change here) ---
    # We get the Kelly fractions (stakes) and 1-unit outcomes, sorted by time
    sorted_stakes_cube = evaluation_cube.stakes[sorted_indices, :, :]
    sorted_outcomes_matrix = evaluation_cube.outcomes[sorted_indices, :]

    # --- 3. Simulate Bankroll Growth (Corrected Logic) ---
    num_sim_matches = length(sorted_match_ids)
    (num_matches, num_markets, num_cs) = size(evaluation_cube.stakes) # Use original cube for dims
    
    wealth_cube = zeros(Float64, num_sim_matches + 1, num_markets, num_cs)
    wealth_cube[1, :, :] .= initial_bankroll
    
    # Loop through time (each match) and update the wealth
    for t in 1:num_sim_matches
        # Get the bankroll from the PREVIOUS step
        previous_wealth = wealth_cube[t, :, :]
        
        # Get the Kelly fraction and 1-unit outcome for the CURRENT match
        kelly_fractions = sorted_stakes_cube[t, :, :]
        unit_outcomes = reshape(sorted_outcomes_matrix[t, :], (num_markets, 1)) # Reshape for broadcasting
        
        # *** THE KEY CORRECTION IS HERE ***
        # Profit = (Current Wealth * Kelly Fraction) * Outcome_per_unit
        profit_this_step = (previous_wealth .* kelly_fractions) .* unit_outcomes
        
        # New Wealth = Previous Wealth + Profit for this step
        wealth_cube[t+1, :, :] = previous_wealth + profit_this_step
    end
    
    return Wealth.WealthCube(
        wealth_cube,
        sorted_match_ids,
        evaluation_cube.markets,
        evaluation_cube.c_values
    )
end
