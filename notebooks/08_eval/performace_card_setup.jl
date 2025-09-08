using StatsBase
using DataFrames

##################################################
#  Module for the EvaluationCube
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

    # Dimension Mappings
    match_ids::Vector{Int}
    markets::Vector{Symbol}
    c_values::Vector{Float64}

    # Helpers for quick lookups
    market_map::Dict{Symbol, Int}
    match_id_map::Dict{Int, Int} # For fast filtering

    function EvaluationCube(stakes, outcomes, match_ids, markets, c_values)
        market_map = Dict(market => i for (i, market) in enumerate(markets))
        match_id_map = Dict(id => i for (i, id) in enumerate(match_ids))
        new(stakes, outcomes, match_ids, markets, c_values, market_map, match_id_map)
    end
end

end # end module Cude


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
#  Module for Performance Scorecard & Analytics
##################################################
module PerformanceAnalytics

using StatsBase, DataFrames

# --- 1. DEFINE THE DATA STRUCTURES ---

"""
Holds a comprehensive set of single-value metrics for an analysis group.
This struct provides the final "scorecard" for a given segment.
"""
struct PerformanceScorecard
    # Metadata
    analysis_name::String
    num_potential_matches::Int
    num_cube_matches::Int
    # ROI Metrics
    max_roi::Float64
    c_at_max_roi::Float64
    roi_auc_positive::Float64
    roi_robustness_range::Float64
    # Sharpe Ratio Metrics
    max_sharpe::Float64
    c_at_max_sharpe::Float64
    sharpe_auc_positive::Float64
    sharpe_robustness_range::Float64
    # Information Ratio Metrics
    max_ir::Float64
    c_at_max_ir::Float64
    ir_auc_positive::Float64      # Added
    ir_robustness_range::Float64 # Added

    # Bet & Drawdown Metrics at Optimal ROI
    num_bets::Int
    num_winning_bets::Int       # Added
    num_losing_bets::Int        # Added

    total_staked::Float64
    total_profit::Float64
    max_drawdown::Float64
end

# Define a container for all market-level scorecards
struct MarketPerformance
    portfolio::PerformanceScorecard
    markets::Dict{Symbol, PerformanceScorecard}
end


# --- 2. HELPER FUNCTIONS FOR METRIC CALCULATION ---
function _calculate_auc_positive(x, y)
    # The 'init=0.0' argument handles cases where the generator is empty.
    return sum(
        (y[i] + y[i-1])/2 * (x[i] - x[i-1]) 
        for i in 2:length(y) if y[i] > 0 || y[i-1] > 0;
        init=0.0
    )
end


function _calculate_robustness_range(x, y)
    idx = findall(y .> 0)
    return isempty(idx) ? 0.0 : x[maximum(idx)] - x[minimum(idx)]
end
function _calculate_max_drawdown(wealth_curve)
    if isempty(wealth_curve) return 0.0 end
    peak = -Inf
    max_dd = 0.0
    for wealth in wealth_curve
        peak = max(peak, wealth)
        dd = peak - wealth
        max_dd = max(max_dd, dd)
    end
    return max_dd
end


# --- 3. CORE CALCULATION LOGIC ---
"""
Internal helper to calculate a single scorecard from sliced stake/outcome data.
"""
function _calculate_single_scorecard(
    analysis_name::String, num_potential_matches::Int,
    stakes_slice::AbstractArray{Float64, 3}, outcomes_slice::AbstractMatrix{Float64},
    c_values::AbstractVector{Float64}
)::Union{PerformanceScorecard, Nothing}

    num_cube_matches, num_markets, num_cs = size(stakes_slice)
    if num_cube_matches < 2 return nothing end

    # --- Calculate Performance Curves ---
    profits_per_match_c = dropdims(sum(stakes_slice .* reshape(outcomes_slice, (num_cube_matches, num_markets, 1)), dims=2), dims=2)
    total_stakes_per_c = vec(sum(stakes_slice, dims=(1, 2)))
    total_profits_per_c = vec(sum(profits_per_match_c, dims=1))
    
    roi_curve = ifelse.(total_stakes_per_c .== 0, 0.0, total_profits_per_c ./ total_stakes_per_c)

    mean_p, std_p = mean(profits_per_match_c, dims=1), std(profits_per_match_c, dims=1)
    ir_curve = vec(ifelse.(std_p .== 0, 0.0, mean_p ./ std_p))

    sharpe_curve = zeros(Float64, num_cs)
    for i in 1:num_cs
        bet_indices = findall(stakes_slice[:, :, i] .> 0)
        if length(bet_indices) >= 2
            bet_returns = outcomes_slice[bet_indices]
            mean_r, std_r = mean(bet_returns), std(bet_returns)
            sharpe_curve[i] = ifelse(std_r == 0, 0.0, mean_r / std_r)
        end
    end

    # --- Extract Single-Value Metrics from Curves ---
    max_roi, idx_roi = findmax(roi_curve)
    max_sharpe, idx_sharpe = findmax(sharpe_curve)
    max_ir, idx_ir = findmax(ir_curve)

    # --- Calculate Metrics at the Optimal ROI C-Value ---
    stakes_at_opt_roi = stakes_slice[:, :, idx_roi]
    profits_at_opt_roi = profits_per_match_c[:, idx_roi]

    # Find winning and losing bets at the optimal ROI
    bet_indices = findall(stakes_at_opt_roi .> 0)
    bet_outcomes = outcomes_slice[bet_indices]
    num_winning_bets = count(>(0), bet_outcomes)
    num_losing_bets = count(<(0), bet_outcomes)

    # Wealth curve and max drawdown
    wealth_curve = cumsum(profits_at_opt_roi)
    max_dd = _calculate_max_drawdown(wealth_curve)
    
    return PerformanceScorecard(
        analysis_name, num_potential_matches, num_cube_matches,
        # ROI Metrics
        max_roi, c_values[idx_roi], 
        _calculate_auc_positive(c_values, roi_curve), 
        _calculate_robustness_range(c_values, roi_curve),
        # Sharpe Metrics
        max_sharpe, c_values[idx_sharpe], 
        _calculate_auc_positive(c_values, sharpe_curve), 
        _calculate_robustness_range(c_values, sharpe_curve),
        # IR Metrics
        max_ir, c_values[idx_ir], 
        _calculate_auc_positive(c_values, ir_curve), # Added
        _calculate_robustness_range(c_values, ir_curve), # Added
        # Bet & Drawdown Metrics
        count(>(0), stakes_at_opt_roi),
        num_winning_bets, # Added
        num_losing_bets,  # Added
        sum(stakes_at_opt_roi), 
        sum(profits_at_opt_roi), 
        max_dd
    )
end
"""
Calculates performance for the portfolio and each individual market.
"""
function calculate_market_performance(
    analysis_name::String, num_potential_matches::Int, cube_slice::NamedTuple
)::MarketPerformance
    
    market_scorecards = Dict{Symbol, PerformanceScorecard}()

    # 1. Calculate for the entire portfolio (all markets combined)
    portfolio_sc = _calculate_single_scorecard(analysis_name, num_potential_matches, cube_slice.stakes, cube_slice.outcomes, cube_slice.c_values)

    # 2. Calculate for each market individually
    for (market_name, market_idx) in cube_slice.market_map
        stakes_market_slice = cube_slice.stakes[:, market_idx:market_idx, :]
        outcomes_market_slice = cube_slice.outcomes[:, market_idx:market_idx]
        
        market_sc = _calculate_single_scorecard(
            "$(analysis_name) - $(string(market_name))", num_potential_matches,
            stakes_market_slice, outcomes_market_slice, cube_slice.c_values
        )
        if !isnothing(market_sc)
            market_scorecards[market_name] = market_sc
        end
    end

    return MarketPerformance(portfolio_sc, market_scorecards)
end

"""
Converts a MarketPerformance struct into a tidy DataFrame for easy analysis.
"""
function performance_to_dataframe(perf_struct::MarketPerformance)
    all_scorecards = [perf_struct.portfolio]
    for sc in values(perf_struct.markets)
        push!(all_scorecards, sc)
    end
    return DataFrame(filter(!isnothing, all_scorecards))
end

end # end module


##################################################
#  Functions to Build the EvaluationCube
##################################################

# (Your existing functions for build_evaluation_cube, _get_base_value, etc. would go here)
# ...
#
#



"""
Runs the performance analysis for a single evaluation cube and returns a tidy DataFrame.
"""
function run_analysis(
    evaluation_cube::Cude.EvaluationCube,
    analysis_groups::Vector,
    target_matches::DataFrame
)::DataFrame
    
    results_df = DataFrame()
    for group in analysis_groups
        # A. Filter match_ids for the current group
        filtered_df = target_matches
        for (col, val) in group.filters
            filtered_df = filter(row -> row[col] == val, filtered_df)
        end
        num_potential_matches = nrow(filtered_df)
        group_match_ids = Set(filtered_df.match_id)
        
        # B. Get the corresponding row indices from the cube
        row_indices = [evaluation_cube.match_id_map[id] for id in evaluation_cube.match_ids if id in group_match_ids]
        
        if isempty(row_indices) continue end

        # C. Slice the cube data
        cube_slice = (
            stakes = evaluation_cube.stakes[row_indices, :, :],
            outcomes = evaluation_cube.outcomes[row_indices, :],
            c_values = evaluation_cube.c_values,
            market_map = evaluation_cube.market_map
        )

        # D. Generate the performance struct
        performance_struct = PerformanceAnalytics.calculate_market_performance(
            group.name, num_potential_matches, cube_slice
        )
        
        # E. Convert to a DataFrame and append
        group_df = PerformanceAnalytics.performance_to_dataframe(performance_struct)
        group_df.group_name = fill(group.name, nrow(group_df))
        append!(results_df, group_df)
    end
    
    return results_df
end
