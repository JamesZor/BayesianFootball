# src/evaluation/scoring.jl
module Scoring

  const Score = Float64
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, Score}
  
  struct MatchHTScores
    home::Score; draw::Score; away::Score
    correct_score::CorrectScore
    under_05::Score; over_05::Score
    under_15::Score; over_15::Score
    under_25::Score; over_25::Score
  end
  
  struct MatchFTScores
    home::Score; draw::Score; away::Score
    correct_score::CorrectScore
    under_05::Score; over_05::Score
    under_15::Score; over_15::Score
    under_25::Score; over_25::Score
    under_35::Score; over_35::Score
    btts_yes::Score; btts_no::Score
  end
  
  struct MatchLineScores
    ht::MatchHTScores
    ft::MatchFTScores
  end

struct AnalysisResult
    model_name::String
    financial_scorecard::Any # Using Any to avoid circular dependency, will hold PerformanceAnalytics.MarketPerformance
    log_likelihood_scorecard::MatchLineScores
end

using ..BayesianFootball
using DataFrames
using StatsBase

# Helper functions for parsing market data from Odds and Results structs
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

function _get_base_value(data_struct, market_symbol::Symbol)
    period, market_type, key = parse_market_symbol(market_symbol)
    period_data = getfield(data_struct, period)
    if market_type == :correct_score
        return get(getfield(period_data, market_type), key, nothing)
    else
        return getfield(period_data, market_type)
    end
end

# NEW HELPER: Specialized for getting values from Predictions struct
function _get_prediction_value(prediction_data, market_symbol::Symbol)
    s_market = string(market_symbol)

    if contains(s_market, "_over_")
        under_symbol = Symbol(replace(s_market, "_over_" => "_under_"))
        under_chain = _get_base_value(prediction_data, under_symbol)
        return isnothing(under_chain) ? nothing : 1.0 .- under_chain
    elseif market_symbol == :ft_btts_yes
        return prediction_data.ft.btts
    elseif market_symbol == :ft_btts_no
        btts_yes_chain = prediction_data.ft.btts
        return isnothing(btts_yes_chain) ? nothing : 1.0 .- btts_yes_chain
    else
        # For all other markets, the field name matches the symbol
        return _get_base_value(prediction_data, market_symbol)
    end
end


function _get_result_value(result_data, market_symbol::Symbol)
    s_market = string(market_symbol)
    if contains(s_market, "_over_")
        under_symbol = Symbol(replace(s_market, "_over_" => "_under_"))
        under_result = _get_base_value(result_data, under_symbol)
        return isnothing(under_result) ? nothing : !under_result
    elseif market_symbol == :ft_btts_yes
        return result_data.ft.btts
    elseif market_symbol == :ft_btts_no
        btts_yes_result = result_data.ft.btts
        return isnothing(btts_yes_result) ? nothing : !btts_yes_result
    else
        return _get_base_value(result_data, market_symbol)
    end
end

include("/home/james/bet_project/models_julia/notebooks/08_eval/performace_card_setup.jl")

function calculate_financial_scorecard(precomputes, c_values, target_matches)
    evaluation_cube = build_evaluation_cube(
        precomputes.matches_kelly,
        precomputes.matches_odds,
        precomputes.matches_results,
        c_values
    )
    
    return PerformanceAnalytics.calculate_market_performance(
        "Total Portfolio",
        nrow(target_matches),
        (stakes=evaluation_cube.stakes, outcomes=evaluation_cube.outcomes, 
         c_values=evaluation_cube.c_values, market_map=evaluation_cube.market_map)
    )
end

function calculate_log_likelihood_scorecard(precomputes, c_values)
    matches_prediction = precomputes.matches_prediction
    matches_results = precomputes.matches_results
    
    markets = [
        :ht_home, :ht_draw, :ht_away, :ht_under_05, :ht_over_05, :ht_under_15, :ht_over_15, :ht_under_25, :ht_over_25,
        :ft_home, :ft_draw, :ft_away, :ft_under_05, :ft_over_05, :ft_under_15, :ft_over_15, :ft_under_25, :ft_over_25, :ft_under_35, :ft_over_35, :ft_btts_yes, :ft_btts_no
    ]
    for h in 0:2, a in 0:2 push!(markets, Symbol("ht_cs_$(h)_$(a)")) end; push!(markets, :ht_any_unquoted)
    for h in 0:3, a in 0:3 push!(markets, Symbol("ft_cs_$(h)_$(a)")) end; push!(markets, :ft_other_home_win, :ft_other_away_win, :ft_other_draw)

    scores = Dict{Symbol, Float64}()
    for market_symbol in markets
        log_likelihood_curve = zeros(length(c_values))
        for (match_id, prediction) in matches_prediction
            # CORRECTED: Use the specialized helper for predictions
            prob_chain = _get_prediction_value(prediction, market_symbol)
            outcome = _get_result_value(matches_results[match_id], market_symbol)
            if isnothing(prob_chain) || isnothing(outcome) continue end
            if outcome
                prob_quantiles = quantile(prob_chain, c_values)
                log_probs = log.(max.(1e-10, prob_quantiles))
                log_likelihood_curve .+= log_probs
            end
        end
        scores[market_symbol] = isempty(log_likelihood_curve) ? 0.0 : maximum(log_likelihood_curve)
    end
    
    get_s(key) = get(scores, key, 0.0)
    ht_cs = Dict{Union{Tuple{Int,Int}, String}, Float64}((h,a) => get_s(Symbol("ht_cs_$(h)_$(a)")) for h in 0:2 for a in 0:2)
    ht_cs["any_unquoted"] = get_s(:ht_any_unquoted)
    ft_cs = Dict{Union{Tuple{Int,Int}, String}, Float64}((h,a) => get_s(Symbol("ft_cs_$(h)_$(a)")) for h in 0:3 for a in 0:3)
    ft_cs["other_home_win"]=get_s(:ft_other_home_win); ft_cs["other_away_win"]=get_s(:ft_other_away_win); ft_cs["other_draw"]=get_s(:ft_other_draw)
    
    ht_scores = MatchHTScores(get_s(:ht_home), get_s(:ht_draw), get_s(:ht_away), ht_cs, get_s(:ht_under_05), get_s(:ht_over_05), get_s(:ht_under_15), get_s(:ht_over_15), get_s(:ht_under_25), get_s(:ht_over_25))
    ft_scores = MatchFTScores(get_s(:ft_home), get_s(:ft_draw), get_s(:ft_away), ft_cs, get_s(:ft_under_05), get_s(:ft_over_05), get_s(:ft_under_15), get_s(:ft_over_15), get_s(:ft_under_25), get_s(:ft_over_25), get_s(:ft_under_35), get_s(:ft_over_35), get_s(:ft_btts_yes), get_s(:ft_btts_no))
    
    return MatchLineScores(ht_scores, ft_scores)
end

end # end module Scoring

