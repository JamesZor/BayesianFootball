export MIQ, MIQResult, MIQStats

using Statistics
using DataFrames
using HypothesisTests

# --- The Triggers ---
struct MIQ <: AbstractScoringRule end

# A reusable component for the edge statistics
struct MIQStats <: AbstractMetricComponent
    mean_gap::Union{Missing, Float64}
    ks_d_stat::Union{Missing, Float64}
    p_value::Union{Missing, Float64}
    n_winners::Int
    n_losers::Int
end

# The specific result payloads - Expanded to cover core markets
struct MIQResult <: AbstractEvaluationResult
    all::MIQStats
    home::MIQStats
    draw::MIQStats
    away::MIQStats
    over_15::MIQStats
    under_15::MIQStats
    over_25::MIQStats
    under_25::MIQStats
    over_35::MIQStats
    under_35::MIQStats
    btts_yes::MIQStats
    btts_no::MIQStats
end

# The Translator needs to know the name of the RESULT struct
function get_metric_method_name(::MIQResult)::String
    return "miq"
end

function get_metric_method_name(::MIQ)::String
    return "miq"
end

# ==============================================================================
# MATH & HELPERS
# ==============================================================================

# 1. Calculate the empirical quantile for a single row
function get_miq(posterior_samples::Vector{Float64}, market_prob::Float64)
    if ismissing(market_prob) || isnan(market_prob)
        return missing
    end
    # Count samples <= market_prob, divide by total samples
    return sum(posterior_samples .<= market_prob) / length(posterior_samples)
end

# 2. Evaluate group edge (Returns our AbstractMetricComponent)
function evaluate_group_edge(market_quantiles::AbstractVector, is_winner::AbstractVector)::MIQStats
    winners_miq = market_quantiles[is_winner .== true]
    losers_miq  = market_quantiles[is_winner .== false]
    
    # Safety Check: If a group has no winners/losers, return missings cleanly
    if length(winners_miq) < 2 || length(losers_miq) < 2
        return MIQStats(
            missing, 
            missing, 
            missing, 
            length(winners_miq), 
            length(losers_miq)
        )
    end
    
    # Metric A: The Mean Gap (Positive is good)
    mean_gap = mean(losers_miq) - mean(winners_miq)
    
    # Metric B: The K-S Test
    ks_result = ApproximateTwoSampleKSTest(winners_miq, losers_miq)
    
    return MIQStats(
        mean_gap, 
        ks_result.δ, 
        pvalue(ks_result), 
        length(winners_miq), 
        length(losers_miq)
    )
end

# ==============================================================================
# MAIN COMPUTE METHOD
# ==============================================================================

function compute_metric(metric::MIQ, exp::ExperimentResults, ds::DataStore)::MIQResult
    # 1. Extract Latents
    latents_raw = Experiments.extract_oos_predictions(ds, exp)

    # 2. Prepare Market Data
    market_data = Data.prepare_market_data(ds)

    # 3. Model Inference (PPD - Posterior Predictive Distribution)
    ppd = Predictions.model_inference(latents_raw)

    # 4. Merge with Market Data
    analysis_df = innerjoin(
        market_data.df,
        ppd.df,
        on = [:match_id, :market_name, :market_line, :selection]
    )

    # 5. Calculate Market-Implied Quantiles
    analysis_df.market_quantile = [
        get_miq(dist, fair_prob) 
        for (dist, fair_prob) in zip(analysis_df.distribution, analysis_df.prob_fair_close)
    ]

    # Clean missing quantiles (e.g., where odds data was missing)
    clean_analysis = dropmissing(analysis_df, :market_quantile)

    # 6. Helper to extract specific selections safely
    function get_selection_stats(sel::Symbol)
        sub = subset(clean_analysis, :selection => ByRow(isequal(sel)))
        return evaluate_group_edge(sub.market_quantile, sub.is_winner)
    end

    # 7. Pack and return the strict Result Struct (Must match the struct field order above)
    return MIQResult(
        evaluate_group_edge(clean_analysis.market_quantile, clean_analysis.is_winner), # all
        get_selection_stats(:home),
        get_selection_stats(:draw),
        get_selection_stats(:away),
        get_selection_stats(:over_15),
        get_selection_stats(:under_15),
        get_selection_stats(:over_25),
        get_selection_stats(:under_25),
        get_selection_stats(:over_35),
        get_selection_stats(:under_35),
        get_selection_stats(:btts_yes),
        get_selection_stats(:btts_no)
    )
end
