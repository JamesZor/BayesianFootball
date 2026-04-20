# current_development/layer_two/l01_calib_utils.jl

using Dates
using DataFrames
using Printf

abstract type AbstractLayerTwoModel end

"""
    CalibrationConfig

The recipe for a Layer 2 recalibration experiment.
"""
Base.@kwdef struct CalibrationConfig
    name::String
    model::AbstractLayerTwoModel
    target_markets::Vector{Symbol}       # e.g., [:over_25, :under_25, :home]
    
    # --- Backtesting / Windowing Controls ---
    min_history_splits::Integer = 4          # Wait for 4 periods of L1 OOS data before applying L2
    max_history_splits::Integer = 0          # 0 = expanding window. >0 = rolling window
    
    time_decay_half_life::Union{Nothing, Number} = nothing 
end

# """
#     CalibrationResults
# """
# struct CalibrationResults
#     config::CalibrationConfig
#     fitted_models::Dict{Symbol, Any}
#     oos_predictions::DataFrame
# end
"""
    CalibrationResults
"""
struct CalibrationResults
    config::CalibrationConfig
    # Changed to track history: Target Market -> Split ID -> Fitted Model
    fitted_models_history::Dict{Symbol, Dict{String, Any}} 
    oos_predictions::DataFrame
end

# ==========================================
# Generic Interfaces (To be extended by models)
# ==========================================

"""
    fit_calibrator(model::AbstractLayerTwoModel, data::DataFrame, config::CalibrationConfig)
Trains the Layer 2 model on historical PPDs and actual outcomes.
"""
function fit_calibrator(model::AbstractLayerTwoModel, data::DataFrame, config::CalibrationConfig)
    error("Not implemented for $(typeof(model))")
end

"""
    apply_shift(fitted_model, new_data::DataFrame)
Applies the learned shift to unobserved PPDs.
"""
function apply_shift(fitted_model, new_data::DataFrame)
    error("Not implemented for $(typeof(fitted_model))")
end


# ---- 

#=
The Elegant Solution: Logit-Space Posterior Shifting

We do not want to fake a distribution by just repeating the scalar [0.6, 0.6, 0.6...] because that destroys the uncertainty your BayesianKelly relies on.

Instead, we calculate the shift between the L1 Mean and the L2 Calibrated Mean in log-odds (logit) space, and we apply that exact shift to every single sample in the MCMC posterior distribution. This anchors the distribution to the new, smarter L2 probability while perfectly preserving the shape and variance of your L1 structural model.
=#



using StatsFuns: logit, logistic
using DataFrames

"""
    shift_posterior_samples(raw_dist::Vector{Float64}, raw_mean::Float64, calib_prob::Float64)

Translates the entire MCMC posterior distribution to align with the calibrated 
point-estimate, preserving the original variance/uncertainty.
"""
function shift_posterior_samples(raw_dist, raw_mean, calib_prob)
    # Clamp to avoid Infinity in logit space
    safe_dist = clamp.(raw_dist, 1e-5, 1 - 1e-5)
    safe_mean = clamp(raw_mean, 1e-5, 1 - 1e-5)
    safe_calib = clamp(calib_prob, 1e-5, 1 - 1e-5)
    
    # Calculate the shift delta in log-odds space
    shift_delta = logit(safe_calib) - logit(safe_mean)
    
    # Apply shift to all samples and convert back to probability space
    return logistic.(logit.(safe_dist) .+ shift_delta)
end

"""
    create_calibrated_ppd(raw_ppd::PPD, l2_results::CalibrationResults)

Takes the raw PPD and the L2 results, applies the logit shift to the distributions,
and returns a completely new PPD struct ready for your Signals engine.
"""
# function create_calibrated_ppd(raw_ppd, l2_results::CalibrationResults)
#     calib_df = l2_results.oos_predictions
#
#     # Join raw PPD with our calibrated outputs
#     joined = innerjoin(raw_ppd.df, calib_df, on=[:match_id, :market_name, :selection])
#
#     # Map the distribution shift across all rows
#     new_dists = map(eachrow(joined)) do row
#         shift_posterior_samples(row.distribution, row.raw_prob, row.calib_prob)
#     end
#
#     # Construct the new PPD DataFrame
#     new_ppd_df = DataFrame(
#         match_id = joined.match_id,
#         market_name = joined.market_name,
#         market_line = joined.market_line, # Preserved from raw PPD
#         selection = joined.selection,
#         distribution = new_dists
#     )
#
#     # Return a native PPD struct
#     return BayesianFootball.Predictions.PPD(new_ppd_df, raw_ppd.model, raw_ppd.config)
# end
#


function create_calibrated_ppd(raw_ppd::BayesianFootball.Predictions.PPD, l2_results::CalibrationResults)
    calib_df = l2_results.oos_predictions
    
    # 1. The crucial fix: Join on market_line and use makeunique=true
    joined = innerjoin(
        raw_ppd.df, 
        calib_df, 
        on=[:match_id, :market_name, :market_line, :selection],
        makeunique=true
    )
    
    # 2. Apply the shift to the MCMC arrays
    new_dists = map(eachrow(joined)) do row
        shift_posterior_samples(row.distribution, row.raw_prob, row.calib_prob)
    end
    
    # 3. Rebuild the DataFrame
    new_ppd_df = DataFrame(
        match_id = joined.match_id,
        market_name = joined.market_name,
        market_line = joined.market_line, 
        selection = joined.selection,
        distribution = new_dists
    )
    
    return BayesianFootball.Predictions.PPD(new_ppd_df, raw_ppd.model, raw_ppd.config)
end



# ----
using MLJBase: log_loss # Or write a quick logloss function

function custom_logloss(y, p)
    p = clamp.(p, 1e-15, 1 - 1e-15)
    return -mean(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
end

function quick_analysis_logloss(y_true, p_raw, p_calib)
    ll_raw = custom_logloss(y_true, p_raw)
    ll_calib = custom_logloss(y_true, p_calib)

    println("\n=== Calibration Results ===")
    println("Raw L1 LogLoss:    ", round(ll_raw, digits=4))
    println("Calib L2 LogLoss:  ", round(ll_calib, digits=4))
    println("Improvement:       ", round(ll_raw - ll_calib, digits=4))
end

function quick_analysis_brier(y_true, p_raw, p_calib)
    # Brier Score is just the Mean Squared Error of the probability
    function brier_score(y, p)
        return mean((y .- p).^2)
    end

    bs_raw = brier_score(y_true, p_raw)
    bs_calib = brier_score(y_true, p_calib)

    println("\n=== Brier Score Results ===")
    println("Raw L1 Brier:    ", round(bs_raw, digits=4))
    println("Calib L2 Brier:  ", round(bs_calib, digits=4))
    println("Improvement:     ", round(bs_raw - bs_calib, digits=4)) 
    # (Positive improvement is better here!)
end

function quick_analysis(results::CalibrationResults) 
    y_true = results.oos_predictions.outcome_hit;
    p_raw = results.oos_predictions.raw_prob;
    p_calib = Float64.(results.oos_predictions.calib_prob);
    
    quick_analysis_logloss(y_true,p_raw,p_calib)
    quick_analysis_brier(y_true,p_raw,p_calib)
end


# ==========================================
# 6. Native Signal Processing & ROI Analysis
# ==========================================

function get_ppd_for_raw_and_calib(ds, exp, results::CalibrationResults)
    latents = Experiments.extract_oos_predictions(ds, exp)
    raw_ppd = Predictions.model_inference(latents) 

    # 2. Convert L2 Output back into a native PPD
    calib_ppd = create_calibrated_ppd(raw_ppd, results)

    # 3. Align the Raw PPD to exactly match the scope of the Calibrated PPD
    # semijoin keeps only the rows in raw_ppd.df that perfectly match the keys in calib_ppd.df
    aligned_raw_df = semijoin(
        raw_ppd.df, 
        calib_ppd.df, 
        on=[:match_id, :market_name, :market_line, :selection]
    )

    # 4. Overwrite raw_ppd with the aligned version
    raw_ppd = BayesianFootball.Predictions.PPD(
        aligned_raw_df, 
        raw_ppd.model, 
        raw_ppd.config 
    )

    println("Raw PPD rows:   ", nrow(raw_ppd.df))
    println("Calib PPD rows: ", nrow(calib_ppd.df))

  return raw_ppd, calib_ppd 
end


# 4. Helper to Compute PnL from your signal ledger
function summarize_ledger(ledger_df)
    # Compute PnL for placed bets
    ledger_df.pnl = map(eachrow(ledger_df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner == 1.0 # Or true, depending on your type
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end
    
    # Financial Metrics
    total_stake = sum(ledger_df.stake)
    total_pnl = sum(ledger_df.pnl)
    roi = total_stake > 0 ? (total_pnl / total_stake) * 100 : 0.0
    
    # Volume & Hit Metrics
    seen = nrow(ledger_df)
    bets = count(x -> x > 0, ledger_df.stake)
    active_rate = seen > 0 ? (bets / seen) * 100 : 0.0
    
    # Win Rate (only counting bets we actually placed)
    won_bets = count(r -> r.stake > 0.0 && r.is_winner == 1.0, eachrow(ledger_df))
    win_rate = bets > 0 ? (won_bets / bets) * 100 : 0.0
    
    return (
        bets = bets, 
        total_stake = total_stake, 
        total_pnl = total_pnl, 
        roi = roi, 
        seen = seen, 
        active_rate = active_rate, 
        win_rate = win_rate
    )
end

function display_result(raw_sig_result, calib_sig_result; min_edge=min_edge)
    raw = summarize_ledger(raw_sig_result.df)
    calib = summarize_ledger(calib_sig_result.df)

    @printf("\n=== L3 Strategy: Bayesian Kelly (Edge > %.2f) ===\n", min_edge)

    @printf("\n[RAW L1 MODEL]\n")
    @printf("  Seen Markets: %d\n", raw.seen)
    @printf("  Bets Placed:  %d (Active Rate: %.2f%%)\n", raw.bets, raw.active_rate)
    @printf("  Win Rate:     %.2f%%\n", raw.win_rate)
    @printf("  Total Stake:  %.2f units\n", raw.total_stake)
    @printf("  Total PnL:    %+.2f units\n", raw.total_pnl)
    @printf("  ROI:          %+.2f%%\n", raw.roi)

    @printf("\n[CALIBRATED L2 MODEL]\n")
    @printf("  Seen Markets: %d\n", calib.seen)
    @printf("  Bets Placed:  %d (Active Rate: %.2f%%)\n", calib.bets, calib.active_rate)
    @printf("  Win Rate:     %.2f%%\n", calib.win_rate)
    @printf("  Total Stake:  %.2f units\n", calib.total_stake)
    @printf("  Total PnL:    %+.2f units\n", calib.total_pnl)
    @printf("  ROI:          %+.2f%%\n", calib.roi)
end




function display_edge_threshold_analysis(ppd, ds)
println("\n=== Model: Edge Threshold Analysis ===")
println("Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%")
println("---------------------------------------------------------")

# for edge_pct in [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
  for edge_pct in 0.0:0.01:0.1

    # Create temporary signals with the new edge
    temp_signals = [BayesianFootball.Signals.BayesianKelly(min_edge=edge_pct)]
    
    # Process just the calibrated PPD
    temp_result = BayesianFootball.Signals.process_signals(ppd, ds.odds, temp_signals; odds_column=:odds_close)
    
    # Summarize
    metrics = summarize_ledger(temp_result.df)
    
    @printf("%5.1f | %4d | %7.2f | %6.2f | %6.2f | %+.2f | %+.2f%%\n", 
            edge_pct * 100, metrics.bets, metrics.active_rate, metrics.win_rate, 
            metrics.total_stake, metrics.total_pnl, metrics.roi)
end

end 

