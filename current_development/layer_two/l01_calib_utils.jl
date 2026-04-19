# current_development/layer_two/l01_calib_utils.jl

using Dates
using DataFrames

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

"""
    CalibrationResults
"""
struct CalibrationResults
    config::CalibrationConfig
    fitted_models::Dict{Symbol, Any}
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
function create_calibrated_ppd(raw_ppd::PPD, l2_results::CalibrationResults)
    calib_df = l2_results.oos_predictions
    
    # Join raw PPD with our calibrated outputs
    joined = innerjoin(raw_ppd.df, calib_df, on=[:match_id, :market_name, :selection])
    
    # Map the distribution shift across all rows
    new_dists = map(eachrow(joined)) do row
        shift_posterior_samples(row.distribution, row.raw_prob, row.calib_prob)
    end
    
    # Construct the new PPD DataFrame
    new_ppd_df = DataFrame(
        match_id = joined.match_id,
        market_name = joined.market_name,
        market_line = joined.market_line, # Preserved from raw PPD
        selection = joined.selection,
        distribution = new_dists
    )
    
    # Return a native PPD struct
    return BayesianFootball.Predictions.PPD(new_ppd_df, raw_ppd.model, raw_ppd.config)
end
