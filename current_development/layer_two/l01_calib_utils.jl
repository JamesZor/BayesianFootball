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
