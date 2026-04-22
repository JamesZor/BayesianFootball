abstract type AbstractLayerTwoModel end

"""
    CalibrationConfig

The recipe for a Layer 2 recalibration experiment.
"""
Base.@kwdef struct CalibrationConfig
    name::String
    model::AbstractLayerTwoModel

    prob_col::Symbol = :prob_mean
    # --- Backtesting / Windowing Controls ---
    min_history_splits::Integer = 4          # Wait for 4 periods of L1 OOS data before applying L2
    max_history_splits::Integer = 0          # 0 = expanding window. >0 = rolling window

    min_market_train::Integer= 10
    min_market_predict::Integer= 0 
end




"""
    CalibrationResults
"""
struct CalibrationResults
    config::CalibrationConfig
    # Changed to track history: Target Market -> Split ID -> Fitted Model
    fitted_models_history::Dict{Symbol, Dict{String, Any}} 
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

function apply_calibration(fitted_model, new_data::DataFrame)
    error("Not implemented for $(typeof(fitted_model))")
end

