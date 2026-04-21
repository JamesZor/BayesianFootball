abstract type AbstractLayerTwoModel end

"""
    CalibrationConfig

The recipe for a Layer 2 recalibration experiment.
"""
Base.@kwdef struct CalibrationConfig
    name::String
    model::AbstractLayerTwoModel
    # --- Backtesting / Windowing Controls ---
    min_history_splits::Integer = 4          # Wait for 4 periods of L1 OOS data before applying L2
    max_history_splits::Integer = 0          # 0 = expanding window. >0 = rolling window

    min_market_train::Integer= 10
    min_market_predict::Integer= 0 
end

