# current_development/layer_two/models/xgboost_shift_v2.jl

using XGBoost
using StatsFuns: logit

Base.@kwdef struct XGBoostCalibratorV2 <: AbstractLayerTwoModel
    features::Vector{Symbol}  # <--- NEW: Dynamically pass which features to use!
    num_rounds::Int = 15      
    max_depth::Int = 2        
    eta::Float64 = 0.05       
    min_child_weight::Float64 = 50.0 
    gamma::Float64 = 1.0      
end

struct FittedXGBoost
    booster::Any
    feature_names::Vector{Symbol} # Save exactly what was trained on
end

function fit_calibrator(model::XGBoostCalibratorV2, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :outcome_hit)
    
    # 1. Build the Numeric Feature Matrix (X) dynamically
    n_rows = nrow(data)
    n_cols = length(model.features)
    X = zeros(Float64, n_rows, n_cols)
    
    for (j, feat) in enumerate(model.features)
        # Safely extract column, converting any `missing` (from early season form) to 0.0
        col_data = coalesce.(data[!, feat], 0.0)
        X[:, j] .= Float64.(col_data)
    end
    
    y = Float64.(data.outcome_hit)
    
    # 2. Extract Layer 1 predictions as the Base Margin (Offset)
    eps = 1e-6
    clamped = clamp.(data.raw_prob, eps, 1.0 - eps)
    l1_logits = Float32.(logit.(clamped)) 
    
    # 3. Train the Booster
    dtrain = DMatrix(X, label=y)
    XGBoost.setinfo!(dtrain, "base_margin", l1_logits)
    
    booster = xgboost(
        dtrain; 
        num_round = model.num_rounds, 
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = model.max_depth,
        eta = model.eta,
        min_child_weight = model.min_child_weight,
        gamma = model.gamma,
        nthread = 1
    )
    
    return FittedXGBoost(booster, model.features)
end

function apply_shift(fitted_model::FittedXGBoost, new_data::DataFrame)
    # 1. Build the X matrix for the new data using the saved feature names
    n_rows = nrow(new_data)
    n_cols = length(fitted_model.feature_names)
    X_new = zeros(Float64, n_rows, n_cols)
    
    for (j, feat) in enumerate(fitted_model.feature_names)
        col_data = coalesce.(new_data[!, feat], 0.0)
        X_new[:, j] .= Float64.(col_data)
    end
    
    # 2. Prepare base_margin 
    eps = 1e-6
    clamped = clamp.(new_data.raw_prob, eps, 1.0 - eps)
    l1_logits = Float32.(logit.(clamped))
    
    dtest = DMatrix(X_new)
    XGBoost.setinfo!(dtest, "base_margin", l1_logits)
    
    # 3. Predict
    calib_probs = XGBoost.predict(fitted_model.booster, dtest)
    
    return Float64.(calib_probs)
end
