# current_development/layer_two/models/xgboost_shift.jl

using XGBoost
using StatsFuns: logit

Base.@kwdef struct XGBoostCalibrator <: AbstractLayerTwoModel
    num_rounds::Int = 15      # Cut trees in half
    max_depth::Int = 2        # Extremely shallow (only looks at 2 variables at a time)
    eta::Float64 = 0.05       # Learn very slowly
    min_child_weight::Float64 = 50.0 # CRITICAL: Leaf must contain ~50 matches!
    gamma::Float64 = 1.0      # Require strong evidence to make a split
end

# We store the booster, AND the dictionary used to encode the string names to numbers
struct FittedXGBoost
    booster::Any
    team_encoder::Dict{String, Float64} 
end

function fit_calibrator(model::XGBoostCalibrator, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :outcome_hit)
    
    # 1. Quick String Encoder (Maps "cove-rangers" -> 1.0, etc.)
    all_teams = unique(vcat(data.home_team, data.away_team))
    encoder = Dict{String, Float64}(t => Float64(i) for (i, t) in enumerate(all_teams))
    
    # 2. Build the Numeric Feature Matrix (X)
    # Features: [Home_Team_ID, Away_Team_ID, Market_Line]
    X = zeros(Float64, nrow(data), 3)
    for (i, row) in enumerate(eachrow(data))
        X[i, 1] = get(encoder, row.home_team, 0.0)
        X[i, 2] = get(encoder, row.away_team, 0.0)
        X[i, 3] = Float64(row.market_line)
    end
    
    y = Float64.(data.outcome_hit)
    
    # 3. Extract Layer 1 predictions as the Base Margin (Offset)
    eps = 1e-6
    clamped = clamp.(data.raw_prob, eps, 1.0 - eps)
    
    # XGBoost strictly requires Float32 for info arrays like base_margin
    l1_logits = Float32.(logit.(clamped)) 
    
    # 4. Train the Booster
    dtrain = DMatrix(X, label=y)
    XGBoost.setinfo!(dtrain, "base_margin", l1_logits) # THE MAGIC HAPPENS HERE
    
    booster = xgboost(
        dtrain; 
        num_round = model.num_rounds, 
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = model.max_depth,
        eta = model.eta,
        min_child_weight = model.min_child_weight, # Pass to XGBoost
        gamma = model.gamma,                       # Pass to XGBoost
        nthread = 1
    )
    
    return FittedXGBoost(booster, encoder)
end

function apply_shift(fitted_model::FittedXGBoost, new_data::DataFrame)
    # 1. Build the X matrix for the new data using the saved encoder
    X_new = zeros(Float64, nrow(new_data), 3)
    for (i, row) in enumerate(eachrow(new_data))
        X_new[i, 1] = get(fitted_model.team_encoder, row.home_team, 0.0) # 0.0 if unseen
        X_new[i, 2] = get(fitted_model.team_encoder, row.away_team, 0.0)
        X_new[i, 3] = Float64(row.market_line)
    end
    
    # 2. Prepare base_margin for the new predictions
    eps = 1e-6
    clamped = clamp.(new_data.raw_prob, eps, 1.0 - eps)
    l1_logits = Float32.(logit.(clamped))
    
    dtest = DMatrix(X_new)
    XGBoost.setinfo!(dtest, "base_margin", l1_logits)
    
    # 3. Predict! 
    # Because we used "binary:logistic", XGBoost automatically converts 
    # the (base_margin + tree_output) back into a probability!
    calib_probs = XGBoost.predict(fitted_model.booster, dtest)
    
    return Float64.(calib_probs)
end
