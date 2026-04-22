# current_development/layer_two/models/smart_glm.jl

using GLM
using StatsFuns: logit, logistic
using StatsModels
using DataFrames

Base.@kwdef struct SmartGLMCalibrator <: AbstractLayerTwoModel
    # Define the exact features we want to regress on.
    features::Vector{Symbol} = [:market_line, :season_progress, :form_points_diff_7]
end

struct FittedSmartGLM
    model::StatsModels.TableRegressionModel
    features::Vector{Symbol}
end

function fit_calibrator(model_config::SmartGLMCalibrator, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :outcome_hit)
    
    # 1. Clamp and convert Layer 1 probabilities to Logit space
    eps = 1e-6
    clamped = clamp.(data.raw_prob, eps, 1.0 - eps)
    
    # 2. Build the clean fitting dataframe
    df_fit = DataFrame(
        actual = Float64.(data.outcome_hit),
        offset_logit = logit.(clamped)
    )
    
    # FIX: Coalesce the missing values to 0.0 FIRST, then convert the safe array to Float64
    for feat in model_config.features
        df_fit[!, feat] = Float64.(coalesce.(data[!, feat], 0.0))
    end
    
    # 3. Construct the formula dynamically
    rhs = sum(term.(model_config.features))
    f = term(:actual) ~ rhs
    
    # 4. Fit the GLM using the L1 prediction as the mathematically fixed offset
    glm_model = glm(f, df_fit, Binomial(), LogitLink(), offset=df_fit.offset_logit)
    
    return FittedSmartGLM(glm_model, model_config.features)
end

function apply_shift(fitted_model::FittedSmartGLM, new_data::DataFrame)
    eps = 1e-6
    clamped = clamp.(new_data.raw_prob, eps, 1.0 - eps)
    
    # Build the prediction dataframe
    df_predict = DataFrame(offset_logit = logit.(clamped))
    
    # FIX: Coalesce the missing values FIRST here as well
    for feat in fitted_model.features
        df_predict[!, feat] = Float64.(coalesce.(new_data[!, feat], 0.0))
    end
    
    # CRITICAL: When predicting with an offset model in GLM.jl, 
    # you must explicitly pass the offset array to the predict function!
    calib_probs = GLM.predict(fitted_model.model, df_predict; offset=df_predict.offset_logit)
    
    return Float64.(calib_probs)
end
