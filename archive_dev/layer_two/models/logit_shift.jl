# current_development/layer_two/models/logit_shift.jl

using GLM
using StatsFuns: logit, logistic
using StatsModels

Base.@kwdef struct PureLogitShift <: AbstractLayerTwoModel
    # No hyperparameters needed for a pure shift
end

# The wrapper for our single calculated coefficient
struct FittedLogitShift
    C_shift::Float64
    model::StatsModels.TableRegressionModel 
end

function fit_calibrator(model::PureLogitShift, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :outcome_hit)
    
    # 1. Clamp and convert to Logit space
    eps = 1e-6
    clamped = clamp.(data.raw_prob, eps, 1.0 - eps)
    
    df_fit = DataFrame(
        actual = Float64.(data.outcome_hit),
        logit_prob = logit.(clamped)
    )
    
    # 2. Fit the GLM with an offset (Fixes slope to 1.0)
    glm_model = glm(@formula(actual ~ 1), df_fit, Binomial(), LogitLink(), offset=df_fit.logit_prob)
    
    C_shift = coef(glm_model)[1]
    
    # Return our lightweight fitted struct
    return FittedLogitShift(C_shift, glm_model)
end

function apply_shift(fitted_model::FittedLogitShift, new_data::DataFrame)
    eps = 1e-6
    clamped = clamp.(new_data.raw_prob, eps, 1.0 - eps)
    logits = logit.(clamped)
    
    # Apply the pure affine shift
    shifted_logits = logits .+ fitted_model.C_shift
    
    return logistic.(shifted_logits)
end
