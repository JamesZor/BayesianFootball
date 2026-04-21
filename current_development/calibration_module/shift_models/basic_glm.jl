using GLM
using StatsFuns: logit, logistic
using StatsModels
using DataFrames

struct BasicLogitShift <: AbstractLayerTwoModel
    # No hyperparameters needed for a pure shift
end

struct FittedLogitShift
    c_shift::Float64
    model::StatsModels.TableRegressionModel 
end

function fit_calibrator(model::BasicLogitShift, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :outcome_hit)
    
    eps = 1e-6
    
    # Fusing the clamp and logit broadcast saves an entire array allocation
    df_fit = DataFrame(
        actual = Float64.(data.outcome_hit),
        logit_prob = logit.(clamp.(data.raw_prob, eps, 1.0 - eps))
    )
    
    # Fit the GLM with an offset (Fixes slope to 1.0)
    glm_model = glm(@formula(actual ~ 1), df_fit, Binomial(), LogitLink(), offset=df_fit.logit_prob)
    
    # Extract intercept
    c_shift = coef(glm_model)[1]
    
    return FittedLogitShift(c_shift, glm_model)
end

function apply_shift(fitted_model::FittedLogitShift, new_data::DataFrame)
    eps = 1e-6
    
    # Fuse the clamp, logit, shift, and logistic operations into a single allocation!
    # This evaluates element-by-element without building any temporary arrays.
    return logistic.(logit.(clamp.(new_data.raw_prob, eps, 1.0 - eps)) .+ fitted_model.c_shift)
end
