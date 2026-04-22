# current_development/calibration_module/shift_models/basic_glm.jl

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
    dropmissing!(data, :is_winner)
    
    eps = 1e-6
    
    # Fusing the clamp and logit broadcast saves an entire array allocation
    df_fit = DataFrame(
        actual = Float64.(data.is_winner),
        # column_data = data[!, target_col]
        logit_prob = logit.(clamp.(data.prob_median, eps, 1.0 - eps))
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
    return logistic.(logit.(clamp.(new_data.prob_median, eps, 1.0 - eps)) .+ fitted_model.c_shift)
end

"""
    apply_calibration(fitted_model::FittedLogitShift, new_data::DataFrame)

Takes the uncalibrated DataFrame and returns both the shifted scalar probabilities
and the shifted MCMC posterior distributions.
"""
function apply_calibration(fitted_model::FittedLogitShift, new_data::DataFrame)
    eps = 1e-6
    c = fitted_model.c_shift # We just use the model's exact shift parameter!
    
    # 1. Shift the scalar probabilities
    shifted_scalars = logistic.(logit.(clamp.(new_data.prob_median, eps, 1.0 - eps)) .+ c)
    
    # 2. Shift the MCMC distributions using the exact same 'c' parameter
    # (Since this is a BasicLogitShift, it doesn't compress variance, it just slides it)
    shifted_dists = map(new_data.distribution) do dist
        logistic.(logit.(clamp.(dist, eps, 1.0 - eps)) .+ c)
    end
    
    return shifted_scalars, shifted_dists
end
