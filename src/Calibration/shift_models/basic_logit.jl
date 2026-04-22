# src/Calibration/shift_models/basic_logic.jl

struct BasicLogitShift <: AbstractLayerTwoModel
    # No hyperparameters needed for a pure shift
end

struct FittedLogitShift
    c_shift::Float64
    model::StatsModels.TableRegressionModel 
    prob_col::Symbol # NEW: The model remembers what column it was trained on!
end

function fit_calibrator(model::BasicLogitShift, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :is_winner)
    eps = 1e-6
    
    # Extract the dynamic column specified in the config
    prob_data = data[!, config.prob_col]
    
    df_fit = DataFrame(
        actual = Float64.(data.is_winner),
        logit_prob = logit.(clamp.(prob_data, eps, 1.0 - eps))
    )
    
    glm_model = glm(@formula(actual ~ 1), df_fit, Binomial(), LogitLink(), offset=df_fit.logit_prob)
    c_shift = coef(glm_model)[1]
    
    # Return the fitted model AND the column name it expects
    return FittedLogitShift(c_shift, glm_model, config.prob_col)
end

"""
    apply_calibration(fitted_model::FittedLogitShift, new_data::DataFrame)
"""
function apply_calibration(fitted_model::FittedLogitShift, new_data::DataFrame)
    eps = 1e-6
    c = fitted_model.c_shift 
    
    # Dynamically grab the correct column based on what the model was trained on
    prob_data = new_data[!, fitted_model.prob_col]
    
    # 1. Shift the scalar probabilities
    shifted_scalars = logistic.(logit.(clamp.(prob_data, eps, 1.0 - eps)) .+ c)
    
    # 2. Shift the MCMC distributions
    shifted_dists = map(new_data.distribution) do dist
        logistic.(logit.(clamp.(dist, eps, 1.0 - eps)) .+ c)
    end
    
    return shifted_scalars, shifted_dists
end
