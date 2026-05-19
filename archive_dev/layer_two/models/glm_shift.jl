# current_development/layer_two/models/glm_shift.jl

using GLM
using StatsFuns: logit, logistic

Base.@kwdef struct TimeWeightedGLM <: AbstractLayerTwoModel
    use_implied_odds::Bool = false
end

# The wrapper for the fitted model returned by fit_calibrator
struct FittedGLM
    model::StatsModels.TableRegressionModel
end

function fit_calibrator(model::TimeWeightedGLM, data::DataFrame, config::CalibrationConfig)
    # 1. Clean Data
    dropmissing!(data, :outcome_hit)
    
    # 2. Compute Time Weights
    weights = ones(nrow(data))
    if !isnothing(config.time_decay_half_life)
        max_date = maximum(data.match_date)
        days_ago = Dates.value.(max_date .- data.match_date)
        weights = 0.5 .^ (days_ago ./ config.time_decay_half_life)
    end
    
    # 3. Create Features (Logit of the L1 probability)
    # We clip probs slightly to avoid logit(0) or logit(1) -> Infinity
    clipped_prob = clamp.(data.raw_prob, 0.001, 0.999)
    
    X = DataFrame(
        y = Float64.(data.outcome_hit),
        l1_logit = logit.(clipped_prob)
    )
    
    if model.use_implied_odds
        X.market_logit = logit.(clamp.(1.0 ./ data.odds_close, 0.001, 0.999))
        fitted = glm(@formula(y ~ l1_logit + market_logit), X, Binomial(), LogitLink(), wts=weights)
    else
        fitted = glm(@formula(y ~ l1_logit), X, Binomial(), LogitLink(), wts=weights)
    end
    
    return FittedGLM(fitted)
end

function apply_shift(fitted_model::FittedGLM, new_data::DataFrame)
    clipped_prob = clamp.(new_data.raw_prob, 0.001, 0.999)
    X_new = DataFrame(l1_logit = logit.(clipped_prob))
    
    # Check if the model expects market_logit
    if "market_logit" in coefnames(fitted_model.model)
        X_new.market_logit = logit.(clamp.(1.0 ./ new_data.odds_close, 0.001, 0.999))
    end
    
    # Predict returns the shifted probability automatically via LogitLink
    return predict(fitted_model.model, X_new)
end
