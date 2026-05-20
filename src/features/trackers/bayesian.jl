# src/features/trackers/bayesian.jl

struct BayesianTracker <: AbstractRatingTracker
    prior_mean::Float64
    prior_var::Float64
    obs_var::Float64
    process_noise::Float64
end

function calculate_player_ratings(config::BayesianTracker, ratings::AbstractVector)
    n = length(ratings)
    out = fill(NaN, n)
    
    curr_mean = config.prior_mean
    curr_var = config.prior_var
    
    for i in 1:n
        out[i] = curr_mean
        
        obs = ratings[i]
        if !ismissing(obs) && !isnan(obs)
            # 1. Prediction Step (Time Update)
            pred_var = curr_var + config.process_noise
            
            # 2. Update Step (Measurement Update)
            kalman_gain = pred_var / (pred_var + config.obs_var)
            curr_mean = curr_mean + kalman_gain * (obs - curr_mean)
            curr_var = (1.0 - kalman_gain) * pred_var
        else
            # If no observation, state evolves with process noise
            curr_var += config.process_noise
        end
    end
    
    return out
end
