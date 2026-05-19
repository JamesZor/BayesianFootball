# current_development/player_tracking_systems/src/trackers/ewma.jl

using ShiftedArrays

struct EWMATracker <: AbstractRatingTracker
    alpha::Float64
end

function calculate_player_ratings(config::EWMATracker, ratings::AbstractVector)
    n = length(ratings)
    out = fill(NaN, n)
    
    if n > 0
        current_val = NaN
        for i in 1:n
            out[i] = current_val
            
            obs = ratings[i]
            if !ismissing(obs) && !isnan(obs)
                if isnan(current_val)
                    current_val = Float64(obs)
                else
                    current_val = (config.alpha * obs) + ((1.0 - config.alpha) * current_val)
                end
            end
        end
    end
    return out
end
