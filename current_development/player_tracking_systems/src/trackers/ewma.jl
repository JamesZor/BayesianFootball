# current_development/player_tracking_systems/src/trackers/ewma.jl

using ShiftedArrays

struct EWMATracker <: AbstractRatingTracker
    alpha::Float64
end

function calculate_player_ratings(config::EWMATracker, ratings::AbstractVector)
    n = length(ratings)
    out = fill(NaN, n)
    
    if n > 1
        # Find first non-NaN rating to initialize
        first_valid_idx = findfirst(!isnan, ratings)
        if isnothing(first_valid_idx) || first_valid_idx == n
            return out
        end
        
        current_ewma = Float64(ratings[first_valid_idx])
        
        # Start filling from the next game
        for i in (first_valid_idx + 1):n
            out[i] = current_ewma
            
            # Update EWMA with the observation from the game that just happened
            if !isnan(ratings[i-1])
                # Note: We update with i-1 because at step i, we just observed game i-1
                # But wait, to be consistent with others, we update at the end of loop
                # Let's use the standard update: S_t = alpha * Y_t + (1-alpha) * S_{t-1}
            end
            
            # Re-implementing more clearly
        end
        
        # Cleaner implementation:
        current_val = NaN
        for i in 1:n
            out[i] = current_val
            
            obs = ratings[i]
            if !isnan(obs)
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
