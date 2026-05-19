# current_development/player_tracking_systems/src/trackers/window_average.jl

using ShiftedArrays
using Statistics

struct WindowAverageTracker <: AbstractRatingTracker
    window_size::Int
    agg_func::Function
end

# Default constructor
WindowAverageTracker(window_size::Int) = WindowAverageTracker(window_size, mean)

function calculate_player_ratings(config::WindowAverageTracker, ratings::AbstractVector)
    n = length(ratings)
    out = fill(NaN, n)
    
    # Lag the ratings first to ensure pre-match
    lagged_ratings = ShiftedArrays.lag(ratings)
    
    for i in 1:n
        start_idx = max(1, i - config.window_size)
        end_idx = i - 1 # Use lagged info
        
        if end_idx >= start_idx
            # Extract window from the original ratings (up to i-1)
            window = ratings[start_idx:end_idx]
            clean_window = filter(!isnan, window)
            if !isempty(clean_window)
                out[i] = config.agg_func(clean_window)
            end
        end
    end
    return out
end
