# src/features/trackers/window_average.jl

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
    
    for i in 1:n
        start_idx = max(1, i - config.window_size)
        end_idx = i - 1 
        
        if end_idx >= start_idx
            # Extract window from the original ratings (up to i-1)
            window = ratings[start_idx:end_idx]
            # Robust to Missing/NaN
            clean_window = filter(x -> !ismissing(x) && !isnan(x), window)
            if !isempty(clean_window)
                out[i] = config.agg_func(clean_window)
            end
        end
    end
    return out
end
