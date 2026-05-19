# src/features/trackers/last_value.jl

using ShiftedArrays

struct LastValueTracker <: AbstractRatingTracker end

function calculate_player_ratings(config::LastValueTracker, ratings::AbstractVector)
    # Simply returns the previous rating. 
    # Use lag to ensure pre-match (value at index i is what we knew before the game at i)
    # Handle both Missing and Float64
    return ShiftedArrays.lag(ratings)
end
