# current_development/player_tracking_systems/src/trackers/last_value.jl

using ShiftedArrays

struct LastValueTracker <: AbstractRatingTracker end

function calculate_player_ratings(config::LastValueTracker, ratings::AbstractVector)
    # Simply returns the previous rating. 
    # Use lag to ensure pre-match (value at index i is what we knew before the game at i)
    return ShiftedArrays.lag(ratings)
end
