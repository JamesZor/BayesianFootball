# current_development/player_tracking_systems/src/types.jl

abstract type AbstractRatingTracker end

"""
    TrackerMetrics

Holds the evaluation results for a specific tracker configuration.
"""
struct TrackerMetrics
    log_loss::Float64
    glm_edge_coef::Float64
    glm_edge_pvalue::Float64
end

"""
    calculate_player_ratings(config::AbstractRatingTracker, ratings_vector::AbstractVector)

Interface function to be implemented by specific trackers. 
Must return a vector of the same length as `ratings_vector` containing pre-match ratings.
"""
function calculate_player_ratings(config::AbstractRatingTracker, ratings_vector::AbstractVector)
    error("calculate_player_ratings not implemented for $(typeof(config))")
end
