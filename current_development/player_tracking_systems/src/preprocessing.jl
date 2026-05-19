# current_development/player_tracking_systems/src/preprocessing.jl

using DataFrames
using Statistics
using Dates
using BayesianFootball.Data

"""
    generate_tracker_features(config::AbstractRatingTracker, ds::Data.DataStore)

Replicates standard feature extraction logic using a specific rating tracker.
"""
function generate_tracker_features(config::AbstractRatingTracker, ds::Data.DataStore)
    # 1. Prepare base data with dates for sorting
    # We need ratings and minutes_played from lineups
    lineups = select(ds.lineups, :match_id, :player_id, :team_side, :position, :rating, :minutes_played)
    matches_dates = select(ds.matches, :match_id, :match_date)
    
    # Join to get dates and sort chronologically
    df_lineups = innerjoin(lineups, matches_dates, on = :match_id)
    sort!(df_lineups, :match_date)

    # 2. Apply the specific tracker calculation via group-by player_id
    gdf = groupby(df_lineups, :player_id)
    # transform! will apply the tracker to each player's rating history
    transform!(gdf, :rating => (r -> calculate_player_ratings(config, r)) => :pre_match_rating)

    # 3. Handle missing/debut ratings (NaNs) by filling with the global mean
    # Filter non-NaNs to find the global average of raw ratings
    valid_ratings = filter(!isnan, df_lineups.rating)
    global_avg = isempty(valid_ratings) ? 0.0 : mean(valid_ratings)
    
    df_lineups.pre_match_rating = coalesce.(replace(df_lineups.pre_match_rating, NaN => global_avg), global_avg)

    # 4. Apply minute-weighting (clamp.(mins, 0.0, 90.0) ./ 90.0)
    mins = coalesce.(df_lineups.minutes_played, 0.0)
    df_lineups.weighted_rating = df_lineups.pre_match_rating .* (clamp.(mins, 0.0, 90.0) ./ 90.0)

    # 5. Positional Aggregation
    function clean_pos(p)
        if ismissing(p) || p == ""
            return "M" # Default to Midfield
        end
        return p
    end
    df_lineups.clean_position = clean_pos.(df_lineups.position)

    # Aggregate by match, side, and position (G, D, M, F)
    agg_df = combine(groupby(df_lineups, [:match_id, :team_side, :clean_position]), 
                     :weighted_rating => sum => :total_rating)

    # 6. Pivot to match level lookup map
    # match_id -> (side, pos) -> rating
    ratings_map = Dict{Int, Dict{Tuple{String, String}, Float64}}()
    
    for row in eachrow(agg_df)
        m_id = Int(row.match_id)
        if !haskey(ratings_map, m_id)
            ratings_map[m_id] = Dict{Tuple{String, String}, Float64}()
        end
        ratings_map[m_id][(row.team_side, row.clean_position)] = row.total_rating
    end

    return ratings_map
end
