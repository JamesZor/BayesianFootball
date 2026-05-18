# current_development/player_model/l00_player_features.jl

using DataFrames
using Statistics
using Dates
using BayesianFootball.Features

"""
    calc_pre_match_ewma(ratings::AbstractVector; alpha=0.15)

Calculates the EWMA of ratings, but shifted so that the value for index i 
is the EWMA of indices 1 to i-1. This ensures no look-ahead bias.
"""
function calc_pre_match_ewma(ratings::AbstractVector; alpha=0.15)
    n = length(ratings)
    baselines = fill(NaN, n)
    
    if n > 1
        current_ewma = Float64(ratings[1])
        for i in 2:n
            baselines[i] = current_ewma
            # Update for next step
            if !isnan(ratings[i])
                current_ewma = (alpha * ratings[i]) + ((1.0 - alpha) * current_ewma)
            end
        end
    end
    return baselines
end

function Features.add_feature!(F_data::Dict, ::Val{:player_ratings}, ordered_ids, team_map::Dict, ds::Data.DataStore)
    println("[INFO] Extracting Player Ratings Feature...")

    # 1. Prepare base data with dates for sorting
    lineups = dropmissing(ds.lineups, :rating)
    matches_dates = select(ds.matches, :match_id, :match_date)
    
    # Join to get dates and sort
    df_lineups = innerjoin(lineups, matches_dates, on = :match_id)
    sort!(df_lineups, :match_date)

    # 2. Calculate Pre-Match EWMA for each player
    # We use a group-by and transform
    gdf = groupby(df_lineups, :player_id)
    transform!(gdf, :rating => (r -> calc_pre_match_ewma(r, alpha=0.15)) => :pre_match_ewma)

    # 3. Handle debuts / cold starts
    global_avg = mean(df_lineups.rating)
    df_lineups.pre_match_ewma = coalesce.(replace(df_lineups.pre_match_ewma, NaN => global_avg), global_avg)

    # 4. Minute Weighting
    # clamp minutes to 90 as per plan
    mins = coalesce.(df_lineups.minutes_played, 0.0)
    df_lineups.weighted_rating = df_lineups.pre_match_ewma .* (clamp.(mins, 0.0, 90.0) ./ 90.0)

    # 5. Positional Aggregation
    # We want home_G, home_D, home_M, home_F, away_G, ...
    
    # Define a helper to map positions to our 4 categories
    function clean_pos(p)
        if ismissing(p) || p == ""
            return "M" # Default to Midfield if missing
        end
        return p
    end
    df_lineups.clean_position = clean_pos.(df_lineups.position)

    # Aggregate by match, side, and position
    agg_df = combine(groupby(df_lineups, [:match_id, :team_side, :clean_position]), 
                     :weighted_rating => sum => :total_rating)

    # Pivot to match level
    # We need a row for every match_id in ordered_ids
    
    # Initialize dictionaries to hold the ratings for each match
    # match_id -> Dict( (side, pos) => rating )
    ratings_map = Dict{Int, Dict{Tuple{String, String}, Float64}}()
    
    for row in eachrow(agg_df)
        m_id = Int(row.match_id)
        if !haskey(ratings_map, m_id)
            ratings_map[m_id] = Dict{Tuple{String, String}, Float64}()
        end
        ratings_map[m_id][(row.team_side, row.clean_position)] = row.total_rating
    end

    # Helper to get rating or 0.0 if not found
    get_r(m_id, side, pos) = get(get(ratings_map, m_id, Dict()), (side, pos), 0.0)

    # 6. Fill F_data with the 8 flat vectors
    positions = ["G", "D", "M", "F"]
    sides = ["home", "away"]

    for side in sides
        for pos in positions
            key = Symbol("flat_$(side)_$(pos)_rating")
            F_data[key] = [get_r(id, side, pos) for id in ordered_ids]
        end
    end
    
    # Store the match IDs (Training sequence)
    F_data[:match_ids] = ordered_ids
    
    # NEW: Store the full lookup map for ALL matches (supports OOS prediction)
    # This allows extract_parameters to look up ratings for any match_id
    F_data[:player_ratings_map] = ratings_map
    
    println("[INFO] Player Ratings Feature extraction complete.")
end
