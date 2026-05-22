# current_development/match_day_inference/src/ratings.jl

using DataFrames
using Statistics
using BayesianFootbal.Features
import BayesianFootbal.Features: AbstractRatingTracker, BayesianTracker, EWMATracker, LastValueTracker, WindowAverageTracker

# ==========================================
# 1. TRACKER-SPECIFIC LATEST RATING EXTRACITON
# ==========================================

function calculate_latest_player_rating(config::BayesianTracker, ratings::AbstractVector)
    n = length(ratings)
    curr_mean = config.prior_mean
    curr_var = config.prior_var
    for i in 1:n
        obs = ratings[i]
        if !ismissing(obs) && !isnan(obs)
            pred_var = curr_var + config.process_noise
            kalman_gain = pred_var / (pred_var + config.obs_var)
            curr_mean = curr_mean + kalman_gain * (obs - curr_mean)
            curr_var = (1.0 - kalman_gain) * pred_var
        else
            curr_var += config.process_noise
        end
    end
    return curr_mean
end

function calculate_latest_player_rating(config::EWMATracker, ratings::AbstractVector)
    n = length(ratings)
    current_val = NaN
    for i in 1:n
        obs = ratings[i]
        if !ismissing(obs) && !isnan(obs)
            if isnan(current_val)
                current_val = Float64(obs)
            else
                current_val = (config.alpha * obs) + ((1.0 - config.alpha) * current_val)
            end
        end
    end
    return current_val
end

function calculate_latest_player_rating(config::LastValueTracker, ratings::AbstractVector)
    clean_ratings = filter(x -> !ismissing(x) && !isnan(x), ratings)
    return isempty(clean_ratings) ? NaN : Float64(last(clean_ratings))
end

function calculate_latest_player_rating(config::WindowAverageTracker, ratings::AbstractVector)
    n = length(ratings)
    start_idx = max(1, n - config.window_size + 1)
    end_idx = n
    if end_idx >= start_idx
        window = ratings[start_idx:end_idx]
        clean_window = filter(x -> !ismissing(x) && !isnan(x), window)
        if !isempty(clean_window)
            return config.agg_func(clean_window)
        end
    end
    return NaN
end

# Generic fallback for any other tracker
function calculate_latest_player_rating(config::AbstractRatingTracker, ratings::AbstractVector)
    clean_ratings = filter(x -> !ismissing(x) && !isnan(x), ratings)
    return isempty(clean_ratings) ? NaN : Float64(mean(clean_ratings))
end

# ==========================================
# 2. RUN PIPELINE & BUILD RATINGS MAP
# ==========================================

"""
    get_latest_player_ratings(ds::Data.DataStore, tracker::AbstractRatingTracker)

Computes the latest ratings for all players based on the historical datastore and the rating tracker.
"""
function get_latest_player_ratings(ds::Data.DataStore, tracker::AbstractRatingTracker)
    lineups = select(ds.lineups, :match_id, :player_id, :team_side, :position, :rating, :minutes_played)
    matches_dates = select(ds.matches, :match_id, :match_date)
    df_lineups = innerjoin(lineups, matches_dates, on = :match_id)
    sort!(df_lineups, :match_date)

    valid_ratings = filter(x -> !ismissing(x) && !isnan(x), df_lineups.rating)
    global_avg = isempty(valid_ratings) ? 6.0 : mean(valid_ratings)

    gdf = groupby(df_lineups, :player_id)
    latest_ratings = Dict{Int, Float64}()

    for df_p in gdf
        player_id = df_p.player_id[1]
        latest_r = calculate_latest_player_rating(tracker, df_p.rating)
        
        if isnan(latest_r)
            if hasproperty(tracker, :prior_mean)
                latest_r = tracker.prior_mean
            else
                latest_r = global_avg
            end
        end
        latest_ratings[player_id] = latest_r
    end

    return latest_ratings, global_avg
end

"""
    build_matchday_ratings_map(ds::Data.DataStore, tracker::AbstractRatingTracker, todays_matches::AbstractDataFrame, json_dir::String)

Computes latest player ratings and aggregates them by position for each matchday lineup.
"""
function build_matchday_ratings_map(ds::Data.DataStore, tracker::AbstractRatingTracker, todays_matches::AbstractDataFrame, json_dir::String)
    println("└── [Ratings] Computing latest player ratings from historical lineups...")
    player_ratings, global_avg = get_latest_player_ratings(ds, tracker)
    println("    └─ Global average player rating: ", round(global_avg, digits=3))
    println("    └─ Total tracked players: ", length(player_ratings))

    ratings_map = Dict{Int, Dict{Tuple{String, String}, Float64}}()

    function clean_pos(pos::String)
        if pos == "G" || pos == "Goalkeeper" || pos == "GK"
            return "G"
        elseif pos == "D" || pos == "Defender" || pos == "DF"
            return "D"
        elseif pos == "M" || pos == "Midfielder" || pos == "MF"
            return "M"
        elseif pos == "F" || pos == "Forward" || pos == "FW" || pos == "A"
            return "F"
        else
            return "M" # Default to Midfielder
        end
    end

    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)
        
        println("└── [Fixture] Match ID: $mid | $home vs $away")
        
        # Load lineups
        lineup = get_matchday_lineup(ds, mid, home, away, json_dir)
        
        m_ratings = Dict{Tuple{String, String}, Float64}()
        
        # Helper to compute positional sums for starters (substitute == false)
        for (side, players) in [("home", lineup.home), ("away", lineup.away)]
            # Filter to starters
            starters = filter(p -> !p.substitute, players)
            
            if isempty(starters)
                @warn "No starters found for $side ($side == home ? $home : $away) in match $mid! Using default 0.0 for all positions."
                for pos in ["G", "D", "M", "F"]
                    m_ratings[(side, pos)] = 0.0
                end
                continue
            end
            
            # Group starters by clean position and sum their ratings
            pos_sums = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
            
            for p in starters
                p_id = p.player_id
                c_pos = clean_pos(p.position)
                
                rating = get(player_ratings, p_id, global_avg)
                
                # Report debutants for transparency
                if !haskey(player_ratings, p_id)
                    println("    └─ [Debut] Player ID: $p_id | $(p.player_name) ($side, $c_pos) - setting to fallback: $global_avg")
                end
                
                pos_sums[c_pos] += rating
            end
            
            for (pos, val) in pos_sums
                m_ratings[(side, pos)] = val
            end
            
            # Print team summary
            println("    └─ $side starters (N=$(length(starters))): G=$(round(pos_sums["G"], digits=1)), D=$(round(pos_sums["D"], digits=1)), M=$(round(pos_sums["M"], digits=1)), F=$(round(pos_sums["F"], digits=1))")
        end
        
        ratings_map[mid] = m_ratings
    end

    return ratings_map
end
