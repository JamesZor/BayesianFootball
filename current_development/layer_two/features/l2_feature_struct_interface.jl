# current_development/layer_two/features/l2_features_struct_interface.jl

using DataFrames
using ShiftedArrays # Required for safely lagging data (preventing data leakage)
using  RollingFunctions

# The core interface
abstract type AbstractFeatureBuilder end

"""
    build_all_features(builders::Vector{AbstractFeatureBuilder}, matches_df::DataFrame)

Executes a pipeline of feature builders and sequentially joins them to the match data.
"""
function build_all_features(builders::Vector{AbstractFeatureBuilder}, matches_df::DataFrame)
    master_df = copy(matches_df)
    
    for builder in builders
        println("Building L2 Feature: $(typeof(builder))")
        feature_df = generate_features(builder, matches_df)
        
        # Join the new features back to the master dataframe based on match_id
        master_df = leftjoin(master_df, feature_df, on=:match_id,)
    end
    
    return master_df
end

# --- Helper: The "Melt" Function ---
# Converts Home/Away match data into a chronological Team-Level ledger
function create_team_ledger(matches_df::DataFrame)
    home = select(matches_df, :match_id, :match_date, :season, 
                  :home_team => :team, :home_score => :scored, :away_score => :conceded)
    home.is_home .= true
    
    away = select(matches_df, :match_id, :match_date, :season, 
                  :away_team => :team, :away_score => :scored, :home_score => :conceded)
    away.is_home .= false
    
    ledger = vcat(home, away)
    
    # Calculate Points
    ledger.points = map(eachrow(ledger)) do r
        if ismissing(r.scored) || ismissing(r.conceded)
            return missing
        end
        r.scored > r.conceded ? 3.0 : (r.scored == r.conceded ? 1.0 : 0.0)
    end
    
    # Sort strictly by Team and then Time
    sort!(ledger, [:team, :match_date])
    return ledger
end


function custom_rolling_sum(v::AbstractVector, window::Int)
        res = Vector{Union{Missing, Float64}}(missing, length(v))
        for i in 1:length(v)
            if i >= window && !any(ismissing, v[i-window+1:i])
                res[i] = sum(v[i-window+1:i])
            end
        end
        return res
    end

function rolling_sum_lagged(v::AbstractVector, window::Int)
        n = length(v)
        res = Vector{Union{Missing, Float64}}(missing, n)
        for i in 1:n
            # We can only calculate if we have enough strictly PREVIOUS games
            if i > window 
                prev_games = v[(i-window):(i-1)]
                if !any(ismissing, prev_games)
                    res[i] = Float64(sum(prev_games))
                end
            end
        end
        return res
    end
