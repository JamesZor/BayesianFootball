# src/features/extractors/stats_extractors.jl

# 1. Shots (Aggregate shots for the whole match)
function add_feature!(F_data::Dict, ::Val{:shots}, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Filter statistics for "ALL" period and aggregate
    # match_id -> (home_shots, away_shots)
    stats_map = Dict(
        row.match_id => (
            coalesce(row.shotsOnGoal_home, 0.0) + coalesce(row.shotsOffGoal_home, 0.0) + coalesce(row.blockedScoringAttempt_home, 0.0),
            coalesce(row.shotsOnGoal_away, 0.0) + coalesce(row.shotsOffGoal_away, 0.0) + coalesce(row.blockedScoringAttempt_away, 0.0)
        ) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    
    F_data[:flat_home_shots] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_shots] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end

# 2. Expected Goals (xG)
function add_feature!(F_data::Dict, ::Val{:xg}, ordered_ids, team_map::Dict, ds::Data.DataStore)
    stats_map = Dict(
        row.match_id => (row.expectedGoals_home, row.expectedGoals_away) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    
    F_data[:flat_home_xg] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_xg] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end
