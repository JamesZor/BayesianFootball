# src/features/extractors/core_extractors.jl

# Fallback error for missing features
function add_feature!(F_data::Dict, ::Val{T}, ordered_ids, team_map::Dict, ds::Data.DataStore) where T
    error("No feature extractor defined for trait :$T")
end

# 1. Team IDs (Mapping match_id to vocabulary indices)
function add_feature!(F_data::Dict, ::Val{:team_ids}, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Match ID -> (HomeTeamName, AwayTeamName)
    match_team_map = Dict(row.match_id => (row.home_team, row.away_team) for row in eachrow(ds.matches))
    
    F_data[:flat_home_ids] = [team_map[match_team_map[id][1]] for id in ordered_ids]
    F_data[:flat_away_ids] = [team_map[match_team_map[id][2]] for id in ordered_ids]
end

# 2. Goals (Actual scores)
function add_feature!(F_data::Dict, ::Val{:goals}, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Match ID -> (HomeScore, AwayScore)
    score_map = Dict(row.match_id => (row.home_score, row.away_score) for row in eachrow(ds.matches))
    
    F_data[:flat_home_goals] = [Int(score_map[id][1]) for id in ordered_ids]
    F_data[:flat_away_goals] = [Int(score_map[id][2]) for id in ordered_ids]
end
