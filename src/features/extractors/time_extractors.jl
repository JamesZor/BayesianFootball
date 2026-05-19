# src/features/extractors/time_extractors.jl

# 1. Dates (Calculates decay deltas relative to the newest match in the split)
function add_feature!(F_data::Dict, ::DatesFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Match ID -> Date
    date_lookup = Dict(row.match_id => row.match_date for row in eachrow(ds.matches))
    
    subset_dates = [date_lookup[id] for id in ordered_ids]
    newest_date = maximum(subset_dates)
    
    # Deltas in days
    F_data[:dates] = (newest_date .- subset_dates) .|> Dates.value
end

# 2. Month (Seasonality)
function add_feature!(F_data::Dict, ::MonthFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    date_lookup = Dict(row.match_id => row.match_date for row in eachrow(ds.matches))
    
    F_data[:flat_months] = [Dates.month(date_lookup[id]) for id in ordered_ids]
    F_data[:n_months] = 12
end

# 3. Midweek (Congestion)
function add_feature!(F_data::Dict, ::MidweekFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    date_lookup = Dict(row.match_id => row.match_date for row in eachrow(ds.matches))
    
    # 1-5 = Mon-Fri
    F_data[:flat_is_midweek] = [Dates.dayofweek(date_lookup[id]) < 6 ? 1 : 0 for id in ordered_ids]
end

# 4. Plastic Pitch (Specific Surface types)
const PLASTIC_TEAMS = Set([
    "airdrieonians", "alloa-athletic", "annan-athletic", "bonnyrigg-rose",
    "clyde-fc", "cove-rangers", "east-kilbride", "edinburgh-city-fc",
    "falkirk-fc", "forfar-athletic", "hamilton-academical", "kelty-hearts-fc",
    "montrose", "queen-of-the-south", "stenhousemuir", "the-spartans-fc"
])

function add_feature!(F_data::Dict, ::PlasticPitchFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    home_team_map = Dict(row.match_id => row.home_team for row in eachrow(ds.matches))
    F_data[:flat_is_plastic] = [home_team_map[id] in PLASTIC_TEAMS ? 1 : 0 for id in ordered_ids]
end

# 5. Time Indices (Sequence indices for dynamic models)
function add_feature!(F_data::Dict, ::TimeIndicesFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Note: time_indices are currently pre-calculated and added directly in Features.create_features 
    # to maintain compatibility with the SplitBoundary history/target logic.
    # This dispatcher exists to satisfy the AbstractFeatureConfig interface.
    return nothing
end
