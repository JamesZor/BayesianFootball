# src/features/extractors.jl  



# ------------------------------------------------------------ 
# Extract features
# ------------------------------------------------------------ 

#Fallback error so we know when we are missing a feature
function add_feature!(F_data::Dict, feature::Val{T}, all_groups, context...) where T 
    error("No feature extractor defined for feature $T")
end 

# 1. Team ids features
function add_feature!(F_data::Dict, ::Val{:team_ids}, all_groups, team_map::Dict)
    round_home_ids = [ [team_map[name] for name in g.home_team] for g in all_groups]
    round_away_ids = [ [team_map[name] for name in g.away_team] for g in all_groups]
    
    F_data[:round_home_ids] = round_home_ids
    F_data[:round_away_ids] = round_away_ids
    F_data[:flat_home_ids]  = vcat(round_home_ids...)
    F_data[:flat_away_ids]  = vcat(round_away_ids...)
end

# 2. Month Feature
function add_feature!(F_data::Dict, ::Val{:month}, all_groups, ::Dict)
    F_data[:n_months] = 12
    month_ids = [ [Dates.month(d) for d in g.match_date] for g in all_groups]
    F_data[:round_month_ids]  = month_ids
    F_data[:flat_months] = vcat(month_ids...)
end

  
# 3. Midweek Feature
function add_feature!(F_data::Dict, ::Val{:midweek}, all_groups, ::Dict)
    is_midweek = [[Dates.dayofweek(d) < 6 ? 1 : 0 for d in g.match_date] for g in all_groups ]
    F_data[:round_is_midweek] = is_midweek
    F_data[:flat_is_midweek] = vcat(is_midweek...)
end


# 4. Time Indices (Calculated from team_ids lengths)
function add_feature!(F_data::Dict, ::Val{:time_indices}, all_groups, ::Dict)
  time_indices = Int[]
  for (t, round_matches) in enumerate( all_groups) 
    append!(time_indices, fill(t, nrow(round_matches)))
  end 
  F_data[:time_indices] = time_indices

end

# 5. Plastic / 3g pitch 
const PLASTIC_TEAMS = Set([
    "airdrieonians",
    "alloa-athletic",
    "annan-athletic",
    "bonnyrigg-rose",
    "clyde-fc",
    "cove-rangers",
    "east-kilbride",
    "edinburgh-city-fc",
    "falkirk-fc",
    "forfar-athletic",
    "hamilton-academical",
    "kelty-hearts-fc",
    "montrose",
    "queen-of-the-south",
    "stenhousemuir",
    "the-spartans-fc"
])

# east_fife ?
# Livingston FC
# Raith Rovers FC

function add_feature!(F_data::Dict, ::Val{:is_plastic}, all_groups, ::Dict)
  is_plastic = [ [h in PLASTIC_TEAMS ? 1 : 0 for h in g.home_team] for g in all_groups]
  F_data[:round_is_plastic] = is_plastic 
  F_data[:flat_is_plastic] = vcat(is_plastic...)
end

# ------------------------------------------------------------ 
# Extract targets
# ------------------------------------------------------------ 

"""
    extract_targets!(F_data, model, grouped_df)
Populates the feature dictionary with target data (goals, shots, etc.).
"""
function extract_targets!(F_data::Dict, ::AbstractFootballModel, grouped)
    # Standard Goal Extraction
    F_data[:round_home_goals] = [g.home_score for g in grouped]
    F_data[:round_away_goals] = [g.away_score for g in grouped]
    
    # Flatten immediately
    F_data[:flat_home_goals] = vcat(F_data[:round_home_goals]...)
    F_data[:flat_away_goals] = vcat(F_data[:round_away_goals]...)
end

function extract_targets!(F_data::Dict, model::AbstractFunnelModel, grouped)
    # 1. Reuse base logic for goals (Layer 3)
    invoke(extract_targets!, Tuple{Dict, AbstractFootballModel, Any}, F_data, model, grouped)
    
    # 2. Extract Shots (Layer 1)
    # Note: We ensure Int conversion here
    F_data[:round_home_shots] = [Int.(g.HS) for g in grouped]
    F_data[:round_away_shots] = [Int.(g.AS) for g in grouped]
    F_data[:flat_home_shots]  = vcat(F_data[:round_home_shots]...)
    F_data[:flat_away_shots]  = vcat(F_data[:round_away_shots]...)

    # 3. Extract Shots on Target (Layer 2)
    F_data[:round_home_sot] = [Int.(g.HST) for g in grouped]
    F_data[:round_away_sot] = [Int.(g.AST) for g in grouped]
    F_data[:flat_home_sot]  = vcat(F_data[:round_home_sot]...)
    F_data[:flat_away_sot]  = vcat(F_data[:round_away_sot]...)
end


# ==============================================================================
# Single Row Extractors (Used for Prediction / Inference)
# ==============================================================================

# Fallback error
function get_feature(feature::Val{T}, row::DataFrameRow, context...) where T
    error("No row extractor defined for feature $T")
end

# 1. Month Extractor
function get_feature(::Val{:month}, row::DataFrameRow)
    return Dates.month(row.match_date)
end

# 2. Midweek Extractor
function get_feature(::Val{:midweek}, row::DataFrameRow)
    # Reminder: < 5 means Mon-Thu. (1=Mon, 2=Tue, 3=Wed, 4=Thu). 
    # Change to < 6 if you want Fridays to count as midweek!
    return Dates.dayofweek(row.match_date) < 6 ? 1 : 0
end

# 3. Plastic Pitch Extractor
function get_feature(::Val{:is_plastic}, row::DataFrameRow)
    # Relies on the PLASTIC_TEAMS constant defined higher up in extractors.jl
    return row.home_team in PLASTIC_TEAMS ? 1 : 0
end





# -------------------------------------------------------------------------
# NEW RELATIONAL EXTRACTORS (Powered by SplitBoundary)
# -------------------------------------------------------------------------

function add_feature!(F_data::Dict, ::Val{:goals}, ordered_ids, team_map::Dict, ds)
    score_map = Dict(row.match_id => (row.home_score, row.away_score) for row in eachrow(ds.matches))
    F_data[:flat_home_goals] = [score_map[id][1] for id in ordered_ids]
    F_data[:flat_away_goals] = [score_map[id][2] for id in ordered_ids]
end

# Extractor B: Team IDs 
function add_feature!(F_data::Dict, ::Val{:team_ids}, ordered_ids, team_map::Dict, ds)
    match_team_map = Dict(row.match_id => (row.home_team, row.away_team) for row in eachrow(ds.matches))
    F_data[:flat_home_ids] = [team_map[match_team_map[id][1]] for id in ordered_ids]
    F_data[:flat_away_ids] = [team_map[match_team_map[id][2]] for id in ordered_ids]
end

# Extractor C: Shots 
function add_feature!(F_data::Dict, ::Val{:shots}, ordered_ids, team_map::Dict, ds)
    stats_map = Dict(
        row.match_id => (
        coalesce(row.shotsOnGoal_home, 0.0) + coalesce(row.shotsOffGoal_home, 0.0) + coalesce(row.blockedScoringAttempt_home, 0.0),
        coalesce(row.shotsOnGoal_away, 0.0) + coalesce(row.shotsOffGoal_away, 0.0) + coalesce(row.blockedScoringAttempt_away, 0.0)
        ) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    F_data[:flat_home_shots] = [get(stats_map, id, (missing, missing))[1] for id in ordered_ids]
    F_data[:flat_away_shots] = [get(stats_map, id, (missing, missing))[2] for id in ordered_ids]
end

# Extractor D: Expected Goals (xG)
function add_feature!(F_data::Dict, ::Val{:xg}, ordered_ids, team_map::Dict, ds)
    stats_map = Dict(
        row.match_id => (row.expectedGoals_home, row.expectedGoals_away) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    F_data[:flat_home_xg] = [get(stats_map, id, (missing, missing))[1] for id in ordered_ids]
    F_data[:flat_away_xg] = [get(stats_map, id, (missing, missing))[2] for id in ordered_ids]
end


# ==============================================================================
# RELATIONAL EXTRACTORS (Match Metadata)
# ==============================================================================

# Extractor E: Midweek
function add_feature!(F_data::Dict, ::Val{:midweek}, ordered_ids, team_map::Dict, ds)
    # Map match_id to match_date
    date_map = Dict(row.match_id => row.match_date for row in eachrow(ds.matches))
    
    # < 6 means Mon-Thu. (1=Mon, ..., 5=Fri)
    F_data[:flat_is_midweek] = [Dates.dayofweek(date_map[id]) < 6 ? 1 : 0 for id in ordered_ids]
end

# Extractor F: Month
function add_feature!(F_data::Dict, ::Val{:month}, ordered_ids, team_map::Dict, ds)
    date_map = Dict(row.match_id => row.match_date for row in eachrow(ds.matches))
    
    F_data[:flat_months] = [Dates.month(date_map[id]) for id in ordered_ids]
    F_data[:n_months] = 12
end

# Extractor G: Plastic Pitch
function add_feature!(F_data::Dict, ::Val{:is_plastic}, ordered_ids, team_map::Dict, ds)
    # Map match_id to home_team to check for plastic pitches
    home_team_map = Dict(row.match_id => row.home_team for row in eachrow(ds.matches))
    
    # PLASTIC_TEAMS is already defined in your file
    F_data[:flat_is_plastic] = [home_team_map[id] in PLASTIC_TEAMS ? 1 : 0 for id in ordered_ids]
end
