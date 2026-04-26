# loader.jl
using DataFrames
using BayesianFootball

# -------------------------------------------------------------------------
# 1. The New Abstract Boundary
# -------------------------------------------------------------------------
struct SplitBoundary
    fold_id::Int
    target_step::Int
    history_match_ids::Vector{Int}
    target_match_ids::Vector{Int}
end

# -------------------------------------------------------------------------
# 2. Shared Split Logic (Returning IDs, not Views)
# -------------------------------------------------------------------------
function _process_tournament_group_ids(
    df::DataFrame, 
    group_ids::Vector{Int}, 
    config::Union{BayesianFootball.Data.CVConfig, BayesianFootball.Data.GroupedCVConfig}, 
    meta_type::Type
)
    splits = Vector{Tuple{SplitBoundary, BayesianFootball.Data.AbstractSplitMetaData}}()
    all_seasons = sort(unique(df.season))
    
    tourn_mask = in.(df.tournament_id, Ref(group_ids))
    if !any(tourn_mask); return splits; end

    for target_season in config.target_seasons
        target_idx = findfirst(==(target_season), all_seasons)
        if isnothing(target_idx)
            @warn "Target season $target_season not found for tournament group $group_ids. Skipping."
            continue
        end
        
        start_idx = max(1, target_idx - config.history_seasons)
        history_seasons_list = all_seasons[start_idx : target_idx-1]
        
        # --- Extract universal match_ids ---
        history_ids = df[tourn_mask .& in.(df.season, Ref(history_seasons_list)), :match_id]
        
        target_pool = df[tourn_mask .& (df.season .== target_season), [:match_id, config.dynamics_col]]
        
        if isempty(target_pool)
            @warn "No data found for target season $target_season (Tournament group $group_ids)."
            continue
        end
        
        season_dynamics = unique(target_pool[!, config.dynamics_col])
        sort!(season_dynamics)
        
        max_week = maximum(season_dynamics)
        effective_end = config.stop_early ? (max_week - 1) : max_week

        valid_steps = filter(t -> t >= config.warmup_period, season_dynamics)
        if !isnothing(config.end_dynamics)
            filter!(t -> t <= config.end_dynamics, valid_steps)
        end
        filter!(t -> t <= effective_end, valid_steps)
       
        # -----------------------------------------------------------------
        # CREATE FOLDS
        # -----------------------------------------------------------------
        fold_counter = 1
        
        # --- 1. Inject the Baseline Fold (History Only, t=0) ---
        boundary_zero = SplitBoundary(
            fold_counter,
            0, # Target step 0 (Baseline)
            copy(history_ids),
            Int[] # No target matches yet!
        )
        
        if meta_type === BayesianFootball.Data.SplitMetaData
            meta_zero = BayesianFootball.Data.SplitMetaData(group_ids[1], target_season, target_season, config.history_seasons, 0, config.warmup_period)
        else
            meta_zero = BayesianFootball.Data.GroupedSplitMetaData(group_ids, target_season, target_season, config.history_seasons, 0, config.warmup_period)
        end
        
        push!(splits, (boundary_zero, meta_zero))
        fold_counter += 1

        # --- 2. Inject the Dynamic Folds (Walk Forward) ---
        for t in valid_steps
            # Get target match IDs up to step t
            current_target_ids = target_pool[target_pool[!, config.dynamics_col] .<= t, :match_id]
            
            boundary = SplitBoundary(
                fold_counter,
                t,
                copy(history_ids),       # frozen history
                copy(current_target_ids) # expanding window
            )
            
            if meta_type === BayesianFootball.Data.SplitMetaData
                meta = BayesianFootball.Data.SplitMetaData(group_ids[1], target_season, target_season, config.history_seasons, t, config.warmup_period)
            else
                meta = BayesianFootball.Data.GroupedSplitMetaData(group_ids, target_season, target_season, config.history_seasons, t, config.warmup_period)
            end
            
            push!(splits, (boundary, meta))
            fold_counter += 1
        end
        # NOTE: The duplicate `for (i, t) in enumerate(valid_steps)` loop has been removed!
    end
    
    return splits
end
# -------------------------------------------------------------------------
# 3. Wrappers
# -------------------------------------------------------------------------
function create_id_boundaries(ds::BayesianFootball.Data.DataStore, config::BayesianFootball.Data.CVConfig)
    splits = Vector{Tuple{SplitBoundary, BayesianFootball.Data.SplitMetaData}}()
    for tourn_id in config.tournament_ids
        group_splits = _process_tournament_group_ids(ds.matches, [tourn_id], config, BayesianFootball.Data.SplitMetaData)
        for (b, m) in group_splits
            push!(splits, (b, m::BayesianFootball.Data.SplitMetaData))
        end
    end
    return splits
end





# ==============================================================================
# loader.jl - Part 2: Relational Feature Building
# ==============================================================================

# --- 1. Dummy Model & Traits ---
struct DevSequentialFunnel <: BayesianFootball.Models.AbstractFootballModel end

# The model tells the builder what it needs!
BayesianFootball.Features.required_features(::DevSequentialFunnel) = [:team_ids, :goals, :shots, :xg]

struct MockFeatureSet
    data::Dict{Symbol, Any}
end

# --- 2. The Macro Loop (Vector Dispatch) ---
function build_features(
    splits::Vector{<:Tuple{SplitBoundary, <:Any}}, 
    ds::BayesianFootball.Data.DataStore, 
    model::BayesianFootball.Models.AbstractFootballModel
)
    return [
        (build_features(boundary, ds, model), meta) 
        for (boundary, meta) in splits
    ]
end

# --- 3. The Micro Builder (Single Boundary Dispatch) ---
function build_features(
    boundary::SplitBoundary, 
    ds::BayesianFootball.Data.DataStore, 
    model::BayesianFootball.Models.AbstractFootballModel
)
    F_data = Dict{Symbol, Any}()
    
    # Store temporal metadata
    F_data[:dynamics_step] = boundary.target_step
    F_data[:n_history_matches] = length(boundary.history_match_ids)
    F_data[:n_target_matches] = length(boundary.target_match_ids)

    # ---> DYNAMIC PIPELINE: Model asks for traits, Builder pulls them <---
    for trait in BayesianFootball.Features.required_features(model)
        add_feature!(F_data, Val(trait), boundary, ds)
    end

    return MockFeatureSet(F_data)
end

# --- 4. The Relational Extractors ---

# Extractor A: Goals (From ds.matches)
function add_feature!(F_data::Dict, ::Val{:goals}, boundary::SplitBoundary, ds)
    # Fast Map: match_id -> (home_score, away_score)
    score_map = Dict(row.match_id => (row.home_score, row.away_score) for row in eachrow(ds.matches))
    
    # Notice we map cleanly over the ordered target IDs
    F_data[:flat_home_goals] = [score_map[id][1] for id in boundary.target_match_ids]
    F_data[:flat_away_goals] = [score_map[id][2] for id in boundary.target_match_ids]
end

# Extractor B: Team Names/IDs (From ds.matches)
function add_feature!(F_data::Dict, ::Val{:team_ids}, boundary::SplitBoundary, ds)
    team_map = Dict(row.match_id => (row.home_team, row.away_team) for row in eachrow(ds.matches))
    
    # In reality you'd map these to Ints using your vocabulary dictionary
    F_data[:flat_home_teams] = [team_map[id][1] for id in boundary.target_match_ids]
    F_data[:flat_away_teams] = [team_map[id][2] for id in boundary.target_match_ids]
end

# Extractor C: Shots (From ds.statistics)
# function add_feature!(F_data::Dict, ::Val{:shots}, boundary::SplitBoundary, ds)
#     # Fast Map: match_id -> (home_shots, away_shots)
#     # We use `missing` in case a match lacks stats
#   #
#     stats_full = subset(ds.statistics, :period => ByRow(isequal("ALL")))
#     stats_map = Dict(row.match_id => (row.shotsOnGoal_home, row.shotsOnGoal_away) for row in eachrow(stats_full))
#
#     F_data[:flat_home_shots] = [get(stats_map, id, (missing, missing))[1] for id in boundary.target_match_ids]
#     F_data[:flat_away_shots] = [get(stats_map, id, (missing, missing))[2] for id in boundary.target_match_ids]
# end
#
# function add_feature!(F_data::Dict, ::Val{:xg}, boundary::SplitBoundary, ds)
#     # Fast Map: match_id -> (home_shots, away_shots)
#     # We use `missing` in case a match lacks stats
#     stats_full = subset(ds.statistics, :period => ByRow(isequal("ALL")))
#     stats_map = Dict(row.match_id => (row.expectedGoals_home, row.expectedGoals_away) for row in eachrow(stats_full))
#
#     F_data[:flat_home_xg] = [get(stats_map, id, (missing, missing))[1] for id in boundary.target_match_ids]
#     F_data[:flat_away_xg] = [get(stats_map, id, (missing, missing))[2] for id in boundary.target_match_ids]
# end
#
# Extractor C: Shots (Calculated & Filtered)
function add_feature!(F_data::Dict, ::Val{:shots}, boundary::SplitBoundary, ds)
    # 1. Filter for "ALL" period inline.
    # 2. Safely sum On Goal + Off Goal (add woodwork/blocked here if your DB separates them).
    stats_map = Dict(
        row.match_id => (
            coalesce(row.shotsOnGoal_home, 0.0) + coalesce(row.shotsOffGoal_home, 0.0),
            coalesce(row.shotsOnGoal_away, 0.0) + coalesce(row.shotsOffGoal_away, 0.0)
        ) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    
    F_data[:flat_home_shots] = [get(stats_map, id, (missing, missing))[1] for id in boundary.target_match_ids]
    F_data[:flat_away_shots] = [get(stats_map, id, (missing, missing))[2] for id in boundary.target_match_ids]
end

# Extractor D: Expected Goals (Filtered)
function add_feature!(F_data::Dict, ::Val{:xg}, boundary::SplitBoundary, ds)
    # Inline filter for "ALL" avoids allocating a new DataFrame via `subset`
    stats_map = Dict(
        row.match_id => (row.expectedGoals_home, row.expectedGoals_away) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    
    F_data[:flat_home_xg] = [get(stats_map, id, (missing, missing))[1] for id in boundary.target_match_ids]
    F_data[:flat_away_xg] = [get(stats_map, id, (missing, missing))[2] for id in boundary.target_match_ids]
end




# ==============================================================================
# loader.jl - Part 2: Relational Feature Building (Updated for GRW)
# ==============================================================================

function build_features(
    boundary::SplitBoundary, 
    ds::BayesianFootball.Data.DataStore, 
    model::BayesianFootball.Models.AbstractFootballModel,
    dynamics_col::Symbol = :match_month # Defaulting based on your config
)
    F_data = Dict{Symbol, Any}()
    
    # 1. COMBINE IDs for the full sequence (History + Target)
    all_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)

    # 2. Extract just the matches for this specific fold
    matches_df = subset(ds.matches, :match_id => ByRow(id -> id in all_ids))
    
    # 3. BUILD VOCABULARY (Strings -> Integers)
    all_teams = unique(vcat(matches_df.home_team, matches_df.away_team))
    team_map = Dict(name => i for (i, name) in enumerate(sort(all_teams)))
    
    F_data[:n_teams] = length(team_map)
    F_data[:team_map] = team_map # Save it for inference later!

    # 4. GENERATE TIME INDICES (Dual Grouping)
    # We must preserve the order: History first, Target second.
    history_df = subset(matches_df, :match_id => ByRow(id -> id in boundary.history_match_ids))
    target_df  = subset(matches_df, :match_id => ByRow(id -> id in boundary.target_match_ids))
    
    # Group History by Season, Target by dynamics_col (e.g., match_month)
    history_groups = groupby(history_df, :season, sort=true)
    target_groups  = groupby(target_df, dynamics_col, sort=true)
    
    time_indices = Int[]
    t_idx = 1
    
    for g in history_groups
        append!(time_indices, fill(t_idx, nrow(g)))
        t_idx += 1
    end
    n_history = length(history_groups)
    
    for g in target_groups
        append!(time_indices, fill(t_idx, nrow(g)))
        t_idx += 1
    end
    n_target = length(target_groups)
    
    # Ensure IDs match the exact grouped order we just created
    ordered_ids = Int.(vcat(history_df.match_id, target_df.match_id))
    
    F_data[:time_indices] = time_indices
    F_data[:n_history_steps] = n_history
    F_data[:n_target_steps] = n_target
    F_data[:n_rounds] = n_history + n_target

    # 5. ---> DYNAMIC PIPELINE <---
    # Notice we pass `ordered_ids` and `team_map` down to the extractors!
    for trait in BayesianFootball.Features.required_features(model)
        add_feature!(F_data, Val(trait), ordered_ids, team_map, ds)
    end

    return MockFeatureSet(F_data)
end



# -------------------------------------------------------------------------
# The Relational Extractors (Updated for all_ids & team_map)
# -------------------------------------------------------------------------


# Extractor A: Goals
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
            coalesce(row.shotsOnGoal_home, 0.0) + coalesce(row.shotsOffGoal_home, 0.0),
            coalesce(row.shotsOnGoal_away, 0.0) + coalesce(row.shotsOffGoal_away, 0.0)
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
