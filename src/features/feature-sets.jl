# src/features/feature-sets.jl 

# -----
# --- Helper Traits for Feature Extraction ---

"""
    target_columns(model)
Returns the list of columns required from the dataframe (e.g. goals, shots)
to ensure we filter out missing rows correctly.
"""
# Default behavior (Existing models): Just needs scores
target_columns(::AbstractFootballModel) = [:home_score, :away_score]

# Funnel behavior: Needs scores + Shot Data
target_columns(::AbstractFunnelModel) = [
    :home_score, :away_score, 
    :HS, :AS,   # Home/Away Shots
    :HST, :AST  # Home/Away Shots on Target
]

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


# ------

# 1. Define the hook (Default behavior: do nothing)

function apply_model_specific_logic(model::AbstractStaticPoissonModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticDixonColesModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticMixCopulaModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticMVPLNModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticBivariatePoissonModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticNegBinModel, df::DataFrame)
    return df
end

# 2. Define the hook for GRW (Behavior: Sort by time)
function apply_model_specific_logic(model::AbstractDynamicPoissonModel, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end

function apply_model_specific_logic(model::AbstractDynamicDixonColesModel, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end

function apply_model_specific_logic(model::AbstractDynamicNegBinModel, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end

function apply_model_specific_logic(model::AbstractDynamicBivariatePoissonModel, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end

function apply_model_specific_logic(model::AbstractFunnelModel, df::DataFrame)
    # 1. Sort by season and date (Crucial for Sequential/GRW models)
    # 2. Return the clean dataframe
    return sort(df, [:season, :match_date])
end

function apply_model_specific_logic(model::AbstractMultiScaledNegBinModel, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end



"""
    build_mappings(df, model)

Internal helper to create the necessary ID maps (e.g. Team A -> 1)
based specifically on the data present in `df`.
"""
function build_mappings(df::AbstractDataFrame, model::AbstractFootballModel)
    keys_needed = required_mapping_keys(model)
    mappings = Dict{Symbol, Any}()

    # --- Team Mapping Factory ---
    if :team_map in keys_needed || :n_teams in keys_needed
        present_teams = Set{String}()
        
        if hasproperty(df, :home_team)
            union!(present_teams, df.home_team)
        end
        if hasproperty(df, :away_team)
            union!(present_teams, df.away_team)
        end

        # Sort for deterministic ordering (Crucial for reproducibility)
        sorted_teams = sort(collect(present_teams))
        
        # Create dense map (1..N)
        team_map = Dict(t => i for (i, t) in enumerate(sorted_teams))
        
        mappings[:team_map] = team_map
        mappings[:n_teams] = length(sorted_teams)
    end

    # --- League/Tournament Factory (Example extension) ---
    if :league_map in keys_needed || :n_leagues in keys_needed
        if hasproperty(df, :tournament_slug)
            leagues = unique(df.tournament_slug)
            mappings[:league_map] = Dict(l => i for (i, l) in enumerate(leagues))
            mappings[:n_leagues] = length(leagues)
        end
    end

    return mappings
end


# ---- 
function create_features(
    data_split::AbstractDataFrame,
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::FeatureSet

    # 1. Initialize data dictionary
    F_data = Dict{Symbol, Any}()

    # 2. Build Mappings
    mappings = build_mappings(data_split, model)
    merge!(F_data, mappings)
    team_map = F_data[:team_map]::Dict{<:AbstractString, Int}
    
    # --- Process Data ---
    
    # [CHANGED] Use the trait to determine which columns we need
    needed_cols = target_columns(model)
    matches_df_filtered = dropmissing(data_split, needed_cols)
    
    # Filter teams map
    matches_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df_filtered, view=false)

    # Ensure integers for scores (generic safety)
    matches_df.home_score = Int.(matches_df.home_score)
    matches_df.away_score = Int.(matches_df.away_score)
    matches_df = apply_model_specific_logic(model, matches_df)
    
    F_data[:matches_df] = matches_df

    # --- DETERMINING THE TIME GROUPING ---
    grouping_col = isnothing(splitter_config.dynamics_col) ? splitter_config.window_col : splitter_config.dynamics_col
    
    if !hasproperty(matches_df, grouping_col)
        error("The time column ':$grouping_col' was not found in the DataFrame.")
    end

    # Group by the Time Column
    grouped = groupby(matches_df, grouping_col, sort=true)
    F_data[:n_rounds] = length(grouped)

    # Extract IDs (Standard)
    F_data[:round_home_ids] = [ [team_map[name] for name in g.home_team] for g in grouped]
    F_data[:round_away_ids] = [ [team_map[name] for name in g.away_team] for g in grouped]
    F_data[:flat_home_ids] = vcat(F_data[:round_home_ids]...)
    F_data[:flat_away_ids] = vcat(F_data[:round_away_ids]...)

    # [CHANGED] Delegate Target Extraction (Goals vs Funnel)
    extract_targets!(F_data, model, grouped)

    # Time Indices (Standard)
    time_indices = Int[]
    for (t, round_matches) in enumerate(F_data[:round_home_ids])
        n_matches_in_round = length(round_matches)
        append!(time_indices, fill(t, n_matches_in_round))
    end
    F_data[:time_indices] = time_indices

    return FeatureSet(F_data)
end

function create_features(
    data_splits::Vector{<:Tuple{<:AbstractDataFrame, M}},
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::FeatureCollection{M} where M  # <--- Update Return Type

    # Generate the vector as before
    raw_vector = [
        (
            create_features(data, model, splitter_config), 
            meta
        ) 
        for (data, meta) in data_splits
    ]

    # Wrap it
    return FeatureCollection(raw_vector)
end


# src/features/feature-sets.jl
function create_features(
    data_split::AbstractDataFrame,
    model::AbstractMultiScaledNegBinModel,
    splitter_config::CVConfig
)::FeatureSet

    # 1. Initialize data dictionary
    F_data = Dict{Symbol, Any}()

    # 2. Build Mappings
    mappings = build_mappings(data_split, model)
    merge!(F_data, mappings)
    team_map = F_data[:team_map]::Dict{<:AbstractString, Int}
    
    # --- Process Data ---
    needed_cols = target_columns(model)
    matches_df_filtered = dropmissing(data_split, needed_cols)
    
    # Filter teams map
    matches_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df_filtered, view=false)

    # Ensure integers for scores generic safety
    matches_df.home_score = Int.(matches_df.home_score)
    matches_df.away_score = Int.(matches_df.away_score)
    
    # Hook for any model specific sorting (like GRW time flows)
    matches_df = apply_model_specific_logic(model, matches_df)
    F_data[:matches_df] = matches_df

    # --- DUAL TIME GROUPING LOGIC ---
    grouping_col = splitter_config.dynamics_col
    
    if !hasproperty(matches_df, grouping_col)
        error("The time column ':$grouping_col' was not found in the DataFrame.")
    end

    # 3. Create the Mask using CVConfig
    # If the season is in our target_seasons list, it belongs to the target df.
    # Otherwise, it belongs to the history df.
    history_mask = .!in.(matches_df.season, Ref(splitter_config.target_seasons))
    
    history_df = matches_df[history_mask, :]
    target_df = matches_df[.!history_mask, :]

    # 4. Perform the Dual Grouping
    # Macro Grouping: History grouped strictly by Season
    history_grouped = groupby(history_df, :season, sort=true)
    
    # Micro Grouping: Target grouped by the dynamics column (e.g., match_month)
    target_grouped = groupby(target_df, grouping_col, sort=true)

    # Combine them sequentially
    all_groups = vcat(collect(history_grouped), collect(target_grouped))
    
    # 5. Store metadata for the Turing Model
    F_data[:n_rounds] = length(all_groups)
    F_data[:n_history_steps] = length(history_grouped)
    F_data[:n_target_steps] = length(target_grouped)

    # --- Extract Data ---
  #  ids 
    F_data[:round_home_ids] = [ [team_map[name] for name in g.home_team] for g in all_groups]
    F_data[:round_away_ids] = [ [team_map[name] for name in g.away_team] for g in all_groups]
  # Extract the calendar month (1-12) for each match
    F_data[:round_month_ids] = [ [Dates.month(d) for d in g.match_date] for g in all_groups]
    F_data[:n_months] = 12
  # Extract Midweek binary Flag - 1 if Mon - Thu, 0 if Fri-Sun 
    F_data[:round_is_midweek] = [ [Dates.dayofweek(d) < 5 ? 1 : 0 for d in g.match_date] for g in all_groups]



    # --- Flatten Data 
    F_data[:flat_home_ids] = vcat(F_data[:round_home_ids]...)
    F_data[:flat_away_ids] = vcat(F_data[:round_away_ids]...)
    F_data[:flat_month_ids] = vcat(F_data[:round_month_ids]...)
    F_data[:flat_is_midweek] = vcat(F_data[:round_is_midweek]...)

    # Delegate Target Extraction (Goals vs Funnel)
    extract_targets!(F_data, model, all_groups)

    # Time Indices (Standard flat mapping)
    time_indices = Int[]
    for (t, round_matches) in enumerate(F_data[:round_home_ids])
        n_matches_in_round = length(round_matches)
        append!(time_indices, fill(t, n_matches_in_round))
    end
    F_data[:time_indices] = time_indices

    return FeatureSet(F_data)
end
