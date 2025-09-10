# splits_module.jl
# Development module for time series splitting functionality
# Usage: include("splits_module.jl") or includet("splits_module.jl") for Revise.jl

module Splits

using DataFrames

export TimeSeriesSplitsConfig, TimeSeriesSplits
export time_series_splits, summarize_splits

# ============================================================================
# Type Definitions
# ============================================================================

"""
    TimeSeriesSplitsConfig

Configuration for time series cross-validation splits.

# Fields
- `base_seasons`: Vector of season strings to use as base training data
- `target_seasons`: Vector of season strings to incrementally add
- `round_col`: Symbol for the column containing round information
"""
struct TimeSeriesSplitsConfig 
    base_seasons::AbstractVector{AbstractString}
    target_seasons::AbstractVector{AbstractString}
    round_col::Symbol
end

# Constructor overload for backward compatibility (single season)
TimeSeriesSplitsConfig(base_seasons::AbstractVector{AbstractString}, 
                       target_season::AbstractString, 
                       round_col::Symbol) = 
    TimeSeriesSplitsConfig(base_seasons, [target_season], round_col)

"""
    TimeSeriesSplits

Iterator for time series cross-validation splits.

# Fields
- `df`: The complete DataFrame
- `base_seasons`: Vector of base season identifiers
- `target_seasons`: Vector of target season identifiers
- `round_col`: Symbol for the round column
- `base_indices`: Indices of base season rows
- `target_rounds_by_season`: Dictionary mapping seasons to their rounds
- `target_round_sequence`: Sequence of (season, round) tuples for iteration
"""
struct TimeSeriesSplits
    df::DataFrame
    base_seasons::Vector
    target_seasons::Vector
    round_col::Symbol
    base_indices::Vector{Int}
    target_rounds_by_season::Dict{String, Vector}
    target_round_sequence::Vector{Tuple{String, Any}}
end

# ============================================================================
# Constructors
# ============================================================================

"""
    TimeSeriesSplits(df, base_seasons, target_seasons, round_col=:round)

Create TimeSeriesSplits with default interleaved ordering.
"""
function TimeSeriesSplits(df::DataFrame, 
                         base_seasons::AbstractVector, 
                         target_seasons::AbstractVector, 
                         round_col::Symbol=:round)
    # Get base indices
    base_indices = findall(row -> row.season in base_seasons, eachrow(df))
    
    # Get rounds for each target season
    target_rounds_by_season = Dict{String, Vector}()
    for season in target_seasons
        season_data = filter(row -> row.season == season, df)
        if nrow(season_data) > 0
            target_rounds_by_season[String(season)] = sort(unique(season_data[!, round_col]))
        else
            target_rounds_by_season[String(season)] = []
        end
    end
    
    # Create sequence of (season, round) pairs for iteration
    target_round_sequence = Tuple{String, Any}[]
    
    # Default: interleaved ordering
    max_rounds = maximum(length(rounds) for rounds in values(target_rounds_by_season); init=0)
    
    for round_idx in 1:max_rounds
        for season in target_seasons
            season_str = String(season)
            if round_idx <= length(target_rounds_by_season[season_str])
                push!(target_round_sequence, (season_str, target_rounds_by_season[season_str][round_idx]))
            end
        end
    end
    
    return TimeSeriesSplits(df, base_seasons, target_seasons, round_col, 
                           base_indices, target_rounds_by_season, target_round_sequence)
end

"""
    TimeSeriesSplits(df, base_seasons, target_seasons, round_col, ordering)

Create TimeSeriesSplits with specified ordering strategy.

# Ordering options
- `:interleaved`: Round-by-round across all seasons (default)
- `:sequential`: Complete each season before moving to next
"""
function TimeSeriesSplits(df::DataFrame, 
                         base_seasons::AbstractVector, 
                         target_seasons::AbstractVector, 
                         round_col::Symbol, 
                         ordering::Symbol)
    base_indices = findall(row -> row.season in base_seasons, eachrow(df))
    
    target_rounds_by_season = Dict{String, Vector}()
    for season in target_seasons
        season_data = filter(row -> row.season == season, df)
        if nrow(season_data) > 0
            target_rounds_by_season[String(season)] = sort(unique(season_data[!, round_col]))
        else
            target_rounds_by_season[String(season)] = []
        end
    end
    
    target_round_sequence = Tuple{String, Any}[]
    
    if ordering == :sequential
        # Complete each season before moving to next
        for season in target_seasons
            season_str = String(season)
            for round in target_rounds_by_season[season_str]
                push!(target_round_sequence, (season_str, round))
            end
        end
    elseif ordering == :interleaved
        # Round-by-round across all seasons
        max_rounds = maximum(length(rounds) for rounds in values(target_rounds_by_season); init=0)
        for round_idx in 1:max_rounds
            for season in target_seasons
                season_str = String(season)
                if round_idx <= length(target_rounds_by_season[season_str])
                    push!(target_round_sequence, (season_str, target_rounds_by_season[season_str][round_idx]))
                end
            end
        end
    else
        error("Unknown ordering: $ordering. Use :sequential or :interleaved")
    end
    
    return TimeSeriesSplits(df, base_seasons, target_seasons, round_col, 
                           base_indices, target_rounds_by_season, target_round_sequence)
end

# ============================================================================
# Iterator Interface
# ============================================================================

Base.length(splits::TimeSeriesSplits) = length(splits.target_round_sequence) + 1

function Base.iterate(splits::TimeSeriesSplits, state=1)
    if state > length(splits)
        return nothing
    end
    
    if state == 1
        # First split: just base data
        train_view = @view splits.df[splits.base_indices, :]
        round_info = "base_only"
        return ((train_view, round_info), state + 1)
    else
        # Subsequent splits: base + incremental rounds from target seasons
        round_idx = state - 1
        current_pairs = splits.target_round_sequence[1:round_idx]
        
        # Group by season for cleaner reporting
        seasons_rounds = Dict{String, Vector}()
        for (season, round) in current_pairs
            if !haskey(seasons_rounds, season)
                seasons_rounds[season] = []
            end
            push!(seasons_rounds[season], round)
        end
        
        # Get indices for all included season-round pairs
        target_indices = Int[]
        for (season, round) in current_pairs
            append!(target_indices, 
                   findall(row -> row.season == season && row[splits.round_col] == round, 
                          eachrow(splits.df)))
        end
        
        # Combine base and target indices  
        combined_indices = vcat(splits.base_indices, target_indices)
        
        # Create informative round_info string
        round_info = _format_round_info(splits.target_seasons, seasons_rounds)
        
        train_view = @view splits.df[combined_indices, :]
        return ((train_view, round_info), state + 1)
    end
end

# ============================================================================
# Public Functions
# ============================================================================

"""
    time_series_splits(data::DataStore, cv_config::TimeSeriesSplitsConfig)

Create time series splits from DataStore using configuration.
"""
function time_series_splits(data, cv_config::TimeSeriesSplitsConfig)
    return TimeSeriesSplits(data.matches, cv_config.base_seasons, 
                           cv_config.target_seasons, cv_config.round_col)
end

"""
    time_series_splits(data::DataStore, cv_config::TimeSeriesSplitsConfig, ordering::Symbol)

Create time series splits with custom ordering strategy.
"""
function time_series_splits(data, cv_config::TimeSeriesSplitsConfig, ordering::Symbol)
    return TimeSeriesSplits(data.matches, cv_config.base_seasons, 
                           cv_config.target_seasons, cv_config.round_col, ordering)
end

"""
    time_series_splits(df::DataFrame, cv_config::TimeSeriesSplitsConfig)

Create time series splits directly from DataFrame.
"""
function time_series_splits(df::DataFrame, cv_config::TimeSeriesSplitsConfig)
    return TimeSeriesSplits(df, cv_config.base_seasons, 
                           cv_config.target_seasons, cv_config.round_col)
end

"""
    time_series_splits(df::DataFrame, cv_config::TimeSeriesSplitsConfig, ordering::Symbol)

Create time series splits directly from DataFrame with custom ordering.
"""
function time_series_splits(df::DataFrame, cv_config::TimeSeriesSplitsConfig, ordering::Symbol)
    return TimeSeriesSplits(df, cv_config.base_seasons, 
                           cv_config.target_seasons, cv_config.round_col, ordering)
end

"""
    summarize_splits(splits::TimeSeriesSplits; max_display::Int=5)

Print summary statistics for the splits configuration.
"""
function summarize_splits(splits::TimeSeriesSplits; max_display::Int=5)
    println("Time Series Splits Configuration:")
    println("  Base seasons: ", join(splits.base_seasons, ", "))
    println("  Target seasons: ", join(splits.target_seasons, ", "))
    println("  Total splits: ", length(splits))
    
    for season in splits.target_seasons
        season_str = String(season)
        n_rounds = length(get(splits.target_rounds_by_season, season_str, []))
        println("  $(season): $n_rounds rounds")
    end
    
    println("\nSplit sequence:")
    
    # Collect all split info
    split_infos = []
    for (i, (train_data, round_info)) in enumerate(splits)
        n_base = length(splits.base_indices)
        n_target = nrow(train_data) - n_base
        push!(split_infos, (i, round_info, nrow(train_data), n_base, n_target))
    end
    
    # Display with truncation if needed
    n_splits = length(split_infos)
    if n_splits <= 2 * max_display + 1
        # Show all
        for (i, round_info, n_total, n_base, n_target) in split_infos
            println("  Split $i ($round_info): $n_total rows (base: $n_base, target: $n_target)")
        end
    else
        # Show first max_display
        for j in 1:max_display
            (i, round_info, n_total, n_base, n_target) = split_infos[j]
            println("  Split $i ($round_info): $n_total rows (base: $n_base, target: $n_target)")
        end
        println("  ...")
        # Show last max_display
        for j in (n_splits - max_display + 1):n_splits
            (i, round_info, n_total, n_base, n_target) = split_infos[j]
            println("  Split $i ($round_info): $n_total rows (base: $n_base, target: $n_target)")
        end
    end
end

# ============================================================================
# Private Helper Functions
# ============================================================================

"""
    _format_round_info(target_seasons, seasons_rounds)

Format the round information string for display.
"""
function _format_round_info(target_seasons, seasons_rounds)
    if length(target_seasons) == 1
        # Single target season - use original format
        season = String(target_seasons[1])
        rounds = get(seasons_rounds, season, [])
        if length(rounds) == 0
            return "base only"
        elseif length(rounds) == 1
            return "base + $(season)_round_$(rounds[1])"
        else
            min_round = minimum(rounds)
            max_round = maximum(rounds)
            # Check if continuous
            if length(rounds) == max_round - min_round + 1
                return "base + $(season)_rounds_$(min_round)-$(max_round)"
            else
                # Non-continuous, show actual rounds
                return "base + $(season)_rounds_[$(join(rounds, ","))]"
            end
        end
    else
        # Multiple target seasons - show summary
        info_parts = String[]
        for season in target_seasons
            season_str = String(season)
            if haskey(seasons_rounds, season_str)
                rounds = seasons_rounds[season_str]
                if length(rounds) == 1
                    push!(info_parts, "$(season_str)[r$(rounds[1])]")
                else
                    min_round = minimum(rounds)
                    max_round = maximum(rounds)
                    if length(rounds) == max_round - min_round + 1
                        push!(info_parts, "$(season_str)[r$(min_round)-$(max_round)]")
                    else
                        push!(info_parts, "$(season_str)[r:$(join(rounds, ","))]")
                    end
                end
            end
        end
        return "base + " * join(info_parts, ", ")
    end
end

end # module Splits
