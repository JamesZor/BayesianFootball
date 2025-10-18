"""
    TimeSeriesSplits

Iterator for time series cross-validation splits.

# Fields
- `base_indices`: Indices of base season rows
- `target_rounds_by_season`: Dictionary mapping seasons to their rounds
- `target_round_sequence`: Sequence of (season, round) tuples for iteration
"""
struct TimeSeriesSplits
    base_indices::Vector{Int}
    target_rounds_by_season::Dict{String, Vector}
    target_round_sequence::Vector{Tuple{String, Any}}
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
    
    return TimeSeriesSplits(base_indices, target_rounds_by_season, target_round_sequence)
end

