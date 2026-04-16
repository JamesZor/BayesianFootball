# src/DataLegacy/io.jl

using CSV
using DataFrames

export load_default_datastore

# --- Internal Helpers ---

function _get_data_matches(file_path)
    matches = CSV.read(file_path, DataFrame; 
        types=MATCHES_COLS_TYPES, 
        dateformat=Dict(:match_date => dateformat"yyyy-mm-dd")
    )
    cols_to_convert = [:home_score, :away_score, :home_score_ht, :away_score_ht, :winner_code, :round]
    for col in cols_to_convert
        matches[!, col] = [ismissing(x) ? missing : Int(x) for x in matches[!, col]]
    end
    return matches 
end

function _get_data_incidents(file_path)
    incidents = CSV.read(file_path, DataFrame; types=INCIDENTS_COLS_TYPES)
    cols_to_convert = [:home_score, :away_score]
    for col in cols_to_convert
        incidents[!, col] = [ismissing(x) ? missing : Int(x) for x in incidents[!, col]]
    end
    return incidents
end

function _get_data_odds(file_path)
    return CSV.read(file_path, DataFrame; types=ODDS_COLS_TYPES)
end

# --- Constructors ---
# HACK: 2026-04-15
function DataStore(data_files::DataFiles)
    matches = _get_data_matches(data_files.match)
    incidents = _get_data_incidents(data_files.incidents)
    odds = _get_data_odds(data_files.odds)
  
return DataStore_wrapper(matches=matches, incidents=incidents , odds= odds)
end 


"""
    DataStore_wrapper(; kwargs...)

A shim to allow legacy code to create a modern DataStore. 
Defaults missing fields to empty DataFrames.
"""
function DataStore_wrapper(;
    matches::AbstractDataFrame   = DataFrame(),
    stats::AbstractDataFrame     = DataFrame(),
    lineups::AbstractDataFrame   = DataFrame(),
    incidents::AbstractDataFrame = DataFrame(),
    odds::AbstractDataFrame      = DataFrame()
)
    # 1. Get your placeholder segment
    segment_holder = Data.DevLegacyDataStore()

    tmp_ds = Data.DataStore(
        segment_holder,
        matches,
        stats,
        odds,
        lineups,
        incidents
    )
    
    md = prepare_market_data(tmp_ds)

    # 2. Return the new Type
    # Note: Order must match your struct: (segment, matches, stats, odds, lineups, incidents)
    return Data.DataStore(
        segment_holder,
        matches,
        stats,
        md.df,
        lineups,
        incidents
    )
end

function load_default_datastore() 
  return DataStore(DataFiles(DataPaths.scotland))
end
