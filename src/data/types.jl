# src/data/types.jl

using DataFrames

export DataFiles, DataStore

"""
    DataFiles
Paths to the raw CSV files required for a DataStore.
"""
struct DataFiles
    base_dir::String
    match::String
    odds::String
    incidents::String
end

# Constructor to easily init from a base directory
function DataFiles(path::String)
    base_dir = path
    match = joinpath(path, "football_data_mixed_matches.csv")
    odds = joinpath(path, "football_data_mixed_odds.csv")
    incidents = joinpath(path, "football_data_mixed_incidents.csv")
    return DataFiles(base_dir, match, odds, incidents)
end

"""
    DataStore
Container for the three core DataFrames used in the football model.
"""
struct DataStore
    matches::DataFrame
    odds::DataFrame
    incidents::DataFrame
end
