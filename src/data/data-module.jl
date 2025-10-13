"""
doc string for the data module 
"""
module Data

using DataFrames, CSV, Dates



# abstracts 
struct DataFiles
    base_dir::String
    match::String
    odds::String
    incidents::String
end

struct DataStore
    matches::DataFrame
    odds::DataFrame
    incidents::DataFrame
end

const DataPaths = (
    scotland = "/home/james/bet_project/football/scotland_football",
    uk_all   = "/home/james/bet_project/football/uk_football_data_20_26", # not on laptop
)


# Data loading functionality
function DataFiles(path::String)
    base_dir = path
    match = joinpath(path, "football_data_mixed_matches.csv")
    odds = joinpath(path, "football_data_mixed_odds.csv")
    incidents = joinpath(path, "football_data_mixed_incidents.csv")
    return DataFiles(base_dir, match, odds, incidents)
end


function DataStore(data_files::DataFiles)
    # Read the CSV files with appropriate date formats for each file
    matches = CSV.read(data_files.match, DataFrame; 
        dateformat=Dict(:match_date => dateformat"yyyy-mm-dd"))
    odds = CSV.read(data_files.odds, DataFrame; 
        dateformat=Dict(:timestamp => dateformat"yyyy-mm-dd HH:MM:SS"))
    incidents = CSV.read(data_files.incidents, DataFrame; )

    return DataStore(matches, odds, incidents)
end



end
