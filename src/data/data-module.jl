"""
doc string for the data module 
"""
module Data





# Data loading functionality
function DataFiles(path::String)
    base_dir = path
    match = joinpath(path, "football_data_mixed_matches.csv")
    odds = joinpath(path, "odds.csv")
    incidents = joinpath(path, "football_data_mixed_incidents.csv")
    return DataFiles(base_dir, match, odds, incidents)
end






end
