# src/data/types.jl

"""
    DBConfig
Configuration struct for establishing a connection to the PostgreSQL database.
"""
struct DBConfig 
    url::String
end 

abstract type DataTournemantSegment end


abstract type FootballDataType end

"""
    DataStore
The central data structure holding all processed DataFrames for a specific segment.
Once the fetchers complete their asynchronous tasks, they populate this struct.
"""
struct DataStore
    segment::DataTournemantSegment
    matches::DataFrame
    statistics::DataFrame
    odds::DataFrame
    lineups::DataFrame
    incidents::DataFrame
end
