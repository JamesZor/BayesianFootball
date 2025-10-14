"""
doc string for the data module 
"""
module Data

using DataFrames, CSV, Dates
using InlineStrings


export load_default_datastore, DataStore


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

const NameType = String63 # <-- Add this line

const MATCHES_COLS_TYPES = Dict(
      :tournament_id => Int64,
      :season_id => Int64,
      :season => String7,
      :match_id => Int64,
      :tournament_slug => String15,
      :home_team => String31,
      :away_team => String31,
      # Change these next 5 lines from Int64 to Float64
      :home_score => Float64,
      :away_score => Float64,
      :home_score_ht => Float64,
      :away_score_ht => Float64,
      :winner_code => Float64,
      # Keep the rest the same
      :match_date => Date,
      :round => Float64,
      :has_xg => Bool,
      :has_stats => Bool,
      :match_hour => Int64,
      :match_dayofweek => Int64,
      :match_month => Int64
  )

const INCIDENTS_COLS_TYPES = Dict(
    :tournament_id => Int64,
    :season_id => Int64,
    :match_id => Int64,
    :incident_type => String15,
    :time => Int64,
    :is_home => Bool,
    :is_live => Bool,
    :period_text => String3,
    :player_in_name => NameType,
    :player_out_name => NameType,
    :is_injury => Bool,
    :player_name => NameType,
    :card_type => String15,
    :assist1_name => NameType,
    :assist2_name => NameType,
    # --- Load these as Float64 to handle potential missing/float values ---
    :home_score => Float64,
    :away_score => Float64,
    :added_time => Float64,
    :time_seconds => Float64,
    :reversed_period_time => Float64,
    :reversed_period_time_seconds => Float64,
    :period_time_seconds => Float64,
    :injury_time_length => Float64,
    # --- Continue with other types ---
    :incident_class => String31,
    :reason => String31,
    :rescinded => Bool,
    :var_confirmed => Bool,
    :var_decision => String31,
    :var_reason => String31,
    :penalty_reason => String31
)

const ODDS_COLS_TYPES = Dict(
    :tournament_id => Int64,
    :season_id => Int64,
    :match_id => Int64,
    :market_id => Int64,
    :market_name => String31,
    :market_group => String31,
    # choice_name can be long (team names), so use a larger string or standard String
    :choice_name => String63, 
    :choice_group => Float64,
    # Fractional values are text, so they must be loaded as strings
    :initial_fractional_value => String7,
    :final_fractional_value => String7,
    :winning => Bool,
    :decimal_odds => Float64
)


##############################
# Helper functions
##############################
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


##############################
# Main functions
##############################

# Data loading functionality
function DataFiles(path::String)
    base_dir = path
    match = joinpath(path, "football_data_mixed_matches.csv")
    odds = joinpath(path, "football_data_mixed_odds.csv")
    incidents = joinpath(path, "football_data_mixed_incidents.csv")
    return DataFiles(base_dir, match, odds, incidents)
end


# main data functions 
function DataStore(data_files::DataFiles)
    matches = _get_data_matches(data_files.match)
    incidents = _get_data_incidents(data_files.incidents)
    odds = _get_data_odds(data_files.odds)
    return DataStore(matches, odds, incidents)
end

function load_default_datastore() 
  return DataStore(DataFiles( DataPaths.scotland))
end

end
