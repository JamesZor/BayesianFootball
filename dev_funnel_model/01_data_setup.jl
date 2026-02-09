using DataFrames
using CSV
using Dates
using InlineStrings


folder_path = "/home/james/bet_project/football/scotland_l12_extra"
files_list::Vector{String} = readdir(folder_path)


const DATA_COLS_TYPES = Dict(
    :Div => String3,
    :Date => Date,
    :Time => Time,
    :HomeTeam => String31,
    :AwayTeam => String31,
    :FTHG => Float64,
    :FTAG => Float64,
    :FTR => String1,
    :HTHG => Float64,
    :HTAG => Float64,
    :HTR => String1,
    :Referee => String15,
    :HS => Float64,
    :AS => Float64,
    :HST => Float64,
    :AST => Float64,
    :HF => Float64,
    :AF => Float64,
    :HC => Float64,
    :AC => Float64,
    :HY => Float64,
    :AY => Float64,
    :HR => Float64,
    :AR => Float64,
)


# --- Internal Helpers ---
function _loaded_dataframe(file_path::AbstractString)::AbstractDataFrame
    matches = CSV.read(file_path, DataFrame; 
    types= DATA_COLS_TYPES,
    dateformat=Dict(:Date => dateformat"dd/mm/yyyy")
    )
    return matches 
end

file_path_1 = joinpath(folder_path, files_list[1])

d1 = _loaded_dataframe(file_path_1)


file_path_2 = joinpath(folder_path, files_list[2])
d2 = _loaded_dataframe(file_path_2)
append!(d1, d2, promote=true)


function _load_dateframes(folder_dir::AbstractString)::AbstractDataFrame
    # Create a list of all loaded DataFrames
    all_dfs = [ _loaded_dataframe(joinpath(folder_dir, f)) 
                for f in readdir(folder_dir) if endswith(f, ".csv") ]
    
    # Vertically concatenate them all at once
    return vcat(all_dfs..., cols=:union)
end

df = _load_dateframes(folder_path)

###
# name mapping
###

# get all the names

team_names_list = unique( df.HomeTeam)

using BayesianFootball
data_store = BayesianFootball.Data.load_default_datastore()
ds_l12 = subset(data_store.matches, :tournament_id => ByRow(in([56,57])), 
                                    :season => ByRow(in(["21/22", "22/23", "23/24", "24/25"])),
                )

projection_team_names_list = unique(ds_l12.home_team)



const TEAM_NAME_MAPPING = Dict(
    # --- The Mismatches & Abbreviations ---
    "Airdrie Utd"    => "airdrieonians",
    "Albion Rvs"     => "albion-rovers",
    "Alloa"          => "alloa-athletic",
    "Clyde"          => "clyde-fc",
    "Dunfermline"    => "dunfermline-athletic",
    "Elgin"          => "elgin-city",
    "Falkirk"        => "falkirk-fc",
    "Forfar"         => "forfar-athletic",
    "Hamilton"       => "hamilton-academical",
    "Inverness C"    => "inverness-caledonian-thistle",
    "Kelty Hearts"   => "kelty-hearts-fc",
    "Queen of Sth"   => "queen-of-the-south",
    "Queens Park"    => "queens-park-fc",
    "Spartans"       => "the-spartans-fc",
    "Stirling"       => "stirling-albion",

    # --- The Duplicate Case (Rebranding) ---
    "Edinburgh City" => "edinburgh-city-fc",
    "FC Edinburgh"   => "edinburgh-city-fc", 

    # --- The Exact or Near-Exact Matches ---
    "Annan Athletic" => "annan-athletic",
    "Arbroath"       => "arbroath",
    "Bonnyrigg Rose" => "bonnyrigg-rose",
    "Cove Rangers"   => "cove-rangers",
    "Cowdenbeath"    => "cowdenbeath",
    "Dumbarton"      => "dumbarton",
    "East Fife"      => "east-fife",
    "Montrose"       => "montrose",
    "Peterhead"      => "peterhead",
    "Stenhousemuir"  => "stenhousemuir",
    "Stranraer"      => "stranraer"
)

# for the df Div
const TOURNAMENT_MAPPING = Dict(
      "SC2" => 56,
      "SC3" => 57
)


ds_l12

names(ds_l12)

# Apply the mapping to your DataFrame
# This creates a new column :team_id that matches the projection format
df.home_team_id = [get(TEAM_NAME_MAPPING, name, missing) for name in df.HomeTeam]
df.away_team_id = [get(TEAM_NAME_MAPPING, name, missing) for name in df.AwayTeam]



ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)








#=
julia> names(d1)
106-element Vector{String}:
 "Div"
 "Date"
 "Time"
 "HomeTeam"
 "AwayTeam"
 "FTHG"
 "FTAG"
 "FTR"
 "HTHG"
 "HTAG"
 "HTR"
 "Referee"
 "HS"
 "AS"
 "HST"
 "AST"
 "HF"
 "AF"
 "HC"
 "AC"
 "HY"
 "AY"
 "HR"
 "AR"

=#






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

function DataStore(data_files::DataFiles)
    matches = _get_data_matches(data_files.match)
    incidents = _get_data_incidents(data_files.incidents)
    odds = _get_data_odds(data_files.odds)
    return DataStore(matches, odds, incidents)
end

function load_default_datastore() 
  return DataStore(DataFiles(DataPaths.scotland))
end
