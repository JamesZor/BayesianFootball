# src/data/scotland_extra.jl

using CSV
using DataFrames
using Dates
using InlineStrings

# --- Constants & Mappings ---

const SCOT_EXTRA_COLS_TYPES = Dict(
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

const SCOT_TEAM_MAPPING = Dict(
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
    "Stranraer"      => "stranraer",
    "Partick"       => "partick-thistle",
    "Brechin"       => "brechin-city",
    "East Kilbride" => "east-kilbride",
)

const SCOT_TOURNAMENT_MAPPING = Dict(
      "SC2" => 56,
      "SC3" => 57
)

# --- Loading Logic ---

function _get_season_from_filename(filename::AbstractString)
    # Extracts "2122" from "l1_2122.csv" and returns "21/22"
    raw_digits = split(split(basename(filename), "_")[2], ".")[1]
    return "$(raw_digits[1:2])/$(raw_digits[3:4])"
end

function _load_single_scot_csv(file_path::AbstractString)
    df = CSV.read(file_path, DataFrame; 
        types = SCOT_EXTRA_COLS_TYPES,
        dateformat = Dict(:Date => dateformat"dd/mm/yyyy")
    )
    season_str = _get_season_from_filename(file_path)
    insertcols!(df, :Season => season_str)
    return df 
end

function _load_scot_folder(folder_dir::AbstractString)
    all_dfs = [ _load_single_scot_csv(joinpath(folder_dir, f)) 
                for f in readdir(folder_dir) if endswith(f, ".csv") ]
    
    if isempty(all_dfs)
        error("No CSV files found in $folder_dir")
    end
    
    return vcat(all_dfs..., cols=:union)
end

# --- Integration Logic ---

"""
    enrich_with_scotland_extra(ds::DataStore, extra_data_folder::String)

Loads the additional Scotland data, merges it with the existing matches to add 
shot statistics (HS, AS, HST, AST), and returns a NEW DataStore.
"""
function enrich_with_scotland_extra(ds::DataStore, folder_path::String)
    println("... Loading Extra Scotland Data from: $folder_path")
    
    # 1. Load Raw Extra Data
    extra_df = _load_scot_folder(folder_path)

    # 2. Map IDs to match DataStore
    extra_df.home_team_id = [get(SCOT_TEAM_MAPPING, name, missing) for name in extra_df.HomeTeam]
    extra_df.away_team_id = [get(SCOT_TEAM_MAPPING, name, missing) for name in extra_df.AwayTeam]
    extra_df.tournament_id = [get(SCOT_TOURNAMENT_MAPPING, div, missing) for div in extra_df.Div]

    # 3. Validation (Optional but recommended)
    # _validate_counts(extra_df, ds.matches)

    # 4. Select Columns to Merge
    # We only want the stats, plus the keys for joining
    cols_to_merge = [:Season, :tournament_id, :Date, :home_team_id, :away_team_id, 
                     :HS, :AS, :HST, :AST, :HC, :AC, :HF, :AF, :Referee, :HY, :AY, :HR, :AY] # Add others if needed
    
    merge_source = select(extra_df, cols_to_merge)
    
    # Rename keys to match DataStore for joining
    # DataStore keys: :season, :tournament_id, :match_date, :home_team, :away_team
    rename!(merge_source, 
        :Season => :season, 
        :Date => :match_date,
        :home_team_id => :home_team, 
        :away_team_id => :away_team
    )

    # 5. Perform the Join
    # We use leftjoin on the original matches to ensure we keep all original rows
    # and just add stats where available.
    enriched_matches = leftjoin(
        Data.add_match_week_column(
              subset(ds.matches, :tournament_id => ByRow(in([56,57])))
              ),
        merge_source, 
        on = [:season, :tournament_id, :match_date, :home_team, :away_team],
        makeunique = true
    )

    # 6. Return new DataStore
    # We keep the original odds and incidents for now
    println("... Merged $(nrow(merge_source)) rows of extra stats.")
    return DataStore(enriched_matches, ds.odds, ds.incidents)
end


function load_extra_ds()
  ds = Data.load_default_datastore()
  # 2. Enrich with New Data
  folder_path = "/home/james/bet_project/football/scotland_l12_extra"
  ds_enriched = Data.enrich_with_scotland_extra(ds, folder_path)
  return ds_enriched
end
