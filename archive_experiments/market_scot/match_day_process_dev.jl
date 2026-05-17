using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

using DataFramesMeta


# load the data store for league 1  and league 2
ds = Data.load_extra_ds()


df_56 = subset(ds.matches, :tournament_id => ByRow(isequal(56)), :season => ByRow(isequal("25/26")))
df_57 =subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))


# loaded the sampled models files 

saved_folders = Experiments.list_experiments("exp/market_runs"; data_dir="./data")
# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

m1 = loaded_results[1]

m2 = loaded_results[1]
m1 = loaded_results[2]

feature_collection1 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m1.config.splitter),
    m1.config.model, 
    m1.config.splitter
)


feature_collection2 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m2.config.splitter),
    m2.config.model, 
    m2.config.splitter
)

last_split_idx = length(m1.training_results)

# Get the Chain from the results
chain1 = m1.training_results[last_split_idx][1]
# Get the corresponding FeatureSet from our regenerated list
# Note: [1] gets the FeatureSet, [2] would be the Metadata
feature_set1 = feature_collection1[last_split_idx][1]
#=
julia> feature_set1.data[:team_map]
Dict{String, Int64} with 10 entries:
  "alloa-athletic"               => 1
  "kelty-hearts-fc"              => 6
  "montrose"                     => 7
  "cove-rangers"                 => 2
  "queen-of-the-south"           => 9
  "hamilton-academical"          => 4
  "peterhead"                    => 8
  "east-fife"                    => 3
  "inverness-caledonian-thistle" => 5
  "stenhousemuir"                => 10
=#
chain2 = m2.training_results[last_split_idx][1]
feature_set2 = feature_collection2[last_split_idx][1]

#=
julia> feature_set2.data[:team_map]
Dict{String, Int64} with 10 entries:
  "edinburgh-city-fc" => 5
  "dumbarton"         => 3
  "east-kilbride"     => 4
  "stranraer"         => 9
  "the-spartans-fc"   => 10
  "clyde-fc"          => 2
  "elgin-city"        => 6
  "forfar-athletic"   => 7
  "stirling-albion"   => 8
  "annan-athletic"    => 1
=#



# ---- ++++++

using Dates

# Represents a match found in the schedule
struct ScraperMatch
    id::String            
    event_name::String    
    start_time::Time      
    date::Date            
    league::String
end

# Represents a single price (Back and Lay) for a selection
struct ScraperRow
    match_id::String
    event_name::String
    market_name::String   # Matches Python: "Match Odds", "Over/Under 2.5 Goals"
    selection::Symbol     # Normalized: :home, :over_25
    back_price::Float64   
    lay_price::Float64    
    timestamp::DateTime
end

using DataFrames
using JSON3
using Dates
using CSV
using Statistics
using SHA

using Dates
using JSON3
using DataFrames

# --- CONFIGURATION (Matched to your Python Script) ---
const PYTHON_EXE = "/home/james/.conda/envs/webscrape/bin/python"
const CLI_PATH = "/home/james/bet_project/whatstheodds/live_odds_cli.py"

# Wrapper to run the command and capture STDOUT (ignoring STDERR logs)
function _run_cli(args::Vector{String})
    # We run in the CLI directory so it can find 'mappings/'
    cmd = Cmd(vcat([PYTHON_EXE, CLI_PATH], args))
    try
        return read(setenv(cmd, dir=dirname(CLI_PATH)), String)
    catch e
        @error "Python CLI Failed: $e"
        return nothing
    end
end

"""
    get_matches(leagues; time_window_hours=nothing)
    
    Fetches today's matches. Optional: filter for matches starting soon.
"""
function get_matches(leagues::Vector{String}; time_window_hours::Union{Float64, Nothing}=nothing)
    output = _run_cli(["list", "-f", leagues...])
    isnothing(output) && return ScraperMatch[]

    matches = ScraperMatch[]
    current_time = now()
    today_date = today()

    # Parse output lines like: "  - 19:45 | East Kilbride v Spartans"
    for line in split(output, '\n')
        m = match(r"^\s*-\s*(\d{2}:\d{2})\s*\|\s*(.+?)\s+v\s+(.+?)$", line)
        if !isnothing(m)
            t_str, home, away = m.captures
            match_time = Time(t_str, "HH:MM")
            
            # --- Time Filter ---
            if !isnothing(time_window_hours)
                match_dt = DateTime(today_date, match_time)
                diff = (match_dt - current_time) / Hour(1)
                # Keep if in future (diff > -0.5) AND within window
                if diff < -0.5 || diff > time_window_hours
                    continue 
                end
            end
            
            # Create a unique ID for this run
            event_name = "$home v $away"
            id_str = bytes2hex(sha256(event_name * string(today_date)))[1:8]
            push!(matches, ScraperMatch(id_str, event_name, match_time, today_date, "Unknown"))
        end
    end
    return matches
end

"""
    fetch_odds(matches)
    
    Calls 'odds -d' for each match and parses the JSON.
"""
function fetch_odds(matches::Vector{ScraperMatch})
    results = ScraperRow[]
    for m in matches
        json_str = _run_cli(["odds", m.event_name, "-d"])
        
        if !isnothing(json_str) && !isempty(json_str)
            try
                # FIX: Read as Dict to ensure keys are Strings, not Symbols
                data = JSON3.read(json_str, Dict)
                _parse_data!(results, m, data)
            catch e
                @warn "Error processing data for $(m.event_name): $e"
            end
        end
    end
    return DataFrame(results)
end

function _parse_data!(out, m, data)
    # Helper to extracting price from the "back"/"lay" dictionaries
    # Keys are now Strings: "back", "price"
    get_p(o) = (
        Float64(get(get(o, "back", Dict()), "price", NaN)), 
        Float64(get(get(o, "lay", Dict()), "price", NaN))
    )

    # We use String keys for lookups now
    # 1. FT Section
    if haskey(data, "ft")
        ft = data["ft"]

        # 1X2 Match Odds
        if haskey(ft, "Match Odds")
            for (t, o) in ft["Match Odds"]
                # 't' is now a String (e.g. "Inverness CT")
                sel = t == "The Draw" ? :draw : (startswith(m.event_name, t) ? :home : :away)
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "1X2", sel, b, l, now()))
            end
        end

        # Over/Under 2.5
        if haskey(ft, "Over/Under 2.5 Goals")
            for (k, o) in ft["Over/Under 2.5 Goals"]
                sel = startswith(k, "Over") ? :over_25 : :under_25
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "OverUnder", sel, b, l, now()))
            end
        end


        if haskey(ft, "Over/Under 1.5 Goals")
            for (k, o) in ft["Over/Under 1.5 Goals"]
                sel = startswith(k, "Over") ? :over_15 : :under_15
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "OverUnder", sel, b, l, now()))
            end
        end


        if haskey(ft, "Over/Under 0.5 Goals")
            for (k, o) in ft["Over/Under 0.5 Goals"]
                sel = startswith(k, "Over") ? :over_05 : :under_05
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "OverUnder", sel, b, l, now()))
            end
        end

        if haskey(ft, "Over/Under 3.5 Goals")
            for (k, o) in ft["Over/Under 3.5 Goals"]
                sel = startswith(k, "Over") ? :over_35 : :under_35
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "OverUnder", sel, b, l, now()))
            end
        end

        if haskey(ft, "Over/Under 4.5 Goals")
            for (k, o) in ft["Over/Under 4.5 Goals"]
                sel = startswith(k, "Over") ? :over_45 : :under_45
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "OverUnder", sel, b, l, now()))
            end
        end

        if haskey(ft, "Over/Under 5.5 Goals")
            for (k, o) in ft["Over/Under 5.5 Goals"]
                sel = startswith(k, "Over") ? :over_55 : :under_55
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "OverUnder", sel, b, l, now()))
            end
        end

        # BTTS
        if haskey(ft, "Both teams to Score?")
            for (k, o) in ft["Both teams to Score?"]
                sel = k == "Yes" ? :btts_yes : :btts_no
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "BTTS", sel, b, l, now()))
            end
        end
    end
end

"""
    load_mappings(mapping_dir)

Reads all .json files in the folder and returns a Dict:
"Scraper Name" => "model-id"
"""
function load_mappings(mapping_dir::String=joinpath(dirname(CLI_PATH), "mappings"))
    # We want:  "Arbroath" -> "arbroath"
    # But JSON is: "arbroath": "Arbroath"
    
    master_map = Dict{String, String}()
    
    if !isdir(mapping_dir)
        @warn "Mapping directory not found: $mapping_dir"
        return master_map
    end
    
    for file in readdir(mapping_dir)
        if endswith(file, ".json")
            try
                path = joinpath(mapping_dir, file)
                data = JSON3.read(read(path, String), Dict)
                
                for (model_id, scraper_name) in data
                    # REVERSE THE MAPPING HERE
                    master_map[scraper_name] = model_id
                end
            catch e
                @warn "Could not read mapping file $file: $e"
            end
        end
    end
    
  master_map["Spartans"] = "the-spartans-fc"
  master_map["East Kilbride"] = "east-kilbride"
    return master_map
end

#### 

# end # module


println("=== 1. Testing Match List Fetching ===")
# We ask for a wide time window (48 hours) to ensure we find *something* for testing purposes.
# Change "Premiership" to whatever league has games today/tomorrow.
leagues_to_test = ["scotland"]
matches = get_matches(leagues_to_test, time_window_hours=48.0)


# ---- matches -> filter [ get the matches for the leages we want ] --> reduce_matches 
# dev 
matches
#= 
julia> matches
17-element Vector{ScraperMatch}:
 ScraperMatch("b2afbfc7", "Arbroath v Ayr", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("430347b5", "Spartans v Edinburgh City", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("0e1d67b7", "Elgin City FC v Forfar", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("91516d1c", "Stirling v Clyde", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("240f1942", "Annan v East Kilbride", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("7a976ad5", "Dumbarton v Stranraer", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("a9bf70b6", "Stenhousemuir v Montrose", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("38f83fa6", "Queen of South v Alloa", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("82e0ff03", "East Fife v Cove Rangers", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("26dd5c53", "Kelty Hearts v Peterhead", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("37343a07", "Hibernian v St Mirren", Time(15), Date("2026-02-14"), "Unknown")
 ScraperMatch("dbcdefb5", "Falkirk v Dundee Utd", Time(15), Date("2026-02-14"), "Unknown")
=#

match_1 = matches[1]
# ScraperMatch("b2afbfc7", "Arbroath v Ayr", Time(15), Date("2026-02-14"), "Unknown")
# map the team name
#  get and regex the event name 
event_name = match_1.event_name
# "Arbroath v Ayr"
home_str, away_str = split(event_name, " v ")
#=
2-element Vector{SubString{String}}:
 "Arbroath"
 "Ayr"
=#

team_mappings = load_mappings() 

home_name = team_mappings[home_str]
away_name = team_mappings[away_str]

home_name = get(team_mappings, home_str, missing) 
# "arbroath" 

away_name = get(team_mappings, away_str, missing) 
# "ayr-united"
team_names_l1 =unique(
  subset(ds.matches, 
         :tournament_id => ByRow(isequal(56)),
         :season => ByRow(isequal("25/26"))).home_team
)
#=
julia> team_names_l1
10-element Vector{String31}:
 "cove-rangers"
 "hamilton-academical"
 "kelty-hearts-fc"
 "peterhead"
 "stenhousemuir"
 "east-fife"
 "inverness-caledonian-thistle"
 "montrose"
 "queen-of-the-south"
 "alloa-athletic"

=#


team_names_l2 =unique(
  subset(ds.matches, 
         :tournament_id => ByRow(isequal(57)),
         :season => ByRow(isequal("25/26"))).home_team
)
#=
10-element Vector{String31}:
 "annan-athletic"
 "dumbarton"
 "east-kilbride"
 "edinburgh-city-fc"
 "forfar-athletic"
 "clyde-fc"
 "elgin-city"
 "the-spartans-fc"
 "stirling-albion"
 "stranraer"
=#



match_league =:other
if home_name in team_names_l1 && away_name in team_names_l1
  match_league = :l1
elseif home_name in team_names_l2 && away_name in team_names_l2
  match_league = :l2
end 

# add to a main DataFrames
# the ScraperMatch datafield plus the home_str, away_str, home_team, away_team (the mapped so we can check )
# the leage Symbol 

# abstract the above steps in to a funciton 
# then a processor function 
# processor [ matches] -> filtered matches 

# so we can get the odds like this since fecth odds use the 
#julia> typeof(matches)
# Vector{ScraperMatch} (alias for Array{ScraperMatch, 1})
# else we can create a wrappper so we fetch odds can take a df
df_odds = fetch_odds(matches)




using DataFrames, Dates, JSON3, CSV



# --- 1. CONFIGURATION: League Definitions ---
# We use Sets for O(1) fast lookups
const TEAM_NAMES_L1 = Set([
    "cove-rangers", "hamilton-academical", "kelty-hearts-fc", "peterhead",
    "stenhousemuir", "east-fife", "inverness-caledonian-thistle", 
    "montrose", "queen-of-the-south", "alloa-athletic"
])

const TEAM_NAMES_L2 = Set([
    "annan-athletic", "dumbarton", "east-kilbride", "edinburgh-city-fc",
    "forfar-athletic", "clyde-fc", "elgin-city", "the-spartans-fc",
    "stirling-albion", "stranraer"
])

# --- 2. CORE PROCESSING FUNCTION ---

"""
    enrich_matches(matches::Vector{ScraperMatch}, mappings::Dict)

Converts raw ScraperMatches into a rich DataFrame with mapped IDs and League info.
"""
function enrich_matches(matches::Vector{ScraperMatch}, mappings::Dict)
    # Pre-allocate vectors for the DataFrame columns
    ids = String[]
    events = String[]
    home_strs = String[]
    away_strs = String[]
    home_ids = Union{String, Missing}[]
    away_ids = Union{String, Missing}[]
    leagues = Symbol[]
    
    for m in matches
        # 1. Parse Strings
        parts = split(m.event_name, " v ")
        if length(parts) != 2
            @warn "Skipping malformed event name: $(m.event_name)"
            continue
        end
        h_str, a_str = parts[1], parts[2]
        
        # 2. Map to Canonical IDs
        # get(dict, key, missing) is the best pattern here
        h_id = get(mappings, h_str, missing)
        a_id = get(mappings, a_str, missing)
        
        # 3. Determine League
        league = :other # Default
        
        if !ismissing(h_id) && !ismissing(a_id)
            if (h_id in TEAM_NAMES_L1) && (a_id in TEAM_NAMES_L1)
                league = :l1
            elseif (h_id in TEAM_NAMES_L2) && (a_id in TEAM_NAMES_L2)
                league = :l2
            end
        elseif ismissing(h_id) || ismissing(a_id)
            # Log missing teams for debugging
            league = :unknown_mapping
            if ismissing(h_id) println("MISSING MAPPING: $h_str") end
            if ismissing(a_id) println("MISSING MAPPING: $a_str") end
        end

        # 4. Push to vectors
        push!(ids, m.id)
        push!(events, m.event_name)
        push!(home_strs, h_str)
        push!(away_strs, a_str)
        push!(home_ids, h_id)
        push!(away_ids, a_id)
        push!(leagues, league)
    end

    return DataFrame(
        :id => ids,
        :event_name => events,
        :home_str => home_strs,
        :away_str => away_strs,
        :home_id => home_ids,
        :away_id => away_ids,
        :league => leagues
    )
end

# --- 3. WRAPPER FOR FETCH ODDS ---

"""
    fetch_odds(df::DataFrame)

Overload of fetch_odds that accepts the filtered DataFrame.
"""
function fetch_odds(df::DataFrame)
    results = ScraperRow[]
    
    if isempty(df)
        @warn "No matches provided to fetch_odds"
        return DataFrame(results)
    end

    # Iterate over DataFrame rows
    for row in eachrow(df)
        println("Fetching odds for: $(row.event_name) [$(row.league)]")
        
        # Call CLI using the event name from the dataframe
        json_str = _run_cli(["odds", row.event_name, "-d"])
        
        if !isnothing(json_str) && !isempty(json_str)
            try
                data = JSON3.read(json_str, Dict)
                # Reconstruct a dummy ScraperMatch for the parser or refactor parser
                # Here we just pass the ID and Event Name to the existing parser
                dummy_match = (id=row.id, event_name=row.event_name) 
                _parse_data!(results, dummy_match, data)
            catch e
                @warn "Error processing odds for $(row.event_name): $e"
            end
        end
    end
    return DataFrame(results)
end

# --- 4. EXECUTION SCRIPT ---

println("--- Starting Pipeline ---")

# A. Load Resources
mappings = load_mappings()

# B. Get Raw Matches
println("Fetching matches...")
raw_matches = get_matches(["scotland"], time_window_hours=48.0)

# C. Enrich & Classify
println("Enriching data...")
df_all = enrich_matches(raw_matches, mappings)

# D. Debug: See what we are missing
missing_map_df = filter(row -> row.league == :unknown_mapping, df_all)
if !isempty(missing_map_df)
    println("\n⚠️  WARNING: The following matches have missing mappings and will be skipped:")
    display(missing_map_df[:, [:event_name, :home_str, :away_str]])
end

# E. Filter for Target Leagues
# We select L1 and L2
target_df = filter(row -> row.league in [:l1, :l2], df_all)
target_df.id = string.(1:nrow(target_df))
println("\n✅ Found $(nrow(target_df)) matches in League 1 & 2:")
display(target_df[:, [:event_name, :league, :home_id, :away_id]])


"""
    fetch_odds(df::DataFrame)

Overload of fetch_odds that accepts the filtered DataFrame.
"""
function fetch_odds(df::DataFrame)
    results = ScraperRow[]
    
    if isempty(df)
        @warn "No matches provided to fetch_odds"
        return DataFrame(results)
    end

    # Iterate over DataFrame rows
    for row in eachrow(df)
        println("Fetching odds for: $(row.event_name) [$(row.league)]")
        
        # Call CLI using the event name from the dataframe
        json_str = _run_cli(["odds", row.event_name, "-d"])
        
        if !isnothing(json_str) && !isempty(json_str)
            try
                data = JSON3.read(json_str, Dict)
                
                # We create a lightweight object (NamedTuple) that acts like ScraperMatch
                # This allows _parse_data! to work without needing to change it.
                proxy_match = (id=row.id, event_name=row.event_name)
                
                _parse_data!(results, proxy_match, data)
            catch e
                @warn "Error processing odds for $(row.event_name): $e"
            end
        end
    end
    odds_df = DataFrame(results)
    meta_data = df[:, [:id, :home_id, :away_id, :league]]
    rename!(meta_data, :id => :match_id)
    final_df = leftjoin(odds_df, meta_data, on = :match_id)

    return final_df
end


# F. Fetch Odds for Targets
odds_df = fetch_odds(target_df)



#  
"""
    get_ppd_for_match(model, home_id, away_id)

Creates the 1-row input DataFrame and generates the PPD using the model.
"""
function get_ppd_for_match(model, home_id::String, away_id::String)
    # 1. Construct the input DataFrame the model expects
    # We use match_week = 999 as a placeholder for "Next Game"
    input_df = DataFrame(
        :match_id => [1], 
        :match_week => [999], 
        :home_team => [home_id], 
        :away_team => [away_id]
    )

    # 2. Generate Prediction
    # ADAPT THIS LINE to however you usually generate 'ppd1'
    # Example: ppd = predict(model, input_df)
    ppd = BayesianFootball.Models.predict(model, input_df) 
    
    return ppd
end



using DataFrames

# Configuration: Map Scraper League Symbols to Tournament IDs
const LEAGUE_TO_TOURNAMENT = Dict(
    :l1 => 56,
    :l2 => 57
)

function process_live_matches(odds_df::DataFrame, models::Vector, signals::Vector)
    all_bets = DataFrame()

    println("--- Processing $(length(unique(odds_df.match_id))) Matches ---")

    # 1. Group by Match so we handle one game at a time
    for gdf in groupby(odds_df, :match_id)
        
        # --- A. Setup Metadata ---
        match_row = first(gdf)
        league = match_row.league
        home_id = match_row.home_id
        away_id = match_row.away_id
        event_name = match_row.event_name
        
        # --- B. Model Selector (The Router) ---
        target_tourn_id = get(LEAGUE_TO_TOURNAMENT, league, nothing)
        
        # Find the model that was trained on this tournament ID
        active_model = nothing
        for m in models
            if !isnothing(target_tourn_id) && target_tourn_id in m.config.splitter.tournament_ids
                active_model = m
                break
            end
        end

        if isnothing(active_model)
            # @warn "No model found for $league ($event_name). Skipping."
            continue
        end

        # --- C. Get PPD (Prediction) ---
        # Wrap in try/catch in case of missing teams in the model
        ppd = try
            get_ppd_for_match(active_model, home_id, away_id)
        catch e
            @warn "Prediction failed for $event_name: $e"
            continue
        end

        # --- D. Prepare Odds DataFrame ---
        # process_signals expects a specific format.
        # We need to ensure column names match what your Signals expect.
        # 1. Copy the group
        current_odds = DataFrame(gdf)
        
        # 2. Rename 'back_price' to 'odds' (as per your example)
        rename!(current_odds, :back_price => :odds)
        
        # 3. Ensure 'selection' is the correct type (Symbol or String) matches your PPD keys
        # (Assuming your scraper returns Symbols :home, :away, your code seems to handle this)

        # --- E. Calculate Signals (The Existing Workflow) ---
        try
            match_results = BayesianFootball.Signals.process_signals(
                ppd, 
                current_odds, 
                signals; 
                odds_column=:odds
            )
            
            # Attach the Real Event Name for readability later
            match_results.event_name .= event_name
            
            append!(all_bets, match_results)
        catch e
            @warn "Signal processing failed for $event_name: $e"
        end
    end

    return all_bets
end



#### ??? 
# --- SETUP MODEL 1 (League 1) ---
println("Preparing Model 1 (L1)...")
feats_m1 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m1.config.splitter),
    m1.config.model, m1.config.splitter
)
# Get the last split (latest data)
chain_m1 = m1.training_results[end][1]
fset_m1  = feats_m1[end][1]

# --- SETUP MODEL 2 (League 2) ---
println("Preparing Model 2 (L2)...")
feats_m2 = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m2.config.splitter),
    m2.config.model, m2.config.splitter
)
chain_m2 = m2.training_results[end][1]
fset_m2  = feats_m2[end][1]

println("Models Ready.")


#=
# get match to predict 
# odds_df -> df_match_predict 
julia> odds_df
117×10 DataFrame
 Row │ match_id  event_name                 market_name  selection  back_price  lay_price  timestamp                home_id              away_id                       league  
     │ String    String                     String       Symbol     Float64     Float64    DateTime                 String?              String?                       Symbol? 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 1         Spartans v Edinburgh City  1X2          draw             4.4        4.7   2026-02-14T12:49:51.631  the-spartans-fc      edinburgh-city-fc             l2
   2 │ 1         Spartans v Edinburgh City  1X2          home             1.59       1.6   2026-02-14T12:49:51.631  the-spartans-fc      edinburgh-city-fc             l2
   3 │ 1         Spartans v Edinburgh City  1X2          away             6.4        7.0   2026-02-14T12:49:51.631  the-spartans-fc      edinburgh-city-fc             l2
   4 │ 1         Spartans v Edinburgh City  OverUnder    under_25         2.28       2.44  2026-02-14T12:49:51.631  the-spartans-fc      edinburgh-city-fc             l2
   5 │ 1         Spartans v Edinburgh City  OverUnder    over_25          1.7        1.78  2026-02-14T12:49:51.631  the-spartans-fc      edinburgh-city-fc             l2

-> 
match_to_predict2 = DataFrame(
    match_id = [1, 2],
    match_week = [999, 999], 
    home_team = ["clyde-fc", "annan-athletic"], 
    away_team = ["elgin-city", "dumbarton"]
)

so group by the league and the match id, 


=#
"""
    get_matches_to_predict(odds_df)

Extracts unique matches from the odds dataframe and formats them for the model.
"""
function get_matches_to_predict(odds_df::DataFrame)
    # 1. Extract unique match metadata (ignore market rows)
    # We only care about the ID, League, and Teams
    unique_matches = unique(odds_df[:, [:match_id, :home_id, :away_id, :league]])

    # 2. Rename columns to match what the Model expects
    # home_id -> home_team, away_id -> away_team
    rename!(unique_matches, :home_id => :home_team, :away_id => :away_team)

    # 3. Add the required 'match_week' column (placeholder 999)
    unique_matches.match_week .= 999

    # 4. Ensure match_id is an Integer (Model usually expects Int, odds_df has String)
    # We parse "1" -> 1
    unique_matches.match_id = parse.(Int, unique_matches.match_id)

    # 5. Reorder columns to be clean
    select!(unique_matches, :match_id, :match_week, :home_team, :away_team, :league)

    return unique_matches
end

matches_to_predict = get_matches_to_predict(odds_df)

matches_to_predict.match_date .= today()

raw_preds1 = BayesianFootball.Models.PreGame.extract_parameters(
    m1.config.model, 
    matches_to_predict,
    fset_m1, 
    chain_m1
)

#=
Dict{Int64, @NamedTuple{λ_h::Vector{Float64}, λ_a::Vector{Float64}, r::Vector{Float64}}} with 5 entries:
  6  => (λ_h = [2.39925, 1.72266, 1.51839, 2.02214, 2.26357, 2.75044, 1.26044, 1.71226, 1.23242, 1.95777  …  3.13775, 0.736602, 2.76817, 1.35182, 2.50275, 2.43883, 1.92165, 1.69706, 1.45698, 1.67891], λ_a = [1.22843, 0.967194, 0.761267, 0.664, 0.471533, 0.877593, 0.8809…
  7  => (λ_h = [1.25739, 1.54305, 0.815999, 0.781085, 0.929177, 1.43895, 2.05015, 2.68339, 0.921894, 3.50317  …  1.38361, 1.61003, 1.38289, 1.24293, 1.00116, 1.20626, 1.37464, 1.8649, 1.18814, 1.03464], λ_a = [0.957058, 2.23711, 1.07324, 1.14033, 1.18951, 1.52937, 0.994…
  10 => (λ_h = [1.2976, 1.03289, 1.02733, 1.13399, 1.05931, 1.47622, 0.884632, 0.790938, 0.873623, 1.51523  …  0.807935, 1.07274, 1.12942, 0.70372, 1.26455, 1.59254, 0.397461, 0.871625, 0.379088, 0.513226], λ_a = [0.890296, 1.39498, 1.36738, 1.5598, 1.45955, 1.48155, 1.…
  9  => (λ_h = [2.18313, 1.61509, 4.23089, 3.04416, 0.870939, 0.96148, 1.48739, 2.74086, 2.00184, 1.42201  …  1.5057, 1.67675, 1.96111, 1.60099, 0.86889, 0.967304, 2.31919, 1.59943, 2.35729, 1.8797], λ_a = [2.36092, 1.82443, 1.90129, 2.38089, 1.49742, 1.62545, 1.57847, …
  8  => (λ_h = [1.2524, 1.27249, 1.03819, 0.619774, 0.768856, 0.665935, 1.28069, 0.569431, 1.84503, 1.44939  …  1.4015, 1.18649, 1.25731, 0.91261, 1.44908, 1.46195, 0.694369, 0.584527, 0.91083, 0.712384], λ_a = [1.5144, 1.6874, 2.24721, 2.18369, 2.02707, 1.75238, 2.9345…
=#

function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    for k in keys(raw_preds[ids[1]]); cols[k] = [raw_preds[i][k] for i in ids]; end
    return DataFrame(cols)
end

raw_preds_to_df(raw_preds1)
#=
5×4 DataFrame
 Row │ match_id  r                                  λ_a                                λ_h                               
     │ Any       Any                                Any                                Any                               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 6         [1.2704, 21.3234, 69.2302, 7.459…  [1.22843, 0.967194, 0.761267, 0.…  [2.39925, 1.72266, 1.51839, 2.02…
   2 │ 7         [2.99689, 329.525, 52.7196, 71.9…  [0.957058, 2.23711, 1.07324, 1.1…  [1.25739, 1.54305, 0.815999, 0.7…
   3 │ 10        [26.311, 0.635927, 3.27399, 3.69…  [0.890296, 1.39498, 1.36738, 1.5…  [1.2976, 1.03289, 1.02733, 1.133…
   4 │ 9         [4.33425, 116.097, 158.632, 7.85…  [2.36092, 1.82443, 1.90129, 2.38…  [2.18313, 1.61509, 4.23089, 3.04…
   5 │ 8         [75.0166, 16.1735, 4.75964, 2.14…  [1.5144, 1.6874, 2.24721, 2.1836…  [1.2524, 1.27249, 1.03819, 0.619…
=#




ppd1 = BayesianFootball.Predictions.model_inference(
  BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds1), m1.config.model)
)

#= 
julia> ppd1 = BayesianFootball.Predictions.model_inference(                                                                                                                                                                                                                    
         BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds1), m1.config.model)                                                                                                                                                                               
       )                                                                                                                                                                                                                                                                       
Running Inference on 5 matches...                                                                                                                                                                                                                                              
135×5 DataFrame                                                                                                                                                                                                                                                                
 Row │ match_id  market_name  market_line  selection  distribution                                                                                                                                                                                                             
     │ Int64     String       Float64      Symbol     Array…                            
─────┼──────────────────────────────────────────────────────────────────────────────────
   1 │        6  1X2                  0.0  away       [0.258366, 0.216526, 0.188877, 0…
   2 │        6  1X2                  0.0  home       [0.525928, 0.546196, 0.55141, 0.…
   3 │        6  1X2                  0.0  draw       [0.204898, 0.237277, 0.259713, 0…
   4 │        6  BTTS                 0.0  btts_yes   [0.420464, 0.494972, 0.412737, 0…
   5 │        6  BTTS                 0.0  btts_no    [0.568728, 0.505026, 0.587263, 0…
   6 │        6  OverUnder            0.5  under_05   [0.110029, 0.0740871, 0.104446, …
   7 │        6  OverUnder            0.5  over_05    [0.879162, 0.925911, 0.895554, 0…
   8 │        6  OverUnder            1.5  over_15    [0.719056, 0.739276, 0.661722, 0…
   9 │        6  OverUnder            1.5  under_15   [0.270136, 0.260722, 0.338277, 0…
  10 │        6  OverUnder            2.5  over_25    [0.555801, 0.498298, 0.39788, 0.…
  11 │        6  OverUnder            2.5  under_25   [0.433391, 0.5017, 0.60212, 0.51…
  12 │        6  OverUnder            3.5  under_35   [0.576502, 0.714224, 0.802167, 0…
  13 │        6  OverUnder            3.5  over_35    [0.41269, 0.285774, 0.197833, 0.…
  14 │        6  OverUnder            4.5  under_45   [0.691937, 0.858175, 0.916821, 0…
  15 │        6  OverUnder            4.5  over_45    [0.297255, 0.141823, 0.0831785, …
  16 │        6  OverUnder            5.5  under_55   [0.780335, 0.938012, 0.969804, 0…
=#



my_signals = [BayesianKelly()]

odds_df.date .= today()
odds_df.is_winner .= true
odds_df.match_id = parse.(Int, odds_df.match_id)


ppd1.df.mean_odds = [mean(1 ./ row) for row in ppd1.df.distribution]

results = BayesianFootball.Signals.process_signals(
    ppd1, 
    odds_df, 
    my_signals; 
    odds_column=:back_price
)

#=
julia> names(results.df)
10-element Vector{String}:
 "match_id"
 "date"
 "market_name"
 "selection"
 "is_winner"
 "signal_name"
 "signal_params"
 "odds_type"
 "odds"
 "stake"

i want to join the pdd1 mean odds as the model odds column in the results 
ppd1.df.mean_odds = [mean(1 ./ row) for row in ppd1.df.distribution]


=#


# 1. Extract Model Prices from PPD
# We create a lightweight DF with just the keys and the calculated odds
model_pricing = select(ppd1.df, :match_id, :market_name, :selection)

# Calculate "Fair Odds" (1 / Mean Probability)
# Note: This is usually more stable than mean(1 ./ p)
model_pricing.model_odds = [1.0 / mean(d) for d in ppd1.df.distribution]

# 2. Join into Results
# We join on the three keys that define a unique betting line
final_df = leftjoin(
    results.df, 
    model_pricing, 
    on = [:match_id, :market_name, :selection]
)

# 3. (Optional) Calculate Edge explicitly for checking
# Edge = (Model Probability * Market Odds) - 1
# We can re-derive it easily now:
final_df.edge_calc = (1.0 ./ final_df.model_odds) .* final_df.odds .- 1.0

# View
display(final_df[:, [:match_id, :selection, :odds, :model_odds, :stake]])

# 1. Get a unique list of IDs and Names from your odds dataframe
match_names = unique(odds_df[:, [:match_id, :event_name]])

# 2. Join it to your final results
final_df = leftjoin(final_df, match_names, on=:match_id)

# 3. Now this display command will work
aa = display(final_df[:, [:match_id, :event_name, :selection, :odds, :model_odds, :stake, :edge_calc]])



# -----
using DataFrames, Statistics

# --- 1. HELPERS (From your snippets) ---

function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    # Extract columns from the first entry's keys
    first_keys = keys(raw_preds[ids[1]])
    for k in first_keys
        cols[k] = [raw_preds[i][k] for i in ids]
    end
    return DataFrame(cols)
end

function get_matches_to_predict(odds_df::DataFrame)
    # Extract unique matches and format for model
    unique_matches = unique(odds_df[:, [:match_id, :home_id, :away_id, :league]])
    rename!(unique_matches, :home_id => :home_team, :away_id => :away_team)
    unique_matches.match_week .= 999
    # Ensure IDs are Int
    if eltype(unique_matches.match_id) == String
        unique_matches.match_id = parse.(Int, unique_matches.match_id)
    end
    return select(unique_matches, :match_id, :match_week, :home_team, :away_team, :league)
end

# --- 2. MAIN PIPELINE WRAPPER ---

function process_betting_pipeline(
    odds_df::DataFrame, 
    m1, fset_m1, chain_m1, # L1 Artifacts
    m2, fset_m2, chain_m2, # L2 Artifacts
    signals
)
    # A. PREPARE ODDS DF
    # Ensure date/winner cols exist for Signals package
    work_odds = copy(odds_df)
    work_odds.date .= today()
    work_odds.is_winner .= true
    if eltype(work_odds.match_id) == String
        work_odds.match_id = parse.(Int, work_odds.match_id)
    end

    # B. PREPARE INPUTS
    matches_to_predict = get_matches_to_predict(work_odds)
    all_results = DataFrame()
    all_model_pricing = DataFrame()

    # --- PROCESS LEAGUE 1 ---
    l1_matches = subset(matches_to_predict, :league => ByRow(isequal(:l1)))
    if !isempty(l1_matches)
        println("Processing $(nrow(l1_matches)) L1 matches...")
        
        # 1. Extract & Convert
        raw_p = BayesianFootball.Models.PreGame.extract_parameters(
            m1.config.model, l1_matches, fset_m1, chain_m1
        )
        preds_df = raw_preds_to_df(raw_p)

        # 2. Inference
        ppd = BayesianFootball.Predictions.model_inference(
            BayesianFootball.Experiments.LatentStates(preds_df, m1.config.model)
        )

        # 3. Signals
        # Filter odds for just these matches to avoid mismatches
        l1_odds = filter(row -> row.match_id in l1_matches.match_id, work_odds)
        res = BayesianFootball.Signals.process_signals(
            ppd, l1_odds, signals; odds_column=:back_price
        )
        append!(all_results, res.df) # Note: res.df based on your snippet

        # 4. Save Model Odds for joining later
        mp = select(ppd.df, :match_id, :market_name, :selection)
        mp.model_odds = [1.0 / mean(d) for d in ppd.df.distribution]
        append!(all_model_pricing, mp)
    end

    # --- PROCESS LEAGUE 2 ---
    l2_matches = subset(matches_to_predict, :league => ByRow(isequal(:l2)))
    if !isempty(l2_matches)
        println("Processing $(nrow(l2_matches)) L2 matches...")
        
        raw_p = BayesianFootball.Models.PreGame.extract_parameters(
            m2.config.model, l2_matches, fset_m2, chain_m2
        )
        preds_df = raw_preds_to_df(raw_p)

        ppd = BayesianFootball.Predictions.model_inference(
            BayesianFootball.Experiments.LatentStates(preds_df, m2.config.model)
        )

        l2_odds = filter(row -> row.match_id in l2_matches.match_id, work_odds)
        res = BayesianFootball.Signals.process_signals(
            ppd, l2_odds, signals; odds_column=:back_price
        )
        append!(all_results, res.df)

        mp = select(ppd.df, :match_id, :market_name, :selection)
        mp.model_odds = [1.0 / mean(d) for d in ppd.df.distribution]
        append!(all_model_pricing, mp)
    end

    # --- C. FINAL JOINING ---
    println("Finalizing Bet Sheet...")
    
    # Join 1: Attach Model Odds
    # Join on match_id, market_name, selection
    final_df = leftjoin(
        all_results, 
        all_model_pricing, 
        on = [:match_id, :market_name, :selection]
    )

    # Join 2: Attach Event Names
    # Get unique map from the original odds
    match_names = unique(work_odds[:, [:match_id, :event_name]])
    final_df = leftjoin(final_df, match_names, on=:match_id)

    # Calculate Edge
    final_df.edge_calc = (1.0 ./ final_df.model_odds) .* final_df.odds .- 1.0

    return final_df
end

# --- 3. EXECUTION ---

# Define Signals
my_signals = [BayesianKelly()]

# Run Pipeline
final_df = process_betting_pipeline(
    odds_df, 
    m1, fset_m1, chain_m1, 
    m2, fset_m2, chain_m2, 
    my_signals
)

# View Betting Opportunities
bets = final_df[:, [:match_id, :event_name, :selection, :odds, :model_odds, :edge_calc, :stake]]
display(filter(r -> r.stake > 0, bets))

bbb = [:over_05, :over_15, :over_25, :under_55, :draw, :home]

subset(bets, :selection => ByRow(in(bbb)), :stake => ByRow( >(0)))

using CSV 

CSV.write("odds_df.csv", odds_df)
