# module MatchDayTypes
    using Dates
    # export ScraperMatch, ScraperRow

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
# end
#
#
# module PythonScraper

using DataFrames
using JSON3
using Dates
using CSV
using Statistics
using SHA

using Dates
using JSON3
using DataFrames
# using ..MatchDayTypes

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


        # BTTS
        # if haskey(ft, "Both teams to Score?")
        #     for (k, o) in ft["Both teams to Score?"]
        #         sel = k == "Yes" ? :btts_yes : :btts_no
        #         b, l = get_p(o)
        #         push!(out, ScraperRow(m.id, m.event_name, "BTTS", sel, b, l, now()))
        #     end
        # end
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
    
    return master_map
end

#### 

# end # module


println("=== 1. Testing Match List Fetching ===")
# We ask for a wide time window (48 hours) to ensure we find *something* for testing purposes.
# Change "Premiership" to whatever league has games today/tomorrow.
leagues_to_test = ["scotland"]
matches = get_matches(leagues_to_test, time_window_hours=48.0)

if isempty(matches)
    println("⚠️  No matches found. \nPossible reasons:")
    println("   - No games scheduled in the next 48h for: $leagues_to_test")
    println("   - Python script path is incorrect in 'python_scraper.jl'")
    println("   - Python environment issues")
else
    println("✅ Found $(length(matches)) matches.")
    
    # Print the first few to visually verify
    for m in first(matches, 3)
        println("   Example: [$(m.start_time)] $(m.event_name) (ID: $(m.id))")
    end

    println("\n=== 2. Testing Odds Fetching ===")
    # We will only fetch odds for the FIRST match to save time/API calls during this test
    target_match = [matches[1]]
    println("   Fetching odds for: $(target_match[1].event_name)...")

    df_odds = fetch_odds(target_match)

    if isempty(df_odds)
        println("⚠️  Matches found, but NO odds returned.")
        println("   - Check if markets (Match Odds, Over/Under) are open.")
        println("   - Check if your Python CLI 'odds' command returns valid JSON.")
    else
        println("✅ Success! Retrieved $(nrow(df_odds)) betting lines.")
        
        # Display the DataFrame to ensure columns are correct (Back vs Lay)
        println("\n--- Sample Output ---")
        display(first(df_odds, 5))
        
        # Check for non-NaN values
        valid_prices = filter(r -> !isnan(r.back_price), df_odds)
        if !isempty(valid_prices)
      println("\n   Verifying Data Quality: Found valid Back prices (e.g., $(valid_prices[1, [:back_price]])")
        else
            println("\n   ⚠️  Warning: All fetched prices seem to be NaN. Check parser logic.")
        end
    end
end




####
using DataFrames
using BayesianFootball
using Dates

# 1. Load Modules
include("../src/matchday/types.jl")
include("../src/matchday/providers/python_scraper.jl")
using .MatchDayTypes
using .PythonScraper

# ==============================================================================
# PHASE 1: FETCH DATA
# ==============================================================================
println("🔍 Scanning for matches...")
matches = PythonScraper.get_matches(["Premiership", "League Two"], time_window_hours=24.0)
matches = get_matches(["scotland"], time_window_hours=24.0)

if isempty(matches)
    println("No matches found in the next 24 hours.")
    exit()
end

println("✅ Found $(length(matches)) matches. Fetching live odds...")
df_odds = fetch_odds(matches)

# Prepare odds for Signals
# We use 'back_price' as the odds we can get
df_odds.odds = df_odds.back_price   
df_odds.is_winner .= missing        

# ==============================================================================
# PHASE 2: PREPARE MODEL INPUT
# ==============================================================================
println("🗺️  Mapping Teams...")
team_mappings = load_mappings() 
team_mappings["Spartans"] = "the-spartans-fc"
team_mappings["East Kilbride"] = "east-kilbride"


model_input_rows = []

for m in matches
    parts = split(m.event_name, " v ")
    if length(parts) == 2
        h_name, a_name = parts[1], parts[2]
        if haskey(team_mappings, h_name) && haskey(team_mappings, a_name)
            push!(model_input_rows, (
                match_id = m.id,   # <--- CRITICAL: Use the Scraper's ID (String)
                match_week = 999,
                home_team = team_mappings[h_name],
                away_team = team_mappings[a_name]
            ))
        end
    end
end


df_model_input = DataFrame(model_input_rows)

df_model_input = df_model_input[[1,6], :]

match_to_predict = DataFrame(
    match_id = [1],
    match_week = [999], 
    home_team = ["east-kilbride"], 
    away_team = ["the-spartans-fc"]
)


# ==============================================================================
# THE TUNNELING FIX: Bypassing the Int64 Requirement
# ==============================================================================

# 1. Create a "Look-up Table" to remember the Real IDs
#    Maps: 1 => "221e32ca", 2 => "2f48ce82", etc.
id_mapping = Dict(i => row.match_id for (i, row) in enumerate(eachrow(df_model_input)))

# 2. Modify the DataFrame to use Temporary Integer IDs
#    The model is happy because it gets Ints (1, 2, 3...)
df_model_input.match_id = collect(1:nrow(df_model_input))

println("temporary IDs assigned. Running inference...")

# 3. Run the Model (Now it won't crash)
raw_preds_int = BayesianFootball.Models.PreGame.extract_parameters(
    m.config.model, 
    df_model_input, 
    feature_set, 
    chain
)

# 4. Swap the Keys BACK to Strings
#    We convert the result {1: Preds} -> {"221e32ca": Preds}
raw_preds = Dict(id_mapping[k] => v for (k, v) in raw_preds_int)

println("✅ Inference Complete. IDs restored.")

# ==============================================================================
# CONTINUE WITH PIPELINE
# ==============================================================================

# 5. Convert to LatentStates (Preserving the restored String IDs)
function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    # Assumes all entries have same keys
    first_entry = raw_preds[ids[1]]
    for k in keys(first_entry)
        cols[k] = [raw_preds[i][k] for i in ids]
    end
    return DataFrame(cols)
end

latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents)
df_odds.date = Date.(df_odds.timestamp)


# 6. Run Signals (Now df_odds and ppd BOTH have String IDs!)
results = BayesianFootball.Signals.process_signals(
    ppd, 
    df_odds, 
    my_signals; 
    odds_column=:odds
)

# 7. Final Filter
profitable_selections = Set([:over_05, :over_15, :over_25, :over_35, :btts_yes])
final_slip = filter(row -> row.stake > 0 && row.selection in profitable_selections, results.df)

if isempty(final_slip)
    println("\n📉 No bets found.")
else
    println("\n💰 RECOMMENDED BETS:")
    display(select(final_slip, :event_name, :market_name, :selection, :odds, :stake, :expected_value))
end
# ==============================================================================
# THE FINAL ALIGNMENT FIX: Convert Everything to Int64
# ==============================================================================

# 1. Create a Master Map: String ID -> Integer ID
#    We use the order of matches in the model input to define ID 1, 2, 3...
unique_ids = unique(df_model_input.match_id)
str_to_int = Dict(id => i for (i, id) in enumerate(unique_ids))

println("Mapping $(length(unique_ids)) matches to Integers...")

# 2. Update Model Input to use Ints (Satisfies the strict Library)
df_model_input.match_id = [str_to_int[x] for x in df_model_input.match_id]

# 3. Update Odds to use THE SAME Ints (Satisfies the Join)
#    We keep the old ID in a temp column just in case
df_odds.match_id_str = df_odds.match_id 
# Only keep odds for matches we are actually modeling
df_odds_clean = filter(row -> haskey(str_to_int, row.match_id), df_odds)
df_odds_clean.match_id = [str_to_int[x] for x in df_odds_clean.match_id]

# 4. RUN MODEL (Now input is purely Int64)
raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m.config.model, 
    df_model_input, 
    feature_set, 
    chain
)

# Convert to DataFrame (Output is now Int64)
function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    for k in keys(raw_preds[ids[1]]); cols[k] = [raw_preds[i][k] for i in ids]; end
    return DataFrame(cols)
end

latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents)

# 5. RUN SIGNALS (Int64 joins with Int64 -> SUCCESS)
results = BayesianFootball.Signals.process_signals(
    ppd, 
    df_odds_clean, 
    my_signals; 
    odds_column=:odds
)

# 6. VIEW PROFITABLE BETS
profitable_selections = Set([:over_05, :over_15, :over_25, :over_35, :btts_yes])
final_slip = filter(row -> row.stake > 0 && row.selection in profitable_selections, results.df)

if isempty(final_slip)
    println("\n📉 No bets found (Calculated stakes are 0).")
else
    println("\n💰 RECOMMENDED BETS:")
    # We select event_name to verify the teams
    display(select(final_slip, :event_name, :market_name, :selection, :odds, :stake, :expected_value))
    
    # Simple Bankroll calc (assuming £1000 bank)
    total_stake = sum(final_slip.stake)
    println("\nStart with a £1000 Bankroll?")
    println("→ Total Investment: £$(round(total_stake * 1000, digits=2))")
end

# ==============================================================================
# PHASE 3: RUN MODEL & SIGNALS
# ==============================================================================
println("🚀 Running Model Inference...")

# 1. Extract Parameters (Assuming 'm', 'feature_set', 'chain' are in global scope)
# If running from fresh REPL, you need to load your experiment 'm' first!
raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m.config.model, 
    df_model_input, 
    feature_set, 
    chain
)

# 2. Convert to LatentStates
function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    for k in keys(raw_preds[ids[1]]); cols[k] = [raw_preds[i][k] for i in ids]; end
    return DataFrame(cols)
end

latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m.config.model)

# 3. Generate Probabilities
ppd = BayesianFootball.Predictions.model_inference(latents)

df_odds.match_id .= 1

# 2. Run Signals again
results = BayesianFootball.Signals.process_signals(
    ppd, 
    df_odds, 
    my_signals; 
    odds_column=:odds
)

# ==============================================================================
# PHASE 4: PORTFOLIO FILTERING (The "Secret Sauce")
# ==============================================================================
# Based on your backtests, we ONLY want to bet on these markets.
# Unders are currently losing money (-16% ROI), so we exclude them.
profitable_selections = Set([
    :over_05, 
    :over_15, 
    :over_25, 
    :over_35, 
    :btts_yes
])

# 1. Filter for Signal Strength (Positive EV)
value_bets = filter(row -> row.stake > 0, results.df)

# 2. Filter for Profitable Strategies (The "Portfolio" Step)
final_slip = filter(row -> row.selection in profitable_selections, value_bets)

if isempty(final_slip)
    println("\n📉 No bets found (either no value, or value was in 'Under' markets which we skip).")
else
    println("\n💰 RECOMMENDED BETS (Overs & BTTS Only):")
    
    # Sort by Start Time (we join back to matches to get time)
    # (Optional nice-to-have, skipping for simplicity)
    
    display(select(final_slip, 
        :event_name, 
        :market_name, 
        :selection, 
        :odds, 
        :stake, 
        :expected_value
    ))
    
    # Calculate Total Exposure
    total_stake = sum(final_slip.stake)
    println("\nTotal Portfolio Risk: $(round(total_stake * 100, digits=2))% of Bankroll")
end



println("🔥 RESETTING PIPELINE...")

# 1. GET DATA (Strings)
# =====================
matches = get_matches(["scotland"], time_window_hours=24.0)
df_odds = fetch_odds(matches)
df_odds.odds = df_odds.back_price;
df_odds.is_winner .= missing;
df_odds.date = Date.(df_odds.timestamp) # Fix missing date;

# 2. BUILD MODEL INPUT (Strings)
# ==============================
team_mappings = load_mappings()
# Force your manual fixes
team_mappings["Spartans"] = "the-spartans-fc"
team_mappings["East Kilbride"] = "east-kilbride"

model_rows = []
valid_string_ids = Set{String}() # Track which IDs we actually mapped

for m in matches
    parts = split(m.event_name, " v ")
    if length(parts) == 2
        h_name, a_name = parts[1], parts[2]
        if haskey(team_mappings, h_name) && haskey(team_mappings, a_name)
            push!(model_rows, (
                match_id = m.id, 
                match_week = 999, 
                home_team = team_mappings[h_name], 
                away_team = team_mappings[a_name]
            ))
            push!(valid_string_ids, m.id)
        end
    end
end
df_model = DataFrame(model_rows)

# 3. ALIGN EVERYTHING TO INTEGERS (The Critical Step)
# ===================================================
# We only care about matches that exist in BOTH (Odds AND Model)
# Filter odds to only keep valid matches
df_odds_clean = filter(row -> row.match_id in valid_string_ids, df_odds)

# Create the Map: String ID -> Int ID (1, 2, 3...)
unique_str_ids = unique(df_model.match_id)
str_to_int = Dict(id => i for (i, id) in enumerate(unique_str_ids))

# Apply Map to BOTH DataFrames immediately
df_model.match_id = [str_to_int[x] for x in df_model.match_id]
df_odds_clean.match_id = [str_to_int[x] for x in df_odds_clean.match_id]

println("✅ ALIGNMENT DONE: $(nrow(df_model)) matches ready for inference.")

df_model1 = df_model[[3], :]

# 4. RUN MODEL (Pure Integers)
# ============================
raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m.config.model, df_model1, feature_set, chain
)

# Convert Preds to DataFrame (Int IDs)
ids = collect(keys(raw_preds))
cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
for k in keys(raw_preds[ids[1]])
    cols[k] = [raw_preds[i][k] for i in ids]
end
latents = BayesianFootball.Experiments.LatentStates(DataFrame(cols), m.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents)

# 5. SIGNALS (Int joins Int -> Success)
# =====================================
results = BayesianFootball.Signals.process_signals(
    ppd, 
    df_odds_clean, 
    my_signals; 
    odds_column=:odds
)

# 6. PROFIT?
# ==========
profitable = Set([:over_05, :over_15, :over_25, :over_35, :btts_yes])
final_slip = filter(row -> row.stake > 0 && row.selection in profitable, results.df)

if isempty(final_slip)
    println("\n📉 No bets found.")
else
    println("\n💰 RECOMMENDED BETS:")
    display(select(final_slip, :event_name, :market_name, :selection, :odds, :stake, :expected_value))
end
