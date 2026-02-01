using BayesianFootball
using DataFrames

data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


saved_folders = Experiments.list_experiments("exp/market_runs"; data_dir="./data")


loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

if isempty(loaded_results)
    error("No results loaded! Did you run runner.jl?")
end

m = loaded_results[1]


match_to_predict = DataFrame(
    match_id = [1, 2, 3],
    match_week = [999, 999, 999], 
    home_team = ["east-kilbride", "stranraer", "dumbarton"], 
    away_team = ["the-spartans-fc", "clyde-fc", "edinburgh-city-fc"]
)



d = subset(ds.matches, :tournament_id => ByRow(isequal(57)), :season => ByRow(isequal("25/26")))

unique(d.home_team)
#+++++++ 


splits = BayesianFootball.Data.create_data_splits(ds, m.config.splitter)

# 2. Re-create the feature sets using the saved model config
# feature_collection is a Vector of tuples: (FeatureSet, SplitMetaData)
feature_collection = BayesianFootball.Features.create_features(
    splits, 
    m.config.model, 
    m.config.splitter
)


# Get the Chain from the results
chain = m.training_results[end][1]

# Get the corresponding FeatureSet from our regenerated list
# Note: [1] gets the FeatureSet, [2] would be the Metadata
feature_set = feature_collection[end][1]





# ==============================================================================
# 3. RUN EXTRACTION
# ==============================================================================

println("Extracting parameters...")

# Now this will work because we are passing the correct types:
# (Model, DataFrame, FeatureSet, Chain)
raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m.config.model,
    match_to_predict,
    feature_set,  # <--- This was the missing piece
    chain
)

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



b = subset(ppd.df, :match_id => ByRow(isequal(3)))
using Statistics

b.prob = mean.(b.distribution)
b.odds = round.(1 ./ b.prob, digits=2)


b
select(b, :market_name, :selection, :prob, :odds)



function make_predictions(data_store, experiment, matches_to_predict)

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

  feature_collection = BayesianFootball.Features.create_features(
      BayesianFootball.Data.create_data_splits(data_store, experiment.config.splitter),
      experiment.config.model, 
      experiment.config.splitter
  )
  feature_set = feature_collection[end][1]

  chain = experiment.training_results[end][1]


  raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
      m.config.model,
      matches_to_predict,
      feature_set,
      chain
  )

  latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m.config.model)
  ppd = BayesianFootball.Predictions.model_inference(latents)

  return ppd

end 



function make_predictions(data_store, experiment, idx, matches_to_predict)

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

  feature_collection = BayesianFootball.Features.create_features(
      BayesianFootball.Data.create_data_splits(data_store, experiment.config.splitter),
      experiment.config.model, 
      experiment.config.splitter
  )
  feature_set = feature_collection[idx][1]

  chain = experiment.training_results[idx][1]


  raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
      m.config.model,
      matches_to_predict,
      feature_set,
      chain
  )

  latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m.config.model)
  ppd = BayesianFootball.Predictions.model_inference(latents)

  return ppd

end 



match_to_predict56= DataFrame(
    match_id = [1, 2, 3, 4, 5],
    match_week = [999, 999, 999, 999, 999], 
    home_team = ["annan-athletic", "elgin-city", "dumbarton", "east-kilbride", "stirling-albion"], 
    away_team = ["clyde-fc", "the-spartans-fc", "edinburgh-city-fc", "forfar-athletic","stranraer"]
)


match_to_predict56= DataFrame(
    match_id = [1, 2, 3, 4, 5],
    match_week = [999, 999, 999, 999, 999], 
    home_team = ["alloa-athletic", "east-fife", "kelty-hearts-fc", "queen-of-the-south", "stenhousemuir"], 
    away_team = ["cove-rangers", "montrose", "hamilton-academical", "inverness-caledonian-thistle","peterhead"]
)


#= 
julia> unique(d.home_team)
10-element Vector{InlineStrings.String31}:
 "annan-athletic"
 "east-kilbride"
 "dumbarton"
 "forfar-athletic"
 "edinburgh-city-fc"
 "clyde-fc"
 "elgin-city"
 "stirling-albion"
 "the-spartans-fc"
 "stranraer"


league one
 "cove-rangers"
 "hamilton-academical"
 "peterhead"
 "kelty-hearts-fc"
 "stenhousemuir"
 "east-fife"
 "inverness-caledonian-thistle"
 "montrose"
 "queen-of-the-south"
 "alloa-athletic"

=#


unique(d.home_team)

pp = make_predictions(ds, m, match_to_predict) 
pp56 = make_predictions(ds, m, 6, match_to_predict56) 
pp56 = make_predictions(ds, m, 4, match_to_predict56) 



pp56.df.prob = mean.(pp56.df.distribution)
pp56.df.odds = round.(1 ./ pp56.df.prob, digits=2)

pp56

id = 5
ppp = subset(pp56.df, :match_id => ByRow(isequal(id)))

select(ppp, :market_name, :selection, :prob, :odds)

match_to_predict56[id, :]



#= 
Dumbarton 0-0 Edinburgh City
East Kilbride 1-0 Forfar
Stirling 0-0 Stranraer



East Fife 0-0 Montrose
Kelty Hearts 0-0 Hamilton
Stenhousemuir 0-0 Peterhead
=#


# ========== 




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
# Wrapper to run the command and capture STDOUT (ignoring STDERR logs)
function _run_cli(args::Vector{String}; 
                  PYTHON_EXE::String = "/home/james/miniconda3/envs/webscraper/bin/python",
                  CLI_PATH::String = "/home/james/bet_project/whatstheodds/live_odds_cli.py"
                  )
    # We run in the CLI directory so it can find 'mappings/'
    cmd = Cmd(vcat([PYTHON_EXE, CLI_PATH], args))
    try
        return read(setenv(cmd, dir=dirname(CLI_PATH)), String)
    catch e
        @error "Python CLI Failed: $e"
        return nothing
    end
end



output = _run_cli(["list", "-f", "england"])



struct ScraperMatch
    id::String            
    event_name::String    
    start_time::Time      
    # date::Date            
    # league::String
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

"""
    get_matches(leagues; time_window_hours=nothing)
    
    Fetches today's matches. Optional: filter for matches starting soon.
"""


function parse_match_line(line::AbstractString, match_date::Date)
    # Regex to capture time, home, and away
    m = match(r"^\s*-\s*(\d{2}:\d{2})\s*\|\s*(.+?)\s+v\s+(.+?)$", line)
    
    if isnothing(m) 
        return nothing
    end

    t_str, home, away = m.captures
    match_time = Time(t_str, "HH:MM")
    event_name = "$home v $away"
    
    # Generate ID using name AND date (prevents collision if teams play twice in a season)
    id_str = bytes2hex(sha256(event_name * string(match_date)))[1:8]
    
    # We removed 'league' from input, so we just default it here or in the struct
    return ScraperMatch(id_str, event_name, match_time, match_date, "Unknown")
end

function get_matches(leagues::Vector{String})
    # 1. Fetch Data
    output = _run_cli(["list", "-f", leagues...])
    
    # Guard clause: if CLI failed, return empty list immediately
    if isnothing(output)
        return ScraperMatch[]
    end

    matches = ScraperMatch[]
    current_date = today() # We set the date ONCE here

    # 2. Process Data
    for line in split(output, '\n')
        # We delegate the messy work to the parser
        parsed_match = parse_match_line(line, current_date)
        
        # Only add to list if parsing succeeded
        if !isnothing(parsed_match)
            push!(matches, parsed_match)
        end
    end

    return matches
end

matches_betfair = get_matches([""])


##

# 1. Struct Definitions (DTOs)
struct PricePoint
    price::Float64
    size::Float64
end

struct MarketOutcome
    back::Union{PricePoint, Nothing} 
    lay::Union{PricePoint, Nothing} 
end

# 2. Helper: Extract Single Price safely
function extract_price(data::AbstractDict)
    # Get the raw values first, defaulting to nothing (lowercase!)
    p_raw = get(data, "price", nothing)
    s_raw = get(data, "size", nothing)

    # Guard Clause: If either is missing, we can't make a PricePoint
    if isnothing(p_raw) || isnothing(s_raw)
        return nothing
    end 

    # Now it is safe to convert to Float64
    return PricePoint(Float64(p_raw), Float64(s_raw))
end 

# 3. Main Extractor
function extract_prices(outcome_data::AbstractDict)
    # 1. Get Back
    back_dict = get(outcome_data, "back", Dict())
    back_pp = extract_price(back_dict)

    # 2. Get Lay (Fixed: changed key from "back" to "lay")
    lay_dict = get(outcome_data, "lay", Dict())
    lay_pp = extract_price(lay_dict)

    return MarketOutcome(back_pp, lay_pp)
end
extract_price(::Nothing) = nothing

# --- TEST ---
test_extract = Dict(
    "back" => Dict("price" => 1.23, "size" => 16.75),
    "lay"  => Dict("price" => 1.25, "size" => 1553.34)
)

println(extract_prices(test_extract))


struct MarketLine
  group::Symbol 
  line::Symbol
  market::MarketOutcome
end

function process_match_odds!(results::Vector{MarketLine}, match::ScraperMatch, market_data::AbstractDict)
    
    # Loop through "Paris St-G", "Newcastle", "The Draw"
    for (selection_name, raw_data) in market_data
        
        # 1. IDENTIFY THE SELECTION (:home, :away, or :draw)
        selection_symbol = if selection_name == "The Draw"
            :draw
        elseif startswith(match.event_name, selection_name)
            :home
        else
            :away
        end

        # 2. EXTRACT PRICES (Using your new helper!)
        # raw_data looks like: Dict("back" => ..., "lay" => ...)
        outcome = extract_prices(raw_data) 
        
        # 3. CONVERT TO FLOATS FOR SCRAPER ROW
        market_line_row = MarketLine(Symbol("1X2"), selection_symbol, outcome ) 

        # 4. PUSH TO BUCKET
        
        push!(results, market_line_row)
        
    end
end



# Run the function with our dummy inputs
process_match_odds!(results, dummy_match, dummy_market_data)

# Show what's in the bucket
display(results)
# Input A: The Bucket
results = MarketLine[]

# Input B: The Match Context (Who is playing?)
dummy_match = ScraperMatch(
    "35155346", 
    "Paris St-G v Newcastle", 
    Time(20,0), 
    today(), 
    "Champions League"
)

# Input C: The Market Data (The specific dictionary for "Match Odds")
# I copied this directly from your CLI output
dummy_market_data = Dict(
    "Paris St-G" => Dict(
        "back" => Dict("price" => 1.63, "size" => 2005.24),
        "lay"  => Dict("price" => 1.64, "size" => 236.53)
    ),
    "Newcastle" => Dict(
        "back" => Dict("price" => 5.2, "size" => 740.57),
        "lay"  => Dict("price" => 5.3, "size" => 430.91)
    ),
    "The Draw" => Dict(
        "back" => Dict("price" => 4.9, "size" => 834.4),
        "lay"  => Dict("price" => 5.0, "size" => 107.08)
    )
)

println("inputs created successfully!")



# --- 
function normalize_market_name(name::String)
    # "Match Odds" -> "MatchOdds"
    # "Over/Under 2.5 Goals" -> "OverUnder25Goals"
    clean_name = replace(name, r"[^a-zA-Z0-9]" => "") 
    return Symbol(clean_name)
end

# The ::Val{T} syntax means "Dispatch on the Value T"
function process_market!(::Val, results, match, data)
    # Do nothing for unknown markets
    # Optional: @debug "Skipping unknown market: $T"
    return nothing
end


# Handler for Match Odds
#
function process_market!(::Val{:MatchOdds}, results::Vector{MarketLine}, match::ScraperMatch, data::AbstractDict)

    for (selection_name, raw_data) in data
        
        # 1. IDENTIFY THE SELECTION (:home, :away, or :draw)
        selection_symbol = if selection_name == "The Draw"
            :draw
        elseif startswith(match.event_name, selection_name)
            :home
        else
            :away
        end

        # 2. EXTRACT PRICES (Using your new helper!)
        outcome = extract_prices(raw_data) 
        
        # 4. PUSH TO BUCKET
        push!(results, 
                  MarketLine(Symbol("1X2"), selection_symbol, outcome ) 
              )
    end
end

# This function is NOT a dispatcher. It is a "Worker" used by dispatchers.
function _process_ou_common!(results, data, market_label::Symbol)
    for (selection_name, raw_data) in data
        # selection_name examples: "Over 2.5 Goals", "Under 0.5 Goals"
        # The logic is identical for all of them: just check for "Over"
        selection_symbol = startswith(selection_name, "Over") ? :over : :under
        
        outcome = extract_prices(raw_data)
        
        if !isnothing(outcome)
            # Use the passed-in 'market_label' (e.g. :OU25) for the group
            push!(results, MarketLine(market_label, selection_symbol, outcome))
        end
    end
end

const OU_MARKETS = [
    (:OverUnder05Goals, :OU05),
    (:OverUnder15Goals, :OU15),
    (:OverUnder25Goals, :OU25),
    (:OverUnder35Goals, :OU35),
    (:OverUnder45Goals, :OU45),
    (:OverUnder55Goals, :OU55)
]

# Generate the functions in a loop
for (valsym, label) in OU_MARKETS
    @eval process_market!(::Val{$(QuoteNode(valsym))}, r, m, d) = _process_ou_common!(r, d, $(QuoteNode(label)))
end


function fetch_odds(matches::Vector{ScraperMatch})
    results = MarketLine[]

    for m in matches
        json_str = _run_cli(["odds", m.event_name, "-d"])
        isnothing(json_str) && continue

        full_data = JSON3.read(json_str, Dict)

        for (period, markets) in full_data
            for (market_name_str, market_data) in markets
                
                # 1. Convert String to Symbol (e.g., :MatchOdds)
                sym = normalize_market_name(market_name_str)
                
                # 2. DISPATCH! 
                # We wrap 'sym' in Val() to turn it into a Type
                process_market!(Val(sym), results, m, market_data)
                
            end
        end
    end
    return results
end


function fetch_odds(matches::ScraperMatch)
    results = MarketLine[]

    json_str = _run_cli(["odds", m.event_name, "-d"])
    isnothing(json_str) && continue

    full_data = JSON3.read(json_str, Dict)

    for (period, markets) in full_data
        for (market_name_str, market_data) in markets
            
            # 1. Convert String to Symbol (e.g., :MatchOdds)
            sym = normalize_market_name(market_name_str)
            
            # 2. DISPATCH! 
            # We wrap 'sym' in Val() to turn it into a Type
            process_market!(Val(sym), results, m, market_data)
            
        end
    end

    return results
end



fetch_odds( [matches_betfair[10]]) 
fetch_odds( [matches_betfair[9]]) 

function process_market!(::Val{:BothteamstoScore}, results::Vector{MarketLine}, match::ScraperMatch, data::AbstractDict ) 
    
  push!(results, MarketLine(:btts, :yes, extract_prices(get(data, "Yes", Dict()) ) ))
  push!(results, MarketLine(:btts, :no, extract_prices(get(data, "No", Dict()) ) ))

end

# 1. Ensure your normalizer handles "Correct Score" -> :CorrectScore (It should already!)

function process_market!(::Val{:CorrectScore}, results::Vector{MarketLine}, match::ScraperMatch, data::AbstractDict)
    
    for (score_key, raw_data) in data
        # score_key examples: "1 - 0", "2 - 2", "Any Unquoted"

        # 1. Transform the string key into a generic Symbol
        selection_symbol = if score_key == "Any Unquoted"
            :cs_other
        else
            # Remove " - " and spaces. 
            # "2 - 1" -> "21"
            clean_score = replace(score_key, r"[^0-9]" => "")
            
            # Prefix with 'cs_' so it's clear (e.g. :cs_21)
            Symbol("cs_", clean_score)
        end

        # 2. Extract Prices
        if isnothing(raw_data)
          continue
        end
        outcome = extract_prices(raw_data)

        # 3. Push to results
        # We use :CS as the group ID
        if !isnothing(outcome)
            push!(results, MarketLine(:CS, selection_symbol, outcome))
        end
    end
end
