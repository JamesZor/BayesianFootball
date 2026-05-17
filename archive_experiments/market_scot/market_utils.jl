# module MatchDayUtils

using BayesianFootball
using DataFrames
using JSON3
using Dates
using CSV
using Statistics

# export fetch_todays_data, save_market_data

# ==============================================================================
# 1. SETTINGS & CONSTANTS
# ==============================================================================

# Adjust this path if necessary or pass it in dynamically
const DEFAULT_CLI_PATH = "/home/james/bet_project/whatstheodds"

# Map the CLI market keys to your BayesianFootball Selection Symbols
const SELECTION_MAP = Dict(
    # 1X2
    :ft_1x2_home => :home,
    :ft_1x2_draw => :draw,
    :ft_1x2_away => :away,
    
    # Over/Under (Standard 2.5)
    :ft_ou_25_over => :over,
    :ft_ou_25_under => :under,
    
    # BTTS
    :ft_btts_yes => :yes,
    :ft_btts_no => :no
)

const MARKET_NAME_MAP = Dict(
    :ft_1x2_home => "1x2",
    :ft_1x2_draw => "1x2", 
    :ft_1x2_away => "1x2",
    :ft_ou_25_over => "ou_25",
    :ft_ou_25_under => "ou_25",
    :ft_btts_yes => "btts",
    :ft_btts_no => "btts"
)

# ==============================================================================
# 2. PYTHON CLI WRAPPERS
# ==============================================================================

function _run_python_cli(cli_path::String, args::Vector{String})
    script_path = joinpath(cli_path, "live_odds_cli.py")
    # Adjust python path if needed, or rely on system python
    python_exe = "/home/james/.conda/envs/webscrape/bin/python" 

    full_command = [python_exe, script_path]
    append!(full_command, args)

    cmd = Cmd(full_command)
    try
        # Run in the directory of the CLI to ensure relative paths work
        return read(setenv(cmd, dir=cli_path), String)
    catch e
        @error "Failed to run Python CLI: $e"
        return nothing
    end
end

function _load_team_mappings(mappings_path::String, leagues::Vector{String})
    reverse_map = Dict{String, String}()
    for league in leagues
        filepath = joinpath(mappings_path, "$league.json")
        if isfile(filepath)
            try
                json_str = read(filepath, String)
                data = JSON3.read(json_str)
                for (model_name, display_name) in data
                    reverse_map[String(display_name)] = String(model_name)
                end
            catch e
                @warn "Could not parse mapping for $league: $e"
            end
        end
    end
    return reverse_map
end

# ==============================================================================
# 3. CORE FETCHING LOGIC
# ==============================================================================

"""
    fetch_todays_data(leagues; cli_path, save_dir)

Main entry point. 
1. Fetches match list.
2. Fetches odds for each match.
3. Returns (matches_df, market_df).
"""
function fetch_todays_data(leagues::Vector{String}; 
                           cli_path::String=DEFAULT_CLI_PATH,
                           save_dir::Union{String, Nothing}=nothing)
    
    # A. Get Matches
    println("🔍 Scanning for matches in: $leagues...")
    matches_df = _get_matches_list(leagues, cli_path)
    
    if isempty(matches_df)
        println("⚠️ No matches found today.")
        return DataFrame(), DataFrame()
    end

    println("✅ Found $(nrow(matches_df)) matches. Fetching odds...")

    # B. Get Odds (Long Format)
    market_df = _fetch_odds_long_format(matches_df, cli_path)

    # C. Save if requested
    if !isnothing(save_dir)
        mkpath(save_dir)
        date_str = Dates.format(today(), "yyyy-mm-dd")
        
        m_path = joinpath(save_dir, "matches_$(date_str).csv")
        o_path = joinpath(save_dir, "odds_$(date_str).csv")
        
        CSV.write(m_path, matches_df)
        CSV.write(o_path, market_df)
        println("💾 Data saved to $save_dir")
    end

    return matches_df, market_df
end

function _get_matches_list(leagues, cli_path)
    mappings_path = joinpath(cli_path, "mappings")
    team_map = _load_team_mappings(mappings_path, leagues)
    
    output = _run_python_cli(cli_path, ["list", "-f", leagues...])
    isnothing(output) && return DataFrame()

    matches = []
    # Simple ID counter for today's session
    id_counter = 1

    for line in split(output, '\n')
        # Regex: "15:00 | Home Team v Away Team"
        m = match(r"^\s*-\s*(\d{2}:\d{2})\s*\|\s*(.+?)\s+v\s+(.+?)$", line)
        if !isnothing(m)
            time_str, home_display, away_display = m.captures
            
            # Map display names to model names
            home_model = get(team_map, home_display, home_display)
            away_model = get(team_map, away_display, away_display)
            
            # Create a simplified ID (or hash) to link odds later
            match_id = id_counter 
            id_counter += 1

            push!(matches, (
                match_id = match_id,
                event_name = "$home_display v $away_display",
                match_time = time_str,
                home_team = home_model,
                away_team = away_model
            ))
        end
    end
    return DataFrame(matches)
end

function _fetch_odds_long_format(matches_df, cli_path)
    # Define the markets we care about (keys for the python scraper)
    target_markets = keys(SELECTION_MAP) |> collect
    
    long_rows = []

    for row in eachrow(matches_df)
        event_name = row.event_name
        # Call the scraper for this single event
        # Note: We reuse the old 'get_live_market_odds' logic internally but reshape immediately
        raw_book = _get_raw_book(event_name, target_markets, cli_path)

        # Convert the Raw Book into Long Rows
        for (i, m_sym) in enumerate(raw_book.markets)
            
            back_price = raw_book.back_odds[i]
            
            # Filter out bad data
            if !isnan(back_price) && back_price > 1.0
                
                # Map to BayesianFootball naming conventions
                sel = SELECTION_MAP[m_sym]      # e.g. :home
                m_name = MARKET_NAME_MAP[m_sym] # e.g. "1x2"

                push!(long_rows, (
                    match_id = row.match_id, # Link back to matches_df
                    event_name = event_name,
                    market_name = m_name,
                    selection = sel,
                    odds = back_price,
                    timestamp = now()
                ))
            end
        end
    end

    if isempty(long_rows)
        return DataFrame()
    end

    return DataFrame(long_rows)
end

# Reusing your existing logic to parse the JSON, but simplified return type
struct RawBook
    markets::Vector{Symbol}
    back_odds::Vector{Float64}
end

function _get_raw_book(event_name, market_list, cli_path)
    market_map = Dict(market => i for (i, market) in enumerate(market_list))
    num_markets = length(market_list)
    back_odds = fill(NaN, num_markets)

    json_str = _run_python_cli(cli_path, ["odds", event_name, "-d"])
    
    if isnothing(json_str) || isempty(json_str)
        return RawBook(Vector{Symbol}(), Vector{Float64}())
    end

    data = JSON3.read(json_str)
    home_team_name = first(split(event_name, " v "))

    # Helper to insert into our flat vector
    function _insert!(sym, price)
        if haskey(market_map, sym)
            idx = market_map[sym]
            back_odds[idx] = price
        end
    end

    # --- FT 1x2 ---
    if haskey(data, "ft") && haskey(data.ft, "Match Odds")
        for (team, odds) in data.ft["Match Odds"]
            !isnothing(odds) || continue
            sym = if team == "The Draw"; :ft_1x2_draw
                  elseif team == home_team_name; :ft_1x2_home
                  else; :ft_1x2_away; end
            
            price = get(get(odds, :back, Dict()), :price, NaN)
            _insert!(sym, Float64(price))
        end
    end

    # --- FT Over/Under 2.5 ---
    # (Simplified for brevity, can expand logic from your original file)
    if haskey(data, "ft") && haskey(data.ft, "Over/Under 2.5 Goals")
        for (sel, odds) in data.ft["Over/Under 2.5 Goals"]
            !isnothing(odds) || continue
            sym = startswith(String(sel), "Over") ? :ft_ou_25_over : :ft_ou_25_under
            price = get(get(odds, :back, Dict()), :price, NaN)
            _insert!(sym, Float64(price))
        end
    end

     # --- FT BTTS ---
     if haskey(data, "ft") && haskey(data.ft, "Both teams to Score?")
        for (sel, odds) in data.ft["Both teams to Score?"]
            !isnothing(odds) || continue
            sym = String(sel) == "Yes" ? :ft_btts_yes : :ft_btts_no
            price = get(get(odds, :back, Dict()), :price, NaN)
            _insert!(sym, Float64(price))
        end
    end

    return RawBook(collect(market_list), back_odds)
end

# end # module
