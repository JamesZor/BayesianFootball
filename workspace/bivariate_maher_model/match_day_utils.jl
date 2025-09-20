# src/utils/MatchDayUtils.jl

module MatchDayUtils

using PythonCall
using JSON3
using DataFrames
using StatsPlots, Distributions

export MarketBook, PredictionMatrix, EVDistribution # Exporting structs for type access
export get_todays_matches, get_live_market_odds, calculate_ev_distributions
export plot_market_distribution_vs_odds, plot_ev_distributions



# ENV["PYTHONCALL_EXE"] = "home/james/.conda/envs/webscrape"
# --------------------------------------------------------------------------- #
# --- 1. CORE DATA STRUCTURES ---
# --------------------------------------------------------------------------- #

"""
    MarketBook

Holds the complete back and lay odds for a single match, structured for
efficient matrix operations.
"""
struct MarketBook
    markets::Vector{Symbol}
    market_map::Dict{Symbol, Int}
    back_odds::Vector{Float64}
    lay_odds::Vector{Float64}
end

"""
    PredictionMatrix

Holds the full posterior probability distributions from a single model for a
match, mirroring the MarketBook structure.
"""
struct PredictionMatrix
    markets::Vector{Symbol}
    market_map::Dict{Symbol, Int}
    probabilities::Matrix{Float64} # Dim: (num_mcmc_samples, num_markets)
end

"""
    EVDistribution

Holds the calculated posterior distribution of Expected Value.
"""
struct EVDistribution
    markets::Vector{Symbol}
    market_map::Dict{Symbol, Int}
    ev::Matrix{Float64} # Dim: (num_mcmc_samples, num_markets)
end

# --------------------------------------------------------------------------- #
# --- 2. INTERNAL HELPER FUNCTIONS ---
# --------------------------------------------------------------------------- #

"""
Runs a command for the whatstheodds Python CLI.
Assumes PythonCall is configured for the 'webscraper' Conda env.
"""
function _run_python_cli(cli_path::String, args::Vector{String})
    script_path = joinpath(cli_path, "live_odds_cli.py")
    python_exe = "/home/james/.conda/envs/webscrape/bin/python"

    # 1. Create a vector containing all parts of the command
    full_command_vector = [python_exe, script_path]
    
    # 2. Append the arguments from the input 'args' vector
    append!(full_command_vector, args)

    # 3. Create the command object from the vector
    cmd = Cmd(full_command_vector)
    try
        output = read(setenv(cmd, dir=cli_path), String)
        return output
    catch e
        @error "Failed to run Python CLI script."
        println("Command: $cmd")
        println("Working Directory: $cli_path")
        println("Error: ", e)
        return nothing
    end
end

"""
Loads team mapping JSON files and creates a reverse map from display name to model name.
"""
function _load_team_mappings(mappings_path::String, leagues::Vector{String})
    reverse_map = Dict{String, String}()
    for league in leagues
        filepath = joinpath(mappings_path, "$league.json")
        if isfile(filepath)
            json_str = read(filepath, String)
            data = JSON3.read(json_str)
            for (model_name, display_name) in data
                reverse_map[String(display_name)] = String(model_name)
            end
        else
            @warn "Mapping file not found for league: $league"
        end
    end
    return reverse_map
end

# --------------------------------------------------------------------------- #
# --- 3. DATA FETCHING FUNCTIONS ---
# --------------------------------------------------------------------------- #

"""
    get_todays_matches(leagues::Vector{String}; cli_path::String)

Fetches today's matches using the Python CLI and returns a structured DataFrame.
"""
function get_todays_matches(leagues::Vector{String}; cli_path::String)
    mappings_path = joinpath(cli_path, "mappings")
    team_map = _load_team_mappings(mappings_path, leagues)
    
    output = _run_python_cli(cli_path, ["list", "-f", leagues...])
    isnothing(output) && return DataFrame()

    matches = []
    for line in split(output, '\n')
        # Regex to capture time and teams robustly
        m = match(r"^\s*-\s*(\d{2}:\d{2})\s*\|\s*(.+?)\s+v\s+(.+?)$", line)
        if !isnothing(m)
            time_str, home_display, away_display = m.captures
            
            home_model = get(team_map, home_display, home_display) # Fallback to display name
            away_model = get(team_map, away_display, away_display)
            
            push!(matches, (
                time = time_str,
                event_name = "$home_display v $away_display",
                home_team = home_model,
                away_team = away_model
            ))
        end
    end
    return DataFrame(matches)
end


"""
    get_live_market_odds(event_name::String, market_list::Vector{Symbol}; cli_path::String)

Fetches live odds for an event and populates a MarketBook struct.
"""
function get_live_market_odds(event_name::String, market_list::Vector{Symbol}; cli_path::String)
    market_map = Dict(market => i for (i, market) in enumerate(market_list))
    num_markets = length(market_list)
    back_odds = fill(NaN, num_markets)
    lay_odds = fill(NaN, num_markets)

    json_str = _run_python_cli(cli_path, ["odds", event_name, "-d"])
    isnothing(json_str) && return MarketBook(market_list, market_map, back_odds, lay_odds)

    data = JSON3.read(json_str)

    # --- This is the complex mapping logic ---
    if haskey(ft_data, "Match Odds")
      # Get the home team name from the event string "TeamA v TeamB"
      home_team_name = first(split(event_name, " v "))

      for (team_key, odds) in ft_data["Match Odds"]
          # Convert team_key (which can be a Symbol or String) to a String
          team_name = String(team_key)
          
          # Determine the market based on the string name
          market_symbol = if team_name == "The Draw"
              :ft_1x2_draw
          elseif team_name == home_team_name
              :ft_1x2_home
          else
              :ft_1x2_away
          end

          # Populate the odds
          if haskey(market_map, market_symbol)
              idx = market_map[market_symbol]
              back_odds[idx] = get(get(odds, :back, Dict()), :price, NaN)
              lay_odds[idx] = get(get(odds, :lay, Dict()), :price, NaN)
          end
      end
  end
        # Over/Under 2.5
        if haskey(ft_data, "Over/Under 2.5 Goals")
            # ... and so on for all other markets you care about ...
            # You would add parsers here for Correct Score, BTTS, etc.
        end
    end
    
    # Half-Time Markets can be parsed similarly from data.ht

    return MarketBook(market_list, market_map, back_odds, lay_odds)
end

# --------------------------------------------------------------------------- #
# --- 4. ANALYSIS AND PLOTTING ---
# --------------------------------------------------------------------------- #

"""
    calculate_ev_distributions(predictions::PredictionMatrix, book::MarketBook)

Calculates the full posterior distribution of EV for all markets.
"""
function calculate_ev_distributions(predictions::PredictionMatrix, book::MarketBook)
    # Ensure alignment
    @assert predictions.markets == book.markets "Market lists do not match between predictions and market book!"

    # Use broadcasting for efficient, vectorized calculation
    market_odds_row_vec = permutedims(book.back_odds)
    ev_matrix = predictions.probabilities .* market_odds_row_vec .- 1
    
    return EVDistribution(predictions.markets, predictions.market_map, ev_matrix)
end


"""
    plot_market_distribution_vs_odds(pred_matrix::PredictionMatrix, book::MarketBook, market::Symbol)

Plots the model's posterior odds distribution against the market back/lay odds.
"""
function plot_market_distribution_vs_odds(pred_matrix::PredictionMatrix, book::MarketBook, market::Symbol)
    idx = get(pred_matrix.market_map, market, 0)
    idx == 0 && error("Market :$market not found in prediction matrix.")

    # Convert probability posterior to odds posterior
    model_odds_dist = 1 ./ pred_matrix.probabilities[:, idx]

    market_back = book.back_odds[idx]
    market_lay = book.lay_odds[idx]

    # Calculate quantiles
    back_quantile = round(100 * mean(model_odds_dist .<= market_back); digits=1)
    
    p = density(model_odds_dist, label="Model Odds Distribution",
              title="Odds Distribution for :$market",
              xlabel="Odds", ylabel="Density")
    
    vline!(p, [market_back], lw=2, ls=:dash, label="Market Back ($market_back) at $back_quantile-th percentile")
    vline!(p, [market_lay], lw=2, ls=:dash, c=:red, label="Market Lay ($market_lay)")
    
    return p
end


"""
    plot_ev_distributions(ev_dists::Dict{String, EVDistribution}, market::Symbol)

Plots and compares the EV distributions from multiple models for a given market.
"""
function plot_ev_distributions(ev_dists::Dict{String, EVDistribution}, market::Symbol)
    title = "Expected Value (EV) Distribution for :$market"
    p = plot(title=title, xlabel="Expected Value (EV)", ylabel="Density")
    
    vline!(p, [0], lw=2, c=:black, ls=:dash, label="Break-even (EV=0)")

    for (model_name, ev_dist) in ev_dists
        idx = get(ev_dist.market_map, market, 0)
        if idx > 0
            ev_posterior = ev_dist.ev[:, idx]
            mean_ev = round(mean(ev_posterior), digits=3)
            density!(p, ev_posterior, label="$model_name (Mean EV: $mean_ev)")
        end
    end
    
    return p
end


end # end module
