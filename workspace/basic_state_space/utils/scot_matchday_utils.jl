using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions


using PythonCall
using JSON3
using DataFrames
using StatsPlots, Distributions
using Dates
using CSV


##################################################
# Models
##################################################
"""
    load_models_from_paths(model_paths::Dict{String, String})

Loads multiple models from a dictionary of paths.

# Arguments
- `model_paths`: A Dictionary mapping a descriptive model name (String) to its file path (String).

# Returns
- A Dictionary mapping the model name to the loaded model object.
"""
function load_models_from_paths(model_paths::Dict{String, String})
    loaded_models = Dict{String, Any}()
    for (name, path) in model_paths
        println("Loading model: '$name' from path: $path")
        # Assuming you have a function `load_model` available
        loaded_models[name] = load_model(path) 
    end
    return loaded_models
end



function print_1x2(predictions)
println("Home Win: mean ", round(mean( 1 ./ predictions.ft.home), digits=2),"  |  median " ,  round(median( 1 ./ predictions.ft.home), digits=2))
println("away Win: mean ", round(mean( 1 ./ predictions.ft.away), digits=2),"  |  median " ,  round(median( 1 ./ predictions.ft.away), digits=2))
println("draw Win: mean ", round(mean( 1 ./ predictions.ft.draw), digits=2),"  |  median " ,  round(median( 1 ./ predictions.ft.draw), digits=2))
end

function print_1x2_ht(predictions)
println("Home Win: mean", round(mean( 1 ./ predictions.ht.home), digits=2),"  | median " ,  round(median( 1 ./ predictions.ht.home), digits=2))
println("away Win: mean", round(mean( 1 ./ predictions.ht.away), digits=2),"  | median " ,  round(median( 1 ./ predictions.ht.away), digits=2))
println("draw Win: mean", round(mean( 1 ./ predictions.ht.draw), digits=2),"  |  median" ,  round(median( 1 ./ predictions.ht.draw), digits=2))
end


function print_under(predictions)
  println("under 05, mean: ", round(mean( 1 ./ predictions.ft.under_05),digits=2), " | ", round(median( 1 ./ predictions.ft.under_05), digits=2) )
  println("under 15, mean: ", round(mean( 1 ./ predictions.ft.under_15),digits=2), " | ", round(median( 1 ./ predictions.ft.under_15), digits=2) )
  println("under 25, mean: ", round(mean( 1 ./ predictions.ft.under_25),digits=2), " | ", round(median( 1 ./ predictions.ft.under_25), digits=2) )
  println("under 35, mean: ", round(mean( 1 ./ predictions.ft.under_35),digits=2), " | ", round(median( 1 ./ predictions.ft.under_35), digits=2) )
end

function print_under_ht(predictions)
  println("under 05, mean: ", round(mean( 1 ./ predictions.ht.under_05),digits=2), " | ", round(median( 1 ./ predictions.ht.under_05), digits=2) )
  println("under 15, mean: ", round(mean( 1 ./ predictions.ht.under_15),digits=2), " | ", round(median( 1 ./ predictions.ht.under_15), digits=2) )
  println("under 25, mean: ", round(mean( 1 ./ predictions.ht.under_25),digits=2), " | ", round(median( 1 ./ predictions.ht.under_25), digits=2) )
end

function print_over(predictions)
  println("over 05, mean: ", round(mean( 1 ./ (1 .- predictions.ft.under_05)),digits=2), " | ", round(median( 1 ./ (1 .- predictions.ft.under_05)), digits=2) )
  println("over 15, mean: ", round(mean( 1 ./ (1 .- predictions.ft.under_15)),digits=2), " | ", round(median( 1 ./ (1 .- predictions.ft.under_15)), digits=2) )
  println("over 25, mean: ", round(mean( 1 ./ (1 .- predictions.ft.under_25)),digits=2), " | ", round(median( 1 ./ (1 .- predictions.ft.under_25)), digits=2) )
  println("over 35, mean: ", round(mean( 1 ./ (1 .- predictions.ft.under_35)),digits=2), " | ", round(median( 1 ./ (1 .- predictions.ft.under_35)), digits=2) )
end

function print_over_ht(predictions)
  println("over 05, mean: ", round(mean( 1 ./ (1 .- predictions.ht.under_05)),digits=2), " | ", round(median( 1 ./ (1 .- predictions.ht.under_05)), digits=2) )
  println("over 15, mean: ", round(mean( 1 ./ (1 .- predictions.ht.under_15)),digits=2), " | ", round(median( 1 ./ (1 .- predictions.ht.under_15)), digits=2) )
  println("over 25, mean: ", round(mean( 1 ./ (1 .- predictions.ht.under_25)),digits=2), " | ", round(median( 1 ./ (1 .- predictions.ht.under_25)), digits=2) )
end

"""
    generate_predictions(
        models::Dict{String, Any}, 
        home_team::String, 
        away_team::String, 
        league_id::Int
    )

Generates match line predictions for a given match across multiple models.

# Arguments
- `models`: A dictionary of loaded model objects from `load_models_from_paths`.
- `home_team`, `away_team`, `league_id`: Details of the match to predict.

# Returns
- A Dictionary mapping the model name to its `MatchLinePredictions` object.
"""
function generate_predictions(models::Dict{String, Any}, home_team::String, away_team::String, league_id::Int)
    
    predictions = Dict{String, Any}()

    for (name, model) in models
        println("Generating predictions for model: '$name'")
        
        # Use 'local' to ensure the variable is accessible after the if/else block
        local match_to_predict::DataFrame

      # --- State Space Model: Requires 'global_round' ---
      println("-> Detected State Space model. Calculating next round.")
      
      chains = model.result.chains_sequence[1]
      mapping = model.result.mapping

      posterior_samples = BayesianFootball.extract_posterior_samples(
          model.config.model_def,
          chains.ft,
          mapping
      )
      last_training_round = posterior_samples.n_rounds
      next_round = last_training_round + 1
      println("   - Predicting for global_round: $next_round")

      match_to_predict = DataFrame(
          home_team=home_team,
          away_team=away_team,
          tournament_id=league_id,
          global_round=next_round, # The crucial addition
          home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
      )

        # --- Common steps for ALL models ---
        
        # 1. Create features from the DataFrame prepared above
        features = BayesianFootball.create_master_features(
            match_to_predict,
            model.result.mapping
        )

        # 2. Call the single, generic prediction function
        preds = BayesianFootball.predict_match_lines(
            model.config.model_def,
            model.result.chains_sequence[1],
            features,
            model.result.mapping
        )
        
        predictions[name] = preds
    end
    
    return predictions
end

"""
    create_odds_dataframe(predictions::Dict{String, Any})

Converts a dictionary of predictions into a DataFrame of mean odds.

# Arguments
- `predictions`: A dictionary of `MatchLinePredictions` from `generate_predictions`.

# Returns
- A DataFrame with rows for each model/time (FT/HT) and columns for market odds.
"""
function create_odds_dataframe(predictions::Dict{String, Any})
    
    df = DataFrame(
        Model=String[], Time=String[], Home=Float64[], Draw=Float64[], Away=Float64[],
        O05=Float64[], U05=Float64[], O15=Float64[], U15=Float64[],
        O25=Float64[], U25=Float64[], BTTS_Yes=Union{Missing, Float64}[], BTTS_No=Union{Missing, Float64}[]
    )

    for (name, pred) in predictions
        # Full Time (FT) Predictions
        ft = pred.ft
        push!(df, (
            Model=name, Time="FT",
            Home = mean(1 ./ ft.home),
            Draw = mean(1 ./ ft.draw),
            Away = mean(1 ./ ft.away),
            U05 = mean(1 ./ ft.under_05),
            O05 = mean(1 ./ (1 .- ft.under_05)),
            U15 = mean(1 ./ ft.under_15),
            O15 = mean(1 ./ (1 .- ft.under_15)),
            U25 = mean(1 ./ ft.under_25),
            O25 = mean(1 ./ (1 .- ft.under_25)),
            BTTS_Yes = mean(1 ./ ft.btts),
            BTTS_No = mean(1 ./ (1 .- ft.btts))
        ))
        
        # Half Time (HT) Predictions
        ht = pred.ht
        push!(df, (
            Model=name, Time="HT",
            Home = mean(1 ./ ht.home),
            Draw = mean(1 ./ ht.draw),
            Away = mean(1 ./ ht.away),
            U05 = mean(1 ./ ht.under_05),
            O05 = mean(1 ./ (1 .- ht.under_05)),
            U15 = mean(1 ./ ht.under_15),
            O15 = mean(1 ./ (1 .- ht.under_15)),
            U25 = mean(1 ./ ht.under_25),
            O25 = mean(1 ./ (1 .- ht.under_25)),
            BTTS_Yes = missing, # BTTS not typically calculated for HT
            BTTS_No = missing
        ))
    end
    
    return df
end


##################################################
# Odds 
##################################################

const CLI_PATH = "/home/james/bet_project/whatstheodds"

const MARKET_LIST = [
    # Full Time 1x2
    :ft_1x2_home, :ft_1x2_draw, :ft_1x2_away,
    
    # Full Time Over/Under
    :ft_ou_05_under, :ft_ou_05_over,
    :ft_ou_15_under, :ft_ou_15_over,
    :ft_ou_25_under, :ft_ou_25_over,
    :ft_ou_35_under, :ft_ou_35_over,
    
    # Full Time BTTS
    :ft_btts_yes, :ft_btts_no,
    
    # Full Time Correct Score (add as many as you want to track)
    :ft_cs_0_0, :ft_cs_1_0, :ft_cs_0_1, :ft_cs_1_1, :ft_cs_2_0, :ft_cs_0_2,
    :ft_cs_2_1, :ft_cs_1_2, :ft_cs_2_2, :ft_cs_3_0, :ft_cs_0_3, :ft_cs_3_1,
    :ft_cs_1_3, :ft_cs_3_2, :ft_cs_2_3, :ft_cs_3_3,
    :ft_cs_other_home, :ft_cs_other_draw, :ft_cs_other_away,

    # Half Time 1x2
    :ht_1x2_home, :ht_1x2_draw, :ht_1x2_away,
    
    # Half Time Over/Under
    :ht_ou_05_under, :ht_ou_05_over,
    :ht_ou_15_under, :ht_ou_15_over,
    :ht_ou_25_under, :ht_ou_25_over,
    
    # Half Time Correct Score
    :ht_cs_0_0, :ht_cs_1_0, :ht_cs_0_1, :ht_cs_1_1, :ht_cs_2_0, :ht_cs_2_1,
    :ht_cs_2_2, :ht_cs_0_2, :ht_cs_1_2, :ht_cs_other
]



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
    fetch_all_market_odds(
        matches_df::DataFrame,
        market_list::Vector{Symbol};
        cli_path::String
    ) -> DataFrame

Fetches live market odds for all matches in a DataFrame and returns them in a single wide-format DataFrame.

# Arguments
- `matches_df`: A DataFrame with match information, including an `event_name` column.
- `market_list`: The master list of market symbols to fetch odds for.
- `cli_path`: The file path to the Python CLI tool.

# Returns
- A `DataFrame` where each row corresponds to a match and columns represent the back/lay odds for each market.
"""
function fetch_all_market_odds(
    matches_df::DataFrame,
    market_list::Vector{Symbol};
    cli_path::String
)
    all_match_odds_list = []
    println("🚀 Starting to fetch market odds for $(nrow(matches_df)) matches...")

    for row in eachrow(matches_df)
        event = row.event_name
        println("Fetching odds for: $event")

        try
            # Fetch the comprehensive odds using the list defined previously
            market_book = get_live_market_odds(event, market_list; cli_path=cli_path)

            # Create a dictionary for this match's row
            match_row = Dict{Symbol, Any}()
            match_row[:event_name] = event
            match_row[:home_team] = row.home_team
            match_row[:away_team] = row.away_team

            for i in 1:length(market_book.markets)
                market_symbol = market_book.markets[i]
                back_col_name = Symbol(market_symbol, "_back")
                lay_col_name = Symbol(market_symbol, "_lay")

                back_odd = market_book.back_odds[i]
                lay_odd = market_book.lay_odds[i]

                # Check if the value is NaN. [cite_start]If it is, use `missing`; otherwise, use the value [cite: 63]
                match_row[back_col_name] = isnan(back_odd) ? missing : back_odd
                match_row[lay_col_name] = isnan(lay_odd) ? missing : lay_odd 
            end

            # Add the completed row to our list
            push!(all_match_odds_list, match_row)

        catch e
            @error "Failed to process match: $event. Error: $e"
            continue # Move to the next match
        end
    end

    println("\n✅ Finished fetching odds.")
    
    if isempty(all_match_odds_list)
        return DataFrame()
    else
        return DataFrame(all_match_odds_list)
    end
end



"""
Safely extracts a price from a nested odds dictionary structure.
Handles cases where the side (:back or :lay) or the price itself is missing or null.
"""
function _safe_get_price(odds_obj, side::Symbol)
    # 1. Safely get the details object for the side (:back or :lay)
    details = get(odds_obj, side, 0)
    
    # 2. If the details object is nothing, we can't proceed. Return NaN.
    if isnothing(details)
        return 0
    end
    
    # 3. If we have a details object, safely get the price from it.
    return get(details, :price, 0)
end

"""
    get_todays_matches(leagues::Vector{String}; cli_path::String)

Fetches today's matches using the Python CLI and returns a structured DataFrame.
"""
function get_live_market_odds(event_name::String, market_list::Vector{Symbol}; cli_path::String)
    market_map = Dict(market => i for (i, market) in enumerate(market_list))
    num_markets = length(market_list)
    back_odds = fill(NaN, num_markets)
    lay_odds = fill(NaN, num_markets)

    json_str = _run_python_cli(cli_path, ["odds", event_name, "-d"])
    if isnothing(json_str) || isempty(json_str)
        @warn "No odds data returned for $event_name"
        return MarketBook(market_list, market_map, back_odds, lay_odds)
    end

    data = JSON3.read(json_str)
    home_team_name = first(split(event_name, " v "))

    # --- Process Full-Time (ft) Markets ---
    if haskey(data, "ft")
        ft_data = data.ft

        # -- FT Match Odds --
        if haskey(ft_data, "Match Odds")
            for (team_key, odds) in ft_data["Match Odds"]
                if !isnothing(odds)
                    team_name = String(team_key)
                    market_symbol = if team_name == "The Draw"
                        :ft_1x2_draw
                    elseif team_name == home_team_name
                        :ft_1x2_home
                    else
                        :ft_1x2_away
                    end

                    if haskey(market_map, market_symbol)
                        idx = market_map[market_symbol]
                        back_odds[idx] = _safe_get_price(odds, :back)
                        lay_odds[idx] = _safe_get_price(odds, :lay)
                    end
                end
            end
        end
        
        # -- FT Over/Under Markets --
        ou_markets = Dict(
            "Over/Under 0.5 Goals" => :ou_05, "Over/Under 1.5 Goals" => :ou_15,
            "Over/Under 2.5 Goals" => :ou_25, "Over/Under 3.5 Goals" => :ou_35
        )
        for (market_name, prefix) in ou_markets
            if haskey(ft_data, market_name)
                for (selection, odds) in ft_data[market_name]
                    if !isnothing(odds)
                        side = startswith(String(selection), "Over") ? :over : :under
                        market_symbol = Symbol(:ft_, prefix, :_, side)
                        if haskey(market_map, market_symbol)
                            idx = market_map[market_symbol]
                            back_odds[idx] = _safe_get_price(odds, :back)
                            lay_odds[idx] = _safe_get_price(odds, :lay)
                        end
                    end
                end
            end
        end

        # -- FT Correct Score --
        if haskey(ft_data, "Correct Score")
            for (score, odds) in ft_data["Correct Score"]
                if !isnothing(odds)
                    s_score = String(score)
                    market_symbol = if occursin(r"\d\s-\s\d", s_score)
                        Symbol(:ft_cs_, replace(s_score, " - " => "_"))
                    elseif s_score == "Any Other Home Win"
                        :ft_cs_other_home
                    elseif s_score == "Any Other Away Win"
                        :ft_cs_other_away
                    elseif s_score == "Any Other Draw"
                        :ft_cs_other_draw
                    else
                        continue
                    end

                    if haskey(market_map, market_symbol)
                        idx = market_map[market_symbol]
                        back_odds[idx] = _safe_get_price(odds, :back)
                        lay_odds[idx] = _safe_get_price(odds, :lay)
                    end
                end
            end
        end

        # -- FT Both Teams To Score --
        if haskey(ft_data, "Both teams to Score?")
            for (selection, odds) in ft_data["Both teams to Score?"]
                if !isnothing(odds)
                    market_symbol = String(selection) == "Yes" ? :ft_btts_yes : :ft_btts_no
                    if haskey(market_map, market_symbol)
                        idx = market_map[market_symbol]
                        back_odds[idx] = _safe_get_price(odds, :back)
                        lay_odds[idx] = _safe_get_price(odds, :lay)
                    end
                end
            end
        end
    end # End FT markets

    # --- Process Half-Time (ht) Markets ---
    if haskey(data, "ht")
        ht_data = data.ht

        # -- HT Match Odds (Half Time) --
        if haskey(ht_data, "Half Time")
            for (team_key, odds) in ht_data["Half Time"]
                if !isnothing(odds)
                    team_name = String(team_key)
                    market_symbol = if team_name == "The Draw"
                        :ht_1x2_draw
                    elseif team_name == home_team_name
                        :ht_1x2_home
                    else
                        :ht_1x2_away
                    end

                    if haskey(market_map, market_symbol)
                        idx = market_map[market_symbol]
                        back_odds[idx] = _safe_get_price(odds, :back)
                        lay_odds[idx] = _safe_get_price(odds, :lay)
                    end
                end
            end
        end

        # -- HT Over/Under Markets --
        ht_ou_markets = Dict(
            "First Half Goals 0.5" => :ou_05, "First Half Goals 1.5" => :ou_15,
            "First Half Goals 2.5" => :ou_25
        )
        for (market_name, prefix) in ht_ou_markets
            if haskey(ht_data, market_name)
                for (selection, odds) in ht_data[market_name]
                    if !isnothing(odds)
                        side = startswith(String(selection), "Over") ? :over : :under
                        market_symbol = Symbol(:ht_, prefix, :_, side)
                        if haskey(market_map, market_symbol)
                            idx = market_map[market_symbol]
                            back_odds[idx] = _safe_get_price(odds, :back)
                            lay_odds[idx] = _safe_get_price(odds, :lay)
                        end
                    end
                end
            end
        end

        # -- HT Correct Score --
        if haskey(ht_data, "Half Time Score")
            for (score, odds) in ht_data["Half Time Score"]
                if !isnothing(odds)
                    s_score = String(score)
                    market_symbol = if occursin(r"\d\s-\s\d", s_score)
                        Symbol(:ht_cs_, replace(s_score, " - " => "_"))
                    elseif s_score == "Any unquoted"
                        :ht_cs_other
                    else
                        continue
                    end

                    if haskey(market_map, market_symbol)
                        idx = market_map[market_symbol]
                        back_odds[idx] = _safe_get_price(odds, :back)
                        lay_odds[idx] = _safe_get_price(odds, :lay)
                    end
                end
            end
        end
    end # End HT markets

    return MarketBook(market_list, market_map, back_odds, lay_odds)
end


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
    save_odds_to_csv(odds_df::DataFrame, output_folder::String)

Saves a DataFrame of market odds to a CSV file with a date-stamped name.

# Arguments
- `odds_df`: The DataFrame containing the market odds.
- `output_folder`: The path to the directory where the CSV should be saved.

# Returns
- The full path to the saved file, or `nothing` if the DataFrame was empty.
"""
function save_odds_to_csv(odds_df::DataFrame, output_folder::String)
    if isempty(odds_df)
         println("No odds were successfully fetched. CSV file not created.")
        return nothing
    end

    # Create the folder if it doesn't exist
    mkpath(output_folder)

    # Create a filename with today's date
    filename = "market_odds_$(today()).csv"
    full_path = joinpath(output_folder, filename)

    CSV.write(full_path, odds_df)

    println("✅ Successfully saved market odds to:")
    println(full_path)

    return full_path
end




##################################################
# Plots 
##################################################

function plot_attack_defence(team1_name, team2_name, loaded_model, posterior_samples)

team1_id = loaded_model.result.mapping.team[team1_name]
team2_id = loaded_model.result.mapping.team[team2_name]

# --- 2. Get the full time-series of the parameters ---
log_α_centered = posterior_samples.log_α_centered
log_β_centered = posterior_samples.log_β_centered
n_rounds = posterior_samples.n_rounds

# --- 3. Calculate the posterior mean AND STANDARD DEVIATION over time ---
# Mean calculations
team1_attack_mean = vec(mean(log_α_centered[:, team1_id, :], dims=1))
team1_defense_mean = vec(mean(log_β_centered[:, team1_id, :], dims=1))
team2_attack_mean = vec(mean(log_α_centered[:, team2_id, :], dims=1))
team2_defense_mean = vec(mean(log_β_centered[:, team2_id, :], dims=1))

# Standard deviation calculations
team1_attack_std = vec(std(log_α_centered[:, team1_id, :], dims=1))
team1_defense_std = vec(std(log_β_centered[:, team1_id, :], dims=1))
team2_attack_std = vec(std(log_α_centered[:, team2_id, :], dims=1))
team2_defense_std = vec(std(log_β_centered[:, team2_id, :], dims=1))


# --- 4. Create the 1x2 plot with ribbons ---
p = plot(
    layout=(1, 2),
    # size=(1400, 500),
    size=(900, 500),
    legend=:bottomleft,
    # link=:y,
    xlabel="Global Round"
)

# Subplot 1: Attacking Strength
plot!(p[1], 1:n_rounds, team1_attack_mean,
    ribbon = 1 .* team1_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,                # Make ribbon transparent
    label = team1_name,
    title = "Attacking Strength (log α)",
    ylabel = "Parameter Value",
    lw = 2
)
plot!(p[1], 1:n_rounds, team2_attack_mean,
    ribbon = 1 .* team2_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
    lw = 2
)

# Subplot 2: Defensive Strength
plot!(p[2], 1:n_rounds, team1_defense_mean,
    ribbon = 1 .* team1_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team1_name,
    title = "Defensive Strength (log β)",
    lw = 2
)
plot!(p[2], 1:n_rounds, team2_defense_mean,
    ribbon = 1 .* team2_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
    lw = 2
)


end 

function plot_1x2(predictions)
p = plot(
    layout=(1, 1),
    # size=(1400, 500),
    size=(900, 500),
    legend=:topleft,
    # link=:y,
    ylabel ="likelihood",
    xlabel="probability"
)
  density!(p, predictions.ft.home, label="home")
  density!(p, predictions.ft.away, label="away")
  density!(p, predictions.ft.draw, label="draw")


end 

function plot_xg(predictions)
p = plot(
    layout=(1, 1),
    # size=(1400, 500),
    size=(900, 500),
    legend=:topleft,
    # link=:y,
    title =" predicted goals",
    ylabel ="likelihood",
    xlabel="probability"
)
  density!(p, predictions.ft.λ_h, label="home")
  density!(p, predictions.ft.λ_a, label="away")
end 

function plot_xg_t(predictions)
p = plot(
    layout=(1, 1),
    # size=(1400, 500),
    size=(900, 500),
    legend=:topleft,
    # link=:y,
    title =" predicted goals",
    ylabel ="likelihood",
    xlabel="probability"
)
  density!(p, predictions.ft.λ_h, label="home")
  density!(p, predictions.ft.λ_a, label="away")
  density!(p, predictions.ft.λ_h .+  predictions.ft.λ_a, label="total")
end 



########################################################
########################################################

using DataFrames
using Statistics

"""
    _get_prediction_vector(predictions::BayesianFootball.Predictions.MatchLinePredictions, market_symbol::Symbol)

A helper function to retrieve the correct probability vector from the nested 
`MatchLinePredictions` struct based on a standardized market symbol.

# Returns
- A `Vector{Float64}` of MCMC probability samples, or `nothing` if the market is not found.
"""
function _get_prediction_vector(predictions::BayesianFootball.Predictions.MatchLinePredictions, market_symbol::Symbol)
    s_market = string(market_symbol)

    # Full-Time 1x2
    if market_symbol == :ft_1x2_home return predictions.ft.home
    elseif market_symbol == :ft_1x2_draw return predictions.ft.draw
    elseif market_symbol == :ft_1x2_away return predictions.ft.away
    # Full-Time Over/Under
    elseif market_symbol == :ft_ou_05_over return 1 .- predictions.ft.under_05
    elseif market_symbol == :ft_ou_05_under return predictions.ft.under_05
    elseif market_symbol == :ft_ou_15_over return 1 .- predictions.ft.under_15
    elseif market_symbol == :ft_ou_15_under return predictions.ft.under_15
    elseif market_symbol == :ft_ou_25_over return 1 .- predictions.ft.under_25
    elseif market_symbol == :ft_ou_25_under return predictions.ft.under_25
    elseif market_symbol == :ft_ou_35_over return 1 .- predictions.ft.under_35
    elseif market_symbol == :ft_ou_35_under return predictions.ft.under_35
    # Full-Time BTTS
    elseif market_symbol == :ft_btts_yes return predictions.ft.btts
    elseif market_symbol == :ft_btts_no return 1 .- predictions.ft.btts
    # Full-Time Correct Score
    elseif startswith(s_market, "ft_cs_")
        if s_market == "ft_cs_other_home"
            return get(predictions.ft.correct_score, "other_home_win", nothing)
        elseif s_market == "ft_cs_other_away"
            return get(predictions.ft.correct_score, "other_away_win", nothing)
        elseif s_market == "ft_cs_other_draw"
            return get(predictions.ft.correct_score, "other_draw", nothing)
        else # It's a numeric score
            score_str = split(s_market, "_")[3:4]
            h, a = parse(Int, score_str[1]), parse(Int, score_str[2])
            return get(predictions.ft.correct_score, (h, a), nothing)
        end
    # Half-Time 1x2
    elseif market_symbol == :ht_1x2_home return predictions.ht.home
    elseif market_symbol == :ht_1x2_draw return predictions.ht.draw
    elseif market_symbol == :ht_1x2_away return predictions.ht.away
    # Half-Time Over/Under
    elseif market_symbol == :ht_ou_05_over return 1 .- predictions.ht.under_05
    elseif market_symbol == :ht_ou_05_under return predictions.ht.under_05
    elseif market_symbol == :ht_ou_15_over return 1 .- predictions.ht.under_15
    elseif market_symbol == :ht_ou_15_under return predictions.ht.under_15
    elseif market_symbol == :ht_ou_25_over return 1 .- predictions.ht.under_25
    elseif market_symbol == :ht_ou_25_under return predictions.ht.under_25
     # Half-Time Correct Score
    elseif startswith(s_market, "ht_cs_")
        if s_market == "ht_cs_other"
            return get(predictions.ht.correct_score, "any_unquoted", nothing)
        else # It's a numeric score
            score_str = split(s_market, "_")[3:4]
            h, a = parse(Int, score_str[1]), parse(Int, score_str[2])
            return get(predictions.ht.correct_score, (h, a), nothing)
        end
    end
    
    return nothing # Return nothing for unhandled markets
end


"""
    calculate_ev_dataframe(
        all_predictions::Dict{String, <:BayesianFootball.Predictions.MatchLinePredictions}, 
        odds_df_row::DataFrameRow,
        market_list::Vector{Symbol}
    ) -> DataFrame

Calculates the Expected Value (EV) for all available markets based on predictions 
from multiple models and a single row of market odds, ordered by `market_list`.
"""
function calculate_ev_dataframe(all_predictions::Dict{String, <:BayesianFootball.Predictions.MatchLinePredictions}, odds_df_row::DataFrameRow)
    
    # Use the master market_list but filter for markets available in this specific match's odds data
    available_markets = filter(m -> hasproperty(odds_df_row, Symbol(m, "_back")), MARKET_LIST)
    
    # Initialize the DataFrame with market info, now in the correct order
    ev_df = DataFrame(
        market = string.(available_markets),
        market_back = [getproperty(odds_df_row, Symbol(m, "_back")) for m in available_markets],
        market_lay = [getproperty(odds_df_row, Symbol(m, "_lay")) for m in available_markets]
    )

    # Loop through each model to calculate and append its EV columns
    for (model_name, predictions) in all_predictions
        mean_evs = Union{Missing, Float64}[]
        std_evs = Union{Missing, Float64}[]

        for market_symbol in available_markets
            p_chain = _get_prediction_vector(predictions, market_symbol)
            back_odds = getproperty(odds_df_row, Symbol(market_symbol, "_back"))

            if !isnothing(p_chain) && !ismissing(back_odds) && back_odds > 0
                ev_chain = (p_chain .* back_odds) .- 1
                push!(mean_evs, mean(ev_chain) * 100)
                push!(std_evs, std(ev_chain) * 100)
            else
                push!(mean_evs, missing)
                push!(std_evs, missing)
            end
        end
        
        # Add columns for the current model
        ev_df[!, Symbol(model_name, "_mean_ev_pct")] = mean_evs
        ev_df[!, Symbol(model_name, "_std_ev_pct")] = std_evs
    end
    
    # Remove rows where all models have missing EV (e.g., unhandled markets)
    model_ev_cols = [col for col in names(ev_df) if endswith(string(col), "_mean_ev_pct")]
    filter!(row -> !all(ismissing, row[col] for col in model_ev_cols), ev_df)

    return ev_df
end

"""
    create_comparison_dataframe(
        predictions::BayesianFootball.Predictions.MatchLinePredictions, 
        odds_df_row::DataFrameRow,
        model_name::String,
        market_list::Vector{Symbol}
    ) -> DataFrame

Compares a model's posterior distributions against the market odds for a single match, ordered by `market_list`.
"""
function create_comparison_dataframe(predictions::BayesianFootball.Predictions.MatchLinePredictions, odds_df_row::DataFrameRow, model_name::String)
    
    # Use the master market_list but filter for markets available in this specific match's odds data
    available_markets = filter(m -> hasproperty(odds_df_row, Symbol(m, "_back")), MARKET_LIST)

    comp_df = DataFrame(
        market = string.(available_markets),
        market_back = [getproperty(odds_df_row, Symbol(m, "_back")) for m in available_markets],
        market_lay = [getproperty(odds_df_row, Symbol(m, "_lay")) for m in available_markets]
    )
    
    model_mean_odds = Union{Missing, Float64}[]
    model_median_odds = Union{Missing, Float64}[]
    market_quantile = Union{Missing, Float64}[]

    for market_symbol in available_markets
        p_chain = _get_prediction_vector(predictions, market_symbol)
        back_odds = getproperty(odds_df_row, Symbol(market_symbol, "_back"))

        if !isnothing(p_chain) && !ismissing(back_odds) && back_odds > 0
            # Ensure no zero probabilities before division
            p_chain_safe = filter(p -> p > 1e-9, p_chain)
            if isempty(p_chain_safe)
                push!(model_mean_odds, missing)
                push!(model_median_odds, missing)
                push!(market_quantile, missing)
                continue
            end
            
            model_odds_chain = 1.0 ./ p_chain_safe
            
      push!(model_mean_odds, round(mean(model_odds_chain), digits=2))
      push!(model_median_odds, round(median(model_odds_chain), digits=2))
            
            # Calculate the proportion of model odds that are HIGHER than the market
            # A high value means the model thinks the market price is too low (good value)
            quantile_val = sum(model_odds_chain .> back_odds) / length(model_odds_chain)
            push!(market_quantile, quantile_val)
        else
            push!(model_mean_odds, missing)
            push!(model_median_odds, missing)
            push!(market_quantile, missing)
        end
    end

    comp_df[!, Symbol(model_name, "_mean_odds")] = model_mean_odds
    comp_df[!, Symbol(model_name, "_median_odds")] = model_median_odds
    comp_df[!, Symbol(model_name, "_market_quantile")] = market_quantile
    
    # Filter out rows with no data
    filter!(row -> !ismissing(row[Symbol(model_name, "_mean_odds")]), comp_df)

    return comp_df
end



