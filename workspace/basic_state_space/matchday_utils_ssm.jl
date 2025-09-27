
module MatchDayUtilsSSM

include("/home/james/bet_project/models_julia/workspace/basic_state_space/setup.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/prediction.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/analysis_functions.jl")


using .AR1NegativeBinomial
using .AR1NegBiPrediction
using .AR1StateSpace
using .AR1Prediction

using .AnalysisSSM

using PythonCall
using JSON3
using DataFrames
using StatsPlots, Distributions
using Dates
using CSV

export MarketBook, PredictionMatrix, EVDistribution # Exporting structs for type access
export get_todays_matches, get_live_market_odds, calculate_ev_distributions
export plot_market_distribution_vs_odds, plot_ev_distributions
export generate_match_analysis, plot_multi_model_odds_distribution
export CLI_PATH, MARKET_LIST

export fetch_all_market_odds, save_odds_to_csv

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

function generate_prediction_matrix(model, home_team, away_team, league_id, market_list)
    # Generate predictions using your existing function
    preds_dict = generate_predictions(
        Dict{String, Any}("temp" => model), 
        home_team, 
        away_team, 
        league_id
    )
    preds = preds_dict["temp"]
    
    # Get MCMC sample size from one of the prediction vectors
    num_samples = length(preds.ft.home)
    market_map = Dict(m => i for (i, m) in enumerate(market_list))
    num_markets = length(market_list)
    
    # Initialize the probability matrix
    prob_matrix = Matrix{Float64}(undef, num_samples, num_markets)

    # --- Expanded Logic to Populate All Prediction Lines ---
    for (market, idx) in market_map
        market_str = String(market)
        prob_vector = nothing # Default to nothing
        
        try
            if startswith(market_str, "ft_")
                time_preds = preds.ft
                # --- FT 1x2 ---
                if market in (:ft_1x2_home, :ft_1x2_draw, :ft_1x2_away)
                    field = Symbol(split(market_str, '_')[end])
                    prob_vector = getfield(time_preds, field)
                # --- FT Over/Under ---
                elseif startswith(market_str, "ft_ou_")
                    parts = split(market_str, '_') # ft, ou, 05, over
                    field = Symbol(parts[4], "_", parts[3]) # :over_05 or :under_05
                    base_prob = getfield(time_preds, Symbol("under_", parts[3])) # always get the 'under' prob
                    prob_vector = (parts[4] == "over") ? (1.0 .- base_prob) : base_prob
                # --- FT BTTS ---
                elseif startswith(market_str, "ft_btts_")
                    side = split(market_str, '_')[end] # yes or no
                    base_prob = time_preds.btts
                    prob_vector = (side == "yes") ? base_prob : (1.0 .- base_prob)
                # --- FT Correct Score ---
                elseif startswith(market_str, "ft_cs_")
                    score_key_str = market_str[7:end]
                    score_key = if occursin(r"\d_\d", score_key_str)
                        Tuple(parse(Int, i) for i in split(score_key_str, '_'))
                    elseif score_key_str == "other_home"
                        "other_home_win"
                    elseif score_key_str == "other_away"
                        "other_away_win"
                    else "other_draw" end
                    prob_vector = get(time_preds.correct_score, score_key, fill(NaN, num_samples))
                end
            elseif startswith(market_str, "ht_")
                time_preds = preds.ht
                # --- HT 1x2 ---
                if market in (:ht_1x2_home, :ht_1x2_draw, :ht_1x2_away)
                    field = Symbol(split(market_str, '_')[end])
                    prob_vector = getfield(time_preds, field)
                # --- HT Over/Under ---
                elseif startswith(market_str, "ht_ou_")
                    parts = split(market_str, '_')
                    field = Symbol(parts[4], "_", parts[3])
                    base_prob = getfield(time_preds, Symbol("under_", parts[3]))
                    prob_vector = (parts[4] == "over") ? (1.0 .- base_prob) : base_prob
                # --- HT Correct Score ---
                elseif startswith(market_str, "ht_cs_")
                    score_key_str = market_str[7:end]
                    score_key = if occursin(r"\d_\d", score_key_str)
                        Tuple(parse(Int, i) for i in split(score_key_str, '_'))
                    else "any_unquoted" end
                    prob_vector = get(time_preds.correct_score, score_key, fill(NaN, num_samples))
                end
            end

            # Assign the found vector or NaN if nothing was found
            prob_matrix[:, idx] = isnothing(prob_vector) ? fill(NaN, num_samples) : prob_vector

        catch e
            # If any parsing or field access fails, mark as NaN
            prob_matrix[:, idx] .= NaN
        end
    end
    
    return PredictionMatrix(market_list, market_map, prob_matrix)
end


"""
    generate_match_analysis(
        match_info::DataFrameRow,
        all_market_odds::DataFrame,
        models::Dict{String, Any},
        market_list::Vector{Symbol},
        generate_prediction_matrix::Function
    )

Generates a comprehensive analysis DataFrame and prediction matrices for a single match across multiple models.

# Arguments
- `match_info`: A row from the `todays_matches` DataFrame.
- `all_market_odds`: The "wide" DataFrame containing market odds for all matches.
- `models`: A dictionary mapping model names to loaded model objects.
- `market_list`: The master list of markets to analyze.
- `generate_prediction_matrix`: The helper function that converts model output to a PredictionMatrix.

# Returns
- A `Tuple{DataFrame, Dict{String, PredictionMatrix}}`:
    1. A `DataFrame` comparing market odds against predictions (mean/std odds, mean/std EV) for each model.
    2. A `Dict` storing the full `PredictionMatrix` for each model, useful for plotting.
"""
function generate_match_analysis(
    match_info::DataFrameRow,
    all_market_odds::DataFrame,
    models::Dict{String, Any},
    market_list::Vector{Symbol},
)
    event_to_find = match_info.event_name
    home_team = match_info.home_team
    away_team = match_info.away_team
    league_id = 1 # NOTE: Assumes a single league ID for now.

    println("🔬 Starting analysis for: $event_to_find")

    # 1. Reconstruct MarketBook for the specific match
    match_market_odds_row = first(filter(row -> row.event_name == event_to_find, all_market_odds))
    market_map = Dict(m => i for (i, m) in enumerate(market_list))
    back_odds_vec = [match_market_odds_row[Symbol(m, "_back")] for m in market_list]
    lay_odds_vec = [match_market_odds_row[Symbol(m, "_lay")] for m in market_list]
    single_match_book = MarketBook(market_list, market_map, back_odds_vec, lay_odds_vec)

    # Base DataFrame with market info, filtered for markets that exist
    base_df = DataFrame(
        market = market_list,
        market_back = single_match_book.back_odds,
        market_lay = single_match_book.lay_odds
    )
    filter!(:market_back => x -> !ismissing(x) && !isnan(x), base_df)

    all_predictions = Dict{String, PredictionMatrix}()
    final_df = base_df

    # 2. Loop through models, generate predictions, calculate stats, and join
    for (model_name, model) in models
        println(" -> Processing model: $model_name")

        # Generate predictions
        pred_matrix = generate_prediction_matrix(model, home_team, away_team, league_id, market_list)
        all_predictions[model_name] = pred_matrix

        # Calculate statistics
        ev_dist = calculate_ev_distributions(pred_matrix, single_match_book)
        odds_matrix = 1 ./ pred_matrix.probabilities

        model_mean_odds = vec(mean(odds_matrix, dims=1))
        model_std_odds = vec(std(odds_matrix, dims=1))
        mean_evs = vec(mean(ev_dist.ev, dims=1))
        std_evs = vec(std(ev_dist.ev, dims=1))

        # Create a temporary DataFrame for this model's results
        model_df = DataFrame(
            :market => market_list,
            Symbol(model_name, "_mean_odds") => round.(model_mean_odds, digits=2),
            Symbol(model_name, "_std_odds") => round.(model_std_odds, digits=2),
            Symbol(model_name, "_mean_ev_pct") => round.(mean_evs .* 100, digits=2),
            Symbol(model_name, "_std_ev_pct") => round.(std_evs .* 100, digits=2)
        )
        # Join with the final DataFrame
        final_df = leftjoin(final_df, model_df, on = :market)
    end

    println("✅ Analysis complete for: $event_to_find")
    return (final_df, all_predictions, single_match_book)
end

"""
    plot_multi_model_odds_distribution(
        all_predictions::Dict{String, PredictionMatrix},
        book::MarketBook,
        market::Symbol
    )

Plots and compares the odds distributions from multiple models for a single market,
overlaid with the live market back/lay odds.

# Arguments
- `all_predictions`: A dictionary mapping model names to their `PredictionMatrix` objects.
- `book`: The `MarketBook` object for the match.
- `market`: The symbol of the market to plot (e.g., `:ft_1x2_home`).

# Returns
- A `Plots.Plot` object.
"""
function plot_multi_model_odds_distribution(
    all_predictions::Dict{String, PredictionMatrix},
    book::MarketBook,
    market::Symbol,
)
    # 1. Setup the plot and market lines
    p = plot(
        title="Odds Distribution Comparison for :$market",
        xlabel="Odds",
        ylabel="Density",
        legend=:outertopright
    )

    market_idx = get(book.market_map, market, 0)
    if market_idx > 0
        market_back = book.back_odds[market_idx]
        market_lay = book.lay_odds[market_idx]

        if !ismissing(market_back) && !isnan(market_back)
            vline!(p, [market_back], lw=2, ls=:dash, c=:blue, label="Market Back ($market_back)")
        end
        if !ismissing(market_lay) && !isnan(market_lay)
            vline!(p, [market_lay], lw=2, ls=:dash, c=:red, label="Market Lay ($market_lay)")
        end
    else
        @warn "Market :$market not found in MarketBook."
    end

    # 2. Loop through models and plot their distributions
    for (model_name, pred_matrix) in all_predictions
        pred_idx = get(pred_matrix.market_map, market, 0)
        if pred_idx > 0
            model_odds_dist = 1 ./ pred_matrix.probabilities[:, pred_idx]

            # Calculate quantile for the back odds to add to the label
            quantile_str = ""
            if market_idx > 0 && !ismissing(book.back_odds[market_idx]) && !isnan(book.back_odds[market_idx])
                back_quantile = round(100 * mean(model_odds_dist .<= book.back_odds[market_idx]); digits=1)
                quantile_str = " (Back at $back_quantile%)"
            end

            density!(p, model_odds_dist, label=model_name * quantile_str)
        else
             @warn "Market :$market not found for model: $model_name"
        end
    end

    return p
end

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

end # end module
