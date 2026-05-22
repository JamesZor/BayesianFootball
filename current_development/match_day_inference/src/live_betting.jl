# current_development/match_day_inference/src/live_betting.jl

using Redis
using JSON3
using DataFrames
using Statistics
using Dates
using Printf
using BayesianFootball
import BayesianFootball.Signals: KellyCriterion, BayesianKelly, compute_stake

"""
    parse_redis_runners(runners, home_team::String, away_team::String; selections=Dict{String,String}())

Extracts Home, Draw, Away back and lay odds from the Betfair runners dict.
Resolves runner identity via:
  - Option 1: `runner_name` field injected directly into the runner payload by the Python streamer.
  - Option 2: `selections` kwarg mapping `selection_id -> runner_name` from `live_market_meta`.
  - Fallback: raw runner key (numeric selection ID).
"""
function parse_redis_runners(runners, home_team::String, away_team::String; selections::Dict{String,String} = Dict{String,String}())
    odds_dict = Dict{Symbol, NamedTuple}()
    
    # Standard roles we map to
    roles = [:home, :draw, :away]
    for r in roles
        odds_dict[r] = (back = NaN, lay = NaN, back_size = 0.0, lay_size = 0.0)
    end
    
    if isempty(runners)
        return odds_dict
    end
    
    # Normalization helper
    normalize_name(name) = lowercase(replace(string(name), "-" => "", "_" => "", " " => ""))
    
    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    
    for (k, runner) in runners
        # Identify runner role
        runner_key = string(k)
        # Resolve runner name: Option 1 (injected by streamer), Option 2 (metadata selections), fallback to key
        runner_name = if haskey(runner, :runner_name) && runner.runner_name !== nothing
            string(runner.runner_name)
        elseif haskey(selections, runner_key)
            selections[runner_key]
        else
            runner_key
        end
        
        norm_key = normalize_name(runner_key)
        norm_name = normalize_name(runner_name)
        
        role = :unknown
        if norm_key == "home" || norm_name == "home" || occursin(home_norm, norm_name) || occursin(norm_name, home_norm)
            role = :home
        elseif norm_key == "away" || norm_name == "away" || occursin(away_norm, norm_name) || occursin(norm_name, away_norm)
            role = :away
        elseif norm_key == "draw" || norm_name == "draw" || occursin("draw", norm_name)
            role = :draw
        end
        
        if role != :unknown
            # Extract top back & lay prices
            back_p = NaN
            back_s = 0.0
            if haskey(runner, :best_back) && !isempty(runner.best_back)
                back_p = Float64(runner.best_back[1].price)
                back_s = haskey(runner.best_back[1], :size) ? Float64(runner.best_back[1].size) : 0.0
            end
            
            lay_p = NaN
            lay_s = 0.0
            if haskey(runner, :best_lay) && !isempty(runner.best_lay)
                lay_p = Float64(runner.best_lay[1].price)
                lay_s = haskey(runner.best_lay[1], :size) ? Float64(runner.best_lay[1].size) : 0.0
            end
            
            odds_dict[role] = (back = back_p, lay = lay_p, back_size = back_s, lay_size = lay_s)
        end
    end
    
    return odds_dict
end

"""
    get_live_market_mappings(redis_conn)

Scans the `live_market_meta` hash in Redis and builds a lookup dictionary:
`Dict{Tuple{String, String, String}, String}` mapping `(home_slug, away_slug, market_type) -> market_id`.
"""
function get_live_market_mappings(redis_conn)
    mapping_lookup = Dict{Tuple{String, String, String}, String}()
    selections_lookup = Dict{String, Dict{String, String}}()  # market_id -> {selection_id -> runner_name}
    try
        # Retrieve all metadata entries from Redis hash
        meta_dict = Redis.hgetall(redis_conn, "live_market_meta")

        # Parse the JSON values and populate the lookup table
        for (market_id, raw_json) in meta_dict
            try
                meta = JSON3.read(raw_json)
                home_slug = haskey(meta, :home_slug) && meta.home_slug !== nothing ? string(meta.home_slug) : ""
                away_slug = haskey(meta, :away_slug) && meta.away_slug !== nothing ? string(meta.away_slug) : ""
                market_type = haskey(meta, :market_type) ? string(meta.market_type) : ""

                if !isempty(home_slug) && !isempty(away_slug) && !isempty(market_type)
                    mapping_lookup[(home_slug, away_slug, market_type)] = string(market_id)
                end

                # Option 2: Extract selections mapping {selection_id -> runner_name}
                if haskey(meta, :selections) && meta.selections !== nothing
                    sel_dict = Dict{String, String}()
                    for (sid, sname) in pairs(meta.selections)
                        sel_dict[string(sid)] = string(sname)
                    end
                    selections_lookup[string(market_id)] = sel_dict
                end
            catch e
                @warn "Failed to parse metadata JSON for market $market_id: $e"
            end
        end
    catch e
        @error "Failed to retrieve live_market_meta from Redis: $e"
    end
    return (markets=mapping_lookup, selections=selections_lookup)
end

"""
    poll_redis_live_odds(redis_conn, market_id_lookup, home_slug::String, away_slug::String, market_type::String = "1X2")

Resolves the Betfair market ID for a match and fetches the runner prices.
`market_id_lookup` is a NamedTuple with `.markets` and `.selections` from `get_live_market_mappings`.
"""
function poll_redis_live_odds(redis_conn, market_id_lookup, home_slug::String, away_slug::String, market_type::String = "1X2")
    # PPD/todays_matches uses "1X2", but Betfair streaming uses "MATCH_ODDS"
    bf_market_type = market_type == "1X2" ? "MATCH_ODDS" : market_type

    # Check if a market ID matches the team slugs
    market_id = get(market_id_lookup.markets, (home_slug, away_slug, bf_market_type), nothing)
    if market_id === nothing
        # Try reversed order just in case
        market_id = get(market_id_lookup.markets, (away_slug, home_slug, bf_market_type), nothing)
    end

    if market_id === nothing
        # Return empty/default runner odds
        return parse_redis_runners(Dict(), home_slug, away_slug)
    end

    # Get Option 2 selections fallback for this market
    sel_dict = get(market_id_lookup.selections, market_id, Dict{String, String}())

    raw_data = Redis.hget(redis_conn, "live_markets", market_id)
    if raw_data === nothing
        return parse_redis_runners(Dict(), home_slug, away_slug)
    end

    try
        data = JSON3.read(raw_data)
        if haskey(data, :runners)
            return parse_redis_runners(data.runners, home_slug, away_slug; selections=sel_dict)
        end
    catch e
        @warn "Failed to parse Redis JSON for market $market_id: $e"
    end

    return parse_redis_runners(Dict(), home_slug, away_slug)
end

"""
    poll_redis_live_odds(redis_conn, match_id::Int, home_team::String, away_team::String)

Loads the live runners prices from Redis for a single fixture using automatic metadata lookup.
"""
function poll_redis_live_odds(redis_conn, match_id::Int, home_team::String, away_team::String)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return poll_redis_live_odds(redis_conn, market_id_lookup, home_team, away_team)
end

"""
    calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame, market_id_lookup; kelly_fraction=0.5, min_edge=0.0)

Generates EV and Kelly stakes for today's matches using market metadata lookup.
`market_id_lookup` is a NamedTuple from `get_live_market_mappings`.
"""
function calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame, market_id_lookup; kelly_fraction=0.5, min_edge=0.0)
    # Instantiate signals from core package
    kelly_std = KellyCriterion(kelly_fraction)
    kelly_bayes = BayesianKelly(min_edge)
    
    df_results = DataFrame(
        match_id = Int[],
        home_team = String[],
        away_team = String[],
        selection = Symbol[],
        prob_model = Float64[],
        odds_back = Float64[],
        odds_lay = Float64[],
        ev = Float64[],
        stake_std_kelly = Float64[],
        stake_bayes_kelly = Float64[]
    )
    
    # standard 1X2 market rows
    df_1X2 = subset(ppd.df, :market_name => ByRow(==("1X2")))
    
    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)
        
        # Get live odds from Redis
        live_odds = poll_redis_live_odds(redis_conn, market_id_lookup, home, away)
        
        # Get model distributions for this match
        match_preds = subset(df_1X2, :match_id => ByRow(==(mid)))
        if isempty(match_preds)
            continue
        end
        
        for sel in [:home, :away, :draw]
            # Find the row in PPD for this selection
            sel_row = subset(match_preds, :selection => ByRow(s -> Symbol(s) == sel))
            if isempty(sel_row)
                continue
            end
            
            dist = first(sel_row).distribution
            p_model = mean(dist)
            
            # Live odds
            odds_info = live_odds[sel]
            back_odds = odds_info.back
            lay_odds = odds_info.lay
            
            ev = NaN
            stake_std = 0.0
            stake_bayes = 0.0
            
            if !isnan(back_odds) && back_odds > 1.0
                ev = (p_model * back_odds) - 1.0
                stake_std = compute_stake(kelly_std, dist, back_odds)
                stake_bayes = compute_stake(kelly_bayes, dist, back_odds)
            end
            
            push!(df_results, (
                match_id = mid,
                home_team = home,
                away_team = away,
                selection = sel,
                prob_model = p_model,
                odds_back = back_odds,
                odds_lay = lay_odds,
                ev = ev,
                stake_std_kelly = stake_std,
                stake_bayes_kelly = stake_bayes
            ))
        end
    end
    
    return df_results
end

"""
    calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)

Generates EV and Kelly stakes for today's matches. Automatically loads metadata mapping.
"""
function calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return calculate_betting_signals(ppd, redis_conn, todays_matches, market_id_lookup; kelly_fraction=kelly_fraction, min_edge=min_edge)
end

"""
    print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame, market_id_lookup; kelly_fraction=0.5, min_edge=0.0)

Interactive console dashboard showing live prices, model probabilities, and Kelly stakes.
`market_id_lookup` is a NamedTuple from `get_live_market_mappings`.
"""
function print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame, market_id_lookup; kelly_fraction=0.5, min_edge=0.0)
    kelly_std = KellyCriterion(kelly_fraction)
    kelly_bayes = BayesianKelly(min_edge)
    
    println("\n" * "="^85)
    println(" 📊 LIVE MATCHDAY BETTING DASHBOARD | Kelly Fraction: $kelly_fraction | Min Edge: $min_edge")
    println("="^85)
    
    # standard 1X2 market rows
    df_1X2 = subset(ppd.df, :market_name => ByRow(==("1X2")))
    
    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)
        
        # Get live odds from Redis
        live_odds = poll_redis_live_odds(redis_conn, market_id_lookup, home, away)
        
        # Get model distributions for this match
        match_preds = subset(df_1X2, :match_id => ByRow(==(mid)))
        if isempty(match_preds)
            continue
        end
        
        println("\n⚽ $home vs $away (ID: $mid)")
        println("   " * "-"^79)
        println("   Selection   |  Model Prob  |  Back Odds  |  Lay Odds  |    EV    |  Std Kelly  |  Bayes Kelly")
        println("   " * "-"^79)
        
        for sel in [:home, :away, :draw]
            # Find the row in PPD for this selection
            sel_row = subset(match_preds, :selection => ByRow(s -> Symbol(s) == sel))
            if isempty(sel_row)
                continue
            end
            
            dist = first(sel_row).distribution
            p_model = mean(dist)
            
            # Live odds
            odds_info = live_odds[sel]
            back_odds = odds_info.back
            lay_odds = odds_info.lay
            
            ev = NaN
            stake_std = 0.0
            stake_bayes = 0.0
            
            if !isnan(back_odds) && back_odds > 1.0
                ev = (p_model * back_odds) - 1.0
                stake_std = compute_stake(kelly_std, dist, back_odds)
                stake_bayes = compute_stake(kelly_bayes, dist, back_odds)
            end
            
            # Format selection label
            sel_label = rpad(uppercase(string(sel)), 12)
            
            # Format numbers
            prob_str  = @sprintf("%10.1f%%", p_model * 100)
            back_str  = isnan(back_odds) ? "    ----  " : @sprintf("%10.2f", back_odds)
            lay_str   = isnan(lay_odds)  ? "    ----  " : @sprintf("%10.2f", lay_odds)
            ev_str    = isnan(ev)        ? "   ---- " : (ev > 0 ? @sprintf("  +%5.1f%%", ev * 100) : @sprintf("   %5.1f%%", ev * 100))
            
            stake_std_str = stake_std > 0   ? @sprintf("%10.1f%%", stake_std * 100) : "      0.0%"
            stake_bayes_str = stake_bayes > 0 ? @sprintf("%10.1f%%", stake_bayes * 100) : "      0.0%"
            
            # Highlight value bets
            prefix = ev > 0.0 ? "🔥" : "  "
            
            println("   $prefix $sel_label |  $prob_str  | $back_str  | $lay_str  |  $ev_str |  $stake_std_str  |  $stake_bayes_str")
        end
    end
    println("="^85 * "\n")
end

"""
    print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)

Interactive console dashboard showing live prices, model probabilities, and Kelly stakes. Automatically loads metadata mapping.
"""
function print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)
    market_id_lookup = get_live_market_mappings(redis_conn)
    return print_live_betting_dashboard(ppd, redis_conn, todays_matches, market_id_lookup; kelly_fraction=kelly_fraction, min_edge=min_edge)
end
