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
    parse_redis_runners(runners, home_team::String, away_team::String)

Extracts Home, Draw, Away back and lay odds from the Betfair runners dict.
Supports team-name-based keys, standard "home"/"draw"/"away" keys, and Betfair selection IDs.
"""
function parse_redis_runners(runners, home_team::String, away_team::String)
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
        runner_name = haskey(runner, :runner_name) ? string(runner.runner_name) : runner_key
        
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
    poll_redis_live_odds(redis_conn, match_id::Int, home_team::String, away_team::String)

Loads the live runners prices from Redis for a single fixture.
"""
function poll_redis_live_odds(redis_conn, match_id::Int, home_team::String, away_team::String)
    raw_data = Redis.hget(redis_conn, "live_markets", string(match_id))
    if raw_data === nothing
        # Try metadata lookup or key search if direct key not found
        return parse_redis_runners(Dict(), home_team, away_team)
    end
    
    try
        data = JSON3.read(raw_data)
        if haskey(data, :runners)
            return parse_redis_runners(data.runners, home_team, away_team)
        end
    catch e
        @warn "Failed to parse Redis JSON for match $match_id: $e"
    end
    
    return parse_redis_runners(Dict(), home_team, away_team)
end

"""
    calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)

Generates EV and Kelly stakes for today's matches.
"""
function calculate_betting_signals(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)
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
        live_odds = poll_redis_live_odds(redis_conn, mid, home, away)
        
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
    print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)

Interactive console dashboard showing live prices, model probabilities, and Kelly stakes.
"""
function print_live_betting_dashboard(ppd::Predictions.PPD, redis_conn, todays_matches::AbstractDataFrame; kelly_fraction=0.5, min_edge=0.0)
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
        live_odds = poll_redis_live_odds(redis_conn, mid, home, away)
        
        # Get model distributions for this match
        match_preds = subset(df_1X2, :match_id => ByRow(==(mid)))
        if isempty(match_preds)
            continue
        end
        
        println("\n⚽ $home vs $away (ID: $mid)")
        println("   " * "-"*79)
        println("   Selection   |  Model Prob  |  Back Odds  |  Lay Odds  |    EV    |  Std Kelly  |  Bayes Kelly")
        println("   " * "-"*79)
        
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
