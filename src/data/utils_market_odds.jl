MARKET_GROUPS = Dict(
        "ft_1x2" => ["home", "draw", "away"],
        "ht_1x2" => ["ht_home", "ht_draw", "ht_away"],
        "ft_ou_05" => ["over_0_5", "under_0_5"],
        "ft_ou_15" => ["over_1_5", "under_1_5"],
        "ft_ou_25" => ["over_2_5", "under_2_5"],
        "ft_ou_35" => ["over_3_5", "under_3_5"],
        "ht_ou_05" => ["ht_over_0_5", "ht_under_0_5"],
        "ht_ou_15" => ["ht_over_1_5", "ht_under_1_5"],
        "ht_ou_25" => ["ht_over_2_5", "ht_under_2_5"],
        "btts" => ["btts_yes", "btts_no"],
        "ft_cs" => ["0_0", "0_1", "0_2", "0_3", "1_1", "1_2", "1_3","2_1", "2_2", "2_3","3_1", "3_2", "3_3","1_0", "2_0", "3_0", "any_other_home", "any_other_away", "any_other_draw"], 
        "ht_cs" => ["ht_0_0", "ht_0_1", "ht_0_2", "ht_1_1", "ht_1_2","ht_2_1", "ht_2_2","ht_1_0", "ht_2_0", "ht_any_unquoted"], 
       )

"""
    get_game_line_odds(data_store::DataStore, match_id::Int)
    
Extracts the kickoff odds (at minutes=0) for a given match from the odds DataFrame.
Returns Odds.MatchLineOdds or nothing if not found.
"""
function get_game_line_odds_simple(data_store::DataStore, match_id::Int)
    # Filter for kickoff odds (minutes = 0)
    odds_rows = filter(row -> row.sofa_match_id == match_id && row.minutes == 0, data_store.odds)
    
    if isempty(odds_rows)
        @warn "No kickoff odds found for match_id: $match_id"
        return nothing
    end
    
    odds_row = odds_rows[1, :]
    
    # Helper to safely extract values
    function get_val(col_name::String)
        if hasproperty(odds_row, Symbol(col_name))
            val = odds_row[Symbol(col_name)]
            return ismissing(val) ? NaN : Float64(val)
        else
            return NaN
        end
    end
    
    # Extract HT correct scores
    ht_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:2, a in 0:2
        ht_correct_scores[(h, a)] = get_val("ht_$(h)_$(a)")
    end
    ht_correct_scores["any_unquoted"] = get_val("ht_any_unquoted")
    
    # Extract FT correct scores
    ft_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:3, a in 0:3
        ft_correct_scores[(h, a)] = get_val("$(h)_$(a)")
    end
    ft_correct_scores["other_home_win"] = get_val("any_other_home")
    ft_correct_scores["other_away_win"] = get_val("any_other_away")
    ft_correct_scores["other_draw"] = get_val("any_other_draw")
    
    # Build HT odds
    ht_odds = Odds.MatchHTOdds(
        get_val("ht_home"),
        get_val("ht_draw"),
        get_val("ht_away"),
        ht_correct_scores,
        get_val("ht_under_0_5"),  # Note: under_05 = 1/over_05 if not directly available
        get_val("ht_over_0_5"),
        get_val("ht_under_1_5"),
        get_val("ht_over_1_5"),
        get_val("ht_under_2_5"),
        get_val("ht_over_2_5"),
    )
    
    # Build FT odds
    ft_odds = Odds.MatchFTOdds(
        get_val("home"),
        get_val("draw"),
        get_val("away"),
        ft_correct_scores,
        get_val("under_0_5"),
        get_val("over_0_5"),
        get_val("under_1_5"),
        get_val("over_1_5"),
        get_val("under_2_5"),
        get_val("over_2_5"),
        get_val("under_3_5"),
        get_val("over_3_5"),
        get_val("btts_yes"),
        get_val("btts_no")
    )
    
    return Odds.MatchLineOdds(ht_odds, ft_odds)
end

"""
    process_matches_odds(data_store::DataStore, target_matches::DataFrame)
    
Process odds for multiple matches.
Returns Dict{Int64, Odds.MatchLineOdds}
"""
function process_matches_odds_simple(data_store::DataStore, target_matches::DataFrame)
    results = Dict{Int64, Odds.MatchLineOdds}()
    
    for match_id in target_matches.match_id
        odds = get_game_line_odds(data_store, match_id)
        if !isnothing(odds)
            results[match_id] = odds
        end
    end
    
    return results
end

"""
Default configuration for typical use case
"""
function default_marketodds_config()
    market_groups = Dict(
        "ft_1x2" => ["home", "draw", "away"],
        "ht_1x2" => ["ht_home", "ht_draw", "ht_away"],
        "ft_ou_0.5" => ["over_0_5", "under_0_5"],
        "ft_ou_1.5" => ["over_1_5", "under_1_5"],
        "ft_ou_2.5" => ["over_2_5", "under_2_5"],
        "ft_ou_3.5" => ["over_3_5", "under_3_5"],
        "ht_ou_0.5" => ["ht_over_0_5", "ht_under_0_5"],
        "ht_ou_1.5" => ["ht_over_1_5", "ht_under_1_5"],
        "ht_ou_2.5" => ["ht_over_2_5", "ht_under_2_5"],
        "btts" => ["btts_yes", "btts_no"],
        "ft_cs" => ["0_0", "0_1", "0_2", "0_3", "1_0", "1_1", "1_2", "1_3",
                    "2_0", "2_1", "2_2", "2_3", "3_0", "3_1", "3_2", "3_3",
                    "any_other_home", "any_other_away", "any_other_draw"],
        "ht_cs" => ["ht_0_0", "ht_0_1", "ht_0_2", "ht_1_0", "ht_1_1", "ht_1_2",
                    "ht_2_0", "ht_2_1", "ht_2_2", "ht_any_unquoted"]
    )
    
    return OddsProcessing.OddsConfig(
        10, 4, 3,  # Window: 5 mins before/after, min 3 mins
        :weighted_mean, 1.4,  # Aggregation with 100% jump filter
        true, :probability_sum,  # Complete missing via probability
        true, 1.001, :proportional,  # Normalize to 1% overround
        0.8, 10,  # Min 80% liquidity, max 10 min staleness
        market_groups
    )
end

# ============================================
# Pipeline Components (Composable Functions)
# ============================================

"""
Filter odds data to relevant time window, accounting for goals
"""
function apply_time_window(odds_df::DataFrame, incidents_df::DataFrame, 
                          match_id::Int, target_minute::Int, config::OddsProcessing.OddsConfig)
    # Get match odds
    match_odds = filter(row -> row.sofa_match_id == match_id, odds_df)
    
    if isempty(match_odds)
        return DataFrame()
    end
    
    # Check for goals in the window
    match_incidents = filter(row -> row.match_id == match_id, incidents_df)
    goals = filter(row -> row.incident_type == "goal", match_incidents)
    
    # Adjust window if goals occurred
    window_start = target_minute - config.window_minutes_before
    window_end = target_minute + config.window_minutes_after
    
    for goal in eachrow(goals)
        goal_time = goal.time
        if window_start <= goal_time <= window_end
            # Shrink window to exclude goal
            if goal_time < target_minute
                window_start = goal_time + 1
            else
                window_end = goal_time - 1
            end
        end
    end
    
    # Check minimum window size
    if window_end - window_start < config.min_window_size
        @warn "Window too small after goal adjustment for match $match_id at minute $target_minute"
        return DataFrame()
    end
    
    # Filter to window
    return filter(row -> window_start <= row.minutes <= window_end, match_odds)
end

"""
Remove outliers based on price jumps between consecutive observations
"""
function filter_outliers(odds_df::DataFrame, columns::Vector{String}, config::OddsProcessing.OddsConfig)
    if nrow(odds_df) <= 1
        return odds_df
    end
    
    odds_sorted = sort(odds_df, :minutes)
    valid_rows = trues(nrow(odds_sorted))
    
    for col in columns
        if !hasproperty(odds_sorted, Symbol(col))
            continue
        end
        
        values = odds_sorted[!, Symbol(col)]
        for i in 2:length(values)
            if !ismissing(values[i]) && !ismissing(values[i-1])
                # Convert odds to probabilities for comparison
                prob_curr = 1.0 / values[i]
                prob_prev = 1.0 / values[i-1]
                
                # Check relative change
                if abs(prob_curr - prob_prev) / prob_prev > config.outlier_threshold
                    valid_rows[i] = false
                end
            end
        end
    end
    
    return odds_sorted[valid_rows, :]
end

"""
Aggregate odds using specified method
"""
function aggregate_odds(odds_df::DataFrame, columns::Vector{String}, config::OddsProcessing.OddsConfig)
    result = Dict{String, Float64}()
    
    for col in columns
        if !hasproperty(odds_df, Symbol(col))
            result[col] = NaN
            continue
        end
        
        values = skipmissing(odds_df[!, Symbol(col)])
        
        if isempty(values)
            result[col] = NaN
            continue
        end
        
        if config.aggregation == :mean
            result[col] = mean(values)
        elseif config.aggregation == :median
            result[col] = median(values)
        elseif config.aggregation == :weighted_mean
            # Weight by recency
            minutes = odds_df.minutes
            weights = exp.(-0.1 * (maximum(minutes) .- minutes))
            valid_idx = .!ismissing.(odds_df[!, Symbol(col)])
            result[col] = sum(values .* weights[valid_idx]) / sum(weights[valid_idx])
        elseif config.aggregation == :last_valid
            result[col] = last(values)
        else
            result[col] = mean(values)  # Default
        end
    end
    
    return result
end

"""
Complete missing odds in a market group
"""
function complete_market_group(odds_dict::Dict{String, Float64}, 
                              group_columns::Vector{String}, 
                              config::OddsProcessing.OddsConfig)
    # Count available odds
    available = [col for col in group_columns if !isnan(get(odds_dict, col, NaN))]
    missing = [col for col in group_columns if isnan(get(odds_dict, col, NaN))]
    
    if isempty(missing)
        return odds_dict
    end
    
    if config.completion_method == :probability_sum && length(missing) == 1
        # Complete single missing odd using probability sum
        available_probs = [1.0 / odds_dict[col] for col in available]
        prob_sum = sum(available_probs)
        
        if prob_sum < config.target_overround - 0.2  # Leave room for missing
            missing_prob = config.target_overround - prob_sum
            odds_dict[missing[1]] = 1.0 / missing_prob
        end
    elseif config.completion_method == :similar_markets
        # Use ratios from similar complete markets
        # This would require historical data - placeholder for now
        @warn "Similar markets completion not yet implemented"
    end
    
    return odds_dict
end

"""
Normalize market group probabilities to target overround
"""
function normalize_market_group(odds_dict::Dict{String, Float64}, 
                               group_columns::Vector{String}, 
                               config::OddsProcessing.OddsConfig)
    # Get available odds
    available = [col for col in group_columns if !isnan(get(odds_dict, col, NaN))]
    
    if length(available) < 2
        return odds_dict  # Can't normalize with less than 2 odds
    end
    
    # Calculate current probability sum
    probs = [1.0 / odds_dict[col] for col in available]
    current_sum = sum(probs)
    
    if current_sum < config.min_liquidity_threshold
        # @warn "Probability sum too low: $current_sum"
        return odds_dict
    end


    if current_sum < config.target_overround
    # Apply normalization
      if config.normalization_method == :proportional
          # Simple proportional scaling
          scale_factor = config.target_overround / current_sum
          for col in available
              odds_dict[col] = odds_dict[col] / scale_factor
          end
      elseif config.normalization_method == :power
          # Power scaling (preserves favorites/longshots relationship better)
          power = log(config.target_overround) / log(current_sum)
          for col in available
              prob = 1.0 / odds_dict[col]
              new_prob = prob^power
              odds_dict[col] = 1.0 / new_prob
          end
      elseif config.normalization_method == :margin_weights
          # Weighted by distance from fair (more margin on longshots)
          fair_probs = probs ./ current_sum
          margins = (config.target_overround - 1.0) .* (1 .- fair_probs)
          new_probs = fair_probs .+ margins
          
          for (i, col) in enumerate(available)
              odds_dict[col] = 1.0 / new_probs[i]
          end
      end
    end

    
    return odds_dict
end

"""
Process correct scores as a special case
"""
function process_correct_scores(odds_dict::Dict{String, Float64}, 
                               prefix::String,  # "ft" or "ht"
                               config::OddsProcessing.OddsConfig)
    if prefix == "ht"
        scores = [(i,j) for i in 0:2 for j in 0:2]
        other_key = "ht_any_unquoted"
    else
        scores = [(i,j) for i in 0:3 for j in 0:3]
        other_keys = ["any_other_home", "any_other_away", "any_other_draw"]
    end
    
    # Collect available scores
    total_prob = 0.0
    for (h, a) in scores
        key = prefix == "ht" ? "ht_$(h)_$(a)" : "$(h)_$(a)"
        if !isnan(get(odds_dict, key, NaN))
            total_prob += 1.0 / odds_dict[key]
        end
    end
    
    # Add "other" probabilities
    if prefix == "ht"
        if !isnan(get(odds_dict, other_key, NaN))
            total_prob += 1.0 / odds_dict[other_key]
        end
    else
        for key in other_keys
            if !isnan(get(odds_dict, key, NaN))
                total_prob += 1.0 / odds_dict[key]
            end
        end
    end
    
    # Normalize if needed
    if config.normalize_groups && total_prob > config.min_liquidity_threshold
        scale = config.target_overround / total_prob
        
        for (h, a) in scores
            key = prefix == "ht" ? "ht_$(h)_$(a)" : "$(h)_$(a)"
            if !isnan(get(odds_dict, key, NaN))
                odds_dict[key] = odds_dict[key] / scale
            end
        end
        
        if prefix == "ht"
            if !isnan(get(odds_dict, other_key, NaN))
                odds_dict[other_key] = odds_dict[other_key] / scale
            end
        else
            for key in other_keys
                if !isnan(get(odds_dict, key, NaN))
                    odds_dict[key] = odds_dict[key] / scale
                end
            end
        end
    end
    
    return odds_dict
end

# ============================================
# Main Pipeline Function
# ============================================

"""
Main pipeline to process odds for a match at a specific minute
"""
function process_odds_pipeline(data_store::DataStore, 
                              match_id::Int, 
                              target_minute::Int,
                              config::OddsProcessing.OddsConfig = default_marketodds_config())
    
    # Step 1: Get time window
    windowed_odds = apply_time_window(
        data_store.odds, 
        data_store.incidents,
        match_id, 
        target_minute, 
        config
    )
    
    if isempty(windowed_odds)
        @warn "No odds data in window for match $match_id at minute $target_minute"
        return nothing
    end
    
    # Step 2: Filter outliers for each market group
    all_columns = Set{String}()
    for (_, cols) in config.market_groups
        union!(all_columns, cols)
    end
    
    filtered_odds = filter_outliers(
        windowed_odds,
        collect(all_columns),
        config
    )
    
    # Step 3: Aggregate odds
    odds_dict = aggregate_odds(
        filtered_odds,
        collect(all_columns),
        config
    )
    
    # Step 4: Process each market group
    for (group_name, group_cols) in config.market_groups
        # Complete missing odds
        if config.complete_missing
            odds_dict = complete_market_group(odds_dict, group_cols, config)
        end
        
        # Normalize group
        if config.normalize_groups
            odds_dict = normalize_market_group(odds_dict, group_cols, config)
        end
    end
    
    # Step 5: Process correct scores
    odds_dict = process_correct_scores(odds_dict, "ht", config)
    odds_dict = process_correct_scores(odds_dict, "ft", config)
    
    return odds_dict
end

"""
Convert processed odds dictionary to Odds.MatchLineOdds structure
"""
function dict_to_predictions(odds_dict::Dict{String, Float64})
    # Helper to safely get values
    get_val(key) = get(odds_dict, key, NaN)
    
    # Build HT correct scores
    ht_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:2, a in 0:2
        ht_correct_scores[(h, a)] = get_val("ht_$(h)_$(a)")
    end
    ht_correct_scores["any_unquoted"] = get_val("ht_any_unquoted")
    
    # Build FT correct scores
    ft_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:3, a in 0:3
        ft_correct_scores[(h, a)] = get_val("$(h)_$(a)")
    end
    ft_correct_scores["other_home_win"] = get_val("any_other_home")
    ft_correct_scores["other_away_win"] = get_val("any_other_away")
    ft_correct_scores["other_draw"] = get_val("any_other_draw")
    
    # Build predictions
    ht_pred = Odds.MatchHTOdds(
        get_val("ht_home"),
        get_val("ht_draw"),
        get_val("ht_away"),
        ht_correct_scores,
        get_val("ht_under_0_5"),
        get_val("ht_over_0_5"),
        get_val("ht_under_1_5"),
        get_val("ht_over_1_5"),
        get_val("ht_under_2_5"),
        get_val("ht_over_2_5"),
    )
    
    ft_pred = Odds.MatchFTOdds(
        get_val("home"),
        get_val("draw"),
        get_val("away"),
        ft_correct_scores,
        get_val("under_0_5"),
        get_val("over_0_5"),
        get_val("under_1_5"),
        get_val("over_1_5"),
        get_val("under_2_5"),
        get_val("over_2_5"),
        get_val("under_3_5"),
        get_val("over_3_5"),
        get_val("btts_yes"),
        get_val("btts_no")
    )
    
    return Odds.MatchLineOdds(ht_pred, ft_pred)
end

"""
Get processed game line odds (at kickoff)
"""
function get_processed_game_line_odds(data_store::DataStore, 
                                     match_id::Int,
                                     config::OddsProcessing.OddsConfig = default_marketodds_config())
    odds_dict = process_odds_pipeline(data_store, match_id, 0, config)
    
    if isnothing(odds_dict)
        return nothing
    end
    
    return dict_to_predictions(odds_dict)
end

"""
Process multiple matches in parallel
"""
function process_matches_odds(data_store::DataStore, 
                             target_matches::DataFrame,
                             target_minute::Int = 0,
                             config::OddsProcessing.OddsConfig = default_marketodds_config())
    match_ids = target_matches.match_id
    n_matches = length(match_ids)
    n_threads = Threads.nthreads()
    
    # Each thread gets its own dictionary
    thread_results = Vector{Dict{Int64, Odds.MatchLineOdds}}(undef, n_threads)
    
    Threads.@threads for tid in 1:n_threads
        local_dict = Dict{Int64, Odds.MatchLineOdds}()
        
        for idx in tid:n_threads:n_matches
            match_id = match_ids[idx]
            try
                odds_dict = process_odds_pipeline(data_store, match_id, target_minute, config)
                if !isnothing(odds_dict)
                    local_dict[match_id] = dict_to_predictions(odds_dict)
                end
            catch e
                @warn "Failed to process odds for match $match_id" exception=e
            end
        end
        
        thread_results[tid] = local_dict
    end
    
    return merge(thread_results...)
end

