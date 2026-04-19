# current_development/layer_two/data_pipeline.jl

using Statistics
using DataFrames

"""
    build_l2_training_df(exp, ds)

Extracts OOS PPDs from Layer 1, joins them with live market odds, 
and evaluates the actual outcome to create the Layer 2 training set.
"""
function build_l2_training_df(exp, ds)
    println("Extracting L1 Out-of-Sample Predictions...")
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp)
    ppd_df = BayesianFootball.Predictions.model_inference(latents)
    
    # The PPD distribution is an array of samples. We need the mean probability.
    ppd_df.df.raw_prob = [mean(dist) for dist in ppd_df.df.distribution]
    
    # Join with Match metadata (to get dates, scores, and splits)
    println("Joining with Match Metadata & Odds...")
    matches_df = ds.matches[:, [:match_id, :match_date, :season, :match_month, :home_score, :away_score]]
    
    # Create a chronological split_id (e.g., Year-Month)
    matches_df.split_id = [string(r.season, "-", lpad(r.match_month, 2, "0")) for r in eachrow(matches_df)]
    
    df_joined = innerjoin(ppd_df.df, matches_df, on=:match_id)
    
    # Join with Odds
    odds_df = ds.odds[:, [:match_id, :market_name, :market_line, :selection, :odds_close]]
    df_final = innerjoin(df_joined, odds_df, on=[:match_id, :market_name, :market_line, :selection])
    
    # Compute the actual target (Did the bet win?)
    println("Resolving Match Outcomes...")
    df_final.outcome_hit = [resolve_outcome(r) for r in eachrow(df_final)]
    
    # Clean up and select final columns for L2
    select!(df_final, 
        :match_id, :match_date, :split_id, :season, :match_month,
        :market_name, :market_line, :selection, 
        :odds_close, :raw_prob, :outcome_hit
    )
    
    # Sort chronologically by split
    sort!(df_final, [:match_date])
    
    return df_final
end

"""
    resolve_outcome(row)
A helper to determine if a specific selection won (1.0) or lost (0.0).
TODO: Replace with BayesianFootball.Markets native resolver if available.
"""
function resolve_outcome(row)
    hg = row.home_score
    ag = row.away_score
    if ismissing(hg) || ismissing(ag); return missing; end
    
    if row.market_name == "1X2"
        if row.selection == :home && hg > ag; return 1.0; end
        if row.selection == :draw && hg == ag; return 1.0; end
        if row.selection == :away && hg < ag; return 1.0; end
        return 0.0
    elseif row.market_name == "OverUnder"
        total_goals = hg + ag
        is_over = occursin("over", string(row.selection))
        if is_over && total_goals > row.market_line; return 1.0; end
        if !is_over && total_goals < row.market_line; return 1.0; end
        return 0.0
    elseif row.market_name == "BTTS"
        btts = (hg > 0 && ag > 0)
        if row.selection == :btts_yes && btts; return 1.0; end
        if row.selection == :btts_no && !btts; return 1.0; end
        return 0.0
    end
    return missing
end
