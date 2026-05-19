# current_development/player_tracking_systems/src/experiment_engine.jl

using DataFrames
using Statistics
using GLM
using BayesianFootball.Data

"""
    evaluate_tracker_on_boundaries(config::AbstractRatingTracker, ds, boundaries)

Evaluates a tracker by fitting a GLM on the generated features across CV boundaries.
"""
function evaluate_tracker_on_boundaries(config::AbstractRatingTracker, ds::Data.DataStore, boundaries)
    # 1. Generate features (match_id -> (side, pos) -> rating)
    ratings_map = generate_tracker_features(config, ds)
    
    metrics_list = TrackerMetrics[]
    
    for (i, boundary_tuple) in enumerate(boundaries)
        # Handle both Tuple and SplitBoundary directly (robustness)
        boundary = boundary_tuple isa Tuple ? boundary_tuple[1] : boundary_tuple
        
        train_ids = boundary.history_match_ids
        test_ids = boundary.target_match_ids
        
        # Build training and testing DataFrames
        function build_df(ids)
            df = DataFrame()
            df.match_id = ids
            
            # Join with matches to get outcomes (goals)
            m_data = filter(row -> row.match_id in ids, ds.matches)
            df = innerjoin(df, select(m_data, :match_id, :home_goals, :away_goals), on = :match_id)
            
            # Target: Home Win (1/0) or goal diff
            function compute_outcome(hg, ag)
                if ismissing(hg) || ismissing(ag)
                    return NaN
                end
                return hg > ag ? 1.0 : 0.0
            end
            df.outcome = compute_outcome.(df.home_goals, df.away_goals)
            
            # Filter out any NaNs from outcome (though typically ds.matches won't have missing goals)
            df = filter(row -> !isnan(row.outcome), df)
            
            # Features: aggregate ratings
            positions = ["G", "D", "M", "F"]
            sides = ["home", "away"]
            
            for side in sides
                for pos in positions
                    col_name = Symbol("$(side)_$(pos)_rating")
                    df[!, col_name] = [get(get(ratings_map, m_id, Dict()), (side, pos), 0.0) for m_id in df.match_id]
                end
            end
            
            # Sum ratings for a simpler 'Total Quality' feature
            df.home_total = df.home_G_rating + df.home_D_rating + df.home_M_rating + df.home_F_rating
            df.away_total = df.away_G_rating + df.away_D_rating + df.away_M_rating + df.away_F_rating
            df.rating_diff = df.home_total - df.away_total
            
            return df
        end
        
        df_train = build_df(train_ids)
        df_test = build_df(test_ids)
        
        if isempty(df_train) || isempty(df_test)
            continue
        end

        # Fit GLM: Logistic regression on Home Win using rating_diff
        # We want to see if our rating_diff has predictive power
        model = glm(@formula(outcome ~ rating_diff), df_train, Bernoulli(), LogitLink())
        
        # Predictions
        preds = predict(model, df_test)
        
        # LogLoss calculation
        y_test = df_test.outcome
        eps = 1e-15
        logloss = -mean(y_test .* log.(preds .+ eps) + (1.0 .- y_test) .* log.(1.0 .- preds .+ eps))
        
        # Extract Edge Coef and P-Value
        coef_table = coeftable(model)
        edge_idx = 2 # Intercept is 1, rating_diff is 2
        edge_coef = coef_table.cols[1][edge_idx]
        edge_pvalue = coef_table.cols[4][edge_idx]
        
        push!(metrics_list, TrackerMetrics(logloss, edge_coef, edge_pvalue))
    end
    
    # Average metrics
    avg_logloss = mean([m.log_loss for m in metrics_list])
    avg_edge_coef = mean([m.glm_edge_coef for m in metrics_list])
    avg_edge_pvalue = mean([m.glm_edge_pvalue for m in metrics_list])
    
    return TrackerMetrics(avg_logloss, avg_edge_coef, avg_edge_pvalue)
end

"""
    run_experiment_grid(configs::Vector{<:AbstractRatingTracker}, ds, boundaries)

Parallel execution of experiment grid.
"""
function run_experiment_grid(configs::Vector{<:AbstractRatingTracker}, ds, boundaries)
    results = Vector{Tuple{AbstractRatingTracker, TrackerMetrics}}(undef, length(configs))
    
    Threads.@threads for i in 1:length(configs)
        config = configs[i]
        println("[INFO] Testing config $(i)/$(length(configs)): $(typeof(config))")
        metrics = evaluate_tracker_on_boundaries(config, ds, boundaries)
        results[i] = (config, metrics)
    end
    
    return results
end
