# current_development/layer_two/runner.jl

"""
    run_calibration_backtest(l1_oos_data::DataFrame, config::CalibrationConfig)
"""
function run_calibration_backtest(l1_oos_data, config)
    # Get chronological unique splits
    split_ids = sort(unique(l1_oos_data.split_id))
    n_splits = length(split_ids)
    
    println("Found $n_splits temporal splits. Starting expanding window backtest...")
    
    all_calibrated_preds = DataFrame()
    final_fitted_models = Dict{Symbol, Any}()
    
    # We start predicting AFTER we accumulate minimum history
    for k in (config.min_history_splits + 1):n_splits
        
        # 1. Train Window
        start_idx = config.max_history_splits == 0 ? 1 : max(1, k - config.max_history_splits)
        train_splits = split_ids[start_idx : (k - 1)]
        df_train = filter(row -> row.split_id in train_splits, l1_oos_data)
        
        # 2. Predict Window
        current_split = split_ids[k]
        df_predict = filter(row -> row.split_id == current_split, l1_oos_data)
        
        println("Processing Split [$k/$n_splits]: $current_split | Train size: $(nrow(df_train)) | Predict size: $(nrow(df_predict))")
        
        calibrated_split = copy(df_predict)
        calibrated_split.calib_prob = Vector{Union{Missing, Float64}}(missing, nrow(calibrated_split))
        
        # 3. Fit & Apply per target market
        for target in config.target_markets
            # Filter data to the specific selection (e.g., :over_25)
            market_train = filter(row -> row.selection == target, df_train)
            market_predict = filter(row -> row.selection == target, df_predict)
            
            if nrow(market_train) > 10 && nrow(market_predict) > 0
                fitted_l2 = fit_calibrator(config.model, market_train, config)
                shifted_probs = apply_shift(fitted_l2, market_predict)
                
                # Update the specific rows in our prediction split
                idx = findall(x -> x == target, calibrated_split.selection)
                calibrated_split.calib_prob[idx] = shifted_probs
                
                # Save the latest model state
                final_fitted_models[target] = fitted_l2
            end
        end
        
        append!(all_calibrated_preds, calibrated_split)
    end
    
    dropmissing!(all_calibrated_preds, :calib_prob)
    
    return CalibrationResults(config, final_fitted_models, all_calibrated_preds)
end
