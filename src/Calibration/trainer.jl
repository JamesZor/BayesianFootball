# src/Calibration/trainer.jl


# ==========================================
# 1. TRAINING LOOP
# ==========================================
"""
    train_calibrators(training_data_l2::DataFrame, config::CalibrationConfig)

Iterates through time splits, trains L2 models on historical data, and returns 
a dictionary of fitted models mapped by Target -> Split ID.
"""
function train_calibrators(training_data_l2::DataFrame, config::CalibrationConfig)
    split_ids = sort(unique(training_data_l2.split_id))
    n_splits = length(split_ids)
    start_k = config.min_history_splits + 1
    all_targets = unique(training_data_l2.selection)

    println("Training Phase: Building models across $(n_splits - start_k + 1) splits on $(Threads.nthreads()) threads...")

    # Thread-safe container
    models_dicts = Vector{Dict{Symbol, Any}}(undef, n_splits)

    Threads.@threads for k in start_k:n_splits
        start_idx = config.max_history_splits == 0 ? 1 : max(1, k - config.max_history_splits)
        train_splits = split_ids[start_idx : (k - 1)]
        
        df_train = subset(training_data_l2, :split_id => ByRow(in(train_splits)), view=true)
        split_models = Dict{Symbol, Any}()

        for target in all_targets
            market_train = subset(df_train, :selection => ByRow(isequal(target)), view=true)

            if nrow(market_train) > config.min_market_train
                fitted_l2 = fit_calibrator(config.model, DataFrame(market_train), config)
                split_models[target] = fitted_l2
            end
        end
        models_dicts[k] = split_models
    end

    # Consolidate into Target -> Split ID -> Model
    fitted_models_history = Dict{Symbol, Dict{String, Any}}()
    for target in all_targets
        fitted_models_history[target] = Dict{String, Any}()
    end

    for k in start_k:n_splits
        split_id = split_ids[k]
        for (target, model) in models_dicts[k]
            fitted_models_history[target][split_id] = model
        end
    end

    return CalibrationResults(config, fitted_models_history)
end



function train_calibrators(ppd::Predictions.PPD, ds::Data.DataStore, config::CalibrationConfig)
  return train_calibrators(
            build_l2_training_df(ds,ppd),
            config) 
end


# ==========================================
# 2. PREDICTION LOOP
# ==========================================
"""
    apply_calibrators(predict_data::DataFrame, models_history::Dict, config::CalibrationConfig)

Applies the pre-trained models to their corresponding time splits.
"""
function apply_calibrators(ppd::Predictions.PPD, ds::Data.DataStore, calibration_results::CalibrationResults)
    predict_data = build_l2_training_df(ds, ppd)
    all_calibrated_preds = apply_calibrators(predict_data, calibration_results.fitted_models_history, calibration_results.config)
    # Return the new PPD (now with properly shifted :distribution arrays!)
    return Predictions.PPD(
        select(all_calibrated_preds, :match_id, :market_name, :selection, :distribution), 
        ppd.model,
        calibration_results.config,
        ppd.config 
    )
end

function apply_calibrators(predict_data::DataFrame, models_history::Dict, config::CalibrationConfig)
    split_ids = sort(unique(predict_data.split_id))
    n_splits = length(split_ids)
    start_k = config.min_history_splits + 1
    all_targets = unique(predict_data.selection)

    println("Prediction Phase: Applying models and shifting posteriors...")

    results_dfs = Vector{DataFrame}(undef, n_splits)

    Threads.@threads for k in start_k:n_splits
        current_split = split_ids[k]
        df_predict = subset(predict_data, :split_id => ByRow(isequal(current_split)), view=true)
        
        # Note: copy() makes a shallow copy, which is fine because we will replace 
        # the distribution arrays with completely new arrays below.
        calibrated_split = copy(df_predict)
        calibrated_split.calib_prob = Vector{Union{Missing, Float64}}(missing, nrow(calibrated_split))

        for target in all_targets
            market_predict = subset(df_predict, :selection => ByRow(isequal(target)), view=true)

            if haskey(models_history[target], current_split) && nrow(market_predict) > config.min_market_predict
                
                model = models_history[target][current_split]
                
                # The model handles both the point estimate AND the uncertainty distribution!
                shifted_probs, shifted_dists = apply_calibration(model, DataFrame(market_predict))

                # Slot them into the master DataFrame
                idx = findall(x -> x == target, calibrated_split.selection)
                calibrated_split.calib_prob[idx] = shifted_probs
                calibrated_split.distribution[idx] = shifted_dists
            end


        end
        results_dfs[k] = calibrated_split
    end

    # Consolidate
    valid_dfs = results_dfs[start_k:n_splits]
    all_calibrated_preds = vcat(valid_dfs...)
    dropmissing!(all_calibrated_preds, :calib_prob)
    return all_calibrated_preds

end

# ==========================================
# 3. HIGH-LEVEL WRAPPER
# ==========================================
"""
    run_calibration_backtest(training_data_l2::DataFrame, config::CalibrationConfig)

Convenience wrapper that runs both training and prediction for backtesting.
"""
function run_calibration_backtest(training_data_l2::DataFrame, config::CalibrationConfig)
    
    # 1. Train
    models_history = train_calibrators(training_data_l2, config)
    
    # 2. Predict
    all_calibrated_preds = apply_calibrators(training_data_l2, models_history, config)
    
    # 3. Return structured results
    return CalibrationResults(config, models_history, all_calibrated_preds)
end


# Add these to the bottom of your trainer.jl file

# ==========================================
# MULTI-CONFIG WRAPPERS
# ==========================================

"""
    train_calibrators(training_data_l2::DataFrame, configs::Vector{CalibrationConfig})

Trains a list of configurations sequentially and returns a Vector of CalibrationResults.
"""
function train_calibrators(training_data_l2::DataFrame, configs::Vector{CalibrationConfig})
    results_list = CalibrationResults[]
    
    for config in configs
        println("\n=== Training L2 Model: $(config.name) ===")
        push!(results_list, train_calibrators(training_data_l2, config))
    end
    
    return results_list
end



function train_calibrators(ppd::Predictions.PPD, ds::Data.DataStore, configs::Vector{CalibrationConfig})
    results_list = CalibrationResults[]
    training_data_l2 = build_l2_training_df(ds, ppd),
    for config in configs
        println("\n=== Training L2 Model: $(config.name) ===")
        push!(results_list, train_calibrators(training_data_l2, config))
    end
    
    return results_list
end


"""
    apply_calibrators(ppd::Predictions.PPD, ds::Data.DataStore, results_list::Vector{CalibrationResults})

Applies a list of fitted calibration results to a PPD. 
Returns a Vector of new, calibrated PPDs corresponding to each config.
"""
function apply_calibrators(ppd::Predictions.PPD, ds::Data.DataStore, results_list::Vector{CalibrationResults})
    
    # 1. Build the DataFrame representation ONCE for all configs to save memory/time
    println("Building master prediction DataFrame...")
    predict_data = build_l2_training_df(ds, ppd)
    
    calibrated_ppds = Predictions.PPD[]
    
    # 2. Loop through each fitted model and generate a new PPD
    for results in results_list
        println("\n=== Applying L2 Model: $(results.config.name) ===")
        
        # Run the core DataFrame application loop
        all_calibrated_preds = apply_calibrators(predict_data, results.fitted_models_history, results.config)
        
        
        # Construct the final PPD
        new_ppd = Predictions.PPD(
            select(all_calibrated_preds, :match_id, :market_name, :selection, :distribution, :match_date, :split_id), 
            ppd.model,
            results.config,
            ppd.config 
        )
        
        push!(calibrated_ppds, new_ppd)
    end
    
    return calibrated_ppds
end
