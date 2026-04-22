# src/Calibration/basic_metrics.jl
# ==========================================
# 1. Scalar Metric Functions
# ==========================================

# Brier Score (MSE of probability)
function calc_brier(y_true::Real, p_pred::Real)
    return (y_true - p_pred)^2
end

# Log Loss (Cross Entropy)
function calc_logloss(y_true::Real, p_pred::Real)
    p_safe = clamp(p_pred, 1e-15, 1 - 1e-15)
    return -(y_true * log(p_safe) + (1 - y_true) * log(1 - p_safe))
end

# Accuracy (Assuming 0.5 threshold for binary classification)
function calc_accuracy(y_true::Real, p_pred::Real; threshold=0.5)
    pred_class = p_pred >= threshold ? 1.0 : 0.0
    return pred_class == y_true ? 1.0 : 0.0
end

# Bias (Difference between prediction and reality)
function calc_bias(y_true::Real, p_pred::Real)
    return p_pred - y_true 
end


function build_evaluation_df(ppd_raw::Predictions.PPD, ppd_calib::Predictions.PPD, ds)
    # 1. Extract means from the Raw PPD
  
    df_raw = select(build_l2_training_df(ds, ppd_raw), :match_id, :selection, :market_name, :split_id,
        :distribution => ByRow(mean) => :raw_prob
    )
    
    # 2. Extract means from the Calibrated PPD
    df_calib = select(ppd_calib.df, :match_id, :selection,
        :distribution => ByRow(mean) => :calib_prob
    )
    
    # 3. Join them together
    df_eval = innerjoin(df_raw, df_calib, on=[:match_id, :selection])
    
    # 4. Bring in the ground truth from ds.odds
    # (Assuming ds.odds has match_id, selection, and is_winner)
    df_eval = innerjoin(df_eval, ds.odds[!, [:match_id, :selection, :is_winner]], on=[:match_id, :selection])
    dropmissing!(df_eval, :is_winner)
    
    # 5. Calculate row-level metrics
    df_eval.raw_brier = calc_brier.(df_eval.is_winner, df_eval.raw_prob)
    df_eval.calib_brier = calc_brier.(df_eval.is_winner, df_eval.calib_prob)
    
    df_eval.raw_logloss = calc_logloss.(df_eval.is_winner, df_eval.raw_prob)
    df_eval.calib_logloss = calc_logloss.(df_eval.is_winner, df_eval.calib_prob)
    
    return df_eval
end

function summarize_metrics(df_eval::DataFrame; groupby_cols=[:selection])
    # Group by whatever columns the user requested and calculate the mean errors
    summary = combine(groupby(df_eval, groupby_cols),
        nrow => :n_predictions,
        :raw_brier => mean => :raw_brier,
        :calib_brier => mean => :calib_brier,
        :raw_logloss => mean => :raw_logloss,
        :calib_logloss => mean => :calib_logloss
    )
    
    # Add Improvement Columns (Positive number = L2 is better than L1)
    summary.brier_imp = summary.raw_brier .- summary.calib_brier
    summary.logloss_imp = summary.raw_logloss .- summary.calib_logloss
    
    # Sort by the highest volume markets first
    sort!(summary, :n_predictions, rev=true)
    
    return summary
end


# ==========================================
# 2. Single PPD Evaluation
# ==========================================

"""
    build_evaluation_df(ppd::Predictions.PPD, ds::Data.DataStore)

Evaluates a single model's predictions against the ground truth.
"""
function build_evaluation_df(ppd::Predictions.PPD, ds)
    # 1. Extract the mean probability from the distributions
    # Assuming you still have build_l2_training_df available to get the split_id
    df_eval = select(build_l2_training_df(ds, ppd), 
        :match_id, :selection, :market_name, :split_id,
        :distribution => ByRow(mean) => :prob
    )
    
    # 2. Bring in the ground truth from ds.odds
    df_eval = innerjoin(df_eval, ds.odds[!, [:match_id, :selection, :is_winner]], on=[:match_id, :selection])
    dropmissing!(df_eval, :is_winner)
    
    # 3. Calculate row-level metrics
    df_eval.brier = calc_brier.(df_eval.is_winner, df_eval.prob)
    df_eval.logloss = calc_logloss.(df_eval.is_winner, df_eval.prob)
    df_eval.accuracy = calc_accuracy.(df_eval.is_winner, df_eval.prob)
    df_eval.bias = calc_bias.(df_eval.is_winner, df_eval.prob)
    
    return df_eval
end


# ==========================================
# 3. Summarize Metrics
# ==========================================

"""
    summarize_metrics(df_eval::DataFrame; groupby_cols=[:selection])

Aggregates row-level metrics to show overall model performance.
"""
function summarize_metrics(df_eval::DataFrame; groupby_cols=[:selection])
    
    summary = combine(groupby(df_eval, groupby_cols),
        nrow => :n_predictions,
        # The actual hit rate of the events
        :is_winner => mean => :actual_win_rate, 
        
        # The model's average predicted probability
        :prob => mean => :avg_predicted_prob,
        
        # Aggregated error metrics
        :brier => mean => :brier,
        :logloss => mean => :logloss,
        :accuracy => mean => :accuracy,
        :bias => mean => :bias
    )
    
    sort!(summary, :n_predictions, rev=true)
    
    return summary
end


"""
    compare_models(eval_baseline::DataFrame, eval_new::DataFrame; groupby_cols=[:selection])

Summarizes two evaluation dataframes and joins them together to show direct improvements.
Positive improvement values mean the new model is performing better.
"""
function compare_models(eval_baseline::DataFrame, eval_new::DataFrame; groupby_cols=[:selection])
    # 1. Get the individual summaries
    sum_base = summarize_metrics(eval_baseline; groupby_cols=groupby_cols)
    sum_new = summarize_metrics(eval_new; groupby_cols=groupby_cols)
    
    # 2. Join them together. 
    # renamecols automatically appends "_raw" and "_calib" to overlapping column names!
    comp = innerjoin(sum_base, sum_new, on=groupby_cols, renamecols = "_raw" => "_calib")
    
    # 3. Calculate Improvements (Positive = Better)
    comp.brier_imp = comp.brier_raw .- comp.brier_calib
    comp.logloss_imp = comp.logloss_raw .- comp.logloss_calib
    
    # For bias, "better" means "closer to zero". 
    # So we compare the absolute values. Positive = New model is closer to 0 bias.
    comp.bias_imp = abs.(comp.bias_raw) .- abs.(comp.bias_calib)
    
    # 4. Clean up the final view to only show what matters
    select!(comp, 
        groupby_cols...,
        :n_predictions_raw => :n_predictions, # Just keep one copy of the counts/win rates
        :actual_win_rate_raw => :actual_win_rate,
        :avg_predicted_prob_raw => :prob_raw,
        :avg_predicted_prob_calib => :prob_calib,
        :brier_imp,
        :logloss_imp,
        :bias_imp,
        :bias_raw,
        :bias_calib
    )
    
    sort!(comp, :n_predictions, rev=true)
    
    return comp
end
