# src/Calibration/basic_metrics.jl

# ==========================================
# 1. Scalar Metric Functions
# ==========================================

function calc_brier(y_true::Real, p_pred::Real)
    return (y_true - p_pred)^2
end

function calc_logloss(y_true::Real, p_pred::Real)
    p_safe = clamp(p_pred, 1e-15, 1 - 1e-15)
    return -(y_true * log(p_safe) + (1 - y_true) * log(1 - p_safe))
end

function calc_accuracy(y_true::Real, p_pred::Real; threshold=0.5)
    pred_class = p_pred >= threshold ? 1.0 : 0.0
    return pred_class == y_true ? 1.0 : 0.0
end

function calc_bias(y_true::Real, p_pred::Real)
    return p_pred - y_true 
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
# 3. Summarize & Compare Metrics
# ==========================================

"""
    summarize_metrics(df_eval::DataFrame; groupby_cols=[:selection])

Aggregates row-level metrics to show overall model performance.
"""
function summarize_metrics(df_eval::DataFrame; groupby_cols=[:selection])
    
    summary = combine(groupby(df_eval, groupby_cols),
        nrow => :n_predictions,
        :is_winner => mean => :actual_win_rate, 
        :prob => mean => :avg_predicted_prob,
        :brier => mean => :brier,
        :logloss => mean => :logloss,
        :accuracy => mean => :accuracy,
        :bias => mean => :bias
    )
    
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
    
    # 2. Join them together
    comp = innerjoin(sum_base, sum_new, on=groupby_cols, renamecols = "_raw" => "_calib")
    
    # 3. Calculate Improvements (Positive = Better)
    comp.brier_imp = comp.brier_raw .- comp.brier_calib
    comp.logloss_imp = comp.logloss_raw .- comp.logloss_calib
    comp.bias_imp = abs.(comp.bias_raw) .- abs.(comp.bias_calib)
    
    # 4. Clean up the final view
    select!(comp, 
        groupby_cols...,
        :n_predictions_raw => :n_predictions, 
        :actual_win_rate_raw => :actual_win_rate,
        :avg_predicted_prob_raw => :prob_raw,
        :avg_predicted_prob_calib => :prob_calib,
        :brier_imp,
        :logloss_imp,
        :bias_imp,
        :bias_raw,
        :bias_calib
    )
    
    return comp
end
