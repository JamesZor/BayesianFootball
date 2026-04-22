using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Statistics
using Dates
using Printf


using GLM
using StatsFuns: logit, logistic
using StatsModels


# files to include:
include("./experiment_utils.jl")
include("./data_l2_prep.jl")
include("./types.jl")
incude("./shift_models/basic_glm.jl")
include("./trainer.jl")



# i. Load DataStore & L1 Experiment (Using your existing code structure)
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
exp_dir = "exp/ablation_study"

# ii: load the experiment results
saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[1])



# 1. experiment_utils  

# latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp)
# ppd = BayesianFootball.Predictions.model_inference(latents)

ppd_raw= model_inference(ds, exp)


# 2. data_l2_prep
# @btime training_data_l2 = build_l2_training_df(ds, ppd_raw)
# Profile.clear()
# @profile build_l2_training_df(ds, ppd_raw)
#
# Profile.print(maxdepth=15)




training_data_l2 = build_l2_training_df(ds, ppd_raw)



# 3. Configs
shift_model_config = CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = BasicLogitShift(), 
    min_history_splits = 8,   
    max_history_splits = 0,   
)



# 4. Training 
# --- calibration_trainer 
#  - inputs 
training_data_l2 
config = shift_model_config 


fitted_model_history = train_calibrators(training_data_l2, shift_model_config);


# 5. applying the calibration to ppd

calib_preds = apply_calibrators(ppd_raw, ds, fitted_model_history)
#
# id = rand(calib_preds.df.match_id)
# target_select = :under_25
#
# d_raw = subset(ppd_raw.df, 
#                 :match_id => ByRow(isequal(id)),
#                :selection => ByRow(isequal(target_select))).distribution[1];
# d_cali = subset(calib_preds.df, 
#                 :match_id => ByRow(isequal(id)),
#                :selection => ByRow(isequal(target_select))).distribution[1];
# mean(d_raw) 
# mean(d_cali)
#

# --- 5 
using DataFrames
using Statistics

# Scalar Brier Score (MSE of probability)
function calc_brier(y_true::Real, p_pred::Real)
    return (y_true - p_pred)^2
end

# Scalar Log Loss (Cross Entropy)
function calc_logloss(y_true::Real, p_pred::Real)
    p_safe = clamp(p_pred, 1e-15, 1 - 1e-15)
    return -(y_true * log(p_safe) + (1 - y_true) * log(1 - p_safe))
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


# 1. Build the massive evaluation frame
df_eval = build_evaluation_df(ppd_raw, calib_preds, ds)

# 2. View the overall performance grouped by selection (home, away, over_15, etc.)
summary_by_selection = summarize_metrics(df_eval, groupby_cols=[:selection])
display(summary_by_selection)

# 3. Want to see if the calibration degraded over time? Group by split_id!
summary_by_time = summarize_metrics(df_eval, groupby_cols=[:split_id])
display(sort(summary_by_time, :split_id))

# 4. Want to isolate a specific market to debug? Just use subset!
over_25_only = subset(df_eval, :selection => ByRow(isequal(:over_25)))
