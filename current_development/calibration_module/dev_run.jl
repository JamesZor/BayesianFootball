using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


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


# - Process
  # Get chronological unique splits
split_ids = sort(unique(training_data_l2.split_id))
n_splits = length(split_ids)
start_k = config.min_history_splits + 1

all_calibrated_preds = DataFrame()
final_fitted_models = Dict{Symbol, Any}()

    # for k in (config.min_history_splits + 1):n_splits
k = 9
start_idx = config.max_history_splits == 0 ? 1 : max(1, k - config.max_history_splits)
train_splits = split_ids[start_idx : (k - 1)]
df_train = subset(training_data_l2, :split_id => ByRow(in(train_splits)))
current_split = split_ids[k]
df_predict = subset(training_data_l2, :split_id => ByRow(isequal(split_ids[k])))

calibrated_split = copy(df_predict)
calibrated_split.calib_prob = Vector{Union{Missing, Float64}}(missing, nrow(calibrated_split))

# for target in unique(training_data_l2.selection)
target = :over_15

market_train = subset(df_train, :selection => ByRow(isequal(target)))
market_predict = subset(df_train, :selection => ByRow(isequal(target)))

if nrow(market_train) > config.min_market_train && nrow(market_predict) > config.min_predict


                fitted_l2 = fit_calibrator(config, market_train)  # create a wrapper for this 
                shifted_probs = apply_shift(fitted_l2, market_predict)

                # Update the specific rows in our prediction split
                idx = findall(x -> x == target, calibrated_split.selection)
                calibrated_split.calib_prob[idx] = shifted_probs

                # Save the latest model state
                final_fitted_models[target] = fitted_l2

end



        # 3. Fit & Apply per target market
        # for target in config.target_markets
            # if nrow(market_train) > 10 && nrow(market_predict) > 0
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

