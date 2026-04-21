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

