# current_development/dev_xg_models/r01_basic_models.jl 
#
#=
Here we want to test the updated feature set and data 
work with the training and running of models via the package 
training-module and the experiment-module.
=#


using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


include("./l01_feature_basic_model.jl")




ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

save_dir::String = "./data/dev_xg_models/"

es = DSExperimentSettings(
  ds,
  "test_featureset",
  save_dir,
  get_target_seasons_string(ds.segment)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)

