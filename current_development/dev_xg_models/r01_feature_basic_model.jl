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



saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);

exp = loaded_results[1]


exo

using Turing
chain_fold_1 = exp.training_results[1][1]
chain_fold_2 = exp.training_results[2][1]

chain_fold_6 = exp.training_results[6][1]


describe(chain_fold_1)

describe(chain_fold_2)
describe(chain_fold_6)
