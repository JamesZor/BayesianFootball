using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


include("./l00_main_utils.jl")


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())



save_dir::String = "./data/bench_models/ireland/"

es = DSExperimentSettings(
  ds,
  "test_batch_1_",
  save_dir,
  get_target_seasons_string(ds.segment)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)

