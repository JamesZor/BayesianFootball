# dev_repl/match_day_inference/r01



# File to include - not repl run 
include("./l00_main_utils.jl")


# ========================================
#  Stage 1 - Training the model
# ========================================
#
# ---- 1. load data - segment
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())


save_dir::String = "./data/match_day/april/ireland/"

es = DSExperimentSettings(
  ds,
  "17_04_26",
  save_dir,
  find_current_cv_parameters(ds)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)

