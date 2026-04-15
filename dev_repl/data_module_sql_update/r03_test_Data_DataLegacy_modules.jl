using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)

# 
include("./l03_test_Data_DataLegacy_modules.jl")
#

# Load from sql
# server is busy... hence blacked out
# ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())

# working local
ds = get_datastore_local_ip()
ds_legacy = get_datastore_legacy()



save_dir::String = "./data/exp/data_module_test"


es_new = DSExperimentSettings(ds, "new", save_dir)
es_legacy = DSExperimentSettings(ds_legacy, "legacy", save_dir)


all_tasks = create_list_experiment_tasks(es_list)

experiment_list_combined = create_list_experiment_configs([es_new, es_legacy])


# results = run_experiment_task.(all_tasks)
