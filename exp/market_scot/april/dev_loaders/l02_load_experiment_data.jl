using Revise
using BayesianFootball
using DataFrames


function load_experiment_data_from_disk(;file_path="exp/market_runs/april", experiment_number=1, data_dir="./data") 
    saved_folders = BayesianFootball.Experiments.list_experiments(file_path; data_dir=data_dir)
    m1 = BayesianFootball.Experiments.load_experiment(saved_folders[experiment_number])
    return m1
end 

