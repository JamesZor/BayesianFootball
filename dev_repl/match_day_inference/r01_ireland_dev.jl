# dev_repl/match_day_inference/r01



# File to include - not repl run 
include("./l00_main_utils.jl")


# ========================================
#  Stage 1 - Training the model
# ========================================
#
# ---- 1. load data - segment
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


save_dir::String = "./data/match_day/april/ireland/"

es = DSExperimentSettings(
  ds,
  "17_04_26",
  save_dir,
  find_current_cv_parameters(ds)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)


# ========================================
#  Stage 2 - Running inference 
# ========================================

# ---- 1. load data and model.
saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);

exp = loaded_results[1] # 

# ---- 2.  fetch_todays_matches.
todays_matches = fetch_todays_matches(ds)


# from l05 ( april dev ) 
best_match = load_same_large_experiment_model(exp)


selections_to_calibrate = [:btts_yes, :draw, :over_15, :over_25, :over_35]
# 2. Get the calibration dictionary
shift_dict = compute_market_shifts(exp_m1, ds, market_data.df, selections_to_calibrate)

# from r04
target_ppd = compute_todays_matches_pdds(data_store,  exp_m1, todays_matches)
calibrated_ppd = apply_market_shifts(target_ppd, shift_dict)


comparison_view = compare_calibrated_odds(target_ppd, calibrated_ppd, todays_matches)



paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market_closing, min_edge=0.0)

# ---- 3.  Compute the PPD.
# ---- 4.  Compute the calibration shift 
# ---- 5.  Compute the shifted PPD.
# ---- 6.  Compute the paper bets
#   - need to just grab from 'whattheodds' another machine 

paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market_closing, min_edge=0.0)


# ========================================
#  Stage 2 - Running inference 
# ========================================

# ---- 1. Load data and model.
saved_folders = BayesianFootball.Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)
exp = loaded_results[end]  # Grabbing the most recent one

# ---- 2. Fetch today's matches.
todays_matches = fetch_todays_matches(ds)
market_data = BayesianFootball.Data.prepare_market_data(ds) # Need this for calibration

# ---- 3. Find Best Model Duration
# explicitly pass `dir=save_dir` so it searches your current folder
#
save_dir_dev::String = "./data/exp/dev_ireland"
best_match = load_same_large_experiment_model(exp, dir=save_dir_dev) 


# ---- 4. Compute Calibration Shifts
selections_to_calibrate = [:draw, :over_15, :over_25, :over_35, :under_25]
shift_dict = compute_market_shifts(best_match, ds, selections_to_calibrate)

#=
Running Inference on 503 matches...
--- Calibrating: :draw ---
>> Calculated Logit Shift: 0.0902
--- Calibrating: :over_15 ---
>> Calculated Logit Shift: 0.092
--- Calibrating: :over_25 ---
>> Calculated Logit Shift: 0.0955
--- Calibrating: :over_35 ---
>> Calculated Logit Shift: -0.0045
--- Calibrating: :under_25 ---
>> Calculated Logit Shift: -0.0955
Dict{Symbol, Float64} with 5 entries:
  :over_35  => -0.00447805
  :draw     => 0.0901502
  :over_15  => 0.0920369
  :over_25  => 0.0955279
  :under_25 => -0.0954918
=#


# ---- 5. Compute PPDs
target_ppd = compute_todays_matches_pdds(ds, exp, todays_matches)
calibrated_ppd = apply_market_shifts(target_ppd, shift_dict)

comparison_view = compare_calibrated_odds(target_ppd, calibrated_ppd, todays_matches)

# ---- 6. Paper Bets 
# (Make sure `live_market_closing` is loaded into a DataFrame before running this)
paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market_closing, min_edge=0.0)




# 1. Load the raw JSONL dump and parse it into our DataFrame
json_filepath = "/root/BayesianFootball/data/raw_odds_history.jsonl"
raw_live_market = load_live_market_jsonl(json_filepath)


# 2. Define exactly what you want to extract
selections_to_calibrate = [:draw, :over_15, :over_25, :over_35, :under_25]

# 3. Filter and Rename
live_market_closing = filter_and_rename_live_markets(raw_live_market, selections_to_calibrate)

first(live_market_closing, 4), 
# 2. Run the paper bets exactly as before
paper_bets_df = generate_paper_bets(
    target_ppd, 
    calibrated_ppd, 
    todays_matches, 
    first(live_market_closing, 4), 
    min_edge=0.0
)
