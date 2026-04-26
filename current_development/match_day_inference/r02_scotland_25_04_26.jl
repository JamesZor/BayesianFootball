#  current_development/match_day_inference/r02_scotland_25_04_26.


# File to include - not repl run 
include("./l00_main_utils.jl")


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())



save_dir::String = "./data/match_day/april/scotland/"

es = DSExperimentSettings(
  ds,
  "25_04_26",
  save_dir,
  find_current_cv_parameters(ds)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)


# ========================================
#  Stage 2 - Running inference 
# ========================================

# ---- 1. Load data and model.
saved_folders = BayesianFootball.Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)
exp = loaded_results[1]  # Grabbing the most recent one



todays_matches = fetch_todays_matches(ds)

target_ppd = compute_todays_matches_pdds(ds, exp, todays_matches)



json_filepath = "/root/BayesianFootball/data/raw_odds_scotland_25_04_26.jsonl"
raw_live_market = load_live_market_jsonl(json_filepath)

selections_to_calibrate = [:over_15, :over_25, :over_35]
live_market_closing = filter_and_rename_live_markets(raw_live_market, selections_to_calibrate)
last(live_market_closing, 10)


paper_bets_df = generate_paper_bets(
    target_ppd, 
    target_ppd, 
    todays_matches, 
    last(live_market_closing, 10), 
    min_edge=0.0
)

# get the shift 
save_dir_dev::String = "./data/exp/ablation_study"
best_match = load_same_large_experiment_model(exp, dir=save_dir_dev) 
shift_dict = compute_market_shifts(best_match, ds, selections_to_calibrate)
calibrated_ppd = apply_market_shifts(target_ppd, shift_dict)

shift_dict1 = shift_dict
shift_dict1[:over_15] = shift_dict1[:over_15]/2
shift_dict1


calibrated_ppd1 = apply_market_shifts(target_ppd, shift_dict1)

paper_bets_df = generate_paper_bets(
    target_ppd, 
    calibrated_ppd, 
    todays_matches, 
    last(live_market_closing, 10), 
    min_edge=0.0
)



#=
julia> paper_bets_df = generate_paper_bets(                                                                                                                                                                                                                                                                                 
           target_ppd,                                                                                                                                                                                                                                                                                                      
           calibrated_ppd,                                                                                                                                                                                                                                                                                                  
           todays_matches,                                                                                                                                                                                                                                                                                                  
           first(live_market_closing, 10),                                                                                                                                                                                                                                                                                  
           min_edge=0.0                                                                                                                                                                                                                                                                                                     
       )                                                                                                                                                                                                                                                                                                                    
============================================================================================================                                                                                                                                                                                                                
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)                                                                                                                                                                                                                                                                    
============================================================================================================                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                            
MATCH: EAST-FIFE                                                                                                                                                                                                                                                                                                            
  draw     (Live: 4.60) | Raw: 4.11 [3.37 - 5.45] | Calib: 3.96 [3.25 - 5.23] | Raw Stake:  1.56% | Calib:  2.63%                                                                                                                                                              
  over_15  (Live: 1.24) | Raw: 1.45 [1.26 - 1.75] | Calib: 1.35 [1.20 - 1.57] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_25  (Live: 1.76) | Raw: 2.29 [1.74 - 3.26] | Calib: 2.16 [1.67 - 3.04] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_35  (Live: 2.86) | Raw: 4.23 [2.80 - 7.42] | Calib: 4.00 [2.67 - 6.94] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  under_35 (Live: 1.47) | Raw: 1.31 [1.16 - 1.55] | Calib: 1.33 [1.17 - 1.60] | Raw Stake: 16.54% | Calib: 12.24%                                                                                                                                                              
                                                                                                                                                                                                                                                                               
MATCH: KELTY-HEARTS-FC                                                                                                                                                                                                                                                         
  draw     (Live: 3.40) | Raw: 3.98 [3.42 - 4.72] | Calib: 3.83 [3.30 - 4.53] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_15  (Live: 1.30) | Raw: 1.37 [1.20 - 1.61] | Calib: 1.28 [1.16 - 1.47] | Raw Stake:  0.00% | Calib:  0.75%                                                                                                                                                              
  over_25  (Live: 1.94) | Raw: 2.04 [1.59 - 2.82] | Calib: 1.94 [1.53 - 2.64] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_35  (Live: 3.30) | Raw: 3.55 [2.42 - 5.97] | Calib: 3.37 [2.32 - 5.60] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  under_35 (Live: 1.39) | Raw: 1.39 [1.20 - 1.70] | Calib: 1.42 [1.22 - 1.76] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
                                                                                                                                                                                                                                                                               
MATCH: QUEEN-OF-THE-SOUTH                                                                                                                                                                                                                                                      
  draw     (Live: 4.80) | Raw: 4.58 [3.88 - 5.83] | Calib: 4.40 [3.74 - 5.58] | Raw Stake:  0.36% | Calib:  1.12%                                                                                                                                                              
  over_15  (Live: 1.15) | Raw: 1.24 [1.12 - 1.40] | Calib: 1.18 [1.09 - 1.30] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_25  (Live: 1.47) | Raw: 1.67 [1.35 - 2.14] | Calib: 1.61 [1.32 - 2.03] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_35  (Live: 2.16) | Raw: 2.59 [1.84 - 3.92] | Calib: 2.47 [1.78 - 3.71] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  under_35 (Live: 1.76) | Raw: 1.63 [1.34 - 2.19] | Calib: 1.68 [1.37 - 2.28] | Raw Stake:  3.79% | Calib:  1.48%                                                                                                                                                              
                                                                                                                                                                                                                                                                               
MATCH: HAMILTON-ACADEMICAL                                                                                                                                                                                                                                                     
  draw     (Live: 4.00) | Raw: 4.60 [3.74 - 6.12] | Calib: 4.42 [3.60 - 5.86] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_15  (Live: 1.25) | Raw: 1.29 [1.15 - 1.50] | Calib: 1.23 [1.12 - 1.38] | Raw Stake:  0.00% | Calib:  1.87%                                                                                                                                                              
  over_25  (Live: 1.76) | Raw: 1.83 [1.45 - 2.47] | Calib: 1.75 [1.40 - 2.33] | Raw Stake:  0.00% | Calib:  0.04%                                                                                                                                                              
  over_35  (Live: 2.96) | Raw: 2.98 [2.06 - 4.85] | Calib: 2.84 [1.98 - 4.56] | Raw Stake:  0.00% | Calib:  0.26%                                                                                                                                                              
  under_35 (Live: 1.45) | Raw: 1.50 [1.26 - 1.94] | Calib: 1.54 [1.28 - 2.02] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
                                                                                                                                                                                                                                                                               
MATCH: DUMBARTON                                                                                                                                                                                                                                                               
  draw     (Live: 4.30) | Raw: 4.76 [4.08 - 5.82] | Calib: 4.57 [3.92 - 5.57] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_15  (Live: 1.15) | Raw: 1.17 [1.08 - 1.31] | Calib: 1.13 [1.06 - 1.24] | Raw Stake:  0.00% | Calib:  2.39%                                                                                                                                                              
  over_25  (Live: 1.47) | Raw: 1.50 [1.25 - 1.90] | Calib: 1.45 [1.23 - 1.81] | Raw Stake:  0.00% | Calib:  0.32%                                                                                                                                                              
  over_35  (Live: 2.14) | Raw: 2.16 [1.60 - 3.23] | Calib: 2.07 [1.56 - 3.07] | Raw Stake:  0.00% | Calib:  0.32%                                                                                                                                                              
  under_35 (Live: 1.76) | Raw: 1.86 [1.45 - 2.66] | Calib: 1.93 [1.48 - 2.79] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
                                                                                                                                                                                                                                                                               
MATCH: CLYDE-FC                                                                                                                                                                                                                                                                
  draw     (Live: 3.75) | Raw: 4.28 [3.69 - 5.18] | Calib: 4.12 [3.55 - 4.97] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_15  (Live: 1.25) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.12 - 1.36] | Raw Stake:  0.00% | Calib:  3.22%                                                                                                                                                              
  over_25  (Live: 1.76) | Raw: 1.80 [1.45 - 2.39] | Calib: 1.73 [1.40 - 2.25] | Raw Stake:  0.00% | Calib:  0.26%                                                                                                                                                              
  over_35  (Live: 2.92) | Raw: 2.93 [2.07 - 4.62] | Calib: 2.79 [1.99 - 4.35] | Raw Stake:  0.00% | Calib:  0.32%                                                                                                                                                              
  under_35 (Live: 1.45) | Raw: 1.52 [1.28 - 1.94] | Calib: 1.56 [1.30 - 2.01] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
                                                                                                                                                                                                                                                                               
MATCH: THE-SPARTANS-FC                                                                                                                                                                                                                                                         
  draw     (Live: 4.00) | Raw: 4.20 [3.56 - 5.24] | Calib: 4.04 [3.43 - 5.03] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_15  (Live: 1.25) | Raw: 1.34 [1.19 - 1.56] | Calib: 1.26 [1.14 - 1.43] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_25  (Live: 1.77) | Raw: 1.97 [1.55 - 2.66] | Calib: 1.88 [1.50 - 2.49] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_35  (Live: 2.94) | Raw: 3.36 [2.33 - 5.44] | Calib: 3.19 [2.23 - 5.11] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  under_35 (Live: 1.44) | Raw: 1.42 [1.23 - 1.75] | Calib: 1.46 [1.24 - 1.81] | Raw Stake:  0.25% | Calib:  0.00%                                                                                                                                                              
                                                                                                                                                                                                                                                                               
MATCH: FORFAR-ATHLETIC                                                                                                                                                                                                                                                         
  draw     (Live: 6.40) | Raw: 4.12 [3.51 - 5.08] | Calib: 3.96 [3.38 - 4.88] | Raw Stake:  9.49% | Calib: 10.67%                                                                                                                                                              
  over_15  (Live: 1.25) | Raw: 1.35 [1.19 - 1.58] | Calib: 1.27 [1.15 - 1.44] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_25  (Live: 1.74) | Raw: 1.98 [1.56 - 2.70] | Calib: 1.89 [1.51 - 2.54] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                              
  over_35  (Live: 2.92) | Raw: 3.40 [2.36 - 5.59] | Calib: 3.22 [2.25 - 5.25] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.45) | Raw: 1.42 [1.22 - 1.74] | Calib: 1.45 [1.24 - 1.80] | Raw Stake:  0.87% | Calib:  0.00%

MATCH: STRANRAER
  draw     (Live: 5.20) | Raw: 3.94 [3.42 - 4.69] | Calib: 3.79 [3.30 - 4.50] | Raw Stake:  6.73% | Calib:  8.04%
  over_15  (Live: 1.20) | Raw: 1.37 [1.20 - 1.61] | Calib: 1.29 [1.16 - 1.46] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.58) | Raw: 2.05 [1.59 - 2.80] | Calib: 1.95 [1.54 - 2.62] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (Live: 2.46) | Raw: 3.60 [2.43 - 5.90] | Calib: 3.41 [2.32 - 5.53] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.59) | Raw: 1.39 [1.20 - 1.70] | Calib: 1.42 [1.22 - 1.76] | Raw Stake: 16.30% | Calib: 12.16%
=#


#=
============================================================================================================                                                
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)                                                                                                    
============================================================================================================                                                
                                                                                                                                                            

MATCH: HAMILTON-ACADEMICAL
  over_15  (Live: 1.25) | Raw: 1.29 [1.15 - 1.50] | Calib: 1.23 [1.12 - 1.38] | Raw Stake:  0.00% | Calib:  1.87%
  over_25  (Live: 1.69) | Raw: 1.83 [1.45 - 2.47] | Calib: 1.75 [1.40 - 2.33] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: DUMBARTON
  over_15  (Live: 1.14) | Raw: 1.17 [1.08 - 1.31] | Calib: 1.13 [1.06 - 1.24] | Raw Stake:  0.00% | Calib:  0.44%
  over_25  (Live: 1.44) | Raw: 1.50 [1.25 - 1.90] | Calib: 1.45 [1.23 - 1.81] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: CLYDE-FC
  over_15  (Live: 1.25) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.12 - 1.36] | Raw Stake:  0.00% | Calib:  3.22%
  over_25  (Live: 1.78) | Raw: 1.80 [1.45 - 2.39] | Calib: 1.73 [1.40 - 2.25] | Raw Stake:  0.00% | Calib:  0.63%
  over_35  (Live: 2.98) | Raw: 2.93 [2.07 - 4.62] | Calib: 2.79 [1.99 - 4.35] | Raw Stake:  0.04% | Calib:  0.64%

=#

