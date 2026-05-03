# File to include - not repl run 
include("./l00_main_utils.jl")


# ========================================
#  Stage 1 - Training the model
# ========================================
#
# ---- 1. load data - segment
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())


save_dir::String = "./data/match_day/april/scotland/"

es = DSExperimentSettings(
  ds,
  "02_05_26",
  save_dir,
  find_current_cv_parameters(ds)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)




saved_folders = BayesianFootball.Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)
expr = loaded_results[1]  # Grabbing the most recent one


function BayesianFootball.Features.required_features(model::BayesianFootball.Models.PreGame.AblationStudy_NB_baseline_month_r)
    return [:team_ids, :goals] 
end



todays_matches = fetch_todays_matches(ds)

target_ppd = compute_todays_matches_pdds(ds, expr, todays_matches)

#  compute todays matches updates for the features 
boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, expr.config.splitter)
feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, expr.config.model)

  last_split_idx = length(expr.training_results)
  chain = expr.training_results[last_split_idx][1]
  feature_set = feature_collection[last_split_idx][1]

  raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
      expr.config.model, todays_matches, feature_set, chain
  )

  latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), expr.config.model)
  ppd = BayesianFootball.Predictions.model_inference(latents)

### 
target_ppd = ppd


json_filepath = "/root/BayesianFootball/data/raw_odds_scotland_02_05_26.jsonl"
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


# claibraiton 

# get the shift 
save_dir_dev::String = "./data/exp/ablation_study"
best_match = load_same_large_experiment_model(expr, dir=save_dir_dev) 
shift_dict = compute_market_shifts(best_match, ds, selections_to_calibrate)
calibrated_ppd = apply_market_shifts(target_ppd, shift_dict)

shift_dict = Dict(
  :under_35 => -0.077246,
  :draw     => 0.0520978,
  :over_15  => 0.269648,
  :over_25  => 0.102343
)

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
           last(live_market_closing, 10),                                                                                                                                                                                                                                                                                   
           min_edge=0.0                                                                                                                                                                                                                                                                                                     
       )                                                                                                                                                                                                                                                                                                                    
============================================================================================================                                                                                                                                                                                                                
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)                                                                                                                                                                                                                                                                    
============================================================================================================                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                            
MATCH: PETERHEAD                                                                                                                                                                                                                                                                                                            
  over_15  (Live: 1.26) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.21 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  6.43%                                                                                                                                                                                                           
  over_25  (Live: 1.81) | Raw: 1.77 [1.43 - 2.36] | Calib: 1.70 [1.39 - 2.22] | Raw Stake:  0.28% | Calib:  2.43%                                                                                                                                                                                                           
  over_35  (Live: 3.00) | Raw: 2.84 [2.01 - 4.49] | Calib: 2.84 [2.01 - 4.49] | Raw Stake:  0.47% | Calib:  0.47%                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                            
MATCH: MONTROSE                                                                                                                                                                                                                                                                                                             
  over_15  (Live: 1.27) | Raw: 1.38 [1.21 - 1.63] | Calib: 1.30 [1.16 - 1.48] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                                                                           
  over_25  (Live: 1.78) | Raw: 2.08 [1.61 - 2.85] | Calib: 1.98 [1.55 - 2.67] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                                                                           
  over_35  (Live: 3.00) | Raw: 3.65 [2.47 - 6.02] | Calib: 3.65 [2.47 - 6.02] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                            
MATCH: EDINBURGH-CITY-FC
  over_15  (Live: 1.22) | Raw: 1.22 [1.12 - 1.37] | Calib: 1.17 [1.09 - 1.28] | Raw Stake:  0.00% | Calib:  9.85%
  over_25  (Live: 1.67) | Raw: 1.63 [1.36 - 2.05] | Calib: 1.57 [1.33 - 1.95] | Raw Stake:  0.61% | Calib:  3.50%
  over_35  (Live: 2.68) | Raw: 2.48 [1.86 - 3.62] | Calib: 2.48 [1.86 - 3.62] | Raw Stake:  1.25% | Calib:  1.25%

MATCH: INVERNESS-CALEDONIAN-THISTLE
  over_15  (Live: 1.22) | Raw: 1.55 [1.32 - 1.92] | Calib: 1.42 [1.24 - 1.70] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.67) | Raw: 2.58 [1.90 - 3.84] | Calib: 2.43 [1.82 - 3.56] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (Live: 2.68) | Raw: 5.05 [3.23 - 9.38] | Calib: 5.05 [3.23 - 9.38] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: EAST-KILBRIDE
  over_15  (Live: 1.19) | Raw: 1.32 [1.18 - 1.52] | Calib: 1.25 [1.14 - 1.40] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.57) | Raw: 1.90 [1.52 - 2.52] | Calib: 1.82 [1.47 - 2.38] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (Live: 2.42) | Raw: 3.18 [2.23 - 5.01] | Calib: 3.18 [2.23 - 5.01] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ALLOA-ATHLETIC
  over_15  (Live: 1.28) | Raw: 1.42 [1.24 - 1.69] | Calib: 1.33 [1.18 - 1.53] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.80) | Raw: 2.20 [1.69 - 3.05] | Calib: 2.08 [1.63 - 2.85] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (Live: 2.96) | Raw: 3.97 [2.67 - 6.69] | Calib: 3.97 [2.67 - 6.69] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ELGIN-CITY
  over_15  (Live: 1.28) | Raw: 1.39 [1.23 - 1.61] | Calib: 1.30 [1.18 - 1.47] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.83) | Raw: 2.10 [1.66 - 2.80] | Calib: 1.99 [1.60 - 2.62] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (Live: 3.05) | Raw: 3.71 [2.60 - 5.84] | Calib: 3.71 [2.60 - 5.84] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: STIRLING-ALBION
  over_15  (Live: 1.20) | Raw: 1.33 [1.19 - 1.51] | Calib: 1.25 [1.15 - 1.39] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.62) | Raw: 1.92 [1.56 - 2.48] | Calib: 1.84 [1.51 - 2.34] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (Live: 2.48) | Raw: 3.24 [2.34 - 4.88] | Calib: 3.24 [2.34 - 4.88] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ANNAN-ATHLETIC
  over_15  (Live: 1.28) | Raw: 1.35 [1.21 - 1.56] | Calib: 1.27 [1.16 - 1.42] | Raw Stake:  0.00% | Calib:  0.19%
  over_25  (Live: 1.89) | Raw: 2.00 [1.61 - 2.63] | Calib: 1.90 [1.55 - 2.47] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (Live: 3.20) | Raw: 3.44 [2.45 - 5.34] | Calib: 3.44 [2.45 - 5.34] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: COVE-RANGERS
  over_15  (Live: 1.30) | Raw: 1.33 [1.18 - 1.55] | Calib: 1.25 [1.14 - 1.42] | Raw Stake:  0.00% | Calib:  4.33%
  over_25  (Live: 1.88) | Raw: 1.93 [1.52 - 2.61] | Calib: 1.84 [1.47 - 2.46] | Raw Stake:  0.00% | Calib:  0.26%
  over_35  (Live: 3.20) | Raw: 3.24 [2.24 - 5.27] | Calib: 3.24 [2.24 - 5.27] | Raw Stake:  0.00% | Calib:  0.00%




----

============================================================================================================                                                                                                                                                                                                                
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)                                                                                                                                                                                                                                                                    
============================================================================================================                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                            
MATCH: PETERHEAD                                                                                                                                                                                                                                                                                                            
  over_15  (Live: 1.26) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.21 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  6.43%                                                                                                                                                                                                           
  over_25  (Live: 1.81) | Raw: 1.77 [1.43 - 2.36] | Calib: 1.70 [1.39 - 2.22] | Raw Stake:  0.28% | Calib:  2.43%                                                                                                                                                                                                           
  over_35  (Live: 3.00) | Raw: 2.84 [2.01 - 4.49] | Calib: 2.84 [2.01 - 4.49] | Raw Stake:  0.47% | Calib:  0.47%                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                            
MATCH: MONTROSE                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                            
MATCH: EDINBURGH-CITY-FC
  over_15  (Live: 1.22) | Raw: 1.22 [1.12 - 1.37] | Calib: 1.17 [1.09 - 1.28] | Raw Stake:  0.00% | Calib:  9.85%
  over_25  (Live: 1.67) | Raw: 1.63 [1.36 - 2.05] | Calib: 1.57 [1.33 - 1.95] | Raw Stake:  0.61% | Calib:  3.50%
  over_35  (Live: 2.68) | Raw: 2.48 [1.86 - 3.62] | Calib: 2.48 [1.86 - 3.62] | Raw Stake:  1.25% | Calib:  1.25%

MATCH: INVERNESS-CALEDONIAN-THISTLE

MATCH: EAST-KILBRIDE

MATCH: ALLOA-ATHLETIC

MATCH: ELGIN-CITY

MATCH: STIRLING-ALBION

MATCH: ANNAN-ATHLETIC
  over_15  (Live: 1.28) | Raw: 1.35 [1.21 - 1.56] | Calib: 1.27 [1.16 - 1.42] | Raw Stake:  0.00% | Calib:  0.19%

MATCH: COVE-RANGERS
  over_15  (Live: 1.30) | Raw: 1.33 [1.18 - 1.55] | Calib: 1.25 [1.14 - 1.42] | Raw Stake:  0.00% | Calib:  4.33%
  over_25  (Live: 1.88) | Raw: 1.93 [1.52 - 2.61] | Calib: 1.84 [1.47 - 2.46] | Raw Stake:  0.00% | Calib:  0.26%
  over_35  (Live: 3.20) | Raw: 3.24 [2.24 - 5.27] | Calib: 3.24 [2.24 - 5.27] | Raw Stake:  0.00% | Calib:  0.00%
=#


#=
========================================================================================================================                                                                                                                                                                                                    
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)                                                                                                                                                                                                                                                                    
========================================================================================================================                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                            
MATCH: PETERHEAD                                                                                                                                                                                                                                                                                                            
  over_15  (B: 1.26 | L: 1.29 | M: 1.27) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.21 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  6.43%                                                                                                                                                                                          
  over_25  (B: 1.81 | L: 1.83 | M: 1.82) | Raw: 1.77 [1.43 - 2.36] | Calib: 1.70 [1.39 - 2.22] | Raw Stake:  0.28% | Calib:  2.43%                                                                                                                                                                                          
  over_35  (B: 3.00 | L: 3.30 | M: 3.15) | Raw: 2.84 [2.01 - 4.49] | Calib: 2.84 [2.01 - 4.49] | Raw Stake:  0.47% | Calib:  0.47%                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                            
MATCH: MONTROSE                                                                                                                                                                                                                                                                                                             
  over_15  (B: 1.27 | L: 1.30 | M: 1.29) | Raw: 1.38 [1.21 - 1.63] | Calib: 1.30 [1.16 - 1.48] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                                                          
  over_25  (B: 1.78 | L: 1.87 | M: 1.83) | Raw: 2.08 [1.61 - 2.85] | Calib: 1.98 [1.55 - 2.67] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                                                          
  over_35  (B: 3.00 | L: 3.30 | M: 3.15) | Raw: 3.65 [2.47 - 6.02] | Calib: 3.65 [2.47 - 6.02] | Raw Stake:  0.00% | Calib:  0.00%                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                            
MATCH: EDINBURGH-CITY-FC                                                                                                                                                                                                                                                                                                    
  over_15  (B: 1.22 | L: 1.25 | M: 1.23) | Raw: 1.22 [1.12 - 1.37] | Calib: 1.17 [1.09 - 1.28] | Raw Stake:  0.00% | Calib:  9.85%                                                                                                                                                                                          
  over_25  (B: 1.67 | L: 1.74 | M: 1.71) | Raw: 1.63 [1.36 - 2.05] | Calib: 1.57 [1.33 - 1.95] | Raw Stake:  0.61% | Calib:  3.50%                                                                                                                                                                                          
  over_35  (B: 2.68 | L: 2.90 | M: 2.79) | Raw: 2.48 [1.86 - 3.62] | Calib: 2.48 [1.86 - 3.62] | Raw Stake:  1.25% | Calib:  1.25%                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                            
MATCH: INVERNESS-CALEDONIAN-THISTLE                                                                                                                                                                                                                                                                                         
  over_15  (B: 1.22 | L: 1.25 | M: 1.23) | Raw: 1.55 [1.32 - 1.92] | Calib: 1.42 [1.24 - 1.70] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (B: 1.67 | L: 1.75 | M: 1.71) | Raw: 2.58 [1.90 - 3.84] | Calib: 2.43 [1.82 - 3.56] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (B: 2.68 | L: 2.84 | M: 2.76) | Raw: 5.05 [3.23 - 9.38] | Calib: 5.05 [3.23 - 9.38] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: EAST-KILBRIDE
  over_15  (B: 1.19 | L: 1.22 | M: 1.21) | Raw: 1.32 [1.18 - 1.52] | Calib: 1.25 [1.14 - 1.40] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (B: 1.57 | L: 1.65 | M: 1.61) | Raw: 1.90 [1.52 - 2.52] | Calib: 1.82 [1.47 - 2.38] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (B: 2.42 | L: 2.64 | M: 2.53) | Raw: 3.18 [2.23 - 5.01] | Calib: 3.18 [2.23 - 5.01] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ALLOA-ATHLETIC
  over_15  (B: 1.28 | L: 1.29 | M: 1.29) | Raw: 1.42 [1.24 - 1.69] | Calib: 1.33 [1.18 - 1.53] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (B: 1.80 | L: 1.86 | M: 1.83) | Raw: 2.20 [1.69 - 3.05] | Calib: 2.08 [1.63 - 2.85] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (B: 2.96 | L: 3.30 | M: 3.13) | Raw: 3.97 [2.67 - 6.69] | Calib: 3.97 [2.67 - 6.69] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ELGIN-CITY
  over_15  (B: 1.28 | L: 1.32 | M: 1.30) | Raw: 1.39 [1.23 - 1.61] | Calib: 1.30 [1.18 - 1.47] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (B: 1.83 | L: 1.93 | M: 1.88) | Raw: 2.10 [1.66 - 2.80] | Calib: 1.99 [1.60 - 2.62] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (B: 3.05 | L: 3.45 | M: 3.25) | Raw: 3.71 [2.60 - 5.84] | Calib: 3.71 [2.60 - 5.84] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: STIRLING-ALBION
  over_15  (B: 1.20 | L: 1.23 | M: 1.21) | Raw: 1.33 [1.19 - 1.51] | Calib: 1.25 [1.15 - 1.39] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (B: 1.62 | L: 1.67 | M: 1.65) | Raw: 1.92 [1.56 - 2.48] | Calib: 1.84 [1.51 - 2.34] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (B: 2.48 | L: 2.72 | M: 2.60) | Raw: 3.24 [2.34 - 4.88] | Calib: 3.24 [2.34 - 4.88] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ANNAN-ATHLETIC
  over_15  (B: 1.28 | L: 1.33 | M: 1.31) | Raw: 1.35 [1.21 - 1.56] | Calib: 1.27 [1.16 - 1.42] | Raw Stake:  0.00% | Calib:  0.19%
  over_25  (B: 1.89 | L: 1.99 | M: 1.94) | Raw: 2.00 [1.61 - 2.63] | Calib: 1.90 [1.55 - 2.47] | Raw Stake:  0.00% | Calib:  0.00%
  over_35  (B: 3.20 | L: 3.60 | M: 3.40) | Raw: 3.44 [2.45 - 5.34] | Calib: 3.44 [2.45 - 5.34] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: COVE-RANGERS
  over_15  (B: 1.30 | L: 1.32 | M: 1.31) | Raw: 1.33 [1.18 - 1.55] | Calib: 1.25 [1.14 - 1.42] | Raw Stake:  0.00% | Calib:  4.33%
  over_25  (B: 1.88 | L: 1.94 | M: 1.91) | Raw: 1.93 [1.52 - 2.61] | Calib: 1.84 [1.47 - 2.46] | Raw Stake:  0.00% | Calib:  0.26%
  over_35  (B: 3.20 | L: 3.45 | M: 3.33) | Raw: 3.24 [2.24 - 5.27] | Calib: 3.24 [2.24 - 5.27] | Raw Stake:  0.00% | Calib:  0.00%
=#

