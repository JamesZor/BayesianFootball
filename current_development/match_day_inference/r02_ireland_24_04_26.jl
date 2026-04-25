#  current_development/match_day_inference/r02_ireland_23_04_26.


# File to include - not repl run 
include("./l00_main_utils.jl")


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())



save_dir::String = "./data/match_day/april/ireland/"

es = DSExperimentSettings(
  ds,
  "24_04_26",
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
exp = loaded_results[2]  # Grabbing the most recent one
exp1 = loaded_results[1]  # Grabbing the most recent one



todays_matches = fetch_todays_matches(ds)

target_ppd = compute_todays_matches_pdds(ds, exp, todays_matches)
target_ppd1 = compute_todays_matches_pdds(ds, exp1, todays_matches)



json_filepath = "/root/BayesianFootball/data/raw_odds_ireland_24_04_26.jsonl"
raw_live_market = load_live_market_jsonl(json_filepath)

selections_to_calibrate = [:over_15, :over_25,:under_25, :over_35, :under_35, :over_45, :under_45]
live_market_closing = filter_and_rename_live_markets(raw_live_market, selections_to_calibrate)
first(live_market_closing, 4)


paper_bets_df = generate_paper_bets(
    target_ppd, 
    target_ppd1, 
    todays_matches, 
    first(live_market_closing, 4), 
    min_edge=0.0
)





#=
# 24/04/26 

raw: HA_r model 
calib: monthly_R


============================================================================================================
 Placed:  EXACT KELLY STAKES (Edge: 0.0%)
============================================================================================================

MATCH: ST-PATRICKS-ATHLETIC  final score: 3-1
  under_35 (Live: 1.26) | Raw: 1.23 [1.12 - 1.40] | Calib: 1.23 [1.12 - 1.40] | Raw Stake:  2.13% | Calib:  2.15% - amount: 1

MATCH: DERRY-CITY final score: 1-0
  under_35 (Live: 1.26) | Raw: 1.23 [1.12 - 1.40] | Calib: 1.23 [1.13 - 1.39] | Raw Stake:  3.00% | Calib:  3.31% - amount: 1

MATCH: WATERFORD-FC final score: 1-0
  under_35 (Live: 1.37) | Raw: 1.31 [1.17 - 1.56] | Calib: 1.31 [1.17 - 1.56] | Raw Stake:  4.31% | Calib:  3.96% - amount: 1 

MATCH: SHELBOURNE: final score: 3-4 
  under_35 (Live: 1.32) | Raw: 1.28 [1.15 - 1.48] | Calib: 1.28 [1.15 - 1.48] | Raw Stake:  2.90% | Calib:  2.97% - amount 1.5



============================================================================================================
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)
============================================================================================================

MATCH: ST-PATRICKS-ATHLETIC
  over_15  (Live: 1.45) | Raw: 1.56 [1.35 - 1.92] | Calib: 1.57 [1.35 - 1.91] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 2.34) | Raw: 2.64 [2.00 - 3.87] | Calib: 2.64 [2.01 - 3.84] | Raw Stake:  0.00% | Calib:  0.00%
  under_25 (Live: 1.72) | Raw: 1.61 [1.35 - 2.00] | Calib: 1.61 [1.35 - 1.99] | Raw Stake:  3.34% | Calib:  3.46%
  over_35  (Live: 4.50) | Raw: 5.31 [3.50 - 9.62] | Calib: 5.31 [3.53 - 9.49] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.26) | Raw: 1.23 [1.12 - 1.40] | Calib: 1.23 [1.12 - 1.40] | Raw Stake:  2.13% | Calib:  2.15%

MATCH: DERRY-CITY
  over_15  (Live: 1.44) | Raw: 1.57 [1.35 - 1.88] | Calib: 1.57 [1.36 - 1.87] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 2.34) | Raw: 2.66 [2.00 - 3.71] | Calib: 2.67 [2.03 - 3.70] | Raw Stake:  0.00% | Calib:  0.00%
  under_25 (Live: 1.70) | Raw: 1.60 [1.37 - 2.00] | Calib: 1.60 [1.37 - 1.97] | Raw Stake:  3.13% | Calib:  3.41%
  over_35  (Live: 4.50) | Raw: 5.38 [3.52 - 9.01] | Calib: 5.41 [3.58 - 8.99] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.26) | Raw: 1.23 [1.12 - 1.40] | Calib: 1.23 [1.13 - 1.39] | Raw Stake:  3.00% | Calib:  3.31%

MATCH: WATERFORD-FC
  over_15  (Live: 1.32) | Raw: 1.44 [1.25 - 1.70] | Calib: 1.44 [1.25 - 1.70] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.99) | Raw: 2.27 [1.73 - 3.12] | Calib: 2.26 [1.73 - 3.11] | Raw Stake:  0.00% | Calib:  0.00%
  under_25 (Live: 1.96) | Raw: 1.79 [1.47 - 2.38] | Calib: 1.79 [1.47 - 2.37] | Raw Stake:  4.12% | Calib:  3.90%
  over_35  (Live: 3.50) | Raw: 4.20 [2.78 - 6.96] | Calib: 4.18 [2.79 - 6.91] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.37) | Raw: 1.31 [1.17 - 1.56] | Calib: 1.31 [1.17 - 1.56] | Raw Stake:  4.31% | Calib:  3.96%

MATCH: SHELBOURNE
  over_15  (Live: 1.37) | Raw: 1.48 [1.29 - 1.75] | Calib: 1.48 [1.30 - 1.75] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 2.12) | Raw: 2.39 [1.83 - 3.28] | Calib: 2.39 [1.85 - 3.27] | Raw Stake:  0.00% | Calib:  0.00%
  under_25 (Live: 1.86) | Raw: 1.72 [1.44 - 2.20] | Calib: 1.72 [1.44 - 2.17] | Raw Stake:  3.75% | Calib:  3.95%
  over_35  (Live: 3.95) | Raw: 4.57 [3.07 - 7.49] | Calib: 4.56 [3.10 - 7.47] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.32) | Raw: 1.28 [1.15 - 1.48] | Calib: 1.28 [1.15 - 1.48] | Raw Stake:  2.90% | Calib:  2.97%

============================================================================================================
20×12 DataFrame
 Row │ match_id  home_team             selection  live_odds  raw_mean_odds  raw_bid_odds  raw_ask_odds  calib_mean_odds  calib_bid_odds  calib_ask_odds  raw_stake_pct  calib_stake_pct 
     │ Int32     String                Symbol     Float64    Float64        Float64       Float64       Float64          Float64         Float64         Float64        Float64         
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 15238069  st-patricks-athletic  over_15         1.45          1.565         1.345         1.917            1.567           1.349           1.912           0.0              0.0
   2 │ 15238069  st-patricks-athletic  over_25         2.34          2.637         1.997         3.867            2.642           2.006           3.844           0.0              0.0
   3 │ 15238069  st-patricks-athletic  under_25        1.72          1.611         1.349         2.003            1.609           1.352           1.994           3.34             3.46
   4 │ 15238069  st-patricks-athletic  over_35         4.5           5.306         3.501         9.62             5.307           3.525           9.489           0.0              0.0
   5 │ 15238069  st-patricks-athletic  under_35        1.26          1.232         1.116         1.4              1.232           1.118           1.396           2.13             2.15
   6 │ 15238067  derry-city            over_15         1.44          1.571         1.347         1.876            1.575           1.355           1.871           0.0              0.0
   7 │ 15238067  derry-city            over_25         2.34          2.66          2.004         3.715            2.67            2.027           3.696           0.0              0.0
   8 │ 15238067  derry-city            under_25        1.7           1.602         1.368         1.996            1.599           1.371           1.974           3.13             3.41
   9 │ 15238067  derry-city            over_35         4.5           5.381         3.525         9.014            5.409           3.584           8.988           0.0              0.0
  10 │ 15238067  derry-city            under_35        1.26          1.228         1.125         1.396            1.227           1.125           1.387           3.0              3.31
  11 │ 15238070  waterford-fc          over_15         1.32          1.443         1.251         1.704            1.442           1.253           1.7             0.0              0.0
  12 │ 15238070  waterford-fc          over_25         1.99          2.266         1.727         3.119            2.26            1.732           3.107           0.0              0.0
  13 │ 15238070  waterford-fc          under_25        1.96          1.79          1.472         2.376            1.794           1.475           2.367           4.12             3.9
  14 │ 15238070  waterford-fc          over_35         3.5           4.203         2.78          6.955            4.181           2.789           6.907           0.0              0.0
  15 │ 15238070  waterford-fc          under_35        1.37          1.312         1.168         1.562            1.314           1.169           1.559           4.31             3.96
  16 │ 15238068  shelbourne            over_15         1.37          1.483         1.288         1.749            1.485           1.296           1.749           0.0              0.0
  17 │ 15238068  shelbourne            over_25         2.12          2.388         1.834         3.276            2.391           1.851           3.27            0.0              0.0
  18 │ 15238068  shelbourne            under_25        1.86          1.72          1.439         2.199            1.719           1.44            2.175           3.75             3.95
  19 │ 15238068  shelbourne            over_35         3.95          4.566         3.069         7.49             4.565           3.096           7.47            0.0              0.0
  20 │ 15238068  shelbourne            under_35        1.32          1.28          1.154         1.483            1.281           1.155           1.477           2.9              2.97
=#



