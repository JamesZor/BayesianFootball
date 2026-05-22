# dev_repl/match_day_inference/r01



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
  "17_04_26",
  save_dir,
  find_current_cv_parameters(ds)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)

# ========================================
#  Stage 2 - Running inference 
# ========================================


saved_folders = BayesianFootball.Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)
exp = loaded_results[end]  # Grabbing the most recent one

# ---- 2. Fetch today's matches.
todays_matches = fetch_todays_matches(ds)

# ---- 3. Find Best Model Duration
# explicitly pass `dir=save_dir` so it searches your current folder
#
save_dir_dev::String = "./data/exp/ablation_study"
best_match = load_same_large_experiment_model(exp, dir=save_dir_dev) 


# ---- 4. Compute Calibration Shifts
selections_to_calibrate = [:draw, :over_15, :over_25, :under_35]
shift_dict = compute_market_shifts(best_match, ds, selections_to_calibrate)

#=
Dict{Symbol, Float64} with 4 entries:
  :under_35 => -0.077246
  :draw     => 0.0520978
  :over_15  => 0.269648
  :over_25  => 0.102343
=#

# ---- 5. Compute PPDs
target_ppd = compute_todays_matches_pdds(ds, exp, todays_matches)
calibrated_ppd = apply_market_shifts(target_ppd, shift_dict)

comparison_view = compare_calibrated_odds(target_ppd, calibrated_ppd, todays_matches)

json_filepath = "/root/BayesianFootball/data/raw_odds_history_scotland.jsonl"
raw_live_market = load_live_market_jsonl(json_filepath)


# 2. Define exactly what you want to extract
selections_to_calibrate = [:draw, :over_15, :over_25, :under_35]

# 3. Filter and Rename
live_market_closing = filter_and_rename_live_markets(raw_live_market, selections_to_calibrate)



# 2. Run the paper bets exactly as before
paper_bets_df = generate_paper_bets(
    target_ppd, 
    calibrated_ppd, 
    todays_matches, 
    first(live_market_closing, 10), 
    min_edge=0.0
)


#=
============================================================================================================
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)
============================================================================================================

MATCH: ALLOA-ATHLETIC
  draw     (Live: 3.80) | Raw: 4.57 [3.74 - 6.07] | Calib: 4.39 [3.60 - 5.82] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.24) | Raw: 1.31 [1.16 - 1.51] | Calib: 1.24 [1.13 - 1.39] | Raw Stake:  0.00% | Calib:  0.05%
  over_25  (Live: 1.73) | Raw: 1.87 [1.48 - 2.49] | Calib: 1.78 [1.43 - 2.35] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.47) | Raw: 1.48 [1.25 - 1.87] | Calib: 1.52 [1.27 - 1.94] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: COVE-RANGERS
  draw     (Live: 3.35) | Raw: 3.61 [3.14 - 4.23] | Calib: 3.48 [3.04 - 4.06] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.35) | Raw: 1.50 [1.28 - 1.81] | Calib: 1.38 [1.22 - 1.62] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 2.06) | Raw: 2.43 [1.82 - 3.47] | Calib: 2.29 [1.74 - 3.23] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.32) | Raw: 1.27 [1.14 - 1.50] | Calib: 1.29 [1.15 - 1.54] | Raw Stake:  3.73% | Calib:  1.13%

MATCH: INVERNESS-CALEDONIAN-THISTLE
  draw     (Live: 6.20) | Raw: 4.48 [3.51 - 6.29] | Calib: 4.31 [3.38 - 6.02] | Raw Stake:  5.60% | Calib:  6.72%
  over_15  (Live: 1.21) | Raw: 1.44 [1.24 - 1.74] | Calib: 1.34 [1.19 - 1.56] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.61) | Raw: 2.23 [1.69 - 3.22] | Calib: 2.12 [1.63 - 3.00] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.55) | Raw: 1.32 [1.16 - 1.60] | Calib: 1.35 [1.17 - 1.65] | Raw Stake: 21.76% | Calib: 17.49%

MATCH: MONTROSE
  draw     (Live: 3.70) | Raw: 4.31 [3.69 - 5.31] | Calib: 4.15 [3.56 - 5.09] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.23) | Raw: 1.28 [1.15 - 1.48] | Calib: 1.21 [1.11 - 1.37] | Raw Stake:  0.00% | Calib:  0.97%
  over_25  (Live: 1.68) | Raw: 1.79 [1.43 - 2.40] | Calib: 1.71 [1.39 - 2.27] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.51) | Raw: 1.53 [1.27 - 1.98] | Calib: 1.57 [1.30 - 2.06] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: PETERHEAD
  draw     (Live: 3.60) | Raw: 4.21 [3.65 - 4.98] | Calib: 4.05 [3.51 - 4.78] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.25) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  3.82%
  over_25  (Live: 1.76) | Raw: 1.79 [1.43 - 2.37] | Calib: 1.72 [1.39 - 2.24] | Raw Stake:  0.00% | Calib:  0.45%
  under_35 (Live: 1.45) | Raw: 1.53 [1.28 - 1.96] | Calib: 1.57 [1.30 - 2.04] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ANNAN-ATHLETIC
  draw     (Live: 3.60) | Raw: 4.12 [3.60 - 4.81] | Calib: 3.96 [3.47 - 4.62] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.25) | Raw: 1.30 [1.16 - 1.50] | Calib: 1.23 [1.12 - 1.38] | Raw Stake:  0.00% | Calib:  0.99%
  over_25  (Live: 1.75) | Raw: 1.86 [1.47 - 2.46] | Calib: 1.77 [1.43 - 2.32] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.49) | Raw: 1.48 [1.26 - 1.88] | Calib: 1.52 [1.28 - 1.95] | Raw Stake:  0.03% | Calib:  0.00%

MATCH: CLYDE-FC
  draw     (Live: 4.00) | Raw: 4.25 [3.68 - 5.10] | Calib: 4.09 [3.54 - 4.89] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.22) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  0.10%
  over_25  (Live: 1.64) | Raw: 1.79 [1.44 - 2.37] | Calib: 1.72 [1.39 - 2.24] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.54) | Raw: 1.53 [1.28 - 1.96] | Calib: 1.57 [1.30 - 2.03] | Raw Stake:  0.08% | Calib:  0.00%

MATCH: EAST-KILBRIDE
  draw     (Live: 9.60) | Raw: 5.91 [4.58 - 8.73] | Calib: 5.66 [4.39 - 8.34] | Raw Stake:  6.11% | Calib:  6.97%
  over_15  (Live: 1.10) | Raw: 1.14 [1.06 - 1.25] | Calib: 1.10 [1.05 - 1.19] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.32) | Raw: 1.39 [1.19 - 1.73] | Calib: 1.35 [1.17 - 1.66] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 2.08) | Raw: 2.11 [1.56 - 3.25] | Calib: 2.19 [1.61 - 3.43] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: STIRLING-ALBION
  draw     (Live: 3.80) | Raw: 4.09 [3.56 - 4.74] | Calib: 3.94 [3.43 - 4.55] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.24) | Raw: 1.30 [1.16 - 1.51] | Calib: 1.23 [1.13 - 1.39] | Raw Stake:  0.00% | Calib:  0.19%
  over_25  (Live: 1.74) | Raw: 1.86 [1.48 - 2.49] | Calib: 1.77 [1.43 - 2.34] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.48) | Raw: 1.48 [1.25 - 1.87] | Calib: 1.52 [1.27 - 1.94] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: STRANRAER
  draw     (Live: 3.20) | Raw: 3.47 [3.02 - 4.03] | Calib: 3.35 [2.92 - 3.87] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.35) | Raw: 1.57 [1.34 - 1.92] | Calib: 1.44 [1.26 - 1.70] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 2.08) | Raw: 2.66 [1.99 - 3.85] | Calib: 2.50 [1.89 - 3.58] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.30) | Raw: 1.23 [1.12 - 1.40] | Calib: 1.25 [1.13 - 1.43] | Raw Stake:  9.28% | Calib:  5.27%

============================================================================================================
40×12 DataFrame
 Row │ match_id  home_team                     selection  live_odds  raw_mean_odds    calib_mean_odds    raw_stake_pct  calib_stake_pct 
     │ Int32     String                        Symbol     Float64    Float64          Float64            Float64        Float64         
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 14035646  alloa-athletic                draw            3.8           4.566              4.386             0.0              0.0
   2 │ 14035646  alloa-athletic                over_15         1.24          1.308              1.236             0.0              0.05
   3 │ 14035646  alloa-athletic                over_25         1.73          1.867              1.785             0.0              0.0
   4 │ 14035646  alloa-athletic                under_35        1.47          1.478              1.516             0.0              0.0
   5 │ 14035647  cove-rangers                  draw            3.35          3.61               3.478             0.0              0.0
   6 │ 14035647  cove-rangers                  over_15         1.35          1.499              1.383             0.0              0.0
   7 │ 14035647  cove-rangers                  over_25         2.06          2.429              2.294             0.0              0.0
   8 │ 14035647  cove-rangers                  under_35        1.32          1.273              1.295             3.73             1.13
   9 │ 14035648  inverness-caledonian-thistle  draw            6.2           4.481              4.306             5.6              6.72
  10 │ 14035648  inverness-caledonian-thistle  over_15         1.21          1.437              1.335             0.0              0.0
  11 │ 14035648  inverness-caledonian-thistle  over_25         1.61          2.234              2.118             0.0              0.0
  12 │ 14035648  inverness-caledonian-thistle  under_35        1.55          1.325              1.35             21.76            17.49
  13 │ 14035650  montrose                      draw            3.7           4.314              4.147             0.0              0.0
  14 │ 14035650  montrose                      over_15         1.23          1.279              1.214             0.0              0.97
  15 │ 14035650  montrose                      over_25         1.68          1.786              1.711             0.0              0.0
  16 │ 14035650  montrose                      under_35        1.51          1.531              1.573             0.0              0.0
  17 │ 14035651  peterhead                     draw            3.6           4.214              4.051             0.0              0.0
  18 │ 14035651  peterhead                     over_15         1.25          1.28               1.215             0.0              3.82
  19 │ 14035651  peterhead                     over_25         1.76          1.791              1.716             0.0              0.45
  20 │ 14035651  peterhead                     under_35        1.45          1.527              1.567             0.0              0.0
  21 │ 14032721  annan-athletic                draw            3.6           4.122              3.964             0.0              0.0
  22 │ 14032721  annan-athletic                over_15         1.25          1.303              1.233             0.0              0.99
  23 │ 14032721  annan-athletic                over_25         1.75          1.855              1.774             0.0              0.0
  24 │ 14032721  annan-athletic                under_35        1.49          1.484              1.521             0.03             0.0
  25 │ 14032715  clyde-fc                      draw            4.0           4.25               4.085             0.0              0.0
  26 │ 14032715  clyde-fc                      over_15         1.22          1.28               1.215             0.0              0.1
  27 │ 14032715  clyde-fc                      over_25         1.64          1.79               1.715             0.0              0.0
  28 │ 14032715  clyde-fc                      under_35        1.54          1.528              1.569             0.08             0.0
  29 │ 14032707  east-kilbride                 draw            9.6           5.912              5.665             6.11             6.97
  30 │ 14032707  east-kilbride                 over_15         1.1           1.136              1.104             0.0              0.0
  31 │ 14032707  east-kilbride                 over_25         1.32          1.39               1.353             0.0              0.0
  32 │ 14032707  east-kilbride                 under_35        2.08          2.108              2.193             0.0              0.0
  33 │ 14032717  stirling-albion               draw            3.8           4.092              3.936             0.0              0.0
  34 │ 14032717  stirling-albion               over_15         1.24          1.303              1.233             0.0              0.19
  35 │ 14032717  stirling-albion               over_25         1.74          1.856              1.775             0.0              0.0
  36 │ 14032717  stirling-albion               under_35        1.48          1.484              1.521             0.0              0.0
  37 │ 14032718  stranraer                     draw            3.2           3.474              3.349             0.0              0.0
  38 │ 14032718  stranraer                     over_15         1.35          1.572              1.439             0.0              0.0
  39 │ 14032718  stranraer                     over_25         2.08          2.657              2.499             0.0              0.0
  40 │ 14032718  stranraer                     under_35        1.3           1.23               1.248             9.28             5.27




============================================================================================================
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)
============================================================================================================

MATCH: ALLOA-ATHLETIC
  over_15  (Live: 1.24) | Raw: 1.31 [1.16 - 1.51] | Calib: 1.24 [1.13 - 1.39] | Raw Stake:  0.00% | Calib:  0.05%

MATCH: COVE-RANGERS
  under_35 (Live: 1.32) | Raw: 1.27 [1.14 - 1.50] | Calib: 1.29 [1.15 - 1.54] | Raw Stake:  3.73% | Calib:  1.13%

MATCH: INVERNESS-CALEDONIAN-THISTLE
  under_35 (Live: 1.55) | Raw: 1.32 [1.16 - 1.60] | Calib: 1.35 [1.17 - 1.65] | Raw Stake: 21.76% | Calib: 17.49%

MATCH: MONTROSE
  over_15  (Live: 1.23) | Raw: 1.28 [1.15 - 1.48] | Calib: 1.21 [1.11 - 1.37] | Raw Stake:  0.00% | Calib:  0.97%

MATCH: PETERHEAD
  over_15  (Live: 1.25) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  3.82%
  over_25  (Live: 1.76) | Raw: 1.79 [1.43 - 2.37] | Calib: 1.72 [1.39 - 2.24] | Raw Stake:  0.00% | Calib:  0.45%

MATCH: ANNAN-ATHLETIC
  over_15  (Live: 1.25) | Raw: 1.30 [1.16 - 1.50] | Calib: 1.23 [1.12 - 1.38] | Raw Stake:  0.00% | Calib:  0.99%
  under_35 (Live: 1.49) | Raw: 1.48 [1.26 - 1.88] | Calib: 1.52 [1.28 - 1.95] | Raw Stake:  0.03% | Calib:  0.00%

MATCH: CLYDE-FC
  over_15  (Live: 1.22) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  0.10%
  under_35 (Live: 1.54) | Raw: 1.53 [1.28 - 1.96] | Calib: 1.57 [1.30 - 2.03] | Raw Stake:  0.08% | Calib:  0.00%


MATCH: STIRLING-ALBION
  over_15  (Live: 1.24) | Raw: 1.30 [1.16 - 1.51] | Calib: 1.23 [1.13 - 1.39] | Raw Stake:  0.00% | Calib:  0.19%

MATCH: STRANRAER
  under_35 (Live: 1.30) | Raw: 1.23 [1.12 - 1.40] | Calib: 1.25 [1.13 - 1.43] | Raw Stake:  9.28% | Calib:  5.27%



# --- place 
MATCH: MONTROSE
  over_15  (Live: 1.23) | Raw: 1.28 [1.15 - 1.48] | Calib: 1.21 [1.11 - 1.37] | Raw Stake:  0.00% | Calib:  0.97%  --- 1 

MATCH: PETERHEAD
  over_15  (Live: 1.25) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  3.82%  --- 3

MATCH: ANNAN-ATHLETIC
  over_15  (Live: 1.25) | Raw: 1.30 [1.16 - 1.50] | Calib: 1.23 [1.12 - 1.38] | Raw Stake:  0.00% | Calib:  0.99%  --- 1 

MATCH: STIRLING-ALBION
  over_15  (Live: 1.26) | Raw: 1.30 [1.16 - 1.51] | Calib: 1.23 [1.13 - 1.39] | Raw Stake:  0.00% | Calib:  0.99% ---  1

=#

last(live_market_closing, 10) 
# 2. Run the paper bets exactly as before
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

MATCH: ALLOA-ATHLETIC
  draw     (Live: 3.90) | Raw: 4.57 [3.74 - 6.07] | Calib: 4.39 [3.60 - 5.82] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.23) | Raw: 1.31 [1.16 - 1.51] | Calib: 1.24 [1.13 - 1.39] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.68) | Raw: 1.87 [1.48 - 2.49] | Calib: 1.78 [1.43 - 2.35] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.53) | Raw: 1.48 [1.25 - 1.87] | Calib: 1.52 [1.27 - 1.94] | Raw Stake:  1.47% | Calib:  0.12%

MATCH: COVE-RANGERS
  draw     (Live: 3.40) | Raw: 3.61 [3.14 - 4.23] | Calib: 3.48 [3.04 - 4.06] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.35) | Raw: 1.50 [1.28 - 1.81] | Calib: 1.38 [1.22 - 1.62] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 2.04) | Raw: 2.43 [1.82 - 3.47] | Calib: 2.29 [1.74 - 3.23] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.32) | Raw: 1.27 [1.14 - 1.50] | Calib: 1.29 [1.15 - 1.54] | Raw Stake:  3.73% | Calib:  1.13%

MATCH: INVERNESS-CALEDONIAN-THISTLE
  draw     (Live: 6.20) | Raw: 4.48 [3.51 - 6.29] | Calib: 4.31 [3.38 - 6.02] | Raw Stake:  5.60% | Calib:  6.72%
  over_15  (Live: 1.21) | Raw: 1.44 [1.24 - 1.74] | Calib: 1.34 [1.19 - 1.56] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.64) | Raw: 2.23 [1.69 - 3.22] | Calib: 2.12 [1.63 - 3.00] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.55) | Raw: 1.32 [1.16 - 1.60] | Calib: 1.35 [1.17 - 1.65] | Raw Stake: 21.76% | Calib: 17.49%

MATCH: MONTROSE
  draw     (Live: 3.70) | Raw: 4.31 [3.69 - 5.31] | Calib: 4.15 [3.56 - 5.09] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.23) | Raw: 1.28 [1.15 - 1.48] | Calib: 1.21 [1.11 - 1.37] | Raw Stake:  0.00% | Calib:  0.97%
  over_25  (Live: 1.68) | Raw: 1.79 [1.43 - 2.40] | Calib: 1.71 [1.39 - 2.27] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.52) | Raw: 1.53 [1.27 - 1.98] | Calib: 1.57 [1.30 - 2.06] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: PETERHEAD
  draw     (Live: 3.65) | Raw: 4.21 [3.65 - 4.98] | Calib: 4.05 [3.51 - 4.78] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.26) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  5.69%
  over_25  (Live: 1.78) | Raw: 1.79 [1.43 - 2.37] | Calib: 1.72 [1.39 - 2.24] | Raw Stake:  0.00% | Calib:  0.90%
  under_35 (Live: 1.46) | Raw: 1.53 [1.28 - 1.96] | Calib: 1.57 [1.30 - 2.04] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: ANNAN-ATHLETIC
  draw     (Live: 3.65) | Raw: 4.12 [3.60 - 4.81] | Calib: 3.96 [3.47 - 4.62] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.25) | Raw: 1.30 [1.16 - 1.50] | Calib: 1.23 [1.12 - 1.38] | Raw Stake:  0.00% | Calib:  0.99%
  over_25  (Live: 1.76) | Raw: 1.86 [1.47 - 2.46] | Calib: 1.77 [1.43 - 2.32] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.49) | Raw: 1.48 [1.26 - 1.88] | Calib: 1.52 [1.28 - 1.95] | Raw Stake:  0.03% | Calib:  0.00%

MATCH: CLYDE-FC
  draw     (Live: 4.00) | Raw: 4.25 [3.68 - 5.10] | Calib: 4.09 [3.54 - 4.89] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.22) | Raw: 1.28 [1.15 - 1.47] | Calib: 1.22 [1.11 - 1.36] | Raw Stake:  0.00% | Calib:  0.10%
  over_25  (Live: 1.64) | Raw: 1.79 [1.44 - 2.37] | Calib: 1.72 [1.39 - 2.24] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.54) | Raw: 1.53 [1.28 - 1.96] | Calib: 1.57 [1.30 - 2.03] | Raw Stake:  0.08% | Calib:  0.00%

MATCH: EAST-KILBRIDE
  draw     (Live: 9.80) | Raw: 5.91 [4.58 - 8.73] | Calib: 5.66 [4.39 - 8.34] | Raw Stake:  6.36% | Calib:  7.22%
  over_15  (Live: 1.10) | Raw: 1.14 [1.06 - 1.25] | Calib: 1.10 [1.05 - 1.19] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 1.32) | Raw: 1.39 [1.19 - 1.73] | Calib: 1.35 [1.17 - 1.66] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 2.08) | Raw: 2.11 [1.56 - 3.25] | Calib: 2.19 [1.61 - 3.43] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: STIRLING-ALBION
  draw     (Live: 3.85) | Raw: 4.09 [3.56 - 4.74] | Calib: 3.94 [3.43 - 4.55] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.25) | Raw: 1.30 [1.16 - 1.51] | Calib: 1.23 [1.13 - 1.39] | Raw Stake:  0.00% | Calib:  0.99%
  over_25  (Live: 1.72) | Raw: 1.86 [1.48 - 2.49] | Calib: 1.77 [1.43 - 2.34] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.48) | Raw: 1.48 [1.25 - 1.87] | Calib: 1.52 [1.27 - 1.94] | Raw Stake:  0.00% | Calib:  0.00%

MATCH: STRANRAER
  draw     (Live: 3.20) | Raw: 3.47 [3.02 - 4.03] | Calib: 3.35 [2.92 - 3.87] | Raw Stake:  0.00% | Calib:  0.00%
  over_15  (Live: 1.35) | Raw: 1.57 [1.34 - 1.92] | Calib: 1.44 [1.26 - 1.70] | Raw Stake:  0.00% | Calib:  0.00%
  over_25  (Live: 2.08) | Raw: 2.66 [1.99 - 3.85] | Calib: 2.50 [1.89 - 3.58] | Raw Stake:  0.00% | Calib:  0.00%
  under_35 (Live: 1.31) | Raw: 1.23 [1.12 - 1.40] | Calib: 1.25 [1.13 - 1.43] | Raw Stake: 11.11% | Calib:  6.87%

============================================================================================================
40×12 DataFrame
 Row │ match_id  home_team                     selection  live_odds  raw_mean_odds  raw_bid_odds  raw_ask_odds  calib_mean_odds  calib_bid_odds  calib_ask_odds  raw_stake_pct  calib_stake_pct 
     │ Int32     String                        Symbol     Float64    Float64        Float64       Float64       Float64          Float64         Float64         Float64        Float64         
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 14035646  alloa-athletic                draw            3.9           4.566         3.743         6.073            4.386           3.604           5.815           0.0              0.0
   2 │ 14035646  alloa-athletic                over_15         1.23          1.308         1.165         1.509            1.236           1.126           1.389           0.0              0.0
   3 │ 14035646  alloa-athletic                over_25         1.68          1.867         1.481         2.491            1.785           1.434           2.346           0.0              0.0
   4 │ 14035646  alloa-athletic                under_35        1.53          1.478         1.254         1.867            1.516           1.275           1.936           1.47             0.12
   5 │ 14035647  cove-rangers                  draw            3.4           3.61          3.144         4.228            3.478           3.035           4.064           0.0              0.0
   6 │ 14035647  cove-rangers                  over_15         1.35          1.499         1.283         1.807            1.383           1.216           1.616           0.0              0.0
   7 │ 14035647  cove-rangers                  over_25         2.04          2.429         1.816         3.466            2.294           1.736           3.226           0.0              0.0
   8 │ 14035647  cove-rangers                  under_35        1.32          1.273         1.14          1.497            1.295           1.151           1.537           3.73             1.13
   9 │ 14035648  inverness-caledonian-thistle  draw            6.2           4.481         3.506         6.29             4.306           3.379           6.022           5.6              6.72
  10 │ 14035648  inverness-caledonian-thistle  over_15         1.21          1.437         1.242         1.736            1.335           1.185           1.562           0.0              0.0
  11 │ 14035648  inverness-caledonian-thistle  over_25         1.64          2.234         1.693         3.216            2.118           1.626           3.0             0.0              0.0
  12 │ 14035648  inverness-caledonian-thistle  under_35        1.55          1.325         1.16          1.598            1.35            1.173           1.646          21.76            17.49
  13 │ 14035650  montrose                      draw            3.7           4.314         3.695         5.309            4.147           3.558           5.09            0.0              0.0
  14 │ 14035650  montrose                      over_15         1.23          1.279         1.145         1.48             1.214           1.111           1.366           0.0              0.97
  15 │ 14035650  montrose                      over_25         1.68          1.786         1.428         2.403            1.711           1.386           2.266           0.0              0.0
  16 │ 14035650  montrose                      under_35        1.52          1.531         1.273         1.982            1.573           1.295           2.061           0.0              0.0
  17 │ 14035651  peterhead                     draw            3.65          4.214         3.649         4.984            4.051           3.514           4.782           0.0              0.0
  18 │ 14035651  peterhead                     over_15         1.26          1.28          1.148         1.473            1.215           1.113           1.361           0.0              5.69
  19 │ 14035651  peterhead                     over_25         1.78          1.791         1.434         2.374            1.716           1.392           2.24            0.0              0.9
  20 │ 14035651  peterhead                     under_35        1.46          1.527         1.279         1.964            1.567           1.302           2.041           0.0              0.0
  21 │ 14032721  annan-athletic                draw            3.65          4.122         3.597         4.811            3.964           3.465           4.618           0.0              0.0
  22 │ 14032721  annan-athletic                over_15         1.25          1.303         1.162         1.5              1.233           1.124           1.382           0.0              0.99
  23 │ 14032721  annan-athletic                over_25         1.76          1.855         1.473         2.465            1.774           1.427           2.322           0.0              0.0
  24 │ 14032721  annan-athletic                under_35        1.49          1.484         1.259         1.88             1.521           1.28            1.951           0.03             0.0
  25 │ 14032715  clyde-fc                      draw            4.0           4.25          3.678         5.102            4.085           3.542           4.894           0.0              0.0
  26 │ 14032715  clyde-fc                      over_15         1.22          1.28          1.148         1.473            1.215           1.113           1.361           0.0              0.1
  27 │ 14032715  clyde-fc                      over_25         1.64          1.79          1.436         2.374            1.715           1.394           2.24            0.0              0.0
  28 │ 14032715  clyde-fc                      under_35        1.54          1.528         1.278         1.956            1.569           1.301           2.032           0.08             0.0
  29 │ 14032707  east-kilbride                 draw            9.8           5.912         4.576         8.728            5.665           4.395           8.336           6.36             7.22
  30 │ 14032707  east-kilbride                 over_15         1.1           1.136         1.06          1.255            1.104           1.046           1.194           0.0              0.0
  31 │ 14032707  east-kilbride                 over_25         1.32          1.39          1.185         1.73             1.353           1.167           1.659           0.0              0.0
  32 │ 14032707  east-kilbride                 under_35        2.08          2.108         1.562         3.246            2.193           1.607           3.426           0.0              0.0
  33 │ 14032717  stirling-albion               draw            3.85          4.092         3.565         4.744            3.936           3.434           4.554           0.0              0.0
  34 │ 14032717  stirling-albion               over_15         1.25          1.303         1.164         1.507            1.233           1.125           1.387           0.0              0.99
  35 │ 14032717  stirling-albion               over_25         1.72          1.856         1.481         2.488            1.775           1.434           2.343           0.0              0.0
  36 │ 14032717  stirling-albion               under_35        1.48          1.484         1.254         1.868            1.521           1.274           1.938           0.0              0.0
  37 │ 14032718  stranraer                     draw            3.2           3.474         3.023         4.025            3.349           2.92            3.872           0.0              0.0
  38 │ 14032718  stranraer                     over_15         1.35          1.572         1.343         1.918            1.439           1.262           1.701           0.0              0.0
  39 │ 14032718  stranraer                     over_25         2.08          2.657         1.99          3.855            2.499           1.894           3.577           0.0              0.0
  40 │ 14032718  stranraer                     under_35        1.31          1.23          1.117         1.403            1.248           1.126           1.435          11.11             6.87
=#

