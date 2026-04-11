# exp/market_scot/april/dev_loaders/r05_compute_shift_ppd.jl

# run this but dont - inlcued to get lsp help
include("./l05_compute_shift_ppd.jl")
#####


# from r02
exp_m1 = load_experiment_data_from_disk()


# from l05
best_match = load_same_large_experiment_model(exp_m1)



# ---- part 2 --- 
# compute the shift logic dic for the markets 

# 1. Prepare data
market_data = Data.prepare_market_data(ds)
selections_to_calibrate = [:btts_yes, :draw, :over_15, :over_25, :over_35]
# 2. Get the calibration dictionary
shift_dict = compute_market_shifts(exp_m1, ds, market_data.df, selections_to_calibrate)


# from r04
target_ppd = compute_todays_matches_pdds(data_store,  exp_m1, todays_matches)

calibrated_ppd = apply_market_shifts(target_ppd, shift_dict)


comparison_view = compare_calibrated_odds(target_ppd, calibrated_ppd, todays_matches)


selections_to_calibrate = [:over_15]
a = subset(comparison_view, :selection => ByRow(in(selections_to_calibrate)))

l1 = subset(a, :tournament_id => ByRow(isequal(56)))



# --- paper trades 
# 1. Your manual Betfair odds entry
# Live Market DataFrame extracted from Betfair exchange screenshots (11 Apr)
live_market = DataFrame(
    home_team = [
        "dumbarton", 
        "edinburgh-city-fc", 
        "elgin-city", 
        "forfar-athletic", 
        "the-spartans-fc",
        "east-fife", 
        "hamilton-academical", 
        "kelty-hearts-fc", 
        "queen-of-the-south", 
        "stenhousemuir"
    ],
    # Back Odds (Blue column) for Over 1.5 Goals
    live_odds_o15 = [
        1.23, # Dumbarton
        1.22, # Edinburgh
        1.24, # Elgin
        1.24, # Forfar
        1.29, # Spartans
        1.27, # East Fife
        1.26, # Hamilton
        1.30, # Kelty
        1.21, # QotS
        1.40  # Stenhousemuir
    ],
    # Back Odds (Blue column) for Over 2.5 Goals
    live_odds_o25 = [
        1.67, # Dumbarton
        1.66, # Edinburgh
        1.71, # Elgin
        1.71, # Forfar
        1.86, # Spartans
        1.80, # East Fife
        1.80, # Hamilton
        1.92, # Kelty
        1.63, # QotS
        2.20  # Stenhousemuir
    ]
)

# 2. Generate the bets (Assuming you have target_ppd and calibrated_ppd from earlier)
paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market, min_edge=0.0)

# 3. Save it down to CSV for your records!
using CSV
CSV.write("bets_2026_04_11.csv", paper_bets_df)


#=

julia> paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market, min_edge=0.0)
==========================================================================================
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)
==========================================================================================

MATCH: DUMBARTON
  over_15  (Live: 1.23) | Model: 1.19 [1.11 - 1.29] | Raw Stake:  0.00% | Calib:  6.81%
  over_25  (Live: 1.67) | Model: 1.63 [1.38 - 2.00] | Raw Stake:  0.00% | Calib:  0.64%

MATCH: EDINBURGH-CITY-FC
  over_15  (Live: 1.22) | Model: 1.19 [1.11 - 1.29] | Raw Stake:  0.00% | Calib:  4.89%
  over_25  (Live: 1.66) | Model: 1.62 [1.39 - 1.99] | Raw Stake:  0.00% | Calib:  0.51%

MATCH: ELGIN-CITY
  over_15  (Live: 1.24) | Model: 1.22 [1.13 - 1.33] | Raw Stake:  0.00% | Calib:  1.86%

MATCH: FORFAR-ATHLETIC
  over_15  (Live: 1.24) | Model: 1.23 [1.13 - 1.37] | Raw Stake:  0.00% | Calib:  0.11%

MATCH: EAST-FIFE
  over_15  (Live: 1.27) | Model: 1.22 [1.13 - 1.35] | Raw Stake:  0.00% | Calib:  5.98%
  over_25  (Live: 1.80) | Model: 1.75 [1.45 - 2.21] | Raw Stake:  0.00% | Calib:  0.72%

MATCH: KELTY-HEARTS-FC
  over_15  (Live: 1.30) | Model: 1.23 [1.14 - 1.35] | Raw Stake:  0.00% | Calib: 10.32%
  over_25  (Live: 1.92) | Model: 1.77 [1.48 - 2.21] | Raw Stake:  0.82% | Calib:  3.72%

==========================================================================================
10×10 DataFrame
 Row │ match_id  home_team          selection  live_odds  raw_mean_odds  calib_mean_odds  calib_bid_odds  calib_ask_odds  raw_stake_pct  calib_stake_pct 
     │ Int64     String             Symbol     Float64    Float64        Float64          Float64         Float64         Float64        Float64         
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 14035821  dumbarton          over_15         1.23          1.245            1.189           1.111           1.295           0.0              6.81
   2 │ 14035821  dumbarton          over_25         1.67          1.693            1.629           1.384           1.995           0.0              0.64
   3 │ 14035826  edinburgh-city-fc  over_15         1.22          1.243            1.188           1.112           1.291           0.0              4.89
   4 │ 14035826  edinburgh-city-fc  over_25         1.66          1.688            1.624           1.386           1.989           0.0              0.51
   5 │ 14035824  elgin-city         over_15         1.24          1.284            1.219           1.132           1.335           0.0              1.86
   6 │ 14035823  forfar-athletic    over_15         1.24          1.304            1.235           1.131           1.375           0.0              0.11
   7 │ 14035642  east-fife          over_15         1.27          1.291            1.225           1.131           1.353           0.0              5.98
   8 │ 14035642  east-fife          over_25         1.8           1.822            1.745           1.446           2.207           0.0              0.72
   9 │ 14035643  kelty-hearts-fc    over_15         1.3           1.301            1.232           1.139           1.355           0.0             10.32
  10 │ 14035643  kelty-hearts-fc    over_25         1.92          1.851            1.772           1.478           2.211           0.82             3.72


=#
