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


# 3 mins to 

# CLOSING LINE Market DataFrame extracted from Betfair (11 Apr - minutes to kickoff)
live_market_closing = DataFrame(
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
        1.21, # Dumbarton
        1.16, # Edinburgh
        1.22, # Elgin
        1.22, # Forfar
        1.29, # Spartans
        1.26, # East Fife
        1.27, # Hamilton
        1.29, # Kelty
        1.22, # QotS
        1.42  # Stenhousemuir
    ],
    # Back Odds (Blue column) for Over 2.5 Goals
    live_odds_o25 = [
        1.65, # Dumbarton
        1.50, # Edinburgh
        1.67, # Elgin
        1.69, # Forfar
        1.86, # Spartans
        1.76, # East Fife
        1.82, # Hamilton
        1.89, # Kelty
        1.68, # QotS
        2.26  # Stenhousemuir
    ]
)

# 2. Generate the bets (Assuming you have target_ppd and calibrated_ppd from earlier)
paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market_closing, min_edge=0.0)

# 3. Save it down to CSV for your records!

#=
julia> paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market, min_edge=0.0)
============================================================================================================
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)
============================================================================================================

MATCH: DUMBARTON
  over_15  (Live: 1.23) | Raw: 1.25 [1.14 - 1.38] | Calib: 1.19 [1.11 - 1.29] | Raw Stake:  0.00% | Calib:  6.81%
  over_25  (Live: 1.67) | Raw: 1.69 [1.42 - 2.10] | Calib: 1.63 [1.38 - 2.00] | Raw Stake:  0.00% | Calib:  0.64%

MATCH: EDINBURGH-CITY-FC
  over_15  (Live: 1.22) | Raw: 1.24 [1.15 - 1.38] | Calib: 1.19 [1.11 - 1.29] | Raw Stake:  0.00% | Calib:  4.89%
  over_25  (Live: 1.66) | Raw: 1.69 [1.43 - 2.09] | Calib: 1.62 [1.39 - 1.99] | Raw Stake:  0.00% | Calib:  0.51%

MATCH: ELGIN-CITY
  over_15  (Live: 1.24) | Raw: 1.28 [1.17 - 1.44] | Calib: 1.22 [1.13 - 1.33] | Raw Stake:  0.00% | Calib:  1.86%

MATCH: FORFAR-ATHLETIC
  over_15  (Live: 1.24) | Raw: 1.30 [1.17 - 1.49] | Calib: 1.23 [1.13 - 1.37] | Raw Stake:  0.00% | Calib:  0.11%

MATCH: EAST-FIFE
  over_15  (Live: 1.27) | Raw: 1.29 [1.17 - 1.46] | Calib: 1.22 [1.13 - 1.35] | Raw Stake:  0.00% | Calib:  5.98%
  over_25  (Live: 1.80) | Raw: 1.82 [1.49 - 2.33] | Calib: 1.75 [1.45 - 2.21] | Raw Stake:  0.00% | Calib:  0.72%

MATCH: KELTY-HEARTS-FC
  over_15  (Live: 1.30) | Raw: 1.30 [1.18 - 1.46] | Calib: 1.23 [1.14 - 1.35] | Raw Stake:  0.00% | Calib: 10.32%
  over_25  (Live: 1.92) | Raw: 1.85 [1.53 - 2.34] | Calib: 1.77 [1.48 - 2.21] | Raw Stake:  0.82% | Calib:  3.72%

============================================================================================================
10×12 DataFrame
 Row │ match_id  home_team          selection  live_odds  raw_mean_odds  raw_bid_odds  raw_ask_odds  calib_mean_odds  calib_bid_odds  calib_ask_odds  raw_stake_pct  calib_stake_pct 
     │ Int64     String             Symbol     Float64    Float64        Float64       Float64       Float64          Float64         Float64         Float64        Float64         
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 14035821  dumbarton          over_15         1.23          1.245         1.145         1.383            1.189           1.111           1.295           0.0              6.81  w
   2 │ 14035821  dumbarton          over_25         1.67          1.693         1.424         2.099            1.629           1.384           1.995           0.0              0.64  w
   3 │ 14035826  edinburgh-city-fc  over_15         1.22          1.243         1.145         1.379            1.188           1.112           1.291           0.0              4.89  w
   4 │ 14035826  edinburgh-city-fc  over_25         1.66          1.688         1.427         2.093            1.624           1.386           1.989           0.0              0.51  f
   5 │ 14035824  elgin-city         over_15         1.24          1.284         1.172         1.435            1.219           1.132           1.335           0.0              1.86  w
   6 │ 14035823  forfar-athletic    over_15         1.24          1.304         1.17          1.488            1.235           1.131           1.375           0.0              0.11  f
   7 │ 14035642  east-fife          over_15         1.27          1.291         1.17          1.46             1.225           1.131           1.353           0.0              5.98  w
   8 │ 14035642  east-fife          over_25         1.8           1.822         1.492         2.334            1.745           1.446           2.207           0.0              0.72  w
   9 │ 14035643  kelty-hearts-fc    over_15         1.3           1.301         1.181         1.462            1.232           1.139           1.355           0.0             10.32  w
  10 │ 14035643  kelty-hearts-fc    over_25         1.92          1.851         1.528         2.338            1.772           1.478           2.211           0.82             3.72  w


3 mins to game start 


julia> paper_bets_df = generate_paper_bets(target_ppd, calibrated_ppd, todays_matches, live_market_closing, min_edge=0.0)
============================================================================================================
 📝 PAPER TRADING BOARD: EXACT KELLY STAKES (Edge: 0.0%)
============================================================================================================

MATCH: DUMBARTON
  over_15  (Live: 1.21) | Raw: 1.25 [1.14 - 1.38] | Calib: 1.19 [1.11 - 1.29] | Raw Stake:  0.00% | Calib:  2.31%
  over_25  (Live: 1.65) | Raw: 1.69 [1.42 - 2.10] | Calib: 1.63 [1.38 - 2.00] | Raw Stake:  0.00% | Calib:  0.19%

MATCH: ELGIN-CITY
  over_15  (Live: 1.22) | Raw: 1.28 [1.17 - 1.44] | Calib: 1.22 [1.13 - 1.33] | Raw Stake:  0.00% | Calib:  0.01%

MATCH: EAST-FIFE
  over_15  (Live: 1.26) | Raw: 1.29 [1.17 - 1.46] | Calib: 1.22 [1.13 - 1.35] | Raw Stake:  0.00% | Calib:  4.05%
  over_25  (Live: 1.76) | Raw: 1.82 [1.49 - 2.33] | Calib: 1.75 [1.45 - 2.21] | Raw Stake:  0.00% | Calib:  0.06%

MATCH: KELTY-HEARTS-FC
  over_15  (Live: 1.29) | Raw: 1.30 [1.18 - 1.46] | Calib: 1.23 [1.14 - 1.35] | Raw Stake:  0.00% | Calib:  8.29%
  over_25  (Live: 1.89) | Raw: 1.85 [1.53 - 2.34] | Calib: 1.77 [1.48 - 2.21] | Raw Stake:  0.29% | Calib:  2.60%

============================================================================================================
7×12 DataFrame
 Row │ match_id  home_team        selection  live_odds  raw_mean_odds  raw_bid_odds  raw_ask_odds  calib_mean_odds  calib_bid_odds  calib_ask_odds  raw_stake_pct  calib_stake_pct 
     │ Int64     String           Symbol     Float64    Float64        Float64       Float64       Float64          Float64         Float64         Float64        Float64         
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 14035821  dumbarton        over_15         1.21          1.245         1.145         1.383            1.189           1.111           1.295           0.0              2.31  w
   2 │ 14035821  dumbarton        over_25         1.65          1.693         1.424         2.099            1.629           1.384           1.995           0.0              0.19  w
   3 │ 14035824  elgin-city       over_15         1.22          1.284         1.172         1.435            1.219           1.132           1.335           0.0              0.01  w
   4 │ 14035642  east-fife        over_15         1.26          1.291         1.17          1.46             1.225           1.131           1.353           0.0              4.05  w
   5 │ 14035642  east-fife        over_25         1.76          1.822         1.492         2.334            1.745           1.446           2.207           0.0              0.06  w
   6 │ 14035643  kelty-hearts-fc  over_15         1.29          1.301         1.181         1.462            1.232           1.139           1.355           0.0              8.29  w
   7 │ 14035643  kelty-hearts-fc  over_25         1.89          1.851         1.528         2.338            1.772           1.478           2.211           0.29             2.6   w




Match,      Market,Result (Goals),1-Hour Odds,1-Hour Stake,1-Hour PnL,3-Min Odds,3-Min Stake,3-Min PnL
Dumbarton,  Over 1.5,2-1 (3) ✅,1.23,6.81u,+1.57u,1.21,2.31u,+0.49u
Dumbarton,  Over 2.5,2-1 (3) ✅,1.67,0.64u,+0.43u,1.65,0.19u,+0.12u
Edinburgh City,Over 1.5,0-2 (2) ✅,1.22,4.89u,+1.08u,No Bet,0.00u,0.00u
Edinburgh City,Over 2.5,0-2 (2) ❌,1.66,0.51u,-0.51u,No Bet,0.00u,0.00u
Elgin City,Over 1.5,0-3 (3) ✅,1.24,1.86u,+0.45u,1.22,0.01u,+0.00u
Forfar Athletic,Over 1.5,1-0 (1) ❌,1.24,0.11u,-0.11u,No Bet,0.00u,0.00u
East Fife,Over 1.5,2-1 (3) ✅,1.27,5.98u,+1.61u,1.26,4.05u,+1.05u
East Fife,Over 2.5,2-1 (3) ✅,1.80,0.72u,+0.58u,1.76,0.06u,+0.05u
Kelty Hearts,Over 1.5,3-1 (4) ✅,1.30,10.32u,+3.10u,1.29,8.29u,+2.40u
Kelty Hearts,Over 2.5,3-1 (4) ✅,1.92,3.72u,+3.42u,1.89,2.60u,+2.31u




📊 EXECUTIVE SUMMARY: System Performance

SCENARIO A: Betting 1 Hour Before Kickoff

    Total Staked: 35.56 Units

    Win/Loss Record: 8 Wins, 2 Losses

    Gross Profit: +12.24 Units

    Gross Losses: -0.62 Units

    Net PnL: +11.62 Units

    ROI (Yield): 32.68%

SCENARIO B: Betting 3 Minutes Before Kickoff (Closing Line)

    Total Staked: 17.51 Units

    Win/Loss Record: 7 Wins, 0 Losses

    Gross Profit: +6.42 Units

    Gross Losses: 0.00 Units

    Net PnL: +6.42 Units

    ROI (Yield): 36.66%
=#
