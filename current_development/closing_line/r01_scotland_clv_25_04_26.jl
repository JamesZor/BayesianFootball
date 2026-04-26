# current_development/closing_line/r01_scotland_clv_25_04_26


#=
Phase 1: Reconstructing the Order Book Timeline

Right now, your load_live_market_jsonl function is strictly designed to grab the best available back price. To understand market movements, we need the full picture of the order book's top layer at every timestamp.

    Action: We will write a specialized parser for this investigation that extracts four key metrics for every market selection at every timestamp:

        back_price and back_size (What we can buy, and how much liquidity is there).

        lay_price and lay_size (What we can sell, and how much liquidity is there).

    Why: We cannot measure the market's "defensiveness" without knowing the lay price. The gap between the two is our primary indicator of market confidence.

Phase 2: Defining the Quant Metrics

Once we have a wide, time-series DataFrame of the order book, we will engineer the following signals for every minute leading up to kickoff:

    The Mid-Price (The Market's "True" Belief):
    We will calculate the mid-point of the spread.
    Pmid​=2Pback​+Play​​

    This allows us to track the market's true intended price, completely stripping away the defensive tax the market makers are applying.

    Spread % (The Liquidity Indicator): We will track the spread width as a percentage. When this drops below a certain threshold (e.g., 2-3%), we know the "smart money" syndicates have entered and price discovery is complete.

    The Edge Trajectory:
    Instead of calculating your Kelly stakes once, we will run BayesianFootball.Signals.compute_stake  at every single timestamp using your static morning PPD against the moving live back_price. We will literally map out the exact minute your edge goes from 0.00% to actionable.

Phase 3: Testing the Calibration Hypothesis

You mentioned wanting to calibrate the model to the market line using your logistic regression shift to improve the activity rate (volume). This is a great idea, but we have to be extremely careful when we calibrate.

    The Trap: If you calibrate your PPD using the 10:30 AM lines, your model (calculate_single_shift ) is going to calculate a C_shift based on pure noise and defensive market maker positions.

    The Test: We will simulate calibration at different intervals (e.g., T-4 hours, T-2 hours, T-15 minutes). We want to see if applying a logit shift based on the highly liquid Closing Line Value (CLV) improves our strike rate without destroying our Expected Value (EV), compared to calibrating on the illiquid morning lines.

Phase 4: Establishing Execution Rules (The Final Output)

Ultimately, this investigation isn't just for plotting pretty charts; it's to build a programmatic execution trigger. We want to conclude this research by defining hard rules for your live runner.

For example, our research might show: "Never query the Kelly criterion unless the match is < 45 minutes to kickoff AND the Spread % is < 3.5% AND the available back_size is > £500."
=#

include("./l00_clv_main_util.jl")


json_filepath = "/root/BayesianFootball/data/raw_odds_scotland_25_04_26.jsonl"

# ====================================================================
# PHASE 1: Order Book Extraction & Time Series Parsing
# ====================================================================
#
ou25_df = parse_order_book_timeline(json_filepath, "Over/Under 2.5 Goals")

ou25_df = add_market_metrics!(ou25_df)

east_fife_o25 = filter(r -> r.home_team == "east-fife" && r.selection_sym == :over_25, ou25_df)



# phase 3
# include 
#  current_development/match_day_inference/r02_scotland_25_04_26.

# ... [Assuming you have loaded your DataStore (ds), Exp, Calib_Exp, and todays_matches] ...
target_ppd = compute_todays_matches_pdds(ds, exp, todays_matches)
# calib_ppd = compute_todays_matches_pdds(ds, exp_calib, todays_matches)
calib_ppd = calibrated_ppd
