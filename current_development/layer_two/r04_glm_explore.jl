using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


# files to include
include("l01_calib_utils.jl")
include("data_pipeline.jl")
include("runner.jl")

include("./models/logit_shift.jl") 
include("./models/smart_glm.jl")


include("./features/l2_feature_struct_interface.jl")
include("./features/feature_builders.jl")

# 1. Load DataStore & L1 Experiment (Using your existing code structure)
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
exp_dir = "exp/ablation_study"


saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[6])



function build_l2_training_df(exp_results, ds)
    println("1. Extracting L1 Predictions & Latents...")
    
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_results)
    ppd_df = BayesianFootball.Predictions.model_inference(latents)

    # 1A. Get mean probability from distributions
    # ppd_df.df.raw_prob = [mean(dist) for dist in ppd_df.df.distribution]
    ppd_df.df.raw_prob = [median(dist) for dist in ppd_df.df.distribution]
    
    # 1B. Extract the dynamic L1 states (Alpha/Beta)
    # l1_features = select(latents.df, :match_id, :home_alpha, :away_alpha, :home_beta, :away_beta)
    l1_features = select(latents.df, :match_id)

    println("2. Building L2 Rolling Features...")
    
    # 2A. Only select the columns that currently exist in the datastore
    matches_df = ds.matches[:, [
        :match_id, :match_date, :season, :match_month, 
        :home_score, :away_score, :home_team, :away_team
    ]]
    
    # 2B. Generate the split_id dynamically on the new dataframe
    matches_df.split_id = [string(r.season, "-", lpad(r.match_month, 2, "0")) for r in eachrow(matches_df)]

    # 2C. The Feature Pipeline
    pipeline = [
        SeasonProgress(),
        RollingForm(window_size = 3),
        RollingForm(window_size = 7)
    ]
    
    # enriched_matches now contains EVERYTHING from matches_df PLUS the new features
    enriched_matches = build_all_features(pipeline, matches_df)

    println("3. Fetching Market Data...")
    
    # CRITICAL: This is the line that went missing!
    raw_odds = ds.odds[:, [:match_id, :market_name, :market_line, :selection, :odds_close]]

    # Deduplicate the odds (Take the Maximum odds / Best Market Price)
    odds_df = combine(
        groupby(raw_odds, [:match_id, :market_name, :market_line, :selection]),
        :odds_close => maximum => :odds_close
    )

    # ==========================================
    # 4. THE GRAND JOIN 
    # ==========================================
    println("4. Consolidating Final Training Set...")
    
    # A. Base Predictions + L1 Latent Features
    df_base = innerjoin(ppd_df.df, l1_features, on=:match_id)
    
    # B. Add the Context & Form Features (Prevents duplicate columns)
    df_context = innerjoin(df_base, enriched_matches, on=:match_id)
    
    # C. Lock it to the Betting Market Lines
    final_df = innerjoin(df_context, odds_df, on=[:match_id, :market_name, :market_line, :selection])

    # D. Resolve Targets
    final_df.outcome_hit = [resolve_outcome(r) for r in eachrow(final_df)]
    
    return final_df
end



l2_data = build_l2_training_df(exp, ds);

println(first(l2_data, 5))


# A/B Test
# 1. Test the "Dumb" Baseline
config_dumb = CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = PureLogitShift(), 
    target_markets = [:over_15, :over_25, :over_35], 
    min_history_splits = 8,   
    max_history_splits = 0,   
    time_decay_half_life = nothing 
)
results_dumb = run_calibration_backtest(l2_data, config_dumb);

# 2. Test the "Smart" GLM
config_smart = CalibrationConfig(
    name = "Smart_GLM",
    model = SmartGLMCalibrator(
        features = [:market_line, :season_progress, :form_points_diff_7]
    ), 
    target_markets = [:over_15], 
    min_history_splits = 8,   
    max_history_splits = 0,   
    time_decay_half_life = nothing 
)
results_smart = run_calibration_backtest(l2_data, config_smart)




quick_analysis(results_dumb)
quick_analysis(results_smart)


#=
ppd_df.df.raw_prob = [median(dist) for dist in ppd_df.df.distribution]
julia> quick_analysis(results_dumb)

=== Calibration Results ===
Raw L1 LogLoss:    0.5237
Calib L2 LogLoss:  0.5199
Improvement:       0.0038

=== Brier Score Results ===
Raw L1 Brier:    0.1705
Calib L2 Brier:  0.1688
Improvement:     0.0016
=#


#=
# Old: Using Expected Value (Mean)
ppd_df.df.raw_prob = [mean(dist) for dist in ppd_df.df.distribution]
julia> quick_analysis(results_dumb)

=== Calibration Results ===
Raw L1 LogLoss:    0.5241
Calib L2 LogLoss:  0.5197
Improvement:       0.0044

=== Brier Score Results ===
Raw L1 Brier:    0.1706
Calib L2 Brier:  0.1687
Improvement:     0.0019

julia> quick_analysis(results_smart)

=== Calibration Results ===
Raw L1 LogLoss:    0.5241
Calib L2 LogLoss:  0.5228
Improvement:       0.0013

=== Brier Score Results ===
Raw L1 Brier:    0.1706
Calib L2 Brier:  0.1697
Improvement:     0.001
=#


function compare_raw_and_cali(ds, exp, results, min_edge)
    raw_ppd, calib_ppd = get_ppd_for_raw_and_calib(ds, exp, results);
    signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

    raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd, ds.odds, signals; odds_column=:odds_close);

    calib_sig_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, signals; odds_column=:odds_close);

    display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)

    println("calib_ppd")
    display_edge_threshold_analysis(calib_ppd, ds)

    println("raw_ppd")
    display_edge_threshold_analysis(raw_ppd, ds)
end

compare_raw_and_cali(ds,exp, results_dumb, 0.0)




#=
 monthly R model, over_15, over_25, over_35

Running Inference on 1226 matches...
Raw PPD rows:   2703
Calib PPD rows: 2703

=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===

[RAW L1 MODEL]
  Seen Markets: 2703
  Bets Placed:  505 (Active Rate: 18.68%)
  Win Rate:     47.72%
  Total Stake:  13.91 units
  Total PnL:    +1.46 units
  ROI:          +10.48%

[CALIBRATED L2 MODEL]
  Seen Markets: 2703
  Bets Placed:  1089 (Active Rate: 40.29%)
  Win Rate:     52.16%
  Total Stake:  47.35 units
  Total PnL:    +1.67 units
  ROI:          +3.53%
calib_ppd

=== Model: Edge Threshold Analysis ===
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%
---------------------------------------------------------
  0.0 | 1089 |   40.29 |  52.16 |  47.35 | +1.67 | +3.53%
  1.0 |  895 |   33.11 |  52.07 |  47.08 | +1.68 | +3.57%
  2.0 |  707 |   26.16 |  49.93 |  45.06 | +1.57 | +3.49%
  3.0 |  566 |   20.94 |  49.29 |  42.02 | +1.65 | +3.93%
  4.0 |  426 |   15.76 |  49.06 |  36.67 | +2.28 | +6.21%
  5.0 |  304 |   11.25 |  47.37 |  29.44 | +2.18 | +7.40%
  6.0 |  230 |    8.51 |  47.39 |  25.15 | +2.07 | +8.24%
  7.0 |  169 |    6.25 |  47.34 |  20.20 | +2.16 | +10.69%
  8.0 |  122 |    4.51 |  49.18 |  16.15 | +2.20 | +13.59%
  9.0 |   85 |    3.14 |  45.88 |  12.60 | +0.98 | +7.74%
 10.0 |   59 |    2.18 |  40.68 |   8.97 | +0.11 | +1.20%
raw_ppd

=== Model: Edge Threshold Analysis ===
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%
---------------------------------------------------------
  0.0 |  505 |   18.68 |  47.72 |  13.91 | +1.46 | +10.48%
  1.0 |  401 |   14.84 |  47.63 |  13.82 | +1.46 | +10.57%
  2.0 |  286 |   10.58 |  47.20 |  13.01 | +1.46 | +11.23%
  3.0 |  225 |    8.32 |  48.44 |  12.21 | +1.57 | +12.82%
  4.0 |  160 |    5.92 |  48.75 |  10.50 | +1.46 | +13.92%
  5.0 |  109 |    4.03 |  49.54 |   8.75 | +1.42 | +16.21%
  6.0 |   82 |    3.03 |  48.78 |   7.35 | +1.21 | +16.44%
  7.0 |   54 |    2.00 |  46.30 |   5.52 | +0.71 | +12.94%
  8.0 |   42 |    1.55 |  47.62 |   4.59 | +0.82 | +17.77%
  9.0 |   27 |    1.00 |  48.15 |   3.14 | +0.70 | +22.36%
 10.0 |   13 |    0.48 |  53.85 |   1.77 | +0.43 | +24.57%
=#



#=
 monthly R model, over_15, over_25
Running Inference on 1226 matches...
Raw PPD rows:   1805
Calib PPD rows: 1805

=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===

[RAW L1 MODEL]
  Seen Markets: 1805
  Bets Placed:  283 (Active Rate: 15.68%)
  Win Rate:     62.90%
  Total Stake:  9.18 units
  Total PnL:    +1.41 units
  ROI:          +15.37%

[CALIBRATED L2 MODEL]
  Seen Markets: 1805
  Bets Placed:  680 (Active Rate: 37.67%)
  Win Rate:     64.71%
  Total Stake:  34.39 units
  Total PnL:    +1.95 units
  ROI:          +5.68%
calib_ppd

=== Model: Edge Threshold Analysis ===
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%
---------------------------------------------------------
  0.0 |  680 |   37.67 |  64.71 |  34.39 | +1.95 | +5.68%
  1.0 |  548 |   30.36 |  65.33 |  34.15 | +1.96 | +5.74%
  2.0 |  420 |   23.27 |  63.57 |  32.34 | +1.89 | +5.85%
  3.0 |  325 |   18.01 |  63.38 |  29.70 | +1.94 | +6.54%
  4.0 |  233 |   12.91 |  63.52 |  25.07 | +2.33 | +9.28%
  5.0 |  155 |    8.59 |  61.94 |  18.91 | +2.19 | +11.58%
  6.0 |  113 |    6.26 |  62.83 |  15.71 | +2.08 | +13.22%
  7.0 |   81 |    4.49 |  62.96 |  12.06 | +2.09 | +17.33%
  8.0 |   55 |    3.05 |  69.09 |   9.20 | +2.31 | +25.16%
  9.0 |   39 |    2.16 |  61.54 |   7.14 | +1.22 | +17.14%
 10.0 |   24 |    1.33 |  62.50 |   4.49 | +1.04 | +23.07%
raw_ppd

=== Model: Edge Threshold Analysis ===
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%
---------------------------------------------------------
  0.0 |  283 |   15.68 |  62.90 |   9.18 | +1.41 | +15.37%
  1.0 |  217 |   12.02 |  62.67 |   9.10 | +1.41 | +15.47%
  2.0 |  149 |    8.25 |  59.06 |   8.44 | +1.33 | +15.75%
  3.0 |  120 |    6.65 |  61.67 |   7.89 | +1.45 | +18.34%
  4.0 |   83 |    4.60 |  63.86 |   6.59 | +1.45 | +22.02%
  5.0 |   52 |    2.88 |  69.23 |   5.27 | +1.47 | +27.98%
  6.0 |   39 |    2.16 |  69.23 |   4.37 | +1.25 | +28.69%
  7.0 |   26 |    1.44 |  69.23 |   3.21 | +1.01 | +31.49%
  8.0 |   19 |    1.05 |  68.42 |   2.61 | +0.79 | +30.15%
  9.0 |   13 |    0.72 |  61.54 |   1.82 | +0.39 | +21.56%
 10.0 |    8 |    0.44 |  62.50 |   1.20 | +0.30 | +24.76%
=#





#=
# New: Using the 50th Percentile Consensus (Median)
ppd_df.df.raw_prob = [median(dist) for dist in ppd_df.df.distribution]

Running Inference on 1226 matches...
Raw PPD rows:   898
Calib PPD rows: 898

=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===

[RAW L1 MODEL]
  Seen Markets: 898
  Bets Placed:  79 (Active Rate: 8.80%)
  Win Rate:     84.81%
  Total Stake:  2.67 units
  Total PnL:    +0.41 units
  ROI:          +15.45%

[CALIBRATED L2 MODEL]
  Seen Markets: 898
  Bets Placed:  320 (Active Rate: 35.63%)
  Win Rate:     77.81%
  Total Stake:  18.69 units
  Total PnL:    +0.76 units
  ROI:          +4.08%
calib_ppd

=== Model: Edge Threshold Analysis ===
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%
---------------------------------------------------------
  0.0 |  320 |   35.63 |  77.81 |  18.69 | +0.76 | +4.08%
  1.0 |  244 |   27.17 |  79.10 |  18.48 | +0.76 | +4.12%
  2.0 |  167 |   18.60 |  77.84 |  16.94 | +0.69 | +4.05%
  3.0 |  119 |   13.25 |  78.15 |  14.92 | +0.72 | +4.83%
  4.0 |   71 |    7.91 |  84.51 |  11.35 | +1.17 | +10.30%
  5.0 |   32 |    3.56 |  84.38 |   6.57 | +0.93 | +14.12%
  6.0 |   22 |    2.45 |  81.82 |   4.99 | +0.64 | +12.88%
  7.0 |   12 |    1.34 |  91.67 |   2.98 | +0.81 | +27.19%
  8.0 |    8 |    0.89 | 100.00 |   2.11 | +0.78 | +37.12%
  9.0 |    6 |    0.67 | 100.00 |   1.64 | +0.61 | +37.41%
 10.0 |    1 |    0.11 | 100.00 |   0.36 | +0.13 | +36.36%
raw_ppd

=== Model: Edge Threshold Analysis ===
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%
---------------------------------------------------------
  0.0 |   79 |    8.80 |  84.81 |   2.67 | +0.41 | +15.45%
  1.0 |   53 |    5.90 |  84.91 |   2.62 | +0.41 | +15.61%
  2.0 |   26 |    2.90 |  80.77 |   2.16 | +0.35 | +16.12%
  3.0 |   18 |    2.00 |  88.89 |   1.85 | +0.40 | +21.55%
  4.0 |    9 |    1.00 | 100.00 |   1.14 | +0.39 | +34.66%
  5.0 |    6 |    0.67 | 100.00 |   0.90 | +0.31 | +34.55%
  6.0 |    3 |    0.33 | 100.00 |   0.54 | +0.19 | +35.47%
  7.0 |    1 |    0.11 | 100.00 |   0.24 | +0.09 | +36.36%
  8.0 |    1 |    0.11 | 100.00 |   0.24 | +0.09 | +36.36%
  9.0 |    0 |    0.00 |   0.00 |   0.00 | +0.00 | +0.00%
 10.0 |    0 |    0.00 |   0.00 |   0.00 | +0.00 | +0.00%
=#


#=
# Old: Using Expected Value (Mean)
ppd_df.df.raw_prob = [mean(dist) for dist in ppd_df.df.distribution]
Running Inference on 1226 matches...                                                                                                                                                                                                                                                                                        
Raw PPD rows:   898                                                                                                                                                                                                                                                                                                         
Calib PPD rows: 898                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                            
=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                            
[RAW L1 MODEL]                                                                                                                                                                                                                                                                                                              
  Seen Markets: 898                                                                                                                                                                                                                                                                                                         
  Bets Placed:  79 (Active Rate: 8.80%)                                                                                                                                                                                                                                                                                     
  Win Rate:     84.81%                                                                                                                                                                                                                                                                                                      
  Total Stake:  2.67 units                                                                                                                                                                                                                                                                                                  
  Total PnL:    +0.41 units                                                                                                                                                                                                                                                                                                 
  ROI:          +15.45%                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                            
[CALIBRATED L2 MODEL]                                                                                                                                                                                                                                                                                                       
  Seen Markets: 898                                                                                                                                                                                                                                                                                                         
  Bets Placed:  343 (Active Rate: 38.20%)                                                                                                                                                                                                                                                                                   
  Win Rate:     76.68%                                                                                                                                                                                                                                                                                                      
  Total Stake:  21.11 units                                                                                                                                                                                                                                                                                                 
  Total PnL:    +0.78 units                                                                                                                                                                                                                                                                                                 
  ROI:          +3.69%                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                            
=== Model: Edge Threshold Analysis ===                                                                                                                                                                                                                                                                                      
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%                                                                                                                                                                                                                                                                     
---------------------------------------------------------                                                                                                                                                                                                                                                                   
  0.0 |  343 |   38.20 |  76.68 |  21.11 | +0.78 | +3.69%                                                                                                                                                                                                                                                                   
  1.0 |  268 |   29.84 |  78.36 |  20.92 | +0.78 | +3.75%                                                                                                                                                                                                                                                                   
  2.0 |  193 |   21.49 |  80.31 |  19.52 | +0.89 | +4.54%                                                                                                                                                                                                                                                                   
  3.0 |  125 |   13.92 |  77.60 |  16.54 | +0.64 | +3.87%
  4.0 |   83 |    9.24 |  81.93 |  13.26 | +1.09 | +8.19%
  5.0 |   43 |    4.79 |  83.72 |   8.49 | +1.01 | +11.87%
  6.0 |   23 |    2.56 |  82.61 |   5.51 | +0.72 | +13.10%
  7.0 |   14 |    1.56 |  92.86 |   3.58 | +0.98 | +27.32%
  8.0 |    8 |    0.89 | 100.00 |   2.19 | +0.81 | +37.13%
  9.0 |    6 |    0.67 | 100.00 |   1.70 | +0.64 | +37.42%
 10.0 |    3 |    0.33 | 100.00 |   0.88 | +0.34 | +38.46%

=== Model: Edge Threshold Analysis ===
Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%
---------------------------------------------------------
  0.0 |   79 |    8.80 |  84.81 |   2.67 | +0.41 | +15.45%
  1.0 |   53 |    5.90 |  84.91 |   2.62 | +0.41 | +15.61%
  2.0 |   26 |    2.90 |  80.77 |   2.16 | +0.35 | +16.12%
  3.0 |   18 |    2.00 |  88.89 |   1.85 | +0.40 | +21.55%
  4.0 |    9 |    1.00 | 100.00 |   1.14 | +0.39 | +34.66%
  5.0 |    6 |    0.67 | 100.00 |   0.90 | +0.31 | +34.55%
  6.0 |    3 |    0.33 | 100.00 |   0.54 | +0.19 | +35.47%
  7.0 |    1 |    0.11 | 100.00 |   0.24 | +0.09 | +36.36%
  8.0 |    1 |    0.11 | 100.00 |   0.24 | +0.09 | +36.36%
  9.0 |    0 |    0.00 |   0.00 |   0.00 | +0.00 | +0.00%
 10.0 |    0 |    0.00 |   0.00 |   0.00 | +0.00 | +0.00%
=#




compare_raw_and_cali(ds,exp, results_smart, 0.0)
