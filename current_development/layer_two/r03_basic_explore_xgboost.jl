# Include the new model

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

# Include our new Layer 2 files
include("l01_calib_utils.jl")
include("data_pipeline.jl")
include("runner.jl")

include("models/xgboost_shift.jl")
include("./features/l2_feature_struct_interface.jl")
include("./features/feature_builders.jl")



# ----- feature l2 pipe line explore  -----
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
exp_dir = "exp/ablation_study"


saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])

function build_l2_training_df(exp_results, ds)
    println("1. Extracting L1 Predictions & Latents...")
    
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_results)
    ppd_df = BayesianFootball.Predictions.model_inference(latents)

    # 1A. Get mean probability from distributions
    ppd_df.df.raw_prob = [mean(dist) for dist in ppd_df.df.distribution]
    
    # 1B. Extract the dynamic L1 states (Alpha/Beta)
    l1_features = select(latents.df, :match_id, :home_alpha, :away_alpha, :home_beta, :away_beta)

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

l2_data_old = build_l2_training_df_old(exp, ds);
size(l2_data_old)
l2_data = build_l2_training_df(exp, ds);
size(l2_data)

first(l2_data, 4)
# 1. Define your ultimate feature array
my_features = [
    :home_alpha, :away_alpha, :home_beta, :away_beta, 
    :season_progress, 
    :form_points_diff_3, :form_points_diff_7, 
    :market_line,
]

# 2. Pass it into the model
config = CalibrationConfig(
    name = "XGBoost_Latents_And_Form",
    model = XGBoostCalibratorV2(
        features = my_features, # <--- Pass them here!
        num_rounds = 100, 
        max_depth = 2, 
        eta = 0.01,
        min_child_weight = 50.0
    ), 
    target_markets = [:over_15], 
    min_history_splits = 8,   
    max_history_splits = 0,   
    time_decay_half_life = nothing 
)

# 3. Run the backtest!
results = run_calibration_backtest(l2_data, config);

quick_analysis(results)



raw_ppd, calib_ppd = get_ppd_for_raw_and_calib(ds, exp, results);

min_edge = 0.00
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd, ds.odds, signals; odds_column=:odds_close);

calib_sig_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, signals; odds_column=:odds_close);

display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)

display_edge_threshold_analysis(calib_ppd, ds)
display_edge_threshold_analysis(raw_ppd, ds)

#- ---
# 1. Access the history for your specific market
target = :over_15
history = results.fitted_models_history[target]

# 2. Grab the model from the very last temporal split (most data)
latest_split_key = sort(collect(keys(history)))[end]
latest_model = history[latest_split_key]

# 3. Generate and display the Importance Report
println("\n=== XGBoost Feature Importance (Split: $latest_split_key) ===")
# Note: XGBoost.importance returns a vector of tuples (Feature, Gain, Cover, Frequency)
importance_report = XGBoost.importance(latest_model.booster)

println("\n=== XGBoost Feature Importance (Split: $latest_split_key) ===")

for (i, feat) in enumerate(latest_model.feature_names)
    # XGBoost's C++ backend uses 0-based indexing, so we subtract 1 from Julia's enumerate index
    xgb_idx = i - 1 
    
    if haskey(importance_report, xgb_idx)
        # The value is a 1-element vector, so we grab the first element [1]
        gain = round(importance_report[xgb_idx][1], digits=4)
        println("Feature: $(rpad(feat, 20)) | Gain: $gain")
    else
        println("Feature: $(rpad(feat, 20)) | Gain: 0.0000 (Ignored by model)")
    end
end

# ---- XGBoost basic  test runner. -----

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
exp_dir = "exp/ablation_study"


saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])



# Rebuild the dataset so it has the team names
l2_data = build_l2_training_df(exp, ds)



ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
exp_dir = "exp/ablation_study"


saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])



# Rebuild the dataset so it has the team names
l2_data = build_l2_training_df(exp, ds)

# Configure for XGBoost
config = CalibrationConfig(
    name = "XGBoost_Basic_Features",
    model = XGBoostCalibrator(num_rounds=15, max_depth=2, eta=0.05), 
    target_markets = [:btts_yes], 
    min_history_splits = 8,   
    max_history_splits = 0,   
    time_decay_half_life = nothing 
)

# Run the backtest!
results = run_calibration_backtest(l2_data, config);


quick_analysis(results)

# Check the PnL exactly as before
raw_ppd, calib_ppd = get_ppd_for_raw_and_calib(ds, exp, results);

min_edge = 0.04
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd, ds.odds, signals; odds_column=:odds_close);

calib_sig_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, signals; odds_column=:odds_close);

display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)

display_edge_threshold_analysis(calib_ppd, ds)
display_edge_threshold_analysis(raw_ppd, ds)


# ------
using XGBoost

# 1. Grab the latest XGBoost model from the history
latest_split = sort(collect(keys(results.fitted_models_history[:over_25])))[end]
latest_model = results.fitted_models_history[:over_25][latest_split]

println("\n=== Feature Importance for Split: $latest_split ===")
# Generate the importance report
importance_report = XGBoost.importance(latest_model.booster)
display(importance_report)


# -------
using Statistics
using StatsFuns: logit

df = results.oos_predictions

# Calculate the Logit Shift XGBoost applied to every single match
eps = 1e-6
raw_logits = logit.(clamp.(df.raw_prob, eps, 1.0 - eps));
calib_logits = logit.(clamp.(Float64.(df.calib_prob), eps, 1.0 - eps));
implied_shifts = calib_logits .- raw_logits;

println("\n=== XGBoost Implied Logit Shifts ===")
println("Mean Shift:   ", round(mean(implied_shifts), digits=4))
println("Median Shift: ", round(median(implied_shifts), digits=4))
println("Min Shift:    ", round(minimum(implied_shifts), digits=4))
println("Max Shift:    ", round(maximum(implied_shifts), digits=4))
println("Std Dev:      ", round(std(implied_shifts), digits=4))

# Let's look at the most extreme shifts!
df.shift = implied_shifts;
extreme_shifts = sort(df, :shift, rev=true);
println("\nTop 5 Most Aggressive Positive Shifts (Model LOVES the Over here):")
display(first(extreme_shifts[:, [:match_date, :home_team, :away_team, :market_line, :raw_prob, :calib_prob, :shift, :outcome_hit]], 5))
