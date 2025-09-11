# scripts/evaluate.jl

using BayesianFootball
using DataFrames, JLD2, CSV, JSON

# --- 1. SETUP ---
const EXPERIMENT_PATH = "/home/james/bet_project/models_julia/experiments/scottish_league_initial_test"
# This is the specific model run folder you want to analyze.
# You would change this name to evaluate a different model.
const RUN_NAME = "maher_basic_20250911-164135" # UPDATE THIS with a real folder name
const RUN_PATH = joinpath(EXPERIMENT_PATH, RUN_NAME)
const DATA_PATH = "/home/james/bet_project/football_data/scot_nostats_20_to_24"

# Load core objects from the training run
println("Loading experiment run: $RUN_NAME")
if !isdir(RUN_PATH) error("Run path not found: $RUN_PATH. Please run the training script first.") end
run_data = load_run(RUN_PATH)
result = run_data.result; config = run_data.config; mapping = result.mapping

# Load full dataset and identify target matches
data_store = DataStore(DataFiles(DATA_PATH))
target_matches = filter(row -> row.season in config.cv_config.target_seasons, data_store.matches)

# Bring in our new scoring module and its dependencies
# TODO: add to src
include("/home/james/bet_project/models_julia/src/evaluation/scoring.jl")
using .Scoring

# --- 2. PRE-COMPUTATION (Two-Level Caching) ---

# Get shared, model-agnostic data
shared_cache_path = joinpath(EXPERIMENT_PATH, "shared_precomputes.jld2")
if isfile(shared_cache_path)
    println("Loading shared precomputes from cache...")
    # Load JLD2 file and convert string keys to symbols
    loaded_data = JLD2.load(shared_cache_path)
    shared_precomputes = Dict(Symbol(k) => v for (k, v) in loaded_data)
else
    println("Generating shared precomputes...")
    # Use symbols for keys
    shared_data = Dict(
        :matches_odds => process_matches_odds(data_store, target_matches),
        :matches_results => process_matches_results(data_store, target_matches)
    )
    # Save using keyword splatting, which requires symbols
    jldsave(shared_cache_path; shared_data...)
    shared_precomputes = shared_data
end

# Get run-specific, model-dependent data
run_specific_cache_path = joinpath(RUN_PATH, "run_specific_precomputes.jld2")
if isfile(run_specific_cache_path)
    println("Loading run-specific precomputes from cache...")
    # Load JLD2 file and convert string keys to symbols
    loaded_data = JLD2.load(run_specific_cache_path)
    run_specific_precomputes = Dict(Symbol(k) => v for (k, v) in loaded_data)
else
    println("Generating run-specific precomputes (this may take a while)...")
    matches_prediction = predict_target_season(target_matches, result, mapping)
    kelly_config = BayesianFootball.Kelly.Config(0.02, 0.05)
    # Access the shared data with a symbol
    matches_kelly = process_matches_kelly(matches_prediction, shared_precomputes[:matches_odds], kelly_config)
    
    # Use symbols for keys
    run_specific_data = Dict(
        :matches_prediction => matches_prediction, 
        :matches_kelly => matches_kelly
    )
    # Save using keyword splatting
    jldsave(run_specific_cache_path; run_specific_data...)
    run_specific_precomputes = run_specific_data
end

# Combine into a single NamedTuple for easy dot-access syntax (e.g., precomputes.matches_odds)
precomputes = (; merge(shared_precomputes, run_specific_precomputes)...);

# --- 3. ANALYSIS ---
println("Running analysis and scoring...")
c_values = 0.01:0.01:0.99

financial_card = Scoring.calculate_financial_scorecard(precomputes, c_values, target_matches);
log_likelihood_card = Scoring.calculate_log_likelihood_scorecard(precomputes, c_values);

analysis_result = Scoring.AnalysisResult(
    config.name, # Use the run's unique name
    financial_card,
    log_likelihood_card
);

# Save the rich analysis object
analysis_path = joinpath(RUN_PATH, "analysis_result.jld2")
jldsave(analysis_path; analysis_result)
println("✅ Detailed analysis object saved to: $analysis_path")

# --- 4. REPORTING (Example) ---
println("\n--- Generating Example Report ---")
loaded_analysis = JLD2.load(analysis_path)["analysis_result"]

# Generate and display the financial performance DataFrame
financial_df = Scoring.PerformanceAnalytics.performance_to_dataframe(loaded_analysis.financial_scorecard)
display(financial_df)

# Display specific, nested log-likelihood scores
println("\n--- Log-Likelihood Scores ---")
ll_scores = loaded_analysis.log_likelihood_scorecard
println("Full-Time 1x2 Scores: Home=$(round(ll_scores.ft.home, digits=2)), Draw=$(round(ll_scores.ft.draw, digits=2)), Away=$(round(ll_scores.ft.away, digits=2))")
println("Full-Time O/U 2.5 Scores: Over=$(round(ll_scores.ft.over_25, digits=2)), Under=$(round(ll_scores.ft.under_25, digits=2))")

