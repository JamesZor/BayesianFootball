using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions
using CSV 


include("/home/james/bet_project/models_julia/workspace/basic_state_space/setup.jl")
# include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/match_day_utils.jl")

include("/home/james/bet_project/models_julia/workspace/basic_state_space/prediction.jl")
# include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/analysis_funcs.jl")
include( "/home/james/bet_project/models_julia/workspace/basic_state_space/matchday_utils_ssm.jl")


include( "/home/james/bet_project/models_julia/workspace/basic_state_space/analysis_functions.jl")
using .MatchDayUtilsSSM

using .AR1NegativeBinomial
using .AR1NegBiPrediction
using .AR1StateSpace
using .AR1Prediction
# using .Analysis
using .AnalysisSSM

all_model_paths = Dict(
  "ssm_bineg" => "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_negbi_2425_to_2526_20250926-173118",
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_poisson_2425_to_2526_20250926-135921"
)



todays_matches = get_todays_matches(["scotland", "england"]; cli_path=CLI_PATH)

patterns_to_exclude = [" U21", r"\bB\b"]
# Filter out rows where ANY of the patterns are found
filter!(todays_matches) do row
    !any(p -> occursin(p, row.event_name), patterns_to_exclude)
end
todays_matches


odds_df = fetch_all_market_odds(
    todays_matches,
    MARKET_LIST;
    cli_path=CLI_PATH
)

MatchDayUtils.save_odds_to_csv(odds_df, "data/")

odds_df = CSV.read("/home/james/bet_project/models_julia/data/market_odds_2025-09-27.csv", DataFrame, header=1)

loaded_models_all = load_models_from_paths(all_model_paths)


match_to_analyze = todays_matches[1, :]

(comparison_df, prediction_matrices, market_book) = generate_match_analysis(
    match_to_analyze,
    odds_df, # Your wide DataFrame of all market odds
    loaded_models_all,
    MARKET_LIST # Use your comprehensive or specific market list here
);



