using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions
using CSV 


include( "/home/james/bet_project/models_julia/workspace/basic_state_space/matchday_utils_ssm.jl")
include( "/home/james/bet_project/models_julia/workspace/basic_state_space/analysis_functions.jl")
using .AnalysisSSM

all_model_paths = Dict(
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/ar1_model_comparison/ar1_poisson_ha_20250930-154846",
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


save_odds_to_csv(odds_df, "data/")



loaded_models_all = load_models_from_paths(all_model_paths)

##

function print_1x2(predictions)
println("Home Win: ", round(mean( 1 ./ predictions.ft.home), digits=2),"  |  " ,  round(median( 1 ./ predictions.ft.home), digits=2))
println("away Win: ", round(mean( 1 ./ predictions.ft.away), digits=2),"  |  " ,  round(median( 1 ./ predictions.ft.away), digits=2))
println("draw Win: ", round(mean( 1 ./ predictions.ft.draw), digits=2),"  |  " ,  round(median( 1 ./ predictions.ft.draw), digits=2))
end


function print_under(predictions)
  println("under 05, mean: ", round(mean( 1 ./ predictions.ft.under_05),digits=2), " | ", round(median( 1 ./ predictions.ft.under_05), digits=2) )
  println("under 15, mean: ", round(mean( 1 ./ predictions.ft.under_15),digits=2), " | ", round(median( 1 ./ predictions.ft.under_15), digits=2) )
  println("under 25, mean: ", round(mean( 1 ./ predictions.ft.under_25),digits=2), " | ", round(median( 1 ./ predictions.ft.under_25), digits=2) )
  println("under 35, mean: ", round(mean( 1 ./ predictions.ft.under_35),digits=2), " | ", round(median( 1 ./ predictions.ft.under_35), digits=2) )
end


##
todays_matches
match_to_analyze = todays_matches[15, :]

m1 = loaded_models_all["ssm_poiss"]
mapping = m1.result.mapping
chain = m1.result.chains_sequence[1]



last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

match_to_predict = DataFrame(
    home_team=match_to_analyze.home_team,
    away_team=match_to_analyze.away_team,
    tournament_id=2,
    global_round = next_round, # Use the calculated next_round
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)



features = BayesianFootball.create_master_features(match_to_predict, mapping);

predictions = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);

println("Home Win: ", round(mean( 1 ./ predictions.ht.home), digits=2))
println("Away Win: ", round(mean( 1 ./ predictions.ht.away), digits=2))
println("Draw:     ", round(mean( 1 ./ predictions.ht.draw), digits=2))

println("Home Win: ", round(median( 1 ./ predictions.ht.home), digits=2))
println("Away Win: ", round(median( 1 ./ predictions.ht.away), digits=2))
println("Draw:     ", round(median( 1 ./ predictions.ht.draw), digits=2))



print_1x2(predictions)
print_under(predictions)

"""
mean 
Home Win: 2.21
Away Win: 4.36
Draw:     4.19
# ft medain
Home Win: 2.09
Away Win: 3.86
Draw:     4.01

# ht
Home Win: 3.68
Away Win: 3.27
Draw:     2.58


"""

odds = filter(row -> row.event_name==match_to_analyze.event_name, odds_df);

odds[: , [:ft_1x2_home_back, :ft_1x2_away_back, :ft_1x2_draw_back, :ht_1x2_home_back, :ht_1x2_away_back, :ht_1x2_draw_back, ]]

odds[: , [:ft_ou_05_under_back, :ft_ou_15_under_back, :ft_ou_25_under_back, :ft_ou_35_under_back]]

density(1 ./ predictions.ft.home)
vline!(odds.ft_1x2_home_back)


density(1 ./ predictions.ft.away)
vline!(odds.ft_1x2_away_back)
match_to_analyze

mapping
todays_matches



odds_df[: , [:event_name, :ft_1x2_home_back, :ft_1x2_away_back, :ft_1x2_draw_back, :ht_1x2_home_back, :ht_1x2_away_back, :ht_1x2_draw_back, ]]
