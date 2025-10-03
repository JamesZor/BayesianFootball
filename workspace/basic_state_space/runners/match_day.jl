using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions
using CSV 


include( "/home/james/bet_project/models_julia/workspace/basic_state_space/matchday_utils_ssm.jl")
include( "/home/james/bet_project/models_julia/workspace/basic_state_space/analysis_functions.jl")
using .AnalysisSSM


include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_poisson_ha.jl")
using .AR1PoissonHA

include("/home/james/bet_project/models_julia/workspace/basic_state_space/models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA


all_model_paths = Dict(
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/ar1_model_comparison/ar1_poisson_ha_20250930-154846",
  "ssm_neg_bin" => "/home/james/bet_project/models_julia/experiments/ar1_model_comparison/ar1_neg_bin_ha_20251001-134452",
)

all_model_paths = Dict(
  "ssm_poiss" => "/home/james/bet_project/models_julia/experiments/scotland_ar1/ar1_poisson_ha_20251003-211114"
)


todays_matches = get_todays_matches(["scotland", "england"]; cli_path=CLI_PATH)

patterns_to_exclude = [" U21", r"\bB\b", " U19"]
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
println("Home Win: mean ", round(mean( 1 ./ predictions.ft.home), digits=2),"  |  median " ,  round(median( 1 ./ predictions.ft.home), digits=2))
println("away Win: mean ", round(mean( 1 ./ predictions.ft.away), digits=2),"  |  median " ,  round(median( 1 ./ predictions.ft.away), digits=2))
println("draw Win: mean ", round(mean( 1 ./ predictions.ft.draw), digits=2),"  |  median " ,  round(median( 1 ./ predictions.ft.draw), digits=2))
end

function print_1x2_ht(predictions)
println("Home Win: mean", round(mean( 1 ./ predictions.ht.home), digits=2),"  | median " ,  round(median( 1 ./ predictions.ht.home), digits=2))
println("away Win: mean", round(mean( 1 ./ predictions.ht.away), digits=2),"  | median " ,  round(median( 1 ./ predictions.ht.away), digits=2))
println("draw Win: mean", round(mean( 1 ./ predictions.ht.draw), digits=2),"  |  median" ,  round(median( 1 ./ predictions.ht.draw), digits=2))
end


function print_under(predictions)
  println("under 05, mean: ", round(mean( 1 ./ predictions.ft.under_05),digits=2), " | ", round(median( 1 ./ predictions.ft.under_05), digits=2) )
  println("under 15, mean: ", round(mean( 1 ./ predictions.ft.under_15),digits=2), " | ", round(median( 1 ./ predictions.ft.under_15), digits=2) )
  println("under 25, mean: ", round(mean( 1 ./ predictions.ft.under_25),digits=2), " | ", round(median( 1 ./ predictions.ft.under_25), digits=2) )
  println("under 35, mean: ", round(mean( 1 ./ predictions.ft.under_35),digits=2), " | ", round(median( 1 ./ predictions.ft.under_35), digits=2) )
end

function print_under_ht(predictions)
  println("under 05, mean: ", round(mean( 1 ./ predictions.ht.under_05),digits=2), " | ", round(median( 1 ./ predictions.ht.under_05), digits=2) )
  println("under 15, mean: ", round(mean( 1 ./ predictions.ht.under_15),digits=2), " | ", round(median( 1 ./ predictions.ht.under_15), digits=2) )
  println("under 25, mean: ", round(mean( 1 ./ predictions.ht.under_25),digits=2), " | ", round(median( 1 ./ predictions.ht.under_25), digits=2) )
end


##
todays_matches

matches__to_view = [11, 8, 7, 6, ]
match_to_analyze = todays_matches[matches__to_view[4], :]

m1 = loaded_models_all["ssm_poiss"]

m1 = loaded_models_all["ssm_neg_bin"]
mapping = m1.result.mapping;
chain = m1.result.chains_sequence[1];



last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

match_to_predict = DataFrame(
    home_team=match_to_analyze.home_team,
    away_team=match_to_analyze.away_team,
    tournament_id=2,
    global_round = 85, # Use the calculated next_round
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)
features = BayesianFootball.create_master_features(match_to_predict, mapping);
predictions = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);


print(match_to_analyze.event_name)
print_1x2(predictions)
print_1x2_ht(predictions)
print_under(predictions)
print_under_ht(predictions)

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

odds[: , [:ht_ou_05_under_back, :ht_ou_15_under_back, :ht_ou_25_under_back]]

density(1 ./ predictions.ft.home)
vline!(odds.ft_1x2_home_back)


density(1 ./ predictions.ft.away)
vline!(odds.ft_1x2_away_back)
match_to_analyze

mapping
todays_matches



odds_df[: , [:event_name, :ft_1x2_home_back, :ft_1x2_away_back, :ft_1x2_draw_back, :ht_1x2_home_back, :ht_1x2_away_back, :ht_1x2_draw_back, ]]

##### scotland 

m1 = loaded_models_all["ssm_poiss"]

# m1 = loaded_models_all["ssm_neg_bin"]
mapping = m1.result.mapping;
chain = m1.result.chains_sequence[1];

posterior_samples = BayesianFootball.extract_posterior_samples(
    m1.config.model_def,
    chain.ft,
    mapping
);


last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1


sc_l2 = filter(row -> row.tournament_id==57 && row.season_id==77045, df)
unique(sc_l2.home_team)
"""
scot league 2 
 "annan-athletic"
 "east-kilbride"
 "dumbarton"
 "forfar-athletic"
 "edinburgh-city-fc"
 "clyde-fc"
 "elgin-city"
 "stirling-albion"
 "the-spartans-fc"
 "stranraer"
"""

past_round =  next_round -2

team_name_home = "the-spartans-fc"
team_name_away = "stirling-albion"
league_id_to_predict = "57"

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=league_id_to_predict,
    # global_round = next_round, # Use the calculated next_round
    global_round = past_round,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)

features = BayesianFootball.create_master_features(match_to_predict, mapping)

predictions = BayesianFootball.predict(m1.config.model_def, chain, features, mapping);

print_1x2(predictions)
print_1x2_ht(predictions)

print_under(predictions)
print_under_ht(predictions)


plot_attack_defence(team_name_home, team_name_away, m1, posterior_samples)

plot_1x2(predictions)

plot_xg(predictions)
plot_xg_t(predictions)

"""
QPR v Oxford Utd
ft 
Home Win: mean2.41  |  median2.27
away Win: mean3.73  |  median3.43
draw Win: mean3.95  |  median3.87

julia> print_1x2_ht(predictions)
Home Win: mean3.75  | median 3.61
away Win: mean2.88  | median 2.81
draw Win: mean2.82  |  median2.79

julia> print_under(predictions)
under 05, mean: 14.33 | 12.59
under 15, mean: 3.88 | 3.57
under 25, mean: 1.97 | 1.88
under 35, mean: 1.38 | 1.34

julia> print_under_ht(predictions)
under 05, mean: 4.55 | 4.34
under 15, mean: 1.87 | 1.8
under 25, mean: 1.29 | 1.26



 Row │ ft_1x2_home_back  ft_1x2_away_back  ft_1x2_draw_back  ht_1x2_home_back  ht_1x2_away_back  ht_1x2_draw_back 
     │ Float64           Float64           Float64           Float64?          Float64?          Float64?         
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │             2.16              3.75               3.6              2.82               4.3               2.3
 Row │ ft_ou_05_under_back  ft_ou_15_under_back  ft_ou_25_under_back  ft_ou_35_under_back 
     │ Float64?             Float64?             Float64?             Float64?            
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │                12.0                  3.7                 1.92                 1.34
 Row │ ht_ou_05_under_back  ht_ou_15_under_back  ht_ou_25_under_back 
     │ Float64?             Float64?             Float64?            
─────┼───────────────────────────────────────────────────────────────
   1 │                 3.1                 1.43                 1.11

result:
ht: 0-0, ft: 0-0

##################################################
#
Norwich v West Brom
julia> print_1x2(predictions)
Home Win: mean 2.52  |  median 2.38
away Win: mean 3.64  |  median 3.35
draw Win: mean 3.82  |  median 3.73

julia> print_1x2_ht(predictions)
Home Win: mean4.54  | median 4.37
away Win: mean2.74  | median 2.66
draw Win: mean2.59  |  median2.58

julia> print_under(predictions)
under 05, mean: 12.89 | 11.04
under 15, mean: 3.59 | 3.25
under 25, mean: 1.87 | 1.76
under 35, mean: 1.34 | 1.29

julia> print_under_ht(predictions)
under 05, mean: 3.72 | 3.62
under 15, mean: 1.65 | 1.63
under 25, mean: 1.2 | 1.19


1×6 DataFrame
 Row │ ft_1x2_home_back  ft_1x2_away_back  ft_1x2_draw_back  ht_1x2_home_back  ht_1x2_away_back  ht_1x2_draw_back 
     │ Float64           Float64           Float64           Float64?          Float64?          Float64?         
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │              3.4              2.34              3.45               4.2              2.98              2.26
 Row │ ft_ou_05_under_back  ft_ou_15_under_back  ft_ou_25_under_back  ft_ou_35_under_back 
     │ Float64?             Float64?             Float64?             Float64?            
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │                11.5                 3.45                 1.83                 1.31
 Row │ ht_ou_05_under_back  ht_ou_15_under_back  ht_ou_25_under_back 
     │ Float64?             Float64?             Float64?            
─────┼───────────────────────────────────────────────────────────────
   1 │                3.05                 1.44                  1.1

results:
ht: 0-1, ft: 0-1

#############################################

Millwall v Coventry
julia> print_1x2(predictions)
Home Win: mean 3.08  |  median 2.87
away Win: mean 2.92  |  median 2.73
draw Win: mean 3.75  |  median 3.68

julia> print_1x2_ht(predictions)
Home Win: mean 3.34  | median 3.2
away Win: mean 3.99  | median 3.78
draw Win: mean 2.42  |  median2.4

julia> print_under(predictions)
under 05, mean: 12.49 | 10.6
under 15, mean: 3.51 | 3.17
under 25, mean: 1.85 | 1.73
under 35, mean: 1.33 | 1.28

julia> print_under_ht(predictions)
under 05, mean: 3.29 | 3.18
under 15, mean: 1.54 | 1.51
under 25, mean: 1.16 | 1.15

 Row │ ft_1x2_home_back  ft_1x2_away_back  ft_1x2_draw_back  ht_1x2_home_back  ht_1x2_away_back  ht_1x2_draw_back 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │              3.9               2.1               3.6               4.3              2.78              2.32
 Row │ ft_ou_05_under_back  ft_ou_15_under_back  ft_ou_25_under_back  ft_ou_35_under_back 
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │                12.5                 3.85                 1.94                 1.36
 Row │ ht_ou_05_under_back  ht_ou_15_under_back  ht_ou_25_under_back 
─────┼───────────────────────────────────────────────────────────────
   1 │                 3.1                 1.47                 1.11

results: 
ht 0-1, ft:0-4

##############################################
Portsmouth v Watford
julia> print_1x2(predictions)
Home Win: mean 2.6  |  median 2.47
away Win: mean 3.73  |  median 3.48
draw Win: mean 3.48  |  median 3.43

julia> print_1x2_ht(predictions)
Home Win: mean3.7  | median 3.55
away Win: mean3.04  | median 2.9
draw Win: mean2.73  |  median2.71

julia> print_under(predictions)
under 05, mean: 8.96 | 8.09
under 15, mean: 2.79 | 2.63
under 25, mean: 1.59 | 1.54
under 35, mean: 1.22 | 1.19

julia> print_under_ht(predictions)
under 05, mean: 4.24 | 4.07
under 15, mean: 1.78 | 1.74
under 25, mean: 1.25 | 1.24

 Row │ ft_1x2_home_back  ft_1x2_away_back  ft_1x2_draw_back  ht_1x2_home_back  ht_1x2_away_back  ht_1x2_draw_back 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │             2.46              3.25               3.4               3.1               4.0              2.22

 Row │ ft_ou_05_under_back  ft_ou_15_under_back  ft_ou_25_under_back  ft_ou_35_under_back 
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │                11.0                 3.35                 1.77                 1.29
 Row │ ht_ou_05_under_back  ht_ou_15_under_back  ht_ou_25_under_back 
─────┼───────────────────────────────────────────────────────────────
   1 │                2.92                 1.41                  1.1

result: 
ht: 1-0, ft: 2-2 

"""
